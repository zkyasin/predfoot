"""Position evaluation on SN-GSR clips, shared by the dev evaluation and the preregistered test run.

``scripts/eval_dev.py`` (leave-one-game-out on dev), ``scripts/freeze_prereg.py`` (models fitted on all six
dev games) and ``scripts/run_test_once.py`` (the locked test games) all go through these functions, so the
three score exactly the same camera, rows and conformal construction:

- :func:`calibrate_detections`: per-frame calibration of a clip's cached detections, label-free, seeded by
  (clip, frame); :func:`attach_labels` then joins the labels (filtered person rows, trusted line oracle,
  health features, line reprojection);
- :func:`smooth_outputs`: the output cameras (per-frame, PTZ projection or PTZ smoother, with noise parameters
  fitted per fold, on all clips, or given);
- :func:`player_rows` and :func:`frame_rows_out`: errors against the references and covariances per player,
  and what the output camera did on each labelled frame;
- :func:`fit_regions` and :func:`logo`: grouped Mondrian conformal regions per covariance source, and the
  leave-one-game-out evaluation that uses them.

References: ``lab``, the SN-GSR raw labels (``x_raw``, ``y_raw``); ``or``, positions from a homography fitted
to the annotated lines of the same frame, kept where it fits those lines within 3 px; ``sm``, the smoothed
labels (``x``, ``y``), for temporal metrics only.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from predfoot.calib import temporal as tm
from predfoot.calib.lines import line_oracle, line_reprojection_px
from predfoot.calib.observations import observations_from_detections
from predfoot.calib.solver import SolverConfig, calibrate
from predfoot.contracts import CalibResult, CalibStatus
from predfoot.geometry import homography as hg
from predfoot.geometry.pitch import pnlcalib_keypoints
from predfoot.health.features import calibration_features, calibration_jump_px
from predfoot.keypoints.cache import cache_path, cached_detections
from predfoot.paths import data_dir
from predfoot.uncertainty.conformal import MondrianConformal, grouped_quantile
from predfoot.uncertainty.delta import (
    circumscribed_radius,
    mahalanobis2,
    position_covariance,
    region_quantile,
)

KP = pnlcalib_keypoints()
ORACLE_MAX_PX = 3.0
RAW_GAP_MAX_M = 5.0
ALPHA = 0.1
NATIVE_FPS = 25.0
IMAGE_SIZE = (1920, 1080)
FAIL_LINE_PX = 10.0
FLAGS = (
    "smoothed", "repaired", "filled", "bridged", "projected", "passthrough", "duplicate", "rejected",
    "unchecked",
)  # fmt: skip
#: Mondrian bins of a covariance source; the sources in POOLED_SOURCES (few rows) get one pooled quantile.
MONDRIAN_BINS = {"mpp": 3, "n_inliers": 2}
POOLED_SOURCES = frozenset({"gap"})


def min_source_clips(alpha: float) -> int:
    """Fewest calibration clips G for which the clip-level conformal level (1 − α)(1 + 1/G) is below 1."""
    return next(g for g in range(1, 10_000) if (1 - alpha) * (1 + 1 / g) < 1)


#: ``min_source_clips(ALPHA)``, 10 at α = 0.1. A source with fewer calibration clips gets the model fitted on
#: all sources: its own quantile would be its largest score.
MIN_SOURCE_CLIPS = min_source_clips(ALPHA)


class Smooth(StrEnum):
    none = "none"
    proj = "proj"
    ptz = "ptz"


class FitScope(StrEnum):
    fold = "fold"
    all = "all"


@dataclass
class ClipCalib:
    """Per-frame calibrations of one clip and what the evaluation needs from its labels."""

    clip_id: str
    game: int
    split: str
    frames: np.ndarray  # (T,) every cached frame (native index)
    results: list[CalibResult]
    duplicate: np.ndarray  # (T,) same detections as the previous cached frame
    frame_rows: list[dict[str, Any]]  # raw calibration: health features, oracle, line_px (labelled frames)
    persons_by_frame: dict[int, pl.DataFrame] = field(default_factory=dict)  # filtered person rows
    oracle_by_frame: dict[int, np.ndarray | None] = field(default_factory=dict)  # trusted line oracle
    lines_by_frame: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    n_corr: dict[int, int] = field(default_factory=dict)


def clip_frame_count(clip_id: str, split: str) -> int:
    """Number of frame images of a clip on disk (the key of its detection cache)."""
    return len(list((data_dir() / "sn-gsr" / split / clip_id / "img1").glob("*.jpg")))


def clip_detections(
    clip_id: str, split: str, cache_root: Path | None = None
) -> dict[int, dict[str, Any]] | None:
    """Cached detections of a clip keyed by image number, or None when the clip is not cached.

    ``cache_root`` selects another detection cache (the test run keeps its own); never runs the networks.
    """
    n = clip_frame_count(clip_id, split)
    if not cache_path(clip_id, n, root=cache_root).exists():
        return None
    clip_dir = data_dir() / "sn-gsr" / split / clip_id
    dets, _, _ = cached_detections(clip_dir, clip_id, root=cache_root)
    return dets


def clip_model_version(clip_id: str, split: str, cache_root: Path | None = None) -> str | None:
    """Model version recorded in a clip's detection cache, or None when the clip is not cached."""
    path = cache_path(clip_id, clip_frame_count(clip_id, split), root=cache_root)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        return str(z["version"])


def calibration_seed(clip_id: str, frame: int) -> int:
    """RANSAC seed of a frame (ADR-0001: seeds derive from the clip and the frame)."""
    return frame * 7919 + int(clip_id.split("-")[1])


def calibrate_detections(
    clip_id: str,
    split: str,
    game: int,
    dets: Mapping[int, dict[str, Any]],
    cached: Sequence[CalibResult] | None = None,
) -> ClipCalib:
    """Calibrate every frame of ``dets`` (seeded by frame and clip) unless ``cached`` calibrations are given.

    Label-free: the result has no label rows until :func:`attach_labels`.
    """
    frames = np.array(sorted(dets), dtype=np.int64)
    if cached is not None and len(cached) != len(frames):
        raise ValueError(f"{clip_id}: {len(cached)} saved calibrations for {len(frames)} cached frames")
    cfg = SolverConfig()
    cc = ClipCalib(clip_id, game, split, frames, [], np.zeros(len(frames), dtype=bool), [])
    cc.duplicate[:] = tm.duplicate_mask([dets[f] for f in frames.tolist()])
    for i, frame in enumerate(frames.tolist()):
        if cached is None:
            xy, uv, w, obs = observations_from_detections(dets[frame])
            rng = np.random.default_rng(calibration_seed(clip_id, frame))
            res, _ = calibrate(xy, uv, rng, weights=w, config=cfg, lines=obs)
        else:
            res = cached[i]
        cc.results.append(res)
    return cc


def attach_labels(
    cc: ClipCalib, dets: Mapping[int, dict[str, Any]], persons: pl.DataFrame, lines: pl.DataFrame
) -> ClipCalib:
    """Join one clip's labels to its per-frame calibrations: filtered person rows, the trusted line oracle,
    health features of the raw calibration and its line reprojection, for every frame with person rows."""
    p_by_frame = persons.partition_by("frame", as_dict=True)
    l_by_frame = lines.partition_by("frame", as_dict=True)
    prev_h = None
    for i, frame in enumerate(cc.frames.tolist()):
        p = p_by_frame.get((frame,))
        if p is None:
            continue
        det = dets[frame]
        xy, uv, _, obs = observations_from_detections(det)
        res = cc.results[i]
        # Ground truth from a failed raw camera (raw and smoothed labels more than 5 m apart) is dropped.
        p = p.filter(
            ~pl.col("truncated")
            & pl.col("x_raw").is_not_null()
            & (
                ((pl.col("x") - pl.col("x_raw")) ** 2 + (pl.col("y") - pl.col("y_raw")) ** 2)
                <= RAW_GAP_MAX_M**2
            )
        )
        frame_lines = {}
        lf = l_by_frame.get((frame,))
        if lf is not None:
            for (cls,), g in lf.filter(pl.col("is_planar")).partition_by("line_class", as_dict=True).items():
                if g.height >= 2:
                    frame_lines[cls] = g.sort("point_idx").select("u", "v").to_numpy()
        h_or = line_oracle(frame_lines)
        oracle_px = line_reprojection_px(h_or, frame_lines)
        trusted = h_or is not None and oracle_px <= ORACLE_MAX_PX
        evidence = np.vstack([uv.reshape(-1, 2), *[o.uv for o in obs]]) if obs else uv
        n_kp = int(sum(KP[int(k)][2] for k in det["kp_ids"]))
        feats = calibration_features(res, evidence, n_kp, len(obs), det["kp_conf"], prev_h)
        prev_h = res.H if res.ok else None
        line_px = line_reprojection_px(res.H, frame_lines) if res.ok else float("nan")
        base = {
            "clip": cc.clip_id,
            "game": cc.game,
            "frame": frame,
            "ok": res.ok,
            "n_corr": len(xy) + len(obs),
            "n_inliers": res.n_inliers,
        }
        cc.frame_rows.append(
            {**base, **feats, "line_px": line_px, "oracle_px": oracle_px, "trusted": trusted}
        )
        cc.persons_by_frame[frame] = p
        cc.oracle_by_frame[frame] = h_or if trusted else None
        cc.lines_by_frame[frame] = frame_lines
        cc.n_corr[frame] = len(xy) + len(obs)
    return cc


def calibrate_clip(
    clip_id: str,
    split: str,
    game: int,
    persons: pl.DataFrame,
    lines: pl.DataFrame,
    cached: Sequence[CalibResult] | None = None,
    cache_root: Path | None = None,
) -> ClipCalib | None:
    """Calibrate every cached frame (or reuse ``cached`` calibrations) and join the labels; None when the clip
    has no detection cache."""
    dets = clip_detections(clip_id, split, cache_root)
    if dets is None:
        return None
    cc = calibrate_detections(clip_id, split, game, dets, cached)
    return attach_labels(cc, dets, persons, lines)


def camera_speed_px_s(frames: np.ndarray, results: Sequence[CalibResult]) -> np.ndarray:
    """Image motion (px/s) of a ground grid between consecutive output cameras; NaN without both."""
    out = np.full(len(frames), np.nan)
    for i in range(1, len(frames)):
        a, b = results[i - 1], results[i]
        if a.ok and b.ok and a.H is not None and b.H is not None:
            dt = (frames[i] - frames[i - 1]) / NATIVE_FPS
            out[i] = calibration_jump_px(b.H, a.H, IMAGE_SIZE) / dt
    return out


def flag_arrays(flags: tm.FrameFlags | None, n: int) -> dict[str, np.ndarray]:
    if flags is None:
        return {name: np.zeros(n, dtype=bool) for name in FLAGS}
    return {name: np.asarray(getattr(flags, name), dtype=bool) for name in FLAGS}


def cov_sources(flags: tm.FrameFlags | None, n: int) -> np.ndarray:
    """Model behind each output covariance, so that each gets its own conformal quantiles:

    - "smoother": smoothed and repaired frames, the RTS state pooled over a segment;
    - "gap": filled and bridged frames, the state across a gap without a measurement, whose steps carry the
      gap factor λ: relative to its error, that covariance is about two orders of magnitude wider than a
      smoothed frame's;
    - "frame": one frame's own calibration, passed through or per-frame, or projected on the PTZ manifold in
      the projection ablation (κ² Σθ and A κ² Σφ Aᵀ, Σφ being Σθ projected on the manifold: the same
      single-frame information and scale).

    A duplicate takes the source of the frame it copies.
    """
    out = np.full(n, "frame", dtype=object)
    if flags is None:
        return out
    for k in range(n):
        if k and flags.duplicate[k]:
            out[k] = out[k - 1]
        elif flags.smoothed[k] or flags.repaired[k]:
            out[k] = "smoother"
        elif flags.filled[k] or flags.bridged[k]:
            out[k] = "gap"
    return out


def player_rows(
    cc: ClipCalib, results: Sequence[CalibResult], flags: tm.FrameFlags | None, fold: int | None
) -> pl.DataFrame:
    """Per-player rows (errors against the three references, covariances) of one clip's output cameras."""
    fl = flag_arrays(flags, len(cc.frames))
    source = cov_sources(flags, len(cc.frames))
    speed = camera_speed_px_s(cc.frames, results)
    blocks: dict[str, list[np.ndarray]] = {}

    def add(name: str, values: np.ndarray) -> None:
        blocks.setdefault(name, []).append(values)

    for i, frame in enumerate(cc.frames.tolist()):
        p = cc.persons_by_frame.get(frame)
        res = results[i]
        if p is None or p.height == 0 or not res.ok or res.H is None or res.cov_theta is None:
            continue
        feet = p.select("foot_u", "foot_v").to_numpy()
        gt = p.select("x_raw", "y_raw").to_numpy()
        sm = p.select("x", "y").to_numpy()
        h_or = cc.oracle_by_frame[frame]
        pred, cov, valid = position_covariance(res.H, res.cov_theta, feet)
        ref_or = hg.backproject(h_or, feet)[0] if h_or is not None else np.full_like(gt, np.nan)
        mpp = np.sqrt(np.abs(np.linalg.det(hg.jacobian_backproject_uv(res.H, feet))))
        v = np.flatnonzero(valid)
        m = len(v)
        if not m:
            continue
        add("clip", np.full(m, cc.clip_id, dtype=object))
        add("game", np.full(m, cc.game, dtype=np.int64))
        add("frame", np.full(m, frame, dtype=np.int64))
        add("ok", np.ones(m, dtype=bool))
        add("n_corr", np.full(m, cc.n_corr[frame], dtype=np.int64))
        add("n_inliers", np.full(m, res.n_inliers, dtype=np.int64))
        add("track_id", p["track_id"].to_numpy()[v].astype(np.int64))
        add("ex_lab", pred[v, 0] - gt[v, 0])
        add("ey_lab", pred[v, 1] - gt[v, 1])
        add("ex_or", pred[v, 0] - ref_or[v, 0])
        add("ey_or", pred[v, 1] - ref_or[v, 1])
        add("c00", cov[v, 0, 0])
        add("c01", cov[v, 0, 1])
        add("c11", cov[v, 1, 1])
        add("mpp", mpp[v])
        add("rms_px", np.full(m, np.nan if res.rms_px is None else res.rms_px))
        add("dof", np.full(m, np.nan if res.dof is None else res.dof))
        add("fold", np.full(m, -1 if fold is None else fold, dtype=np.int64))
        for name in FLAGS:
            add(name, np.full(m, bool(fl[name][i])))
        add("cov_source", np.full(m, source[i], dtype=object))
        add("x_pred", pred[v, 0])
        add("y_pred", pred[v, 1])
        add("ex_sm", pred[v, 0] - sm[v, 0])
        add("ey_sm", pred[v, 1] - sm[v, 1])
        add("cam_speed_px_s", np.full(m, speed[i]))
    if not blocks:
        return pl.DataFrame()
    df = pl.DataFrame({name: np.concatenate(parts) for name, parts in blocks.items()})
    df = df.with_columns(pl.col("clip").cast(pl.String), pl.col("cov_source").cast(pl.String))
    return df.with_columns(pl.when(pl.col("fold") >= 0).then(pl.col("fold")).otherwise(None).alias("fold"))


def frame_rows_out(
    cc: ClipCalib,
    results: Sequence[CalibResult],
    flags: tm.FrameFlags | None,
    diag: dict[str, np.ndarray],
    fold: int | None,
) -> list[dict[str, Any]]:
    """Today's frame rows (raw calibration) plus what the output camera did on each labelled frame."""
    fl = flag_arrays(flags, len(cc.frames))
    speed = camera_speed_px_s(cc.frames, results)
    index = {f: i for i, f in enumerate(cc.frames.tolist())}
    rows = []
    for row in cc.frame_rows:
        frame = row["frame"]
        i = index[frame]
        res, raw = results[i], cc.results[i]
        p = cc.persons_by_frame[frame]
        h_or = cc.oracle_by_frame[frame]
        e90_lab = e90_or = line_out = jump = log_sigma = np.nan
        if res.ok and res.H is not None and res.cov_theta is not None:
            line_out = line_reprojection_px(res.H, cc.lines_by_frame[frame])
            if raw.ok and raw.H is not None:
                jump = calibration_jump_px(res.H, raw.H, IMAGE_SIZE)
            centre = np.array([[IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] * 0.75]])
            _, c, ok = position_covariance(res.H, res.cov_theta, centre)
            if ok[0]:
                log_sigma = float(np.log(np.sqrt(max(np.linalg.eigvalsh(c[0])[-1], 1e-12))))
            if p.height:
                feet = p.select("foot_u", "foot_v").to_numpy()
                pred, _, valid = position_covariance(res.H, res.cov_theta, feet)
                e_lab = np.linalg.norm(pred - p.select("x_raw", "y_raw").to_numpy(), axis=1)[valid]
                e90_lab = float(np.quantile(e_lab, 0.9)) if len(e_lab) else np.nan
                if h_or is not None and len(e_lab):
                    e_or = np.linalg.norm(pred - hg.backproject(h_or, feet)[0], axis=1)[valid]
                    e90_or = float(np.quantile(e_or, 0.9))
        out = {**row, "err_p90_lab": e90_lab, "err_p90_or": e90_or, "fold": fold, "ok_out": res.ok}
        out |= {name: bool(fl[name][i]) for name in FLAGS}
        out |= {name: float(v[i]) for name, v in diag.items()}
        out |= {"line_px_out": line_out, "jump_out_raw_px": jump, "log_sigma_out": log_sigma}
        out |= {"cam_speed_px_s": float(speed[i])}
        rows.append(out)
    return rows


# --- saved calibrations ---------------------------------------------------------------------------------


def _opt(values: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
    has = np.array([v is not None for v in values], dtype=bool)
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64), has


def _result_arrays(
    clip_id: str, frames: np.ndarray, duplicate: np.ndarray, results: Sequence[CalibResult]
) -> dict[str, np.ndarray]:
    n = len(frames)
    h = np.full((n, 3, 3), np.nan)
    cov = np.full((n, 8, 8), np.nan)
    has_h, has_cov = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for i, r in enumerate(results):
        if r.H is not None:
            h[i], has_h[i] = r.H, True
        if r.cov_theta is not None:
            cov[i], has_cov[i] = r.cov_theta, True
    key = f"{clip_id}__"
    arrays = {key + "frames": frames, key + "duplicate": duplicate, key + "H": h, key + "cov": cov}
    arrays |= {key + "has_H": has_h, key + "has_cov": has_cov}
    arrays[key + "status"] = np.array([r.status.value for r in results])
    arrays[key + "n_inliers"] = np.array([r.n_inliers for r in results], dtype=np.int64)
    for name in ("s2", "rms_px", "cond", "dof"):
        arrays[key + name], arrays[key + "has_" + name] = _opt([getattr(r, name) for r in results])
    return arrays


def savez_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """``np.savez_compressed`` to a temporary file renamed over ``path``: a killed write never leaves a
    truncated archive behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)  # type: ignore[arg-type]
    tmp.replace(path)


def save_calib(path: Path, ccs: Sequence[ClipCalib], meta: dict[str, Any]) -> None:
    """Every clip's per-frame calibrations, exactly (None fields are stored with a mask)."""
    arrays: dict[str, np.ndarray] = {
        "meta": np.array(json.dumps({**meta, "clips": [c.clip_id for c in ccs]}))
    }
    for cc in ccs:
        arrays |= _result_arrays(cc.clip_id, cc.frames, cc.duplicate, cc.results)
    savez_atomic(path, arrays)


def _load_results(z: Any, clip_id: str) -> list[CalibResult]:
    key = f"{clip_id}__"
    results = []
    for i, status in enumerate(z[key + "status"].tolist()):
        opt = {
            name: float(z[key + name][i]) if z[key + "has_" + name][i] else None
            for name in ("s2", "rms_px", "cond", "dof")
        }
        results.append(
            CalibResult(
                CalibStatus(status),
                H=z[key + "H"][i].copy() if z[key + "has_H"][i] else None,
                cov_theta=z[key + "cov"][i].copy() if z[key + "has_cov"][i] else None,
                n_inliers=int(z[key + "n_inliers"][i]),
                **opt,
            )
        )
    return results


def load_calib(path: Path) -> tuple[dict[str, list[CalibResult]], dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return {clip_id: _load_results(z, clip_id) for clip_id in meta["clips"]}, meta


# --- temporal smoothing ----------------------------------------------------------------------------------


@dataclass
class Output:
    """Output cameras of one clip under one fold's parameters (fold None: no folds)."""

    cc: ClipCalib
    fold: int | None
    results: list[CalibResult]
    flags: tm.FrameFlags | None
    diag: dict[str, np.ndarray]
    track: tm.SmoothedTrack | None = None


@dataclass
class SmoothingInfo:
    fits: dict[str, dict[str, Any]] = field(default_factory=dict)
    own: dict[str, tm.CentreEstimate | None] = field(default_factory=dict)
    chosen: dict[str, tm.CentreEstimate | None] = field(default_factory=dict)
    refused: dict[str, bool] = field(default_factory=dict)  # the game centre did not explain the clip
    game: dict[int, tm.CentreEstimate | None] = field(default_factory=dict)
    meas: dict[str, tm.ClipMeasurements] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)


def _diag(meas: tm.ClipMeasurements | None, track: tm.SmoothedTrack | None, n: int) -> dict[str, np.ndarray]:
    nan = np.full(n, np.nan)
    if meas is None:
        return {}
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = meas.d2 / meas.kappa2_clip if np.isfinite(meas.kappa2_clip) else nan
    return {
        "segment": track.segment.astype(float) if track is not None else np.full(n, -1.0),
        "d2_rel": rel,
        "deletion": track.deletion if track is not None else nan,
        "nis": track.nis if track is not None else nan,
    }


def _smooth_with(
    ccs: Sequence[ClipCalib],
    params: tm.NoiseParams,
    cfg: tm.SmootherConfig,
    info: SmoothingInfo,
    fold: int | None,
) -> Iterator[Output]:
    t0 = time.perf_counter()
    for cc in ccs:
        meas = info.meas[cc.clip_id]
        if meas.accepted.any():
            track = tm.smooth_clip(meas, params, cfg)
        else:
            track = tm.empty_track(meas, params, cfg)
        res, flags = tm.to_calib_results(cc.results, meas, track, params, cfg)
        yield Output(cc, fold, res, flags, _diag(meas, track, len(cc.frames)), track)
    info.seconds[f"smooth_{fold}"] = time.perf_counter() - t0


def smooth_outputs(
    ccs: Sequence[ClipCalib],
    smooth: Smooth,
    fit_scope: FitScope,
    cfg: tm.SmootherConfig,
    info: SmoothingInfo,
    params: tm.NoiseParams | None = None,
) -> Iterator[Output]:
    """Output cameras of every clip (and every fold for the fold-wise fit); fills ``info`` on the way.

    With ``params`` (PTZ smoother only), the noise is not fitted: every clip is smoothed once with the given
    parameters (fold None), as the preregistered test run does with the frozen ones. Projection centres and
    game centres always come from ``ccs`` themselves, without labels.
    """
    if smooth is Smooth.none:
        for cc in ccs:
            yield Output(cc, None, cc.results, None, {})
        return
    t0 = time.perf_counter()
    for cc in ccs:
        info.own[cc.clip_id] = tm.estimate_centre(cc.results, cc.duplicate, cfg)
    for g in sorted({cc.game for cc in ccs}):
        info.game[g] = tm.game_centre([info.own[cc.clip_id] for cc in ccs if cc.game == g])
    for cc in ccs:
        meas, refused = tm.measure_at_centre(
            cc.clip_id, cc.frames, cc.results, cc.duplicate, info.own[cc.clip_id], info.game[cc.game], cfg
        )
        info.meas[cc.clip_id], info.chosen[cc.clip_id], info.refused[cc.clip_id] = meas, meas.centre, refused
    info.seconds["centres_and_measurements"] = time.perf_counter() - t0
    if smooth is Smooth.proj:
        for cc in ccs:
            meas = info.meas[cc.clip_id]
            res, flags = tm.projection_results(cc.results, meas, cfg)
            yield Output(cc, None, res, flags, _diag(meas, None, len(cc.frames)))
        return
    if params is not None:
        info.fits["frozen"] = {"params": params.to_dict(), "params_digest": params.digest()}
        yield from _smooth_with(ccs, params, cfg, info, None)
        return
    games = sorted({cc.game for cc in ccs})
    folds: list[int | None] = [None] if fit_scope is FitScope.all else list(games)
    for fold in folds:
        t0 = time.perf_counter()
        train = [info.meas[cc.clip_id] for cc in ccs if fold is None or cc.game != fold]
        fitted, fit = tm.fit_noise(train, cfg)
        info.seconds[f"fit_{fold}"] = time.perf_counter() - t0
        info.fits["all" if fold is None else str(fold)] = {
            "params": fitted.to_dict(),
            "params_digest": fitted.digest(),
            "training_clips": len(train),
            "training_games": [g for g in games if fold is None or g != fold],
            **fit,
        }
        yield from _smooth_with(ccs, fitted, cfg, info, fold)


# --- conformal regions -------------------------------------------------------------------------------------


def clip_bootstrap(
    hit: np.ndarray, clip_ids: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    """95 % interval of a coverage rate, resampling whole clips."""
    uniq = np.unique(clip_ids)
    idx = {c: np.flatnonzero(clip_ids == c) for c in uniq}
    stats = [
        np.mean(np.concatenate([hit[idx[c]] for c in rng.choice(uniq, size=len(uniq), replace=True)]))
        for _ in range(n_boot)
    ]
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def clip_weighted_coverage(
    hit: np.ndarray, clip_ids: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, tuple[float, float]]:
    """Coverage with every clip weighted equally (the mean of the clips' own coverage rates), as the grouped
    conformal quantile weighs its calibration clips, with a 95 % interval resampling whole clips."""
    _, inv = np.unique(clip_ids, return_inverse=True)
    per_clip = np.bincount(inv, weights=np.asarray(hit, dtype=float)) / np.bincount(inv)
    stats = [per_clip[rng.integers(0, len(per_clip), len(per_clip))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(per_clip.mean()), (float(lo), float(hi))


def score_table(d: pl.DataFrame, ref: str) -> dict[str, np.ndarray]:
    """Conformal inputs of player rows against reference ``ref``: covariances, squared Mahalanobis scores,
    error norms, clips (groups), games, Mondrian features and covariance sources."""
    e = d.select(f"ex_{ref}", f"ey_{ref}").to_numpy()
    cov = np.stack([d["c00"], d["c01"], d["c01"], d["c11"]], axis=1).reshape(-1, 2, 2)
    return {
        "cov": cov,
        "score": mahalanobis2(e, cov),
        "norm": np.linalg.norm(e, axis=1),
        "groups": d["clip"].to_numpy(),
        "games": d["game"].to_numpy(),
        "mpp": d["mpp"].to_numpy(),
        "n_inliers": d["n_inliers"].to_numpy().astype(float),
        "dof": d["dof"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        "source": d["cov_source"].to_numpy() if "cov_source" in d.columns else np.full(d.height, "frame"),
    }


FEATURE_NAMES = tuple(MONDRIAN_BINS)


@dataclass
class RegionModels:
    """Conformal regions of one camera and one reference, fitted on calibration clips.

    ``models``: Mondrian models per covariance source, key None for the model of all sources (used by a source
    without a model of its own); ``q_pool``: the pooled quantile; ``r_const``: the radius of the constant
    circle at the same level (conformal quantile of the error norms).
    """

    models: dict[str | None, MondrianConformal]
    q_pool: float
    r_const: float

    def quantile_for(self, feats: Mapping[str, np.ndarray], source: np.ndarray) -> np.ndarray:
        """q̂ of each row, from the Mondrian model of its covariance source."""
        q = np.full(len(source), np.nan)
        for src in np.unique(source):
            sel = source == src
            model = self.models.get(str(src), self.models[None])
            q[sel] = model.quantile_for({k: v[sel] for k, v in feats.items()})
        return q

    def to_dict(self) -> dict[str, Any]:
        models = {("__all__" if k is None else k): json.loads(m.to_json()) for k, m in self.models.items()}
        return {"models": dict(sorted(models.items())), "q_pool": self.q_pool, "r_const": self.r_const}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RegionModels:
        models: dict[str | None, MondrianConformal] = {
            (None if k == "__all__" else k): MondrianConformal.from_json(json.dumps(m))
            for k, m in d["models"].items()
        }
        if None not in models:
            raise ValueError("region models need the model of all sources ('__all__')")
        return cls(models=models, q_pool=float(d["q_pool"]), r_const=float(d["r_const"]))


def _pool(
    s: dict[str, np.ndarray], xs: dict[str, np.ndarray] | None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Calibration pool of the per-source models: the main rows, then the extra rows (base marks the main)."""
    pool = {k: s[k] for k in ("score", "groups", "source", "games", *FEATURE_NAMES)}
    if xs is not None:
        pool = {k: np.concatenate([v, xs[k]]) for k, v in pool.items()}
    return pool, np.arange(len(pool["score"])) < len(s["score"])


def fit_regions(
    s: dict[str, np.ndarray],
    cal: np.ndarray,
    pool: dict[str, np.ndarray],
    cal_pool: np.ndarray,
    base: np.ndarray,
    alpha: float = ALPHA,
) -> RegionModels:
    """Region models at level 1 − ``alpha`` from the calibration rows ``cal`` of ``s`` (pooled quantile,
    constant circle, model of all sources) and ``cal_pool`` of the pool (per-source models; see
    :func:`mondrian_by_source`)."""
    q_pool = grouped_quantile(s["score"][cal], s["groups"][cal], alpha)
    models = mondrian_by_source(
        pool["score"],
        pool["groups"],
        {k: pool[k] for k in FEATURE_NAMES},
        pool["source"],
        cal_pool,
        base,
        alpha=alpha,
    )
    r_const = grouped_quantile(s["norm"][cal], s["groups"][cal], alpha)
    return RegionModels(models, q_pool, r_const)


def fit_frozen_regions(df: pl.DataFrame, ref: str, extra: pl.DataFrame | None = None) -> RegionModels:
    """Region models calibrated on every row of ``df`` (and of ``extra`` for their own covariance source): the
    construction of one :func:`logo` fold with every game of ``df`` on the calibration side."""
    d = df.filter(pl.col(f"ex_{ref}").is_not_nan())
    s = score_table(d, ref)
    x = extra.filter(pl.col(f"ex_{ref}").is_not_nan()) if extra is not None else None
    xs = score_table(x, ref) if x is not None and x.height else None
    pool, base = _pool(s, xs)
    return fit_regions(s, np.ones(d.height, dtype=bool), pool, np.ones(len(base), dtype=bool), base)


def logo(
    df: pl.DataFrame,
    ref: str,
    rng: np.random.Generator,
    n_boot: int = 300,
    fold_col: str | None = None,
    extra: pl.DataFrame | None = None,
    models_out: dict[int, RegionModels] | None = None,
    alpha: float = ALPHA,
) -> tuple[list[dict[str, Any]], dict[str, Any], pl.DataFrame, pl.DataFrame | None]:
    """Leave-one-game-out coverage and size of our ellipses against a constant circle, at level 1 − ``alpha``.

    Returns one row per held-out game, a summary over all held-out predictions (each player-frame is predicted
    with quantiles fitted on the other games), including coverage by distance tercile, the held-out rows with
    their hits and Mondrian quantile (``q_mond``: the region is {e : eᵀ Σp⁻¹ e ≤ q_mond}), and ``extra`` rows
    (e.g. filled frames) scored with the same fold models. With ``fold_col``,
    fold g calibrates on rows (fold == g) & (game != g) and tests on rows (fold == g) & (game == g).
    Mondrian models are fitted within each covariance source (column ``cov_source``, see :func:`cov_sources`
    and :func:`mondrian_by_source`); the ``extra`` rows of the other games calibrate their own source and
    nothing else. The pooled quantile, the constant circle and the fallback model use the ``df`` rows only.
    ``models_out`` receives each fold's :class:`RegionModels`, keyed by held-out game.
    """
    d = df.filter(pl.col(f"ex_{ref}").is_not_nan())
    s = score_table(d, ref)
    score, norm, groups, games, cov = s["score"], s["norm"], s["groups"], s["games"], s["cov"]
    feats = {k: s[k] for k in FEATURE_NAMES}
    source = s["source"]
    raw_q = np.array([region_quantile(2, 1 - alpha, v) for v in s["dof"]])
    folds = d[fold_col].to_numpy() if fold_col else None
    hit_mond, hit_const = np.zeros(len(d), bool), np.zeros(len(d), bool)
    rad_mond, area_mond, r_const_all = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    q_mond_all = np.full(len(d), np.nan)
    held = np.zeros(len(d), bool)
    x = extra.filter(pl.col(f"ex_{ref}").is_not_nan()) if extra is not None else None
    xs = score_table(x, ref) if x is not None and x.height else None
    x_mond = np.zeros(x.height if x is not None else 0, bool)
    x_const, x_held, x_area = x_mond.copy(), x_mond.copy(), np.zeros(len(x_mond))
    x_q = np.full(len(x_mond), np.nan)
    x_folds = x[fold_col].to_numpy() if fold_col and x is not None else None
    pool, base = _pool(s, xs)
    pool_folds = None if folds is None else np.concatenate([folds, x_folds]) if x_folds is not None else folds
    rows = []
    for g in np.unique(games):
        if folds is None:
            cal, tst = games != g, games == g
            cal_pool = pool["games"] != g
        else:
            cal, tst = (folds == g) & (games != g), (folds == g) & (games == g)
            cal_pool = (pool_folds == g) & (pool["games"] != g)
        if not cal.any() or not tst.any():
            warnings.warn(f"fold {g}: empty calibration or test set, skipped", stacklevel=2)
            continue
        held |= tst
        rm = fit_regions(s, cal, pool, cal_pool, base, alpha=alpha)
        if models_out is not None:
            models_out[int(g)] = rm
        q_pool, r_const = rm.q_pool, rm.r_const
        q_mond = rm.quantile_for({k: v[tst] for k, v in feats.items()}, source[tst])
        q_mond_all[tst] = q_mond
        hit_mond[tst] = score[tst] <= q_mond
        hit_const[tst] = norm[tst] <= r_const
        rad_mond[tst] = np.sqrt(q_mond * np.linalg.eigvalsh(cov[tst])[:, -1])
        area_mond[tst] = np.pi * q_mond * np.sqrt(np.clip(np.linalg.det(cov[tst]), 0, None))
        r_const_all[tst] = r_const
        if xs is not None:
            xt = (xs["games"] == g) & ((x_folds == g) if x_folds is not None else True)
            if xt.any():
                qx = rm.quantile_for({k: xs[k][xt] for k in FEATURE_NAMES}, xs["source"][xt])
                x_mond[xt], x_const[xt] = xs["score"][xt] <= qx, xs["norm"][xt] <= r_const
                x_held[xt] = True
                x_q[xt] = qx
                x_area[xt] = np.pi * qx * np.sqrt(np.clip(np.linalg.det(xs["cov"][xt]), 0, None))
        rows.append(
            {
                "game": int(g),
                "clips": len(np.unique(groups[tst])),
                "n": int(tst.sum()),
                "cov_raw": float(np.mean(score[tst] <= raw_q[tst])),
                "cov_pool": float(np.mean(score[tst] <= q_pool)),
                "cov_mond": float(hit_mond[tst].mean()),
                "cov_mond_ci": clip_bootstrap(hit_mond[tst], groups[tst], rng, n_boot),
                "cov_const": float(hit_const[tst].mean()),
                "r_pool": float(np.median(circumscribed_radius(cov[tst], q_pool))),
                "r_mond": float(np.median(rad_mond[tst])),
                "a_mond": float(np.median(area_mond[tst])),
                "r_const": float(r_const),
                "a_const": float(np.pi * r_const**2),
            }
        )
    h = held
    terciles = np.quantile(feats["mpp"][h], [1 / 3, 2 / 3])
    bins = np.searchsorted(terciles, feats["mpp"][h])
    summary = {
        "n": int(h.sum()),
        "clips": len(np.unique(groups[h])),
        "cov_raw": float(np.mean(score[h] <= raw_q[h])),
        "cov_mond": float(hit_mond[h].mean()),
        "cov_mond_ci": clip_bootstrap(hit_mond[h], groups[h], rng, n_boot),
        "cov_const": float(hit_const[h].mean()),
        "cov_const_ci": clip_bootstrap(hit_const[h], groups[h], rng, n_boot),
        "r_mond": float(np.median(rad_mond[h])),
        "a_mond": float(np.median(area_mond[h])),
        "a_const": float(np.median(np.pi * r_const_all[h] ** 2)),
        "by_tercile": [
            (float(hit_mond[h][bins == b].mean()), float(hit_const[h][bins == b].mean())) for b in range(3)
        ],
        "tercile_edges": terciles.tolist(),
    }
    held_df = d.with_columns(
        pl.Series("hit_mond", hit_mond),
        pl.Series("hit_const", hit_const),
        pl.Series("area_mond", area_mond),
        pl.Series("q_mond", q_mond_all),
        pl.Series("r_const", r_const_all),
    ).filter(pl.Series(held))
    extra_df = None
    if x is not None and xs is not None:
        extra_df = x.with_columns(
            pl.Series("hit_mond", x_mond),
            pl.Series("hit_const", x_const),
            pl.Series("area_mond", x_area),
            pl.Series("q_mond", x_q),
        ).filter(pl.Series(x_held))
    return rows, summary, held_df, extra_df


def mondrian_by_source(
    score: np.ndarray,
    groups: np.ndarray,
    feats: dict[str, np.ndarray],
    source: np.ndarray,
    cal: np.ndarray,
    base: np.ndarray | None = None,
    alpha: float = ALPHA,
) -> dict[str | None, MondrianConformal]:
    """Mondrian models fitted on the calibration rows of each covariance source, plus one on the calibration
    rows of ``base`` (default: all rows; key None) for a source without a model of its own.

    A source gets its own model when its calibration rows come from at least ``min_source_clips(alpha)``
    clips (``MIN_SOURCE_CLIPS`` at the default level): with fewer, the clip-level level (1 − α)(1 + 1/G)
    reaches 1 and the quantile would be the source's largest score (the PTZ-projected source of the first full
    run had 5-9 calibration clips per fold, hence regions of 60,000 m²). Sources in ``POOLED_SOURCES`` get one
    pooled quantile instead of bins. With a single source and no ``base`` restriction the two models are the
    same fit.
    """
    keep = np.ones(len(score), dtype=bool) if base is None else np.asarray(base, dtype=bool)
    min_clips = min_source_clips(alpha)

    def fit(sel: np.ndarray, bins: dict[str, int]) -> MondrianConformal:
        return MondrianConformal.fit(
            score[sel], groups[sel], {k: v[sel] for k, v in feats.items()}, bins, alpha=alpha
        )

    present = np.unique(source[cal])
    if len(present) == 1 and keep[cal].all():
        model = fit(cal, MONDRIAN_BINS)
        return {None: model, str(present[0]): model}
    models: dict[str | None, MondrianConformal] = {None: fit(cal & keep, MONDRIAN_BINS)}
    for src in present:
        sel = cal & (source == src)
        if len(np.unique(groups[sel])) >= min_clips:
            models[str(src)] = fit(sel, {} if src in POOLED_SOURCES else MONDRIAN_BINS)
    return models


# --- small report helpers --------------------------------------------------------------------------------


def q3(x: np.ndarray) -> str:
    v = np.asarray(x, dtype=np.float64)
    if not np.isfinite(v).any():
        return "– | – | –"
    return " | ".join(f"{np.nanquantile(v, p):.2f}" for p in (0.5, 0.9, 0.99))


def error_norm(d: pl.DataFrame, ref: str) -> np.ndarray:
    """Position error norms (m) of player rows against reference ``ref``."""
    return np.asarray(np.hypot(d[f"ex_{ref}"].to_numpy(), d[f"ey_{ref}"].to_numpy()), dtype=np.float64)
