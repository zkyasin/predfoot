"""Dev evaluation (tickets T1.5 and T2.3, temporal smoothing): calibrate every cached dev frame, optionally
smooth the camera over time, predict positions with their uncertainty, and measure leave-one-game-out (LOGO)
conformal coverage against two references.

References:
- ``labels``: the SN-GSR raw ground truth (``x_raw``, ``y_raw``), produced by the annotators' camera;
- ``oracle``: positions from a homography fitted to the annotated lines of the same frame, kept only where
  it fits those lines within 3 px (see reports/spike_s0.md for why both are needed).

For each held-out game, conformal quantiles are fitted on the five other dev games only. Two regions are
compared at the same nominal level: our ellipses {e : eᵀ Σp⁻¹ e ≤ q̂} (pooled and Mondrian) and a constant
circle whose radius is the conformal quantile of the error norms. Bootstrap intervals resample clips.

Camera (``--smooth``):
- ``none``: per-frame calibrations;
- ``proj``: each frame projected onto its clip's fixed-centre pan-tilt-zoom manifold, no smoothing (ablation);
- ``ptz``: the PTZ Kalman/RTS smoother of :mod:`predfoot.calib.temporal`. With ``--fit-scope fold`` its noise
  parameters are fitted, without labels, on the five calibration games of each LOGO fold; every clip is
  smoothed with every fold's parameters, fold g's quantiles are fitted on the other games' rows of fold g and
  game g is only scored. ``--fit-scope all`` fits one label-free set on all games (exchangeability check).
Frames whose raw calibration failed but which the smoother fills are kept out of the headline coverage and
reported separately. With smoothing, the covariances of smoothed frames, of frames bridged across a gap and of
single frames come from different models, so conformal quantiles are fitted within each source
(:func:`cov_sources`). Calibrations are saved to ``runs/<run>/calib.npz``; ``--calib-from`` reuses them, so
variants differ only by the camera.

Usage:
    uv run python scripts/eval_dev.py                  # full per-frame run, writes reports/m2_logo_dev.md
    uv run python scripts/eval_dev.py --smooth ptz --calib-from runs/<run>/calib.npz   # runs/<run>/report.md
Needs the index (scripts/build_index.py) and the detection cache (scripts/cache_keypoints.py).
"""

from __future__ import annotations

import json
import subprocess
import time
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import typer

from predfoot.calib import temporal as tm
from predfoot.calib.lines import line_oracle, line_reprojection_px
from predfoot.calib.observations import observations_from_detections
from predfoot.calib.solver import SolverConfig, calibrate
from predfoot.contracts import CalibResult, CalibStatus
from predfoot.geometry import homography as hg
from predfoot.geometry.pitch import pnlcalib_keypoints
from predfoot.health.features import calibration_features, calibration_jump_px
from predfoot.keypoints.cache import cache_path, cached_detections
from predfoot.paths import data_dir, index_dir
from predfoot.uncertainty.conformal import MondrianConformal, grouped_quantile
from predfoot.uncertainty.delta import (
    circumscribed_radius,
    mahalanobis2,
    position_covariance,
    region_quantile,
)

app = typer.Typer(add_completion=False)
KP = pnlcalib_keypoints()
ORACLE_MAX_PX = 3.0
RAW_GAP_MAX_M = 5.0
ALPHA = 0.1
NATIVE_FPS = 25.0
IMAGE_SIZE = (1920, 1080)
FAIL_LINE_PX = 10.0
STEP_MAX_ERROR_M = 10.0  # as scripts/error_process.py: larger errors are calibration failures (abstention)
FLAGS = (
    "smoothed", "repaired", "filled", "bridged", "projected", "passthrough", "duplicate", "rejected",
    "unchecked",
)  # fmt: skip
DEFAULT_REPORT = Path("reports/m2_logo_dev.md")
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


def calibrate_clip(
    clip_id: str,
    split: str,
    game: int,
    persons: pl.DataFrame,
    lines: pl.DataFrame,
    cached: Sequence[CalibResult] | None = None,
) -> ClipCalib | None:
    """Calibrate every cached frame (seeded by frame and clip) unless ``cached`` calibrations are given."""
    clip_dir = data_dir() / "sn-gsr" / split / clip_id
    n = len(list((clip_dir / "img1").glob("*.jpg")))
    if not cache_path(clip_id, n).exists():
        return None
    dets, _, _ = cached_detections(clip_dir, clip_id)
    frames = np.array(sorted(dets), dtype=np.int64)
    if cached is not None and len(cached) != len(frames):
        raise typer.BadParameter(
            f"{clip_id}: {len(cached)} saved calibrations for {len(frames)} cached frames"
        )
    cfg = SolverConfig()
    cc = ClipCalib(clip_id, game, split, frames, [], np.zeros(len(frames), dtype=bool), [])
    p_by_frame = persons.partition_by("frame", as_dict=True)
    l_by_frame = lines.partition_by("frame", as_dict=True)
    cc.duplicate[:] = tm.duplicate_mask([dets[f] for f in frames.tolist()])
    prev_h = None
    for i, frame in enumerate(frames.tolist()):
        det = dets[frame]
        xy, uv, w, obs = observations_from_detections(det)
        if cached is None:
            rng = np.random.default_rng(frame * 7919 + int(clip_id.split("-")[1]))
            res, _ = calibrate(xy, uv, rng, weights=w, config=cfg, lines=obs)
        else:
            res = cached[i]
        cc.results.append(res)
        p = p_by_frame.get((frame,))
        if p is None:
            continue
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
            "clip": clip_id,
            "game": game,
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


def camera_speed_px_s(frames: np.ndarray, results: Sequence[CalibResult]) -> np.ndarray:
    """Image motion (px/s) of a ground grid between consecutive output cameras; NaN without both."""
    out = np.full(len(frames), np.nan)
    for i in range(1, len(frames)):
        a, b = results[i - 1], results[i]
        if a.ok and b.ok and a.H is not None and b.H is not None:
            dt = (frames[i] - frames[i - 1]) / NATIVE_FPS
            out[i] = calibration_jump_px(b.H, a.H, IMAGE_SIZE) / dt
    return out


def _flag_arrays(flags: tm.FrameFlags | None, n: int) -> dict[str, np.ndarray]:
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
    fl = _flag_arrays(flags, len(cc.frames))
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
    fl = _flag_arrays(flags, len(cc.frames))
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


def save_calib(path: Path, ccs: Sequence[ClipCalib], meta: dict[str, Any]) -> None:
    """Every clip's per-frame calibrations, exactly (None fields are stored with a mask)."""
    arrays: dict[str, np.ndarray] = {
        "meta": np.array(json.dumps({**meta, "clips": [c.clip_id for c in ccs]}))
    }
    for cc in ccs:
        n = len(cc.frames)
        h = np.full((n, 3, 3), np.nan)
        cov = np.full((n, 8, 8), np.nan)
        has_h, has_cov = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        for i, r in enumerate(cc.results):
            if r.H is not None:
                h[i], has_h[i] = r.H, True
            if r.cov_theta is not None:
                cov[i], has_cov[i] = r.cov_theta, True
        key = f"{cc.clip_id}__"
        arrays |= {key + "frames": cc.frames, key + "duplicate": cc.duplicate, key + "H": h, key + "cov": cov}
        arrays |= {key + "has_H": has_h, key + "has_cov": has_cov}
        arrays[key + "status"] = np.array([r.status.value for r in cc.results])
        arrays[key + "n_inliers"] = np.array([r.n_inliers for r in cc.results], dtype=np.int64)
        for name in ("s2", "rms_px", "cond", "dof"):
            arrays[key + name], arrays[key + "has_" + name] = _opt([getattr(r, name) for r in cc.results])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_calib(path: Path) -> tuple[dict[str, list[CalibResult]], dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    out = {}
    for clip_id in meta["clips"]:
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
        out[clip_id] = results
    return out, meta


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


def smooth_outputs(
    ccs: Sequence[ClipCalib], smooth: Smooth, fit_scope: FitScope, cfg: tm.SmootherConfig, info: SmoothingInfo
) -> Iterator[Output]:
    """Output cameras of every clip (and every fold for the fold-wise fit); fills ``info`` on the way."""
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
    games = sorted({cc.game for cc in ccs})
    folds: list[int | None] = [None] if fit_scope is FitScope.all else list(games)
    for fold in folds:
        t0 = time.perf_counter()
        train = [info.meas[cc.clip_id] for cc in ccs if fold is None or cc.game != fold]
        params, fit = tm.fit_noise(train, cfg)
        info.seconds[f"fit_{fold}"] = time.perf_counter() - t0
        info.fits["all" if fold is None else str(fold)] = {
            "params": params.to_dict(),
            "params_digest": params.digest(),
            "training_clips": len(train),
            "training_games": [g for g in games if fold is None or g != fold],
            **fit,
        }
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


# --- LOGO conformal ----------------------------------------------------------------------------------------


def _clip_bootstrap(
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


def _scores(d: pl.DataFrame, ref: str) -> dict[str, np.ndarray]:
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


def logo(
    df: pl.DataFrame,
    ref: str,
    rng: np.random.Generator,
    n_boot: int = 300,
    fold_col: str | None = None,
    extra: pl.DataFrame | None = None,
    alpha: float = ALPHA,
) -> tuple[list[dict[str, Any]], dict[str, Any], pl.DataFrame, pl.DataFrame | None]:
    """Leave-one-game-out coverage and size of our ellipses against a constant circle, at level 1 − ``alpha``.

    Returns one row per held-out game, a summary over all held-out predictions (each player-frame is predicted
    with quantiles fitted on the other games), including coverage by distance tercile, the held-out rows with
    their hits and Mondrian quantile (``q_mond``: the region is {e : eᵀ Σp⁻¹ e ≤ q_mond}), and ``extra`` rows
    (e.g. filled frames) scored with the same fold models. With ``fold_col``,
    fold g calibrates on rows (fold == g) & (game != g) and tests on rows (fold == g) & (game == g).
    Mondrian models are fitted within each covariance source (column ``cov_source``, see :func:`cov_sources`
    and :func:`_mondrian_by_source`); the ``extra`` rows of the other games calibrate their own source and
    nothing else. The pooled quantile, the constant circle and the fallback model use the ``df`` rows only.
    """
    d = df.filter(pl.col(f"ex_{ref}").is_not_nan())
    s = _scores(d, ref)
    score, norm, groups, games, cov = s["score"], s["norm"], s["groups"], s["games"], s["cov"]
    feats = {"mpp": s["mpp"], "n_inliers": s["n_inliers"]}
    source = s["source"]
    raw_q = np.array([region_quantile(2, 1 - alpha, v) for v in s["dof"]])
    folds = d[fold_col].to_numpy() if fold_col else None
    hit_mond, hit_const = np.zeros(len(d), bool), np.zeros(len(d), bool)
    rad_mond, area_mond, r_const_all = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    q_mond_all = np.full(len(d), np.nan)
    held = np.zeros(len(d), bool)
    x = extra.filter(pl.col(f"ex_{ref}").is_not_nan()) if extra is not None else None
    xs = _scores(x, ref) if x is not None and x.height else None
    x_mond = np.zeros(x.height if x is not None else 0, bool)
    x_const, x_held, x_area = x_mond.copy(), x_mond.copy(), np.zeros(len(x_mond))
    x_q = np.full(len(x_mond), np.nan)
    x_folds = x[fold_col].to_numpy() if fold_col and x is not None else None
    # Calibration pool of the per-source models: the df rows, then the extra rows (base marks the df rows).
    pool = {"score": score, "groups": groups, "source": source, "games": games, **feats}
    if xs is not None:
        pool = {k: np.concatenate([v, xs[k]]) for k, v in pool.items()}
    base = np.arange(len(pool["score"])) < len(score)
    pool_folds = None if folds is None else np.concatenate([folds, x_folds]) if x_folds is not None else folds
    pool_feats = {k: pool[k] for k in feats}
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
        q_pool = grouped_quantile(score[cal], groups[cal], alpha)
        models = _mondrian_by_source(
            pool["score"], pool["groups"], pool_feats, pool["source"], cal_pool, base, alpha=alpha
        )
        q_mond = np.full(len(d), np.nan)
        for src in np.unique(source[tst]):
            sel = tst & (source == src)
            model = models.get(str(src), models[None])
            q_mond[sel] = model.quantile_for({k: v[sel] for k, v in feats.items()})
        q_mond = q_mond[tst]
        q_mond_all[tst] = q_mond
        r_const = grouped_quantile(norm[cal], groups[cal], alpha)
        hit_mond[tst] = score[tst] <= q_mond
        hit_const[tst] = norm[tst] <= r_const
        rad_mond[tst] = np.sqrt(q_mond * np.linalg.eigvalsh(cov[tst])[:, -1])
        area_mond[tst] = np.pi * q_mond * np.sqrt(np.clip(np.linalg.det(cov[tst]), 0, None))
        r_const_all[tst] = r_const
        if xs is not None:
            xt = (xs["games"] == g) & ((x_folds == g) if x_folds is not None else True)
            if xt.any():
                qx = np.full(len(xt), np.nan)
                for src in np.unique(xs["source"][xt]):
                    sel = xt & (xs["source"] == src)
                    model = models.get(str(src), models[None])
                    qx[sel] = model.quantile_for({k: xs[k][sel] for k in feats})
                x_mond[xt], x_const[xt] = xs["score"][xt] <= qx[xt], xs["norm"][xt] <= r_const
                x_held[xt] = True
                x_q[xt] = qx[xt]
                x_area[xt] = np.pi * qx[xt] * np.sqrt(np.clip(np.linalg.det(xs["cov"][xt]), 0, None))
        rows.append(
            {
                "game": int(g),
                "clips": len(np.unique(groups[tst])),
                "n": int(tst.sum()),
                "cov_raw": float(np.mean(score[tst] <= raw_q[tst])),
                "cov_pool": float(np.mean(score[tst] <= q_pool)),
                "cov_mond": float(hit_mond[tst].mean()),
                "cov_mond_ci": _clip_bootstrap(hit_mond[tst], groups[tst], rng, n_boot),
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
        "cov_mond_ci": _clip_bootstrap(hit_mond[h], groups[h], rng, n_boot),
        "cov_const": float(hit_const[h].mean()),
        "cov_const_ci": _clip_bootstrap(hit_const[h], groups[h], rng, n_boot),
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


def _mondrian_by_source(
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


# --- report ------------------------------------------------------------------------------------------------


def q3(x: np.ndarray) -> str:
    v = np.asarray(x, dtype=np.float64)
    if not np.isfinite(v).any():
        return "– | – | –"
    return " | ".join(f"{np.nanquantile(v, p):.2f}" for p in (0.5, 0.9, 0.99))


def _err(d: pl.DataFrame, ref: str) -> np.ndarray:
    return np.hypot(d[f"ex_{ref}"].to_numpy(), d[f"ey_{ref}"].to_numpy())


def common_steps(d: pl.DataFrame, ref: str = "sm", max_error_m: float | None = None) -> np.ndarray:
    """Lag-1 steps (n, 2) of the per-frame median error over consecutive cached frames, pooled over clips.

    With ``max_error_m``, player-frames whose error is at least that large are left out first, as in
    scripts/error_process.py.
    """
    if max_error_m is not None:
        d = d.filter((pl.col(f"ex_{ref}") ** 2 + pl.col(f"ey_{ref}") ** 2) < max_error_m**2)
    parts = [np.zeros((0, 2))]
    for (_clip,), g in d.partition_by("clip", as_dict=True).items():
        c = g.group_by("frame").agg(pl.col(f"ex_{ref}").median(), pl.col(f"ey_{ref}").median()).sort("frame")
        f = c["frame"].to_numpy()
        e = c.select(f"ex_{ref}", f"ey_{ref}").to_numpy()
        step = np.diff(e, axis=0)[np.diff(f) == 5]
        parts.append(step[np.isfinite(step).all(axis=1)])
    return np.concatenate(parts)


def step_rms(d: pl.DataFrame, ref: str = "sm", max_error_m: float | None = None) -> float:
    """RMS per axis of the common-mode steps (see :func:`common_steps`)."""
    s = common_steps(d, ref, max_error_m)
    return float(np.sqrt(np.mean(s**2))) if s.size else float("nan")


def step_median(d: pl.DataFrame, ref: str = "sm") -> float:
    """Median absolute common-mode step per axis: a few huge steps (one failed frame) do not move it."""
    s = common_steps(d, ref)
    return float(np.median(np.abs(s))) if s.size else float("nan")


def step_stats(d: pl.DataFrame, ref: str = "sm") -> tuple[float, float, float]:
    """Common-mode step RMS, RMS without errors of ``STEP_MAX_ERROR_M`` or more, and median absolute step."""
    return step_rms(d, ref), step_rms(d, ref, STEP_MAX_ERROR_M), step_median(d, ref)


def temporal_section(
    smooth: Smooth,
    fit_scope: FitScope,
    cfg: tm.SmootherConfig,
    info: SmoothingInfo,
    ccs: Sequence[ClipCalib],
    held_outputs: Sequence[Output],
    players: pl.DataFrame,
    raw_players: pl.DataFrame,
    frames: pl.DataFrame,
    logo_extra: dict[str, tuple[dict[str, Any], pl.DataFrame, pl.DataFrame | None]],
) -> list[str]:
    md = [
        "",
        "## Temporal smoothing",
        "",
        f"Camera `{smooth}`"
        + (f", noise fitted per `{fit_scope}`" if smooth is Smooth.ptz else "")
        + f"; smoother config digest `{cfg.digest()}` (β = {cfg.beta:g}). Label-free per clip: projection"
        " centre, PTZ projection, misfit gate"
        + (", segments, deletions and the RTS smoother." if smooth is Smooth.ptz else " (no smoothing).")
        + " Conformal quantiles are refitted on these outputs.",
    ]
    if info.fits:
        md += [
            "",
            "Noise parameters (fitted without labels; fold g excludes game g):",
            "",
            "| Fold | Training clips | κ² | √q pan (°/s^1.5) | √q tilt (°/s^1.5) | √q log f (1/s^1.5) |"
            " Gap factor λ (innovations after a gap, their median NIS) | NLL per term |"
            " NIS mean (truncated) / median | Evaluations |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for fold, f in info.fits.items():
            p = f["params"]
            sq_pan, sq_tilt = np.degrees(np.sqrt(p["q_pan"])), np.degrees(np.sqrt(p["q_tilt"]))
            nis = f"{f.get('mean_nis_truncated', np.nan):.2f} / {f.get('median_nis', np.nan):.2f}"
            evals = f"{f['nfev']}" + ("" if f["converged"] else " (not converged)")
            held = "" if f.get("gap_fitted") else ", held"
            gap = (
                f"{p['gap']:.2f} ({f.get('gap_terms', 0)}, {f.get('median_nis_after_gap', np.nan):.2f}{held})"
            )
            md.append(
                f"| {fold} | {f['training_clips']} | {p['kappa2']:.2f} | {sq_pan:.2f} | {sq_tilt:.3f} |"
                f" {np.sqrt(p['q_logf']):.3f} | {gap} | {f['nll'] / max(f['n_terms'], 1):.2f} | {nis} |"
                f" {evals} |"
            )
        if len(info.fits) > 1:
            names = ("kappa2", "q_pan", "q_tilt", "q_logf", "gap")
            logs = np.log([[f["params"][k] for k in names] for f in info.fits.values()])
            spread = np.exp(np.abs(logs - np.median(logs, axis=0)).max(axis=0))
            md += [
                "",
                "Largest ratio of a fold's parameter to the median over folds"
                " (κ², q pan, q tilt, q log f, λ): "
                + ", ".join(f"×{v:.2f}" for v in spread)
                + " (target: within ×1.5).",
            ]
    md += [
        "",
        "Projection centres per game (own estimate reliable when its largest sd ≤"
        f" {cfg.centre_max_sd_m:g} m with ≥ {cfg.centre_min_frames} PTZ inliers; otherwise the game centre,"
        f" refused when the clip's misfit scale at it exceeds {cfg.max_misfit_ratio:g}× the reference;"
        " without a centre the clip passes through):",
        "",
        "| Game | Reliable / clips | Median centre (m) | MAD (m) | Game centre used | Game centre refused |"
        " None |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in sorted({cc.game for cc in ccs}):
        ids = [cc.clip_id for cc in ccs if cc.game == g]
        rel = [e for c in ids if (e := info.own[c]) is not None and e.reliable]
        cs = np.array([e.centre for e in rel]) if rel else np.full((1, 3), np.nan)
        med = np.median(cs, axis=0)
        mad = np.median(np.abs(cs - med), axis=0)
        src = [info.chosen[c] for c in ids]
        n_game = sum(1 for c in src if c is not None and c.source == "game")
        n_refused = sum(1 for c in ids if info.refused.get(c, False))
        n_none = sum(1 for c in src if c is None)
        md.append(
            f"| {g} | {len(rel)} / {len(ids)} | ({med[0]:.1f}, {med[1]:.1f}, {med[2]:.1f}) |"
            f" ({mad[0]:.1f}, {mad[1]:.1f}, {mad[2]:.1f}) | {n_game} | {n_refused} | {n_none} |"
        )
    # frame outcomes over every cached frame of the held-out outputs
    counts = dict.fromkeys((*FLAGS, "abstain"), 0)
    per_clip = []
    for o in held_outputs:
        fl = _flag_arrays(o.flags, len(o.cc.frames))
        flagged = np.any([fl[k] for k in FLAGS], axis=0)
        for k in FLAGS:
            counts[k] += int(fl[k].sum())
        counts["abstain"] += int((~flagged & ~np.array([r.ok for r in o.results])).sum())
        meas = info.meas[o.cc.clip_id]
        base = meas.raw_ok & ~meas.duplicate
        per_clip.append(
            {
                "segments": len(np.unique(o.track.segment[o.track.segment >= 0]))
                if o.track is not None
                else 0,
                "deleted": int(o.track.rejected.sum()) if o.track is not None else 0,
                "misfit_share": float(meas.misfit[base].mean()) if base.any() else float("nan"),
                "short": int(o.track.short.sum()) if o.track is not None else 0,
                "manoeuvre": int(o.track.manoeuvre.sum()) if o.track is not None else 0,
            }
        )
    total = sum(counts.values())
    md += [
        "",
        f"Frame outcomes over {total} cached frames (one per frame):",
        "",
        "| " + " | ".join(counts) + " |",
        "|" + "---|" * len(counts),
        "| " + " | ".join(f"{v / max(total, 1):.1%}" for v in counts.values()) + " |",
    ]
    if smooth is Smooth.ptz:
        pc = pl.DataFrame(per_clip)
        md += [
            "",
            "Per clip (median / max): segments"
            f" {pc['segments'].median():.0f} / {pc['segments'].max()}, deleted measurements"
            f" {pc['deleted'].median():.0f} / {pc['deleted'].max()}, frames in short segments"
            f" {pc['short'].median():.0f} / {pc['short'].max()}, frames in manoeuvre segments"
            f" {pc['manoeuvre'].median():.0f} / {pc['manoeuvre'].max()}, misfit share"
            f" {pc['misfit_share'].median():.1%} / {pc['misfit_share'].max():.1%}.",
        ]
    # errors against three references, per-frame camera vs output camera, same player-frames
    keys = ["clip", "frame", "track_id"]
    out_rows = players.filter(~pl.col("filled"))
    err_cols = [f"e{a}_{r}" for r in ("lab", "or", "sm") for a in "xy"]
    joined = out_rows.join(raw_players.select(*keys, *err_cols), on=keys, suffix="_raw")
    md += [
        "",
        f"Position error (m) on the same {joined.height} player-frames (raw calibration OK), and jitter of"
        " the common-mode error (per-frame median over players) against the smoothed labels:",
        "",
        "| Camera | Labels p50 | p90 | p99 | Line oracle p50 | p90 | p99 | Smoothed labels p50 | p90 | p99 |"
        f" Common-mode step RMS (m) | RMS, errors < {STEP_MAX_ERROR_M:g} m | Median abs. step (m) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    raw_view = joined.select(*keys, *[pl.col(f"{c}_raw").alias(c) for c in err_cols])
    s_raw, s_out = step_stats(raw_view), step_stats(joined)
    for name, view, st in (("per-frame", raw_view, s_raw), (f"{smooth}", joined, s_out)):
        md.append(
            f"| {name} | {q3(_err(view, 'lab'))} | {q3(_err(view, 'or'))} | {q3(_err(view, 'sm'))} |"
            f" {st[0]:.3f} | {st[1]:.3f} | {st[2]:.3f} |"
        )
    ratios = []
    for (clip,), g in joined.partition_by("clip", as_dict=True).items():
        a, b = step_stats(raw_view.filter(pl.col("clip") == clip)), step_stats(g)
        ratios.append((clip, a, b))
    per_clip = np.array([[b[i] / a[i] for i in range(3)] for _, a, b in ratios])
    md += [
        "",
        f"Common-mode step ratio {smooth} / per-frame, pooled over clips: RMS {s_out[0] / s_raw[0]:.2f},"
        f" RMS without errors of {STEP_MAX_ERROR_M:g} m or more {s_out[1] / s_raw[1]:.2f}, median absolute"
        f" step {s_out[2] / s_raw[2]:.2f}; RMS ratio per clip: median {np.nanmedian(per_clip[:, 0]):.2f},"
        f" worst {np.nanmax(per_clip[:, 0]):.2f}. One failed frame makes steps of tens of metres that"
        " dominate the plain RMS: read the robust ratios.",
    ]
    if len(ratios) <= 10:
        md += [
            "",
            "| Clip | Step RMS per-frame (m) | Step RMS output (m) | Ratio |"
            f" Ratio, errors < {STEP_MAX_ERROR_M:g} m | Median-step ratio |",
            "|---|---|---|---|---|---|",
        ]
        md += [
            f"| {c} | {a[0]:.3f} | {b[0]:.3f} | {b[0] / a[0]:.2f} | {b[1] / a[1]:.2f} | {b[2] / a[2]:.2f} |"
            for c, a, b in sorted(ratios)
        ]
    # filled frames
    filled = players.filter(pl.col("filled"))
    md += [
        "",
        f"Filled frames (raw calibration failed, camera from the smoother): {filled.height} player-frames",
    ]
    if filled.height:
        md[-1] += (
            f" on {filled.select('clip', 'frame').n_unique()} frames; error p50 | p90 | p99 against labels"
            f" {q3(_err(filled, 'lab'))} m."
        )
        for ref, (_, _, extra) in logo_extra.items():
            if extra is not None and extra.height:
                md[-1] += (
                    f" Coverage ({ref}, gap source): Mondrian {extra['hit_mond'].mean():.1%},"
                    f" constant circle {extra['hit_const'].mean():.1%} (n = {extra.height})."
                )
    else:
        md[-1] += "."
    bridged = joined.filter(pl.col("bridged"))
    md += [
        "",
        f"Bridged frames (run of off-model frames of at most {cfg.max_misfit_run_s:g} s inside a segment, raw"
        f" camera replaced by the smoothed state): {bridged.height} player-frames",
    ]
    if bridged.height:
        raw_bridged = bridged.select(*keys, *[pl.col(f"{c}_raw").alias(c) for c in err_cols])
        md[-1] += (
            f" on {bridged.select('clip', 'frame').n_unique()} frames; error p50 | p90 | p99 against labels"
            f" {q3(_err(raw_bridged, 'lab'))} m per-frame, {q3(_err(bridged, 'lab'))} m output."
        )
    else:
        md[-1] += "."
    # coverage per output type (quantiles are fitted within each covariance source)
    md += [
        "",
        "Coverage per output type (quantiles fitted within each covariance source: smoother for smoothed and"
        " repaired frames, gap for filled and bridged frames, single frame for the others), Mondrian ellipses"
        " / constant circle, and median Mondrian area:",
        "",
        "| Reference | Type | Player-frames | Mondrian | Constant circle | Median area (m²) |",
        "|---|---|---|---|---|---|",
    ]
    for ref, (_, held, extra) in logo_extra.items():
        parts = [(name, held.filter(pl.col(name))) for name in FLAGS if name != "filled"]
        if extra is not None:
            parts.append(("filled", extra))
        for name, part in parts:
            if part.height:
                md.append(
                    f"| {ref} | {name} | {part.height} | {part['hit_mond'].mean():.1%} |"
                    f" {part['hit_const'].mean():.1%} | {part['area_mond'].median():.2f} |"
                )
    # coverage per camera-speed tercile
    md += ["", "Coverage per camera-speed tercile (image motion of a ground grid between output frames),"
           " Mondrian ellipses / constant circle:", "",
           "| Reference | Slow | Middle | Fast |", "|---|---|---|---|"]  # fmt: skip
    for ref, (_, held, _) in logo_extra.items():
        sp = held["cam_speed_px_s"].to_numpy()
        ok = np.isfinite(sp)
        edges = np.quantile(sp[ok], [1 / 3, 2 / 3])
        b = np.searchsorted(edges, sp[ok])
        hm, hc = held["hit_mond"].to_numpy()[ok], held["hit_const"].to_numpy()[ok]
        cells = [f"{hm[b == i].mean():.1%} / {hc[b == i].mean():.1%}" for i in range(3)]
        md.append(f"| {ref} (edges {edges[0]:.0f}, {edges[1]:.0f} px/s) | " + " | ".join(cells) + " |")
    # line reprojection of the output camera
    f = frames.filter(pl.col("trusted") & pl.col("ok") & pl.col("ok_out"))
    lr, lo = f["line_px"].to_numpy().astype(float), f["line_px_out"].to_numpy().astype(float)

    def line_stats(v: np.ndarray) -> str:
        return (
            f"p50 {np.nanquantile(v, 0.5):.1f} px, p90 {np.nanquantile(v, 0.9):.1f} px,"
            f" > {FAIL_LINE_PX:.0f} px {np.nanmean(v > FAIL_LINE_PX):.1%}"
        )

    md += [
        "",
        f"Line reprojection on {f.height} frames with self-consistent line annotations (raw calibration OK):"
        f" per-frame {line_stats(lr)}; output {line_stats(lo)}.",
        "",
        f"Published-position rate (output camera OK): {frames['ok_out'].mean():.1%} of labelled frames"
        f" (per-frame calibration: {frames['ok'].mean():.1%}).",
    ]
    return md


def track_stats(track: tm.SmoothedTrack | None) -> dict[str, Any]:
    if track is None:
        return {"segments": 0, "deleted": 0, "short": 0, "manoeuvre": 0, "mean_nis": None, "passes": 0}
    nis = track.nis[np.isfinite(track.nis)]
    return {
        "segments": len(np.unique(track.segment[track.segment >= 0])),
        "deleted": int(track.rejected.sum()),  # measurements deleted (repaired or rejected frames)
        "short": int(track.short.sum()),
        "manoeuvre": int(track.manoeuvre.sum()),
        "mean_nis": float(nis.mean()) if nis.size else None,
        "passes": track.n_iter,
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def smoothing_json(
    smooth: Smooth,
    fit_scope: FitScope,
    cfg: tm.SmootherConfig,
    info: SmoothingInfo,
    held_outputs: Sequence[Output],
) -> dict[str, Any]:
    clips = {}
    for o in held_outputs:
        cid = o.cc.clip_id
        own, chosen, meas = info.own.get(cid), info.chosen.get(cid), info.meas.get(cid)
        fl = _flag_arrays(o.flags, len(o.cc.frames))
        base = meas.raw_ok & ~meas.duplicate if meas is not None else np.zeros(0, bool)
        clips[cid] = {
            "game": o.cc.game,
            "fold": o.fold,
            "centre": chosen.centre.tolist() if chosen is not None else None,
            "centre_sd_m": chosen.sd_m if chosen is not None else None,
            "centre_source": chosen.source if chosen is not None else None,
            "centre_reliable": chosen.reliable if chosen is not None else None,
            "own_centre": own.centre.tolist() if own is not None else None,
            "own_sd_m": own.sd_m if own is not None else None,
            "own_reliable": own.reliable if own is not None else None,
            "game_centre_refused": info.refused.get(cid, False),
            "kappa2_clip": meas.kappa2_clip if meas is not None else None,
            "misfit_share": float(meas.misfit[base].mean()) if meas is not None and base.any() else None,
            **track_stats(o.track),
            **{name: int(fl[name].sum()) for name in FLAGS},
        }  # fmt: skip
    games = {}
    for g, gc in info.game.items():
        rel = [
            e
            for c, e in info.own.items()
            if e is not None and e.reliable and clips.get(c, {}).get("game") == g
        ]
        cs = np.array([e.centre for e in rel]) if rel else np.full((1, 3), np.nan)
        med = np.median(cs, axis=0)
        games[str(g)] = {
            "median_centre": med.tolist(),
            "mad": np.median(np.abs(cs - med), axis=0).tolist(),
            "n_reliable": len(rel),
            "game_centre": gc.centre.tolist() if gc is not None else None,
        }
    return {
        "smooth": str(smooth),
        "fit_scope": str(fit_scope) if smooth is Smooth.ptz else None,
        "config": asdict(cfg),
        "config_digest": cfg.digest(),
        "folds": info.fits,
        "clips": clips,
        "games": games,
        "seconds": info.seconds,
    }


# --- command line ------------------------------------------------------------------------------------------


def select_clips(dev: pl.DataFrame, ids: str, max_clips: int) -> pl.DataFrame:
    """Dev clips to evaluate: all, or the comma-separated ``ids`` (dev only, at least two games)."""
    dev = dev.sort("clip_id")
    if ids:
        wanted = [c.strip() for c in ids.split(",") if c.strip()]
        unknown = sorted(set(wanted) - set(dev["clip_id"].to_list()))
        if unknown:
            raise typer.BadParameter(f"not dev clips (unknown or locked test clips): {', '.join(unknown)}")
        dev = dev.filter(pl.col("clip_id").is_in(wanted))
        if dev["game_id"].n_unique() < 2:
            raise typer.BadParameter("leave-one-game-out needs clips from at least two games")
    if max_clips:
        dev = dev.head(max_clips)
    return dev


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


@app.command()
def main(
    out: Path | None = typer.Option(  # noqa: B008
        None,
        help=f"Report path (default: {DEFAULT_REPORT} for a full per-frame run, else runs/<run>/report.md)",
    ),
    max_clips: int = 0,
    smooth: Smooth = typer.Option(Smooth.none, help="Camera: per-frame, PTZ projection or PTZ smoother."),  # noqa: B008
    clips: str = typer.Option("", help="Comma-separated dev clip ids (default: every cached dev clip)."),
    calib_from: Path | None = typer.Option(None, help="calib.npz of an earlier run: skip calibration."),  # noqa: B008
    fit_scope: FitScope = typer.Option(FitScope.fold, help="Noise fit per LOGO fold or on all games."),  # noqa: B008
    beta: float = typer.Option(0.0, help="Ablation: add beta * R_k to the smoothed covariance."),
) -> None:
    t_start = time.perf_counter()
    dev = pl.read_parquet(index_dir() / "clips.parquet").filter(pl.col("split").is_in(["train", "valid"]))
    selected = select_clips(dev, clips, max_clips)
    persons = pl.read_parquet(index_dir() / "persons.parquet")
    lines = pl.read_parquet(index_dir() / "lines.parquet")
    saved, saved_meta = load_calib(calib_from) if calib_from else ({}, {})
    sha = git_sha()
    run_dir = Path("runs") / (
        f"{datetime.now(UTC):%Y%m%d-%H%M%S}_{sha}_{smooth}" + ("-all" if fit_scope is FitScope.all else "")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    ccs: list[ClipCalib] = []
    for c in selected.iter_rows(named=True):
        cid = c["clip_id"]
        if calib_from and cid not in saved:
            raise typer.BadParameter(f"{cid} is not in {calib_from}")
        cp = persons.filter(pl.col("clip_id") == cid)
        cl = lines.filter(pl.col("clip_id") == cid)
        cc = calibrate_clip(cid, c["split"], int(c["game_id"]), cp, cl, saved.get(cid))
        if cc is not None:
            ccs.append(cc)
    if not ccs:
        typer.echo("No cached clips found: run scripts/cache_keypoints.py first.")
        raise typer.Exit(1)
    t_calib = time.perf_counter() - t_start
    save_calib(
        run_dir / "calib.npz",
        ccs,
        {
            "git_sha": sha,
            "solver": asdict(SolverConfig()),
            "calib_from": str(calib_from) if calib_from else None,
            "source_meta": {k: v for k, v in saved_meta.items() if k != "clips"} or None,
        },
    )

    cfg = tm.SmootherConfig(beta=beta)
    info = SmoothingInfo()
    tables, frame_rows, held_outputs = [], [], []
    for o in smooth_outputs(ccs, smooth, fit_scope, cfg, info):
        tables.append(player_rows(o.cc, o.results, o.flags, o.fold))
        if o.fold is None or o.fold == o.cc.game:
            frame_rows.extend(frame_rows_out(o.cc, o.results, o.flags, o.diag, o.fold))
            held_outputs.append(o)
    if not any(t.height for t in tables):
        typer.echo("No player rows: no labelled frame of the selected clips has a calibrated camera.")
        raise typer.Exit(1)
    all_players = pl.concat([t for t in tables if t.height], how="vertical")
    fold_col = "fold" if smooth is Smooth.ptz and fit_scope is FitScope.fold else None
    players = all_players.filter(pl.col("fold") == pl.col("game")) if fold_col else all_players
    frames = pl.DataFrame(frame_rows, infer_schema_length=None)
    players.write_parquet(run_dir / "players.parquet")
    frames.write_parquet(run_dir / "frames.parquet")

    headline = players.filter(~pl.col("filled"))
    n_clips, n_games = players["clip"].n_unique(), players["game"].n_unique()
    err_lab, err_or = _err(headline, "lab"), _err(headline, "or")
    rng = np.random.default_rng(0)
    line_px = frames["line_px_out"].to_numpy().astype(float)
    command = "uv run python scripts/eval_dev.py" + (
        f" --smooth {smooth}" if smooth is not Smooth.none else ""
    )
    if smooth is Smooth.ptz and fit_scope is FitScope.all:
        command += " --fit-scope all"
    if beta:
        command += f" --beta {beta:g}"
    if clips:
        command += f" --clips {clips}"
    md = [
        "# Dev evaluation: calibration, positions and leave-one-game-out coverage",
        "",
        f"`{command}` · commit `{sha}` · per-player table in `{run_dir}` (not in git).",
        "",
        f"Dev clips with cached detections: {n_clips} from {n_games} games. Frames: {frames.height}"
        f" (one in five), calibrated {frames['ok'].mean():.1%}. Player-frames: {headline.height}"
        " (ground-truth boxes; truncated boxes and frames without a labelled camera excluded).",
        "",
        "## Position error (metres)",
        "",
        "| Reference | p50 | p90 | p99 |",
        "|---|---|---|---|",
        f"| labels (`x_raw`, `y_raw`) | {q3(err_lab)} |",
        f"| trusted line oracle ({np.isfinite(err_or).mean():.0%} of player-frames) | {q3(err_or)} |",
        "",
        "Calibration (mean distance of annotated line points to the projected template):"
        f" p50 {np.nanquantile(line_px, 0.5):.1f} px, p90 {np.nanquantile(line_px, 0.9):.1f} px.",
    ]
    header = (
        "| Held-out game | Clips | Player-frames | Raw Laplace | Conformal, pooled | Conformal, Mondrian"
        " [95 % CI] | Constant circle | Median radius pooled / Mondrian / constant (m) |"
        " Median area Mondrian / constant (m²) |"
    )
    head_all = all_players.filter(~pl.col("filled"))
    filled_all = all_players.filter(pl.col("filled"))
    logo_extra: dict[str, tuple[dict[str, Any], pl.DataFrame, pl.DataFrame | None]] = {}
    for ref, title in (("lab", "labels"), ("or", "trusted line oracle")):
        res, summ, held, extra = logo(head_all, ref, rng, fold_col=fold_col, extra=filled_all)
        logo_extra[title] = (summ, held, extra)
        lo, hi = summ["cov_mond_ci"]
        clo, chi = summ["cov_const_ci"]
        e1, e2 = summ["tercile_edges"]
        md += [
            "",
            f"## LOGO coverage at 90 %, reference: {title}",
            "",
            f"All held-out predictions ({summ['n']} player-frames, {summ['clips']} clips): Mondrian"
            f" ellipses cover **{summ['cov_mond']:.1%}** [{lo:.1%}, {hi:.1%}] with a median area of"
            f" {summ['a_mond']:.2f} m²;"
            f" the constant circle covers {summ['cov_const']:.1%} [{clo:.1%}, {chi:.1%}] with"
            f" {summ['a_const']:.2f} m². Uncalibrated Laplace ellipses cover {summ['cov_raw']:.1%}.",
            "",
            "Coverage by distance (metres per pixel at the feet), Mondrian ellipses / constant circle:",
            "",
            f"| Near (< {e1:.3f} m/px) | Middle | Far (> {e2:.3f} m/px) |",
            "|---|---|---|",
            "| " + " | ".join(f"{m:.1%} / {c:.1%}" for m, c in summ["by_tercile"]) + " |",
            "",
            "Per held-out game (its quantiles come from the five other games):",
            "",
            header,
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in res:
            glo, ghi = r["cov_mond_ci"]
            md.append(
                f"| {r['game']} | {r['clips']} | {r['n']} | {r['cov_raw']:.1%} | {r['cov_pool']:.1%} |"
                f" {r['cov_mond']:.1%} [{glo:.1%}, {ghi:.1%}] | {r['cov_const']:.1%} | {r['r_pool']:.2f} /"
                f" {r['r_mond']:.2f} / {r['r_const']:.2f} | {r['a_mond']:.2f} / {r['a_const']:.2f} |"
            )
    md += [
        "",
        "Notes. Exploratory dev numbers, not the preregistered test. Frames are subsampled at 5 fps and",
        "remain correlated within a clip, hence clip-level bootstrap intervals. Mondrian bins: metres per",
        "pixel at the feet (3 bins) × number of RANSAC inliers (2 bins); small bins use the pooled",
        "quantile"
        + (
            " of their source. With smoothing, quantiles are fitted within each covariance source"
            " (smoother; gap, one pooled quantile; single frame); a source with fewer than"
            f" {MIN_SOURCE_CLIPS} calibration clips uses the model fitted on all sources."
            if smooth is not Smooth.none
            else "."
        ),
    ]
    if smooth is not Smooth.none:
        raw_players = pl.concat(
            [t for t in (player_rows(cc, cc.results, None, None) for cc in ccs) if t.height]
        )
        md += temporal_section(
            smooth, fit_scope, cfg, info, ccs, held_outputs, players, raw_players, frames, logo_extra
        )
        (run_dir / "smoothing.json").write_text(
            json.dumps(
                smoothing_json(smooth, fit_scope, cfg, info, held_outputs), indent=2, default=_json_default
            )
        )
        md += [
            "",
            f"Runtime: {time.perf_counter() - t_start:.0f} s (calibration or loading {t_calib:.0f} s).",
        ]
    full_none = smooth is Smooth.none and not clips and not max_clips
    report = out or (DEFAULT_REPORT if full_none else run_dir / "report.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(md) + "\n")
    typer.echo("\n".join(md))
    typer.echo(f"\nRun folder: {run_dir} ({time.perf_counter() - t_start:.0f} s)")


if __name__ == "__main__":
    app()
