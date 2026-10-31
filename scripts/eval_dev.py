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
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import typer

from predfoot.calib import temporal as tm
from predfoot.calib.solver import SolverConfig
from predfoot.eval.positions import (
    ALPHA,
    FAIL_LINE_PX,
    FLAGS,
    IMAGE_SIZE,
    MIN_SOURCE_CLIPS,
    MONDRIAN_BINS,
    NATIVE_FPS,
    ORACLE_MAX_PX,
    POOLED_SOURCES,
    RAW_GAP_MAX_M,
    ClipCalib,
    FitScope,
    Output,
    Smooth,
    SmoothingInfo,
    calibrate_clip,
    cov_sources,
    frame_rows_out,
    load_calib,
    logo,
    player_rows,
    q3,
    save_calib,
    smooth_outputs,
)
from predfoot.eval.positions import error_norm as _err
from predfoot.eval.positions import flag_arrays as _flag_arrays
from predfoot.eval.positions import mondrian_by_source as _mondrian_by_source
from predfoot.paths import index_dir
from predfoot.uncertainty.conformal import MondrianConformal

__all__ = [  # re-exported for tests and for other scripts
    "ALPHA",
    "IMAGE_SIZE",
    "MIN_SOURCE_CLIPS",
    "MONDRIAN_BINS",
    "NATIVE_FPS",
    "ORACLE_MAX_PX",
    "POOLED_SOURCES",
    "RAW_GAP_MAX_M",
    "ClipCalib",
    "MondrianConformal",
    "_mondrian_by_source",
    "cov_sources",
    "load_calib",
    "logo",
    "save_calib",
]

app = typer.Typer(add_completion=False)
STEP_MAX_ERROR_M = 10.0  # as scripts/error_process.py: larger errors are calibration failures (abstention)
DEFAULT_REPORT = Path("reports/m2_logo_dev.md")


# --- report ------------------------------------------------------------------------------------------------


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
        try:
            cc = calibrate_clip(cid, c["split"], int(c["game_id"]), cp, cl, saved.get(cid))
        except ValueError as err:  # saved calibrations that do not match the cached frames
            raise typer.BadParameter(str(err)) from err
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
