"""Measure predfoot's position-error process on SN-GSR dev (ticket T4.2), for the KPI error budget (T4.3).

Reads ``players.parquet`` written by scripts/eval_dev.py (frames one in five, so 5 fps). For each clip the
errors are split into a common part (per-frame median over players: mostly the camera) and an individual part,
then each part gets a marginal standard deviation and a lag-one autocorrelation per axis. Autocorrelations
measured at 0.2 s are converted to the native 25 Hz assuming an exponential decay: ρ₂₅ = ρ₅^(1/5).

Usage:
    uv run python scripts/error_process.py runs/<run>/players.parquet --out reports/error_process_dev.md
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import typer

from predfoot.kpi.errors import fit_ar1, split_common_mode

app = typer.Typer(add_completion=False)
STEP_FRAMES = 5  # cached frames are one in five of the native 25 fps
MAX_ERROR_M = 10.0  # errors this large are calibration failures, handled by abstention, not by the budget


def clip_tensor(d: pl.DataFrame, ref: str) -> np.ndarray:
    """Errors of one clip as (n_frames, n_tracks, 2), NaN where a track is missing."""
    frames = np.sort(d["frame"].unique().to_numpy())
    tracks = np.sort(d["track_id"].unique().to_numpy())
    fi = {f: i for i, f in enumerate(frames)}
    ti = {t: i for i, t in enumerate(tracks)}
    e = np.full((len(frames), len(tracks), 2), np.nan)
    for row in d.select("frame", "track_id", f"ex_{ref}", f"ey_{ref}").iter_rows():
        e[fi[row[0]], ti[row[1]]] = row[2], row[3]
    return e


def measure(players: pl.DataFrame, ref: str) -> dict[str, float]:
    d = players.filter(pl.col(f"ex_{ref}").is_not_nan())
    d = d.filter((pl.col(f"ex_{ref}") ** 2 + pl.col(f"ey_{ref}") ** 2) < MAX_ERROR_M**2)
    stats: dict[str, list[tuple[float, float, int]]] = {"common": [], "individual": []}
    for (_clip,), g in d.partition_by("clip", as_dict=True).items():
        e = clip_tensor(g, ref)
        if e.shape[0] < 10:
            continue
        common, indiv = split_common_mode(e)
        for axis in range(2):
            s, r = fit_ar1(common[:, axis])
            stats["common"].append((s, r, e.shape[0]))
            for t in range(indiv.shape[1]):
                series = indiv[:, t, axis]
                if np.isfinite(series).sum() >= 10:
                    s, r = fit_ar1(series)
                    stats["individual"].append((s, r, int(np.isfinite(series).sum())))
    out: dict[str, float] = {"n_player_frames": float(d.height), "n_clips": float(d["clip"].n_unique())}
    for part, rows in stats.items():
        arr = np.array([r for r in rows if np.isfinite(r[0]) and np.isfinite(r[1])])
        w = arr[:, 2]
        sigma = float(np.sqrt(np.average(arr[:, 0] ** 2, weights=w)))  # pooled RMS of the per-series std
        rho5 = float(np.clip(np.average(arr[:, 1], weights=w), 0.0, 0.999))
        out[f"sigma_{part}"] = sigma
        out[f"rho5_{part}"] = rho5
        out[f"rho25_{part}"] = rho5 ** (1 / STEP_FRAMES)
    return out


@app.command()
def main(players_path: Path, out: Path = Path("reports/error_process_dev.md")) -> None:
    players = pl.read_parquet(players_path)
    git = ["git", "rev-parse", "--short", "HEAD"]
    sha = subprocess.run(git, capture_output=True, text=True).stdout.strip()
    md = [
        "# Position-error process on SN-GSR dev (T4.2)",
        "",
        f"`uv run python scripts/error_process.py {players_path}` · commit `{sha}`",
        "",
        "Errors of predfoot's positions (ground-truth boxes, our calibration), split into a common part",
        "shared by the players of a frame (mostly the camera) and an individual part. σ is the marginal",
        "standard deviation per axis; ρ is the lag-one autocorrelation at 5 fps, converted to 25 Hz.",
        f"Errors above {MAX_ERROR_M:.0f} m are calibration failures, left to abstention, and excluded here.",
        "",
        "| Reference | Player-frames | Clips | σ common (m) | ρ common 5 fps → 25 Hz | σ individual (m) |"
        " ρ individual 5 fps → 25 Hz |",
        "|---|---|---|---|---|---|---|",
    ]
    for ref, title in (("lab", "labels"), ("or", "trusted line oracle")):
        m = measure(players, ref)
        md.append(
            f"| {title} | {m['n_player_frames']:.0f} | {m['n_clips']:.0f} | {m['sigma_common']:.2f} |"
            f" {m['rho5_common']:.2f} → {m['rho25_common']:.3f} | {m['sigma_individual']:.2f} |"
            f" {m['rho5_individual']:.2f} → {m['rho25_individual']:.3f} |"
        )
    md += [
        "",
        "Reading: a ρ close to 1 at 25 Hz means slow errors (a camera that is wrong for seconds at a time),",
        "which the KPI study found to be the least harmful to distances; ρ ≈ 0.9 (about 0.4 s) was the most",
        "harmful. These parameters are the input of the KPI budget with measured errors (T4.3).",
        "",
        "Caveats. (1) Errors against the labels include the labels' own error, so σ is an upper bound of",
        "predfoot's. (2) Each frame is calibrated independently, so part of the error is white jitter at any",
        "frame rate: the exponential conversion from 5 fps to 25 Hz overstates the 25 Hz correlation of that",
        "part. A direct measurement on clips decoded at the full 25 fps (planned in T1.6) will settle it.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    typer.echo("\n".join(md))


if __name__ == "__main__":
    app()
