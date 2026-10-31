"""Measure predfoot's position-error process on SN-GSR dev (ticket T4.2), for the KPI error budget (T4.3), and
compare two cameras (per-frame calibration against temporal smoothing).

Reads ``players.parquet`` written by scripts/eval_dev.py (frames one in five, so 5 fps). For each clip the
errors are split into a common part (per-frame median over players: mostly the camera) and an individual
part. For each reference (labels, trusted line oracle, smoothed labels when present) and part: the marginal
standard deviation per axis, the autocorrelation at lags 1-5 (0.2-1.0 s, pooled over series and weighted by
their number of pairs), the RMS of the lag-1 steps, and a bracket of the 25 Hz lag-one correlation under an
exponential decay, ρ₂₅ ∈ [ρ(0.2 s)^(1/5), ρ(1 s)^(1/25)].

KPI proxy (model-free, the primary evidence for the KPI cost): per track, runs of at least 8 consecutive
cached frames with every error against the labels below 10 m; Savitzky-Golay positions (window 5, order 2)
and velocities (same filter, first derivative) for the prediction, the smoothed labels and the raw labels.
Distance and high-speed-running distance (speed ≥ 5.56 m/s) of the prediction are compared with the smoothed
labels; the raw labels' own bias is the noise floor of that reference.

Usage:
    uv run python scripts/error_process.py runs/<run>/players.parquet --out reports/error_process_dev.md
    uv run python scripts/error_process.py runs/<ptz>/players.parquet --compare runs/<none>/players.parquet \\
        --out runs/error_process_smoothed.md
With ``--compare``, both tables are restricted to the player-frames where both have finite errors.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import typer
from scipy.signal import savgol_filter

from predfoot.kpi.errors import split_common_mode

app = typer.Typer(add_completion=False)
STEP_FRAMES = 5  # cached frames are one in five of the native 25 fps
DT_S = STEP_FRAMES / 25.0
MAX_ERROR_M = 10.0  # errors this large are calibration failures, handled by abstention, not by the budget
LAGS = (1, 2, 3, 4, 5)
MIN_SERIES = 10
MIN_RUN = 8
SG_WINDOW, SG_ORDER = 5, 2
HSR_MS = 5.56  # 20 km/h
KEYS = ["clip", "frame", "track_id"]
TITLES = {"lab": "labels", "or": "trusted line oracle", "sm": "smoothed labels"}


def references(players: pl.DataFrame) -> list[str]:
    return [r for r in ("lab", "or", "sm") if f"ex_{r}" in players.columns]


def clip_tensor(d: pl.DataFrame, ref: str) -> np.ndarray:
    """Errors of one clip as (n_frames, n_tracks, 2) on the regular grid of cached frames, NaN if missing."""
    frames = d["frame"].to_numpy()
    f0 = int(frames.min())
    tracks = np.unique(d["track_id"].to_numpy())
    e = np.full(((int(frames.max()) - f0) // STEP_FRAMES + 1, len(tracks), 2), np.nan)
    fi = (frames - f0) // STEP_FRAMES
    ti = np.searchsorted(tracks, d["track_id"].to_numpy())
    e[fi, ti] = d.select(f"ex_{ref}", f"ey_{ref}").to_numpy()
    return e


class _Pool:
    """Accumulates pooled statistics over many 1-D series."""

    def __init__(self) -> None:
        self.sd: list[tuple[float, int]] = []
        self.acf: dict[int, list[tuple[float, int]]] = {lag: [] for lag in LAGS}
        self.step_sq, self.step_n = 0.0, 0

    def add(self, x: np.ndarray) -> None:
        n = int(np.isfinite(x).sum())
        if n < MIN_SERIES:
            return
        self.sd.append((float(np.nanstd(x)), n))
        for lag in LAGS:
            a, b = x[:-lag], x[lag:]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                self.acf[lag].append((float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())))
        d1 = np.diff(x)
        d1 = d1[np.isfinite(d1)]
        self.step_sq += float(np.sum(d1**2))
        self.step_n += d1.size

    def summary(self) -> dict[str, Any]:
        sd = np.array(self.sd) if self.sd else np.full((1, 2), np.nan)
        acf = {}
        for lag, rows in self.acf.items():
            arr = np.array(rows) if rows else np.full((1, 2), np.nan)
            acf[lag] = float(np.average(arr[:, 0], weights=arr[:, 1])) if rows else float("nan")
        r1, r5 = (max(acf[lag], 0.0) for lag in (1, 5))
        return {
            "sigma": float(np.sqrt(np.average(sd[:, 0] ** 2, weights=sd[:, 1]))) if self.sd else float("nan"),
            "acf": acf,
            "step_rms": float(np.sqrt(self.step_sq / self.step_n)) if self.step_n else float("nan"),
            "rho25": (r1 ** (1 / 5), r5 ** (1 / 25)),
            "n_series": len(self.sd),
        }


def measure(players: pl.DataFrame, ref: str) -> dict[str, Any]:
    d = players.filter(pl.col(f"ex_{ref}").is_not_nan() & pl.col(f"ex_{ref}").is_not_null())
    d = d.filter((pl.col(f"ex_{ref}") ** 2 + pl.col(f"ey_{ref}") ** 2) < MAX_ERROR_M**2)
    pools = {"common": _Pool(), "individual": _Pool()}
    for (_clip,), g in d.partition_by("clip", as_dict=True).items():
        e = clip_tensor(g, ref)
        if e.shape[0] < MIN_SERIES:
            continue
        common, indiv = split_common_mode(e)
        for axis in range(2):
            pools["common"].add(common[:, axis])
            for t in range(indiv.shape[1]):
                pools["individual"].add(indiv[:, t, axis])
    out: dict[str, Any] = {"n_player_frames": d.height, "n_clips": d["clip"].n_unique()}
    for part, pool in pools.items():
        out[part] = pool.summary()
    return out


def kpi_proxy(players: pl.DataFrame) -> dict[str, float] | None:
    """Distance, HSR distance and speed of the prediction against the smoothed labels (5 fps, SG filters)."""
    if not {"x_pred", "y_pred", "ex_sm", "ey_sm"} <= set(players.columns):
        return None
    d = players.filter(pl.col("ex_lab").is_not_nan() & pl.col("ex_sm").is_not_nan()).sort(KEYS)
    tot = {"pred": 0.0, "sm": 0.0, "raw": 0.0, "hsr_pred": 0.0, "hsr_sm": 0.0, "hsr_raw": 0.0}
    speed_err: list[np.ndarray] = []
    n_runs = n_samples = 0
    for (_clip, _track), g in d.partition_by("clip", "track_id", as_dict=True).items():
        frames = g["frame"].to_numpy()
        pred = g.select("x_pred", "y_pred").to_numpy()
        sm = pred - g.select("ex_sm", "ey_sm").to_numpy()
        raw = pred - g.select("ex_lab", "ey_lab").to_numpy()
        good = np.hypot(*(pred - raw).T) < MAX_ERROR_M
        breaks = np.flatnonzero((np.diff(frames) != STEP_FRAMES) | ~good[1:] | ~good[:-1]) + 1
        for idx in np.split(np.arange(len(frames)), breaks):
            idx = idx[good[idx]]
            if len(idx) < MIN_RUN:
                continue
            n_runs += 1
            n_samples += len(idx)
            speeds = {}
            for name, p in (("pred", pred[idx]), ("sm", sm[idx]), ("raw", raw[idx])):
                pos = savgol_filter(p, SG_WINDOW, SG_ORDER, axis=0)
                vel = savgol_filter(p, SG_WINDOW, SG_ORDER, deriv=1, delta=DT_S, axis=0)
                speed = np.linalg.norm(vel, axis=1)
                steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
                fast = 0.5 * (speed[1:] + speed[:-1]) >= HSR_MS
                tot[name] += float(steps.sum())
                tot[f"hsr_{name}"] += float(steps[fast].sum())
                speeds[name] = speed
            speed_err.append(speeds["pred"] - speeds["sm"])
    if not n_runs:
        return None
    err = np.concatenate(speed_err)
    return {
        "runs": n_runs,
        "samples": n_samples,
        "distance_bias": tot["pred"] / tot["sm"] - 1,
        "raw_distance_bias": tot["raw"] / tot["sm"] - 1,
        "hsr_bias": tot["hsr_pred"] / tot["hsr_sm"] - 1 if tot["hsr_sm"] > 0 else float("nan"),
        "raw_hsr_bias": tot["hsr_raw"] / tot["hsr_sm"] - 1 if tot["hsr_sm"] > 0 else float("nan"),
        "hsr_sm_m": tot["hsr_sm"],
        "speed_rms": float(np.sqrt(np.mean(err**2))),
        "speed_p90": float(np.quantile(np.abs(err), 0.9)),
    }


def _load(path: Path, include_filled: bool) -> pl.DataFrame:
    d = pl.read_parquet(path)
    if "filled" in d.columns and not include_filled:
        d = d.filter(~pl.col("filled"))
    return d


def _restrict(a: pl.DataFrame, b: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Both tables on the player-frames where both have finite errors for every shared reference."""
    refs = [r for r in references(a) if r in references(b)]
    finite = [pl.col(f"ex_{r}").is_not_nan() for r in refs if r != "or"]
    ka = a.filter(*finite).select(KEYS)
    kb = b.filter(*finite).select(KEYS)
    both = ka.join(kb, on=KEYS)
    return a.join(both, on=KEYS), b.join(both, on=KEYS)


def camera_labels(paths: list[Path]) -> list[str]:
    """Distinct table labels: the folder name, else folder/stem, else the full path."""
    for label in (lambda p: p.parent.name or str(p), lambda p: f"{p.parent.name}/{p.stem}", str):
        names = [label(p) for p in paths]
        if len(set(names)) == len(names):
            return names
    return [f"{p} ({i + 1})" for i, p in enumerate(paths)]


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


@app.command()
def main(
    players_path: Path,
    compare: Path | None = typer.Option(None, help="players.parquet of another camera (e.g. per-frame)."),  # noqa: B008
    include_filled: bool = typer.Option(False, help="Keep frames whose raw calibration failed (filled)."),
    out: Path = Path("reports/error_process_dev.md"),
) -> None:
    main_t = _load(players_path, include_filled)
    tables = {camera_labels([players_path])[0]: main_t}
    if compare is not None:
        main_t, other_t = _restrict(main_t, _load(compare, include_filled))
        name_other, name_main = camera_labels([compare, players_path])
        tables = {name_other: other_t, name_main: main_t}
    cmd = f"uv run python scripts/error_process.py {players_path}"
    cmd += f" --compare {compare}" if compare is not None else ""
    cmd += " --include-filled" if include_filled else ""
    md = [
        "# Position-error process on SN-GSR dev (T4.2)",
        "",
        f"`{cmd}` · commit `{git_sha()}`",
        "",
        "Errors of predfoot's positions (ground-truth boxes, our calibration), split into a common part",
        "shared by the players of a frame (mostly the camera) and an individual part. σ is the marginal",
        "standard deviation per axis; ACF the autocorrelation at 0.2-1.0 s (frames are one in five);",
        "ρ₂₅ the 25 Hz lag-one correlation implied by the 0.2 s and by the 1 s lag under an exponential",
        "decay.",
        f"Errors above {MAX_ERROR_M:.0f} m are calibration failures, left to abstention, and excluded here."
        + (
            " Frames filled by the smoother (raw calibration failed) are excluded."
            if not include_filled
            else ""
        ),
    ]
    if compare is not None:
        md += ["", f"Both tables restricted to the same {next(iter(tables.values())).height} player-frames."]
    measured: dict[str, dict[str, dict[str, Any]]] = {}
    for ref in references(next(iter(tables.values()))):
        if not all(f"ex_{ref}" in t.columns for t in tables.values()):
            continue
        md += [
            "",
            f"## Reference: {TITLES[ref]}",
            "",
            "| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s |"
            " Lag-1 step RMS (m) | ρ₂₅ bracket |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, t in tables.items():
            m = measure(t, ref)
            measured.setdefault(name, {})[ref] = m
            for part in ("common", "individual"):
                s = m[part]
                acf = " / ".join(f"{s['acf'][lag]:.2f}" for lag in LAGS)
                rho = f"{s['rho25'][0]:.3f} – {s['rho25'][1]:.3f}"
                md.append(
                    f"| {name} | {m['n_player_frames']} | {part} | {s['sigma']:.3f} | {acf} |"
                    f" {s['step_rms']:.3f} | {rho} |"
                )
    proxies = {name: kpi_proxy(t) for name, t in tables.items()}
    if any(p is not None for p in proxies.values()):
        md += [
            "",
            "## KPI proxy on SN-GSR (5 fps, Savitzky-Golay window 5, order 2)",
            "",
            "| Camera | Runs | Samples | Distance bias vs smoothed labels | HSR-distance bias |"
            " Speed error RMS / p90 (m/s) |",
            "|---|---|---|---|---|---|",
        ]
        for name, p in proxies.items():
            if p is None:
                md.append(f"| {name} | – | – | – | – | – |")
                continue
            md.append(
                f"| {name} | {p['runs']} | {p['samples']} | {p['distance_bias']:+.1%} |"
                f" {p['hsr_bias']:+.1%} | {p['speed_rms']:.2f} / {p['speed_p90']:.2f} |"
            )
        p0 = next(p for p in proxies.values() if p is not None)
        md += [
            "",
            "Noise floor of the reference: the raw labels' own distance bias against the smoothed labels is"
            f" {p0['raw_distance_bias']:+.1%} (HSR distance {p0['raw_hsr_bias']:+.1%}; the smoothed labels"
            f" cover {p0['hsr_sm_m']:.0f} m at high speed).",
        ]
    md += ["", "## Arguments for kpi_sensitivity.py (labels reference)", ""]
    for name, refs in measured.items():
        if "lab" not in refs:
            continue
        ci, cc = refs["lab"]["individual"], refs["lab"]["common"]
        for i, bracket in enumerate(("0.2 s lag", "1 s lag")):
            md.append(
                f"- {name}, {bracket}: `--measured {ci['sigma']:.2f} {ci['rho25'][i]:.3f}"
                f" {cc['sigma']:.2f} {cc['rho25'][i]:.3f}`"
            )
    md += [
        "",
        "Caveats. (1) Errors against the labels include the labels' own error, so σ is an upper bound of",
        "predfoot's; the smoothed labels remove the annotators' jitter but carry track-smoothing artefacts,",
        "so use them for temporal metrics only. (2) The 25 Hz correlation is bracketed, not measured: the",
        "two lags disagree when the process is not a single exponential (white jitter plus slow drift), and",
        "only clips decoded at the full 25 fps (planned in T1.6) can settle it.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    typer.echo("\n".join(md))


if __name__ == "__main__":
    app()
