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
``--json`` also writes the numbers (per camera, reference, part and axis: σ, ACF with pair counts, lag-1 step
RMS, series lengths, and the KPI proxy with its run lengths) for scripts/kpi_spectral.py.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import typer
from scipy.special import ndtri

from predfoot.kpi.errors import split_common_mode
from predfoot.kpi.proxy import PROXY_MIN_RUN, ProxyTotals

app = typer.Typer(add_completion=False)
STEP_FRAMES = 5  # cached frames are one in five of the native 25 fps
DT_S = STEP_FRAMES / 25.0
MAX_ERROR_M = 10.0  # errors this large are calibration failures, handled by abstention, not by the budget
LAGS = (1, 2, 3, 4, 5)
MIN_SERIES = 10
MIN_RUN = PROXY_MIN_RUN
AXES = ("x", "y")
# Probabilities at which the marginal of the errors is stored (dense in the tails).
_TAIL_PROBS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025)
_PROBS = np.concatenate([_TAIL_PROBS, np.linspace(0.05, 0.95, 91), 1.0 - np.array(_TAIL_PROBS[::-1])])
MARGINAL_PROBS = tuple(float(p) for p in _PROBS.round(6))
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
        self.lengths: Counter[int] = Counter()

    def add(self, x: np.ndarray) -> None:
        n = int(np.isfinite(x).sum())
        if n < MIN_SERIES:
            return
        self.lengths[n] += 1
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
            "n_pairs": {lag: int(sum(n for _, n in rows)) for lag, rows in self.acf.items()},
            "step_rms": float(np.sqrt(self.step_sq / self.step_n)) if self.step_n else float("nan"),
            "rho25": (r1 ** (1 / 5), r5 ** (1 / 25)),
            "n_series": len(self.sd),
            "series_lengths": dict(sorted(self.lengths.items())),
        }


def measure(players: pl.DataFrame, ref: str) -> dict[str, Any]:
    d = players.filter(pl.col(f"ex_{ref}").is_not_nan() & pl.col(f"ex_{ref}").is_not_null())
    d = d.filter((pl.col(f"ex_{ref}") ** 2 + pl.col(f"ey_{ref}") ** 2) < MAX_ERROR_M**2)
    parts = ("common", "individual")
    series: dict[tuple[str, str], list[np.ndarray]] = {(part, axis): [] for part in parts for axis in AXES}
    for (_clip,), g in d.partition_by("clip", as_dict=True).items():
        e = clip_tensor(g, ref)
        if e.shape[0] < MIN_SERIES:
            continue
        common, indiv = split_common_mode(e)
        for axis, name in enumerate(AXES):
            series["common", name].append(common[:, axis])
            series["individual", name].extend(indiv[:, t, axis] for t in range(indiv.shape[1]))
    out: dict[str, Any] = {"n_player_frames": d.height, "n_clips": d["clip"].n_unique()}
    for part in parts:
        pooled = _Pool()
        per_axis: dict[str, Any] = {}
        for name in AXES:
            pool, scores = _Pool(), _Pool()
            kept = [x for x in series[part, name] if np.isfinite(x).sum() >= MIN_SERIES]
            for x in kept:
                pool.add(x)
                pooled.add(x)
            # Marginal of the errors around each series' mean (the σ above uses the same centring), and the
            # ACF of their normal scores Φ⁻¹(F̂(e)): the correlation of a Gaussian copula with that marginal.
            centred = [x - np.nanmean(x) for x in kept]
            values = np.sort(np.concatenate([x[np.isfinite(x)] for x in centred]))
            for x in centred:
                scores.add(normal_scores(x, values))
            per_axis[name] = pool.summary()
            per_axis[name]["quantiles"] = np.quantile(values, MARGINAL_PROBS).tolist()
            ns = scores.summary()
            per_axis[name]["acf_nscore"], per_axis[name]["n_pairs_nscore"] = ns["acf"], ns["n_pairs"]
        out[part] = pooled.summary()  # both axes pooled, as in the tables
        out[part].update(per_axis)
    out["marginal_probs"] = list(MARGINAL_PROBS)
    return out


def normal_scores(x: np.ndarray, sorted_values: np.ndarray) -> np.ndarray:
    """Φ⁻¹ of the pooled empirical CDF (mid-ranks) at each value of ``x``; NaN stays NaN."""
    n = sorted_values.size
    lo = np.searchsorted(sorted_values, x, side="left")
    hi = np.searchsorted(sorted_values, x, side="right")
    u = (0.5 * (lo + hi)) / n
    out = ndtri(np.clip(u, 0.5 / n, 1.0 - 0.5 / n))
    return np.where(np.isfinite(x), out, np.nan)


def kpi_proxy(players: pl.DataFrame) -> dict[str, Any] | None:
    """Distance, HSR distance and speed of the prediction against the smoothed labels (5 fps, SG filters)."""
    if not {"x_pred", "y_pred", "ex_sm", "ey_sm"} <= set(players.columns):
        return None
    d = players.filter(pl.col("ex_lab").is_not_nan() & pl.col("ex_sm").is_not_nan()).sort(KEYS)
    pred_vs_sm, raw_vs_sm = ProxyTotals(), ProxyTotals()
    run_lengths: Counter[int] = Counter()
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
            run_lengths[len(idx)] += 1
            pred_vs_sm.add(pred[idx], sm[idx], DT_S)
            raw_vs_sm.add(raw[idx], sm[idx], DT_S)
    if not pred_vs_sm.runs:
        return None
    p, r = pred_vs_sm.summary(), raw_vs_sm.summary()
    return {
        "runs": pred_vs_sm.runs,
        "samples": pred_vs_sm.samples,
        "distance_bias": p["distance_bias"],
        "raw_distance_bias": r["distance_bias"],
        "hsr_bias": p["hsr_bias"],
        "raw_hsr_bias": r["hsr_bias"],
        "hsr_sm_m": p["ref_hsr_m"],
        "sm_distance_m": p["ref_distance_m"],
        "sm_duration_s": p["duration_s"],
        "sm_mean_speed_ms": p["ref_mean_speed_ms"],
        "speed_rms": p["speed_rms"],
        "speed_p90": p["speed_p90"],
        "run_lengths": dict(sorted(run_lengths.items())),
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


def to_jsonable(obj: Any) -> Any:
    """Plain JSON types: dict keys as strings, tuples as lists, NaN as null."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float | np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


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
    json_out: Path | None = typer.Option(None, "--json", help="Also write the numbers as JSON here."),  # noqa: B008
    name: str | None = typer.Option(None, help="Camera label of PLAYERS_PATH (default: its folder)."),
    compare_name: str | None = typer.Option(None, help="Camera label of --compare (default: its folder)."),
) -> None:
    main_t = _load(players_path, include_filled)
    sources = {name or camera_labels([players_path])[0]: players_path}
    tables = {next(iter(sources)): main_t}
    if compare is not None:
        main_t, other_t = _restrict(main_t, _load(compare, include_filled))
        name_other, name_main = camera_labels([compare, players_path])
        name_other, name_main = compare_name or name_other, name or name_main
        if name_other == name_main:
            raise typer.BadParameter("the two cameras need distinct labels (--name / --compare-name)")
        tables = {name_other: other_t, name_main: main_t}
        sources = {name_other: compare, name_main: players_path}
    cmd = f"uv run python scripts/error_process.py {players_path}"
    cmd += f" --compare {compare}" if compare is not None else ""
    cmd += " --include-filled" if include_filled else ""
    cmd += f" --name {name}" if name else ""
    cmd += f" --compare-name {compare_name}" if compare_name else ""
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
    if json_out is not None:
        payload = {
            "command": cmd,
            "git_sha": git_sha(),
            "include_filled": include_filled,
            "dt_s": DT_S,
            "lags_s": [round(lag * DT_S, 6) for lag in LAGS],
            "max_error_m": MAX_ERROR_M,
            "cameras": {
                cam: {
                    "path": str(sources[cam]),
                    "references": measured.get(cam, {}),
                    "kpi_proxy": proxies[cam],
                }
                for cam in tables
            },
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(to_jsonable(payload), indent=1) + "\n")
        typer.echo(f"json: {json_out}")


if __name__ == "__main__":
    app()
