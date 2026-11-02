"""Spectral position-error model: fit, check against the measured KPI proxy, KPI budget at 10 fps.

    uv run python scripts/kpi_spectral.py                                  # all seven matches for the budget
    uv run python scripts/kpi_spectral.py --budget-matches J03WMX J03WN1   # quicker

1. **Fit.** For each camera of ``reports/error_process_temporal_dev.json`` (per-frame calibration ``none``,
   PTZ-smoothed ``ptz``), part (common, individual) and axis, the kernel families of ``predfoot.kpi.noise``
   (nugget + two scales of one kernel) are fitted to the 5 fps ACF measured against the smoothed labels:
   Gaussian processes with the measured σ, and Gaussian copulas with the measured marginal (heavy-tailed)
   fitted to the ACF of the normal scores. The previous model (one Gaussian Ornstein-Uhlenbeck process through
   the 0.2 s lag) is kept for comparison. Round trip: the RMS and median of the 0.2 s steps of simulated
   errors against the measured ones.
2. **Simulation-to-measurement check.** The SN-GSR KPI proxy protocol on IDSSE: clean 25 Hz positions
   subsampled at 5 fps and cut into runs drawn from the measured run-length distribution; simulated errors
   (sampled at 25 Hz, then subsampled) added; Savitzky-Golay window 5, order 2; distance and HSR distance
   (5.56 m/s) of the noisy runs against the same protocol on the clean runs (``predfoot.kpi.proxy``, the code
   that produced the SN-GSR numbers). Every model sees the same runs. Done on all frames and on ball-in-play
   frames only (SN-GSR clips are live play). Verdict: the check passes when the chosen model's ball-in-play
   distance and HSR-distance biases are both within a factor of ``PROXY_TOLERANCE`` of the measured ones; when
   it fails, the report says so above the budget.
3. **Budget at the export rate.** Errors at 25 Hz, then the 10 fps pipeline of ``predfoot.kpi.speed`` and the
   KPIs of ``predfoot.kpi.metrics``; MAPE, bias and Spearman per KPI against the clean 25 Hz reference, with
   intervals from a bootstrap over matches (``predfoot.kpi.report``), for every kernel and both nugget shapes.

Per-player tables and the fitted models go to ``runs/<date>_<sha>_spectral/``; the report to
``reports/kpi_budget_spectral.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from predfoot.io import idsse
from predfoot.kpi.config import KpiConfig, load_kpi_config
from predfoot.kpi.metrics import kpi_names
from predfoot.kpi.noise import AxisProcess, SpectralErrorModel, fit_error_model, sample_axis, sample_errors
from predfoot.kpi.proxy import PROXY_MIN_RUN, ProxyTotals
from predfoot.kpi.report import compare_to_reference
from predfoot.kpi.simulate import PeriodTracks, period_tracks, player_kpis, seeded_rng
from predfoot.kpi.speed import pipeline, segments
from predfoot.paths import REPO_ROOT, reports_dir

FAMILIES = ("ou", "exp", "matern32", "matern52", "se")
KERNELS = FAMILIES[1:]
NUGGET_SHAPES = ("white", "bandlimited")
# (family, marginal) pairs seen by the 5 fps proxy check; every model there has a white nugget, since the
# nugget's shape is invisible at 5 fps.
PROXY_SPECS = (
    ("ou", "gauss"),
    ("matern32", "gauss"),
    ("exp", "copula"),
    ("matern32", "copula"),
    ("se", "copula"),
)
MARGINAL_LABELS = {"gauss": "Gaussian", "copula": "measured marginal"}
KERNEL_LABELS = {
    "ou": "single OU through 0.2 s (previous)",
    "exp": "nugget + exponential",
    "matern32": "nugget + Matérn-3/2",
    "matern52": "nugget + Matérn-5/2",
    "se": "nugget + squared exponential",
}
PARTS = ("common", "individual")
AXES = ("x", "y")
STEP = 5  # native frames per 5 fps sample
#: The proxy check passes when simulated / measured bias is within this factor (distance and HSR distance).
PROXY_TOLERANCE = 2.0
GOALKEEPER = "TW"
BUDGET_KPIS = ("distance_m", "hi_distance_m", "dist_gt25_m", "hsr_efforts", "sprint_efforts", "psv99_kmh")
KPI_LABELS = {
    "distance_m": "Distance",
    "dist_lt15_m": "Dist. < 15",
    "dist_15_20_m": "Dist. 15-20",
    "dist_20_25_m": "Dist. 20-25",
    "dist_gt25_m": "Dist. > 25",
    "hi_distance_m": "HI dist. ≥ 20",
    "hsr_efforts": "HSR efforts",
    "sprint_efforts": "Sprint efforts",
    "psv99_kmh": "PSV-99",
    "explosive_to_hsr": "Expl. acc. to HSR",
    "explosive_to_sprint": "Expl. acc. to sprint",
}


# --- models ----------------------------------------------------------------------------------------------


def fit_family(
    stats: dict[str, Any], camera: str, family: str, ref: str, marginal: str
) -> tuple[SpectralErrorModel, dict[tuple[str, str], float]]:
    """The model of one family and marginal for one camera, and the max |ACF residual| per (part, axis)."""
    return fit_error_model(stats, camera, family=family, marginal=marginal, reference=ref)


def model_label(name: str) -> str:
    camera, family, marginal, nugget = name.split(":")
    if family == "ou":
        return f"{camera}: {KERNEL_LABELS[family]}"
    return f"{camera}: {KERNEL_LABELS[family]}, {MARGINAL_LABELS[marginal]}, {nugget} nugget"


# --- IDSSE -----------------------------------------------------------------------------------------------


def ball_alive(match_id: str, tracks: Sequence[PeriodTracks]) -> list[np.ndarray]:
    """Ball-in-play flag on each period's frame grid (False where the ball feed has no frame)."""
    ball = idsse.load_ball(match_id)
    out = []
    for t in tracks:
        b = ball.filter(pl.col("period") == t.period)
        alive = np.zeros(t.xy.shape[0], dtype=bool)
        rows = b["frame"].to_numpy() - t.frame0
        ok = (rows >= 0) & (rows < alive.size)
        alive[rows[ok]] = b["alive"].to_numpy()[ok]
        out.append(alive)
    return out


def cut_runs(start: int, stop: int, lengths: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Consecutive runs inside [start, stop) with lengths drawn from ``lengths``; leftovers shorter than
    the proxy's minimum are dropped."""
    out = []
    pos = start
    while stop - pos >= PROXY_MIN_RUN:
        n = min(int(rng.choice(lengths)), stop - pos)
        out.append((pos, pos + n))
        pos += n
    return out


def proxy_check(
    matches: dict[str, list[PeriodTracks]],
    alive: dict[str, list[np.ndarray]],
    model: SpectralErrorModel | None,
    run_lengths: np.ndarray,
    seed: int,
    in_play: bool,
) -> dict[str, float]:
    """The SN-GSR KPI proxy on IDSSE (module docstring, step 2). ``model=None`` returns the clean protocol."""
    totals = ProxyTotals()
    for match_id, tracks in matches.items():
        for p, t in enumerate(tracks):
            x5 = t.xy[::STEP]
            y5 = x5
            if model is not None:
                rng = seeded_rng(seed, match_id, model.name, p, "proxy")
                y5 = x5 + sample_errors(model, t.xy.shape[0], t.xy.shape[1], rng)[::STEP]
            keep = alive[match_id][p][::STEP] if in_play else np.ones(x5.shape[0], dtype=bool)
            runs_rng = seeded_rng(seed, match_id, p, "runs", int(in_play))  # the same runs for every model
            for i in range(x5.shape[1]):
                starts, stops = segments(np.isfinite(x5[:, i, 0]) & keep)
                for s, e in zip(starts.tolist(), stops.tolist(), strict=True):
                    for a, b in cut_runs(s, e, run_lengths, runs_rng):
                        totals.add(y5[a:b, i], x5[a:b, i])
    return totals.summary()


def ghost_per_min(model: SpectralErrorModel, cfg: KpiConfig, seed: int, minutes: int = 20) -> float:
    """Distance per minute of a motionless player through the 10 fps pipeline (one player's full error)."""
    n = minutes * 60 * cfg.native_hz + 1
    rng = seeded_rng(seed, "ghost", model.name)
    e = sample_errors(model, n, 8, rng)
    total = 0.0
    for i in range(e.shape[1]):
        kin = pipeline(e[:, i], cfg, resample_to=cfg.output_hz)
        total += float(np.nansum(np.hypot(*np.diff(kin.xy, axis=0).T)))
    return total / (e.shape[1] * minutes)


def step_round_trip(
    stats: dict[str, Any],
    reference: str,
    models: dict[str, dict[str, SpectralErrorModel]],
    seed: int,
    minutes: int = 10,
    n_paths: int = 16,
) -> list[dict[str, Any]]:
    """Measured RMS and median |step| of the 0.2 s error steps per camera, part and axis, next to the same
    statistics of ``minutes`` of samples (25 Hz, taken one in five) of each labelled model."""
    rows = []
    for camera, labelled in models.items():
        for part in PARTS:
            for axis in AXES:
                s = stats["cameras"][camera]["references"][reference][part][axis]
                row: dict[str, Any] = {
                    "camera": camera,
                    "part": part,
                    "axis": axis,
                    "measured_rms": s["step_rms"],
                    "measured_median": s.get("step_abs_median", math.nan),
                }
                for label, model in labelled.items():
                    proc = getattr(model, part).axis(axis)
                    rng = seeded_rng(seed, "steps", model.name, part, axis)
                    d = np.diff(sample_axis([proc], minutes * 60 * 25, n_paths, rng)[::STEP], axis=0)
                    row[f"{label}_rms"] = float(np.sqrt(np.mean(d**2)))
                    row[f"{label}_median"] = float(np.median(np.abs(d)))
                rows.append(row)
    return rows


def tail_ranges(stats: dict[str, Any], cameras: Sequence[str], reference: str) -> dict[str, float]:
    """Range over cameras, parts and axes of the robust σ (IQR / 1.349) as a fraction of the SD, and of the
    kurtosis, from the measured marginals."""
    robust, kurt = [], []
    for camera in cameras:
        r = stats["cameras"][camera]["references"][reference]
        probs = np.asarray(r["marginal_probs"], dtype=float)
        for part in PARTS:
            for axis in AXES:
                s = r[part][axis]
                q25, q75 = np.interp([0.25, 0.75], probs, np.asarray(s["quantiles"], dtype=float))
                robust.append(float(q75 - q25) / 1.349 / s["sigma"])
                if s.get("kurtosis") is not None:
                    kurt.append(float(s["kurtosis"]))
    return {
        "robust_lo": min(robust),
        "robust_hi": max(robust),
        "kurt_lo": min(kurt) if kurt else math.nan,
        "kurt_hi": max(kurt) if kurt else math.nan,
    }


def proxy_verdict(measured: dict[str, Any], simulated: dict[str, Any]) -> dict[str, Any]:
    """Simulated / measured ratios of the distance and HSR-distance biases and whether both are within
    ``PROXY_TOLERANCE``. A ratio is undefined when the measured bias is not positive; the check then fails."""
    out: dict[str, Any] = {}
    for key in ("distance_bias", "hsr_bias"):
        m, s = measured[key], simulated[key]
        out[key] = s / m if m is not None and s is not None and m > 0 else math.nan
    out["passes"] = all(
        math.isfinite(out[k]) and 1.0 / PROXY_TOLERANCE <= out[k] <= PROXY_TOLERANCE
        for k in ("distance_bias", "hsr_bias")
    )
    return out


# --- report helpers --------------------------------------------------------------------------------------


def fmt(v: float | None, digits: int = 2, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def pct(v: float | None, signed: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{100 * v:+.1f} %" if signed else f"{100 * v:.1f} %"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def ci(row: dict[str, Any], name: str, digits: int = 2, signed: bool = False) -> str:
    return (
        f"{fmt(row[name], digits, signed)} [{fmt(row[name + '_lo'], digits, signed)}, "
        f"{fmt(row[name + '_hi'], digits, signed)}]"
    )


def git_info() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)reports"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return sha.stdout.strip() or "unknown", bool(status.stdout.strip())


def model_table(models: dict[str, SpectralErrorModel]) -> list[list[str]]:
    rows = []
    for camera, model in models.items():
        for part in PARTS:
            pp = getattr(model, part)
            for axis in AXES:
                a: AxisProcess = pp.axis(axis)
                rows.append(
                    [
                        camera,
                        part,
                        axis,
                        fmt(a.sigma, 3),
                        fmt(a.w_nugget, 2),
                        fmt(a.length_s, 2),
                        fmt(a.w_slow, 2),
                        fmt(a.slow_s, 1) if a.w_slow > 0 else "–",
                        fmt(math.sqrt(a.derivative_variance()), 2)
                        if math.isfinite(a.derivative_variance())
                        else "∞",
                    ]
                )
    return rows


# --- main ------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stats", type=Path, default=reports_dir() / "error_process_temporal_dev.json")
    parser.add_argument("--reference", default="sm", help="error reference in the stats file")
    parser.add_argument("--family", default="matern32", choices=KERNELS, help="kernel of the chosen model")
    parser.add_argument("--marginal", default="copula", choices=("gauss", "copula"))
    parser.add_argument("--proxy-matches", nargs="*", default=["J03WMX"])
    parser.add_argument("--budget-matches", nargs="*", default=sorted(idsse.IDSSE_MATCHES))
    parser.add_argument("--min-minutes", type=float, default=45.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--report", type=Path, default=reports_dir() / "kpi_budget_spectral.md")
    args = parser.parse_args()

    t_start = time.perf_counter()
    cfg = load_kpi_config()
    stats = json.loads(args.stats.read_text())
    cameras = list(stats["cameras"])
    sha, dirty = git_info()
    run_dir = REPO_ROOT / "runs" / f"{dt.date.today():%Y-%m-%d}_{sha}_spectral"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fit every (family, marginal) model used below.
    specs = {("ou", "gauss"), ("matern32", "gauss"), (args.family, args.marginal)}
    specs |= {(k, "copula") for k in KERNELS} | {(k, "gauss") for k in KERNELS}
    models: dict[tuple[str, str, str], SpectralErrorModel] = {}
    residuals: dict[tuple[str, str, str], dict[tuple[str, str], float]] = {}
    for camera in cameras:
        for family, marginal in sorted(specs):
            key = (camera, family, marginal)
            models[key], residuals[key] = fit_family(stats, camera, family, args.reference, marginal)
    (run_dir / "models.json").write_text(json.dumps({m.name: asdict(m) for m in models.values()}, indent=1))
    print(f"fitted {len(models)} models in {time.perf_counter() - t_start:.1f} s", flush=True)

    # 2. Proxy check (white nugget everywhere: its shape is invisible at 5 fps).
    all_matches = sorted(set(args.proxy_matches) | set(args.budget_matches))
    tracks: dict[str, list[PeriodTracks]] = {}
    alive: dict[str, list[np.ndarray]] = {}
    for match_id in all_matches:
        idsse.prepare_match(match_id)
        tracks[match_id], _, _ = period_tracks(idsse.load_players(match_id), cfg)
        alive[match_id] = ball_alive(match_id, tracks[match_id])
    proxy_tracks = {m: tracks[m] for m in args.proxy_matches}
    lengths = {
        c: np.repeat(
            [int(k) for k in stats["cameras"][c]["kpi_proxy"]["run_lengths"]],
            list(stats["cameras"][c]["kpi_proxy"]["run_lengths"].values()),
        )
        for c in cameras
    }
    proxy_specs = list(dict.fromkeys([*PROXY_SPECS, (args.family, args.marginal)]))
    proxy_rows: list[dict[str, Any]] = []
    for in_play in (False, True):
        clean = proxy_check(proxy_tracks, alive, None, lengths[cameras[0]], args.seed, in_play)
        proxy_rows.append({"model": "clean", "camera": "clean", "in_play": in_play, **clean})
        for camera in cameras:
            for family, marginal in proxy_specs:
                model = models[camera, family, marginal]
                res = proxy_check(proxy_tracks, alive, model, lengths[camera], args.seed, in_play)
                proxy_rows.append({"model": model.name, "camera": camera, "in_play": in_play, **res})
        print(f"proxy check (in play: {in_play}) at {time.perf_counter() - t_start:.1f} s", flush=True)
    proxy = pl.DataFrame(proxy_rows)
    proxy.write_parquet(run_dir / "proxy_check.parquet")

    # 3. Budget at 10 fps.
    budget: list[SpectralErrorModel] = []
    for camera in cameras:
        budget += [models[camera, "ou", "gauss"], models[camera, "matern32", "gauss"]]
        for family in KERNELS:
            for nugget in NUGGET_SHAPES:
                m = models[camera, family, args.marginal]
                budget.append(m.with_nugget(nugget, name=f"{camera}:{family}:{args.marginal}:{nugget}"))
    budget = list({m.name: m for m in budget}.values())
    rows: list[dict[str, Any]] = []
    for match_id in args.budget_matches:
        t0 = time.perf_counter()
        tr = tracks[match_id]
        for name, resample_to in (("reference", None), ("control", cfg.output_hz)):
            for pid, k in player_kpis(tr, cfg, resample_to=resample_to).items():
                rows.append({"match_id": match_id, "player_id": pid, "condition": name, **k})
        for model in budget:
            errs = []
            for p, t in enumerate(tr):
                rng = seeded_rng(args.seed, match_id, model.name, p)
                errs.append(sample_errors(model, t.xy.shape[0], t.xy.shape[1], rng))
            for pid, k in player_kpis(tr, cfg, resample_to=cfg.output_hz, errors=errs).items():
                rows.append({"match_id": match_id, "player_id": pid, "condition": model.name, **k})
        print(f"{match_id}: budget in {time.perf_counter() - t0:.1f} s", flush=True)
    table = pl.DataFrame(rows)
    roster = pl.concat(
        [idsse.load_roster(m).with_columns(pl.lit(m).alias("match_id")) for m in args.budget_matches]
    ).select("match_id", "player_id", "position")
    ref_minutes = table.filter(pl.col("condition") == "reference").select("match_id", "player_id", "minutes")
    table = (
        table.drop("minutes")
        .join(ref_minutes, on=["match_id", "player_id"])
        .join(roster, on=["match_id", "player_id"], how="left")
    )
    table.write_parquet(run_dir / "kpis.parquet")
    population = table.filter((pl.col("position") != GOALKEEPER) & (pl.col("minutes") >= args.min_minutes))
    kpis = [k for k in kpi_names(cfg) if k != "minutes"]
    per90 = [k for k in kpis if k != "psv99_kmh"]

    def compare(test: str) -> pl.DataFrame:
        return compare_to_reference(
            population.filter(pl.col("condition") == "reference"),
            population.filter(pl.col("condition") == test),
            kpis,
            per90=per90,
            n_boot=args.n_boot,
            seed=args.seed,
        ).with_columns(pl.lit(test).alias("condition"))

    comp = pl.concat([compare("control")] + [compare(m.name) for m in budget])
    comp.write_parquet(run_dir / "comparisons.parquet")
    ghosts = {m.name: ghost_per_min(m, cfg, args.seed) for m in budget}
    print(f"comparisons done at {time.perf_counter() - t_start:.1f} s", flush=True)

    stats_path = args.stats.resolve()
    meta = {
        "command": "uv run python scripts/kpi_spectral.py " + " ".join(shlex.quote(a) for a in sys.argv[1:]),
        "git_sha": sha,
        "git_dirty": dirty,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "stats": str(stats_path.relative_to(REPO_ROOT))
        if stats_path.is_relative_to(REPO_ROOT)
        else str(stats_path),
        "stats_command": stats["command"],
        "reference": args.reference,
        "chosen": f"{args.family}:{args.marginal}:white",
        "proxy_matches": args.proxy_matches,
        "budget_matches": args.budget_matches,
        "n_player_matches": population.filter(pl.col("condition") == "reference").height,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    chosen = {c: models[c, args.family, args.marginal] for c in cameras}
    steps = step_round_trip(
        stats,
        args.reference,
        {c: {"gauss": models[c, args.family, "gauss"], "chosen": chosen[c]} for c in cameras},
        args.seed,
    )
    report = render(
        meta, stats, cameras, chosen, residuals, proxy, comp, ghosts, args.family, args.marginal, steps
    )
    args.report.write_text(report, encoding="utf-8")
    print(f"report: {args.report}\nrun: {run_dir}\n{meta['runtime_s']} s")


def render(
    meta: dict[str, Any],
    stats: dict[str, Any],
    cameras: list[str],
    chosen: dict[str, SpectralErrorModel],
    residuals: dict[tuple[str, str, str], dict[tuple[str, str], float]],
    proxy: pl.DataFrame,
    comp: pl.DataFrame,
    ghosts: dict[str, float],
    family: str,
    marginal: str,
    steps: list[dict[str, Any]],
) -> str:
    dirty = " (with uncommitted changes)" if meta["git_dirty"] else ""
    measured = {c: stats["cameras"][c]["kpi_proxy"] for c in cameras}
    tails = tail_ranges(stats, cameras, meta["reference"])
    kurtosis = (
        f" and kurtosis {tails['kurt_lo']:.0f} to {tails['kurt_hi']:.0f}"
        if math.isfinite(tails["kurt_lo"])
        else ""
    )

    def cell(cond: str, kpi: str, metric: str) -> float:
        sub = comp.filter((pl.col("condition") == cond) & (pl.col("kpi") == kpi))
        return float(sub[metric][0])

    def prow(model: str, in_play: bool) -> dict[str, Any]:
        sub = proxy.filter((pl.col("model") == model) & (pl.col("in_play") == in_play))
        return sub.row(0, named=True)

    chosen_name = {c: f"{c}:{family}:{marginal}:white" for c in cameras}
    lines = [
        "# KPI error budget with a spectral error model",
        "",
        "> **Simulation, not measurement.** Every KPI error below comes from errors *simulated* by a model",
        "> fitted to predfoot's error process on SN-GSR dev (5 fps), injected into clean IDSSE tracking. The",
        "> only measured KPI numbers are the SN-GSR proxy values quoted in section 2.",
        "",
        f"- **Command**: `{meta['command']}`",
        f"- **Git commit**: `{meta['git_sha']}`{dirty}; date {meta['date']}; runtime {meta['runtime_s']} s",
        f"- **Error process**: `{meta['stats']}` (from `{meta['stats_command']}`), errors against the"
        " smoothed labels, frames one in five (5 fps), per axis (x along the touchlines, y across) and part.",
        f"- **IDSSE**: proxy check on {', '.join(meta['proxy_matches'])}; budget on"
        f" {len(meta['budget_matches'])} matches ({', '.join(meta['budget_matches'])}),"
        f" {meta['n_player_matches']} player-matches (outfield, ≥ 45 min). IDSSE: Bassek, Rein, Weber &"
        " Memmert 2025, CC BY 4.0.",
        f"- **Seed** {meta['seed']}, {meta['n_boot']} bootstrap resamples over matches; one noise draw per"
        " condition and match.",
        "",
        "## 1. Error model",
        "",
        "Per axis and part, a stationary process with autocorrelation ρ(τ) = w_n·ν(τ) + w_m·k(τ/ℓ) +"
        " w_s·k(τ/T) (`predfoot.kpi.noise`): a nugget ν plus a main and a slow scale of one kernel k, fitted"
        " by weighted least squares to the five measured lags (0.2-1.0 s). Scales are kept at or above the"
        " 0.2 s sampling interval: anything faster is not resolved by the measurement and is carried by the"
        " nugget, whose shape at 25 Hz is an assumption (section 3). Two marginals: **Gaussian** with the"
        " measured σ, or the **measured marginal** through a Gaussian copula, ρ being then fitted to the ACF"
        " of the normal scores. The measured errors are heavy-tailed: robust σ (IQR / 1.349) ="
        f" {tails['robust_lo']:.2f} to {tails['robust_hi']:.2f} of the SD{kurtosis} on these series, so a"
        f" Gaussian with the measured σ makes the typical error {1 / tails['robust_hi']:.1f} to"
        f" {1 / tails['robust_lo']:.1f} times too large.",
        "",
        "Largest |ACF residual| over the five lags (model minus measurement; normal-score ACF for the"
        " copula fits):",
        "",
    ]
    res_rows = []
    cols = [("ou", "gauss")] + [(k, "gauss") for k in KERNELS] + [(k, "copula") for k in KERNELS]
    for camera in cameras:
        for part in PARTS:
            for axis in AXES:
                res_rows.append(
                    [camera, part, axis, *[fmt(residuals[camera, f, m][part, axis], 3) for f, m in cols]]
                )
    header = [f"{KERNEL_LABELS[f]}" + ("" if f == "ou" else f" ({m})") for f, m in cols]
    lines += [md_table(["Camera", "Part", "Axis", *header], res_rows), ""]
    reason = (
        " The kernel is a prior choice, not a fit result: the smooth kernels fit the five lags about as well"
        " as each other (table above), and Matérn-3/2 is the roughest of them with differentiable paths, as"
        " expected of the error of a smoothed camera with inertia."
        if family == "matern32"
        else " The kernel was chosen on the command line (`--family`)."
    )
    lines += [
        f"Chosen model: **{KERNEL_LABELS[family]}, {MARGINAL_LABELS[marginal]}, white nugget**.{reason}"
        " Parameters (σ in m, scales in s; the last column is the RMS of the time derivative of a Gaussian"
        " process with the same σ and ρ, in m/s, infinite when the nugget is white or the kernel"
        " exponential):",
        "",
        md_table(
            ["Camera", "Part", "Axis", "σ", "w_n", "ℓ", "w_s", "T", "RMS d/dt"],
            model_table(chosen),
        ),
        "",
    ]
    rms_ratio = [r["measured_rms"] / r["chosen_rms"] for r in steps]
    median_ratio = [r["measured_median"] / r["chosen_median"] for r in steps]
    lines += [
        "Round trip at 5 fps: RMS and median |step| of the 0.2 s error steps (m), measured on SN-GSR dev and"
        f" in 10 min of samples of the fitted processes (25 Hz, one sample in five): the Gaussian"
        f" {KERNEL_LABELS[family]} and the chosen model. The ACF is pooled over series (`error_process.py`),"
        " so the Gaussian fit reproduces the measured step RMS up to its ACF residual; the copula fit"
        " matches the normal scores instead, so its step RMS can differ.",
        "",
        md_table(
            [
                "Camera",
                "Part",
                "Axis",
                "Step RMS: measured",
                "Gaussian",
                "chosen",
                "Median |step|: measured",
                "Gaussian",
                "chosen",
            ],
            [
                [
                    r["camera"],
                    r["part"],
                    r["axis"],
                    fmt(r["measured_rms"], 3),
                    fmt(r["gauss_rms"], 3),
                    fmt(r["chosen_rms"], 3),
                    fmt(r["measured_median"], 3),
                    fmt(r["gauss_median"], 3),
                    fmt(r["chosen_median"], 3),
                ]
                for r in steps
            ],
        ),
        "",
        "Measured / chosen model: step RMS"
        f" {fmt(min(rms_ratio))} to {fmt(max(rms_ratio))}, median |step| {fmt(min(median_ratio))} to"
        f" {fmt(max(median_ratio))}. A median ratio below the RMS ratio means real steps are intermittent:"
        " mostly smaller than the model's, with rarer large jumps.",
        "",
        "## 2. Simulation against measurement: the KPI proxy at 5 fps",
        "",
        "The SN-GSR proxy compares predfoot's positions with the smoothed labels; on IDSSE the same code"
        " compares clean tracks plus simulated errors with the clean tracks, on runs drawn from the measured"
        " run lengths. At 5 fps a model is only seen through its correlations at multiples of 0.2 s and its"
        " marginal, so this check tests the fitted lags, the marginal and the transfer from SN-GSR clips to"
        " IDSSE matches, **not** the 25 Hz shape: models that differ only above 2.5 Hz give the same proxy.",
        "",
    ]
    clean_all, clean_play = prow("clean", False), prow("clean", True)
    m0 = measured[cameras[0]]
    lines += [
        md_table(
            ["Reference tracks", "Mean speed (m/s)", "Share of distance ≥ 5.56 m/s", "Runs", "Hours"],
            [
                [
                    "SN-GSR smoothed labels (measured)",
                    fmt(m0["sm_mean_speed_ms"]),
                    pct(m0["hsr_sm_m"] / m0["sm_distance_m"], signed=False),
                    str(m0["runs"]),
                    fmt(m0["sm_duration_s"] / 3600, 1),
                ],
                *[
                    [
                        f"IDSSE clean, {label}",
                        fmt(c["ref_mean_speed_ms"]),
                        pct(c["ref_hsr_m"] / c["ref_distance_m"], signed=False),
                        str(int(c["runs"])),
                        fmt(c["duration_s"] / 3600, 1),
                    ]
                    for label, c in (("all frames", clean_all), ("ball in play", clean_play))
                ],
            ],
        ),
        "",
    ]
    prox_rows = []
    models_seen = proxy.filter(pl.col("model") != "clean")["model"].unique(maintain_order=True).to_list()
    for camera in cameras:
        m = measured[camera]
        prox_rows.append(
            [
                f"**{camera}: measured on SN-GSR**",
                pct(m["distance_bias"]),
                pct(m["hsr_bias"]),
                f"{fmt(m['speed_rms'])} / {fmt(m['speed_p90'])}",
            ]
        )
        for name in [n for n in models_seen if n.startswith(f"{camera}:")]:
            a, b = prow(name, False), prow(name, True)
            prox_rows.append(
                [
                    model_label(name),
                    f"{pct(a['distance_bias'])} / {pct(b['distance_bias'])}",
                    f"{pct(a['hsr_bias'])} / {pct(b['hsr_bias'])}",
                    " · ".join(f"{fmt(r['speed_rms'])} / {fmt(r['speed_p90'])}" for r in (a, b)),
                ]
            )
    lines += [
        "Simulated values are *all frames / ball in play* (speed error: RMS / p90 of |error|, all frames ·"
        " ball in play):",
        "",
        md_table(["Camera and model", "Distance bias", "HSR-distance bias", "Speed error (m/s)"], prox_rows),
        "",
    ]
    verdicts = {c: proxy_verdict(measured[c], prow(chosen_name[c], True)) for c in cameras}
    failed = [c for c in cameras if not verdicts[c]["passes"]]
    keys = ("distance_bias", "hsr_bias")
    high = [
        c
        for c in cameras
        if c not in failed and all(math.isfinite(verdicts[c][k]) and verdicts[c][k] > 1.0 for k in keys)
    ]

    def ratio_text(c: str) -> str:
        v = verdicts[c]
        return f"{c}: distance ×{fmt(v['distance_bias'], 1)}, HSR distance ×{fmt(v['hsr_bias'], 1)}"

    every = [verdicts[c][k] for c in cameras for k in keys]
    lines += [
        f"**Verdict.** The check passes when the chosen model's ball-in-play distance and HSR-distance biases"
        f" are both within a factor of {PROXY_TOLERANCE:g} of the measured ones.",
        "",
        md_table(
            [
                "Camera",
                "Distance bias: simulated / measured",
                "HSR-distance bias: simulated / measured",
                "Check",
            ],
            [
                [
                    c,
                    f"×{fmt(verdicts[c]['distance_bias'], 1)}",
                    f"×{fmt(verdicts[c]['hsr_bias'], 1)}",
                    "passes" if verdicts[c]["passes"] else "**fails**",
                ]
                for c in cameras
            ],
        ),
        "",
    ]
    if all(math.isfinite(r) and r > 1.0 for r in every):
        lines += [
            f"Every simulated bias exceeds the measured one (×{fmt(min(every), 1)} to ×{fmt(max(every), 1)}):"
            " the model overstates the 5 fps KPI cost even where the check passes.",
            "",
        ]
    lines += [
        "What the simulation does not reproduce: the SN-GSR reference is the smoothed labels, which carry"
        " their own smoothing error, while IDSSE's is the clean tracking; real errors are heteroscedastic"
        " (larger far from the camera, along the viewing direction) and can correlate with motion (fast pans"
        " during fast play), while the simulated ones are stationary and independent of motion; and the two"
        " data sets move differently (table above). Nothing was tuned on the proxy: models are fitted to the"
        " ACF and the marginal only.",
        "",
        "## 3. Budget at the export rate (10 fps pipeline)",
        "",
    ]
    if failed or high:
        caveat = "> **Not predfoot's KPI error.**"
        if failed:
            caveat += (
                f" The check of section 2 fails for the {', '.join(failed)} camera"
                f"{'s' if len(failed) > 1 else ''}: the chosen model's simulated 5 fps proxy biases are"
                f" several times the measured ones ({'; '.join(ratio_text(c) for c in failed)}, ball in"
                " play)."
            )
        if high:
            caveat += (
                f" For the {', '.join(high)} camera{'s' if len(high) > 1 else ''} the check passes, but the"
                " simulated biases are still above the measured ones"
                f" ({'; '.join(ratio_text(c) for c in high)})."
            )
        caveat += (
            " The numbers below are the budget of the fitted stationary error model; they overstate the KPI"
            " error of predfoot's tracking and must not be quoted as such."
        )
        lines += [caveat, ""]
    lines += [
        "Errors are sampled at 25 Hz and the 10 fps pipeline runs as for exported tracking. The nugget"
        f" ({nugget_range(chosen)} of the variance) is invisible at 5 fps; two shapes bound what it can do at"
        " 25 Hz: **white** (independent frames, power spread to 12.5 Hz, mostly removed by the 10 fps"
        " smoothing after aliasing) and **band-limited** (flat spectrum up to 2.5 Hz: the smoothest process"
        " still uncorrelated at 0.2 s, with all its power where the 1 s smoothing lets speed noise through)."
        " The kernels bracket the resolved part between its measured lags, from exponential (rough) to"
        " squared exponential (smooth).",
        "",
    ]
    main_rows = []
    for camera in cameras:
        for kpi in BUDGET_KPIS:
            sub = comp.filter((pl.col("condition") == chosen_name[camera]) & (pl.col("kpi") == kpi)).row(
                0, named=True
            )
            main_rows.append(
                [
                    camera,
                    KPI_LABELS[kpi],
                    fmt(sub["ref_mean"], 1),
                    ci(sub, "mape"),
                    ci(sub, "bias", signed=True),
                    ci(sub, "spearman", 3),
                ]
            )
    lines += [
        f"### Chosen model ({KERNEL_LABELS[family]}, {MARGINAL_LABELS[marginal]}, white nugget) against the"
        " clean 25 Hz reference",
        "",
        md_table(
            ["Camera", "KPI", "Ref. mean", "MAPE % [95 % CI]", "Bias % [95 % CI]", "Spearman [95 % CI]"],
            main_rows,
        ),
        "",
        "### Sensitivity: bias % of each KPI under every model",
        "",
        "Ghost: distance per minute of a motionless player through the 10 fps pipeline (one player's full"
        " error).",
        "",
    ]
    sens_rows = [
        ["clean 10 fps (control)", "–", *[fmt(cell("control", k, "bias"), 1, True) for k in BUDGET_KPIS]]
    ]
    for name in (
        comp.filter(pl.col("condition") != "control")["condition"].unique(maintain_order=True).to_list()
    ):
        sens_rows.append(
            [
                model_label(name),
                fmt(ghosts[name], 1),
                *[fmt(cell(name, k, "bias"), 1, True) for k in BUDGET_KPIS],
            ]
        )
    lines += [md_table(["Model", "Ghost (m/min)", *[KPI_LABELS[k] for k in BUDGET_KPIS]], sens_rows), ""]
    bound_rows = []
    for camera in cameras:
        names = [f"{camera}:{f}:{marginal}:{n}" for f in KERNELS for n in NUGGET_SHAPES]
        for kpi in BUDGET_KPIS:
            vals = [cell(n, kpi, "bias") for n in names]
            bound_rows.append(
                [
                    camera,
                    KPI_LABELS[kpi],
                    fmt(cell(chosen_name[camera], kpi, "bias"), 1, True),
                    f"{fmt(min(vals), 1, True)} to {fmt(max(vals), 1, True)}",
                    fmt(cell(f"{camera}:matern32:gauss:white", kpi, "bias"), 1, True),
                    fmt(cell(f"{camera}:ou:gauss:white", kpi, "bias"), 1, True),
                ]
            )
    lines += [
        "Sensitivity bound: the range of the bias over the four kernels and both nugget shapes (same"
        " marginal as the chosen model), next to the Gaussian Matérn-3/2 and the previous single OU:",
        "",
        md_table(
            ["Camera", "KPI", "Chosen", "Range, kernels × nuggets", "Gaussian Matérn-3/2", "Single OU"],
            bound_rows,
        ),
        "",
        "All KPIs, chosen model, MAPE % / bias % against the clean 25 Hz reference:",
        "",
    ]
    kpis_all = comp["kpi"].unique(maintain_order=True).to_list()
    all_rows = []
    for label, cond in [(c, chosen_name[c]) for c in cameras] + [("clean 10 fps", "control")]:
        all_rows.append(
            [
                label,
                *[
                    f"{fmt(cell(cond, k, 'mape'), 1)} / {fmt(cell(cond, k, 'bias'), 1, True)}"
                    for k in kpis_all
                ],
            ]
        )
    lines += [md_table(["Model", *[KPI_LABELS[k] for k in kpis_all]], all_rows), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def nugget_range(chosen: dict[str, SpectralErrorModel]) -> str:
    weights = [
        getattr(m, part).axis(axis).w_nugget for m in chosen.values() for part in PARTS for axis in AXES
    ]
    return f"{min(weights):.2f} to {max(weights):.2f}"


if __name__ == "__main__":
    main()
