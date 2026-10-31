"""KPI error budget, first sensitivity study (ticket T4.1) -> reports/kpi_sensitivity.md.

    uv run python scripts/kpi_sensitivity.py --matches J03WPY J03WOH   # two matches
    uv run python scripts/kpi_sensitivity.py                           # all seven

Reference: KPIs on clean IDSSE positions at 25 Hz with the standard pipeline. Control (gate G4): the same
clean positions through the 10 fps pipeline. Then simulated position errors (i.i.d., AR(1), common-mode +
individual) are added to the 25 Hz positions before the 10 fps pipeline. This is a sensitivity analysis: the
real error process of predfoot is measured on SN-GSR later (ticket T4.2).

Per-player tables, the configuration and metadata go to ``runs/<date>_<git-sha>/`` (never committed); the
report with aggregate numbers goes to ``reports/kpi_sensitivity.md``. Run ``scripts/prepare_idsse.py`` first.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from predfoot.io import idsse
from predfoot.kpi.config import DEFAULT_CONFIG_PATH, KpiConfig, load_kpi_config
from predfoot.kpi.errors import ErrorModel, ar1_noise, ghost_distance_per_min, sample_errors
from predfoot.kpi.metrics import compute_kpis, kpi_names, path_length
from predfoot.kpi.report import compare_to_reference
from predfoot.kpi.simulate import period_tracks, seeded_rng
from predfoot.kpi.speed import Kinematics, pipeline
from predfoot.paths import REPO_ROOT, reports_dir

SIGMAS = (0.05, 0.1, 0.2, 0.5)
RHOS = (0.9, 0.99)
GOALKEEPER = "TW"  # DFL position code
G4_MAPE_PCT = 0.5  # EXECUTION.md, ticket T4.1

KPI_LABELS = {
    "distance_m": "Distance",
    "dist_lt15_m": "Dist. < 15",
    "dist_15_20_m": "Dist. 15-20",
    "dist_20_25_m": "Dist. 20-25 (HSR)",
    "dist_gt25_m": "Dist. > 25 (sprint)",
    "hi_distance_m": "HI dist. > 20",
    "hsr_efforts": "HSR efforts",
    "sprint_efforts": "Sprint efforts",
    "psv99_kmh": "PSV-99",
    "explosive_to_hsr": "Expl. acc. to HSR",
    "explosive_to_sprint": "Expl. acc. to sprint",
}


@dataclass(frozen=True)
class Condition:
    """One way of computing KPIs: pipeline rate, configuration and injected error."""

    name: str
    label: str
    group: str
    cfg: KpiConfig
    resample_to: int | None
    error: ErrorModel | None = None


def build_conditions(cfg: KpiConfig) -> list[Condition]:
    out = [
        Condition("reference", "clean, 25 Hz", "reference", cfg, None),
        Condition("control", "clean, 10 fps", "control", cfg, cfg.output_hz),
    ]
    for sigma in SIGMAS:
        model = ErrorModel(f"iid_{sigma:g}", sigma=sigma)
        out.append(Condition(model.name, f"i.i.d. σ={sigma:g} m", "noise", cfg, cfg.output_hz, model))
    for rho in RHOS:
        for sigma in SIGMAS:
            model = ErrorModel(f"ar1_{rho:g}_{sigma:g}", sigma=sigma, rho=rho)
            out.append(
                Condition(model.name, f"AR(1) ρ={rho:g}, σ={sigma:g} m", "noise", cfg, cfg.output_hz, model)
            )
    half = 0.2 / math.sqrt(2.0)
    mix = ErrorModel("common_0.2", sigma=half, rho=0.9, sigma_common=half, rho_common=0.99)
    out.append(
        Condition(
            mix.name, "common+indiv. σ=0.2 m (half common, ρ 0.99/0.9)", "noise", cfg, cfg.output_hz, mix
        )
    )

    # Smoothing window, with and without noise (the noisy conditions share their noise with the main grid).
    for window in (0.5, 2.0, None):
        smoothing = (
            cfg.smoothing.model_copy(update={"method": "none"})
            if window is None
            else cfg.smoothing.model_copy(update={"window_s": window})
        )
        variant = cfg.model_copy(update={"smoothing": smoothing})
        tag = "none" if window is None else f"{window:g}s"
        out.append(Condition(f"smooth_{tag}", f"clean, window {tag}", "smoothing", variant, cfg.output_hz))
        for model in (ErrorModel("iid_0.2", sigma=0.2), ErrorModel("ar1_0.99_0.2", sigma=0.2, rho=0.99)):
            out.append(
                Condition(
                    f"smooth_{tag}_{model.name}",
                    f"{model.name}, window {tag}",
                    "smoothing",
                    variant,
                    cfg.output_hz,
                    model,
                )
            )

    # Unpublished definitions, on clean data at 10 fps.
    definitions = {
        "min_effort_0.5s": cfg.model_copy(update={"min_effort_s": 0.5}),
        "min_effort_2s": cfg.model_copy(update={"min_effort_s": 2.0}),
        "psv_run_peaks": cfg.model_copy(
            update={"psv99": cfg.psv99.model_copy(update={"statistic": "run_peaks"})}
        ),
        "psv_min_activity_0.5s": cfg.model_copy(
            update={"psv99": cfg.psv99.model_copy(update={"min_activity_s": 0.5})}
        ),
        "psv_min_activity_2s": cfg.model_copy(
            update={"psv99": cfg.psv99.model_copy(update={"min_activity_s": 2.0})}
        ),
    }
    for name, variant in definitions.items():
        out.append(Condition(f"def_{name}", name, "definition", variant, cfg.output_hz))
    return out


def period_arrays(
    players: pl.DataFrame, cfg: KpiConfig
) -> tuple[list[tuple[list[str], np.ndarray]], int, int]:
    """One (n_frames, n_players, 2) array per period on the period's frame grid, NaN where a player is absent.

    Tracking glitches of the source feed are cut out first (:func:`drop_glitches`). Also returns the number of
    samples dropped and the number of valid samples before cleaning.
    """
    tracks, dropped, total = period_tracks(players, cfg)
    return [(list(t.player_ids), t.xy) for t in tracks], dropped, total


def condition_kpis(
    match_id: str, periods: list[tuple[list[str], np.ndarray]], cond: Condition, seed: int
) -> list[dict]:
    parts: dict[str, list[Kinematics]] = {}
    for p, (ids, cube) in enumerate(periods, start=1):
        xy_all = cube
        if cond.error is not None:
            rng = seeded_rng(seed, match_id, cond.error.name, p)
            xy_all = cube + sample_errors(cond.error, cube.shape[0], cube.shape[1], rng)
        for i, pid in enumerate(ids):
            valid = np.flatnonzero(np.isfinite(cube[:, i, 0]))
            if valid.size == 0:
                continue
            xy = xy_all[valid[0] : valid[-1] + 1, i]
            parts.setdefault(pid, []).append(pipeline(xy, cond.cfg, resample_to=cond.resample_to))
    return [
        {"match_id": match_id, "player_id": pid, "condition": cond.name, **compute_kpis(kins, cond.cfg)}
        for pid, kins in parts.items()
    ]


def ghost_table(cfg: KpiConfig, seed: int) -> list[dict]:
    """Ghost distance of a motionless player: closed form, raw 25 Hz simulation, and the 10 fps pipeline."""
    rows = []
    minutes = 30
    n = minutes * 60 * cfg.native_hz + 1
    for rho in (0.0, *RHOS):
        for sigma in SIGMAS:
            rng = seeded_rng(seed, "ghost", f"{rho:g}", f"{sigma:g}")
            noise = ar1_noise(n, sigma, rho, rng, (2,))
            kin = pipeline(noise, cfg, resample_to=cfg.output_hz)
            rows.append(
                {
                    "rho": rho,
                    "sigma": sigma,
                    "closed_form": ghost_distance_per_min(sigma, cfg.native_hz, rho),
                    "raw_25hz": path_length(noise) / minutes,
                    "pipeline_10fps": path_length(kin.xy) / minutes,
                }
            )
    return rows


def git_info() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    # Tracked changes outside reports/ (the report being regenerated does not make the code dirty).
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)reports"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return sha.stdout.strip() or "unknown", bool(status.stdout.strip())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --- report ----------------------------------------------------------------------------------------------


def fmt(v: float, digits: int = 2, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def ci(row: dict, name: str, digits: int = 2, signed: bool = False) -> str:
    lo, hi = fmt(row[name + "_lo"], digits, signed), fmt(row[name + "_hi"], digits, signed)
    return f"{fmt(row[name], digits, signed)} [{lo}, {hi}]"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def metric_matrix(
    comp: pl.DataFrame, conds: list[Condition], kpis: list[str], metric: str, digits: int, signed: bool
) -> str:
    header = ["Condition", *[KPI_LABELS[k] for k in kpis]]
    rows = []
    for cond in conds:
        sub = comp.filter(pl.col("condition") == cond.name)
        values = {r["kpi"]: r[metric] for r in sub.iter_rows(named=True)}
        rows.append([cond.label, *[fmt(values[k], digits, signed) for k in kpis]])
    return md_table(header, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", nargs="*", default=sorted(idsse.IDSSE_MATCHES))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--min-minutes", type=float, default=45.0, help="minimum playing time of a player-match"
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--report", type=Path, default=reports_dir() / "kpi_sensitivity.md")
    parser.add_argument(
        "--measured",
        nargs=4,
        type=float,
        metavar=("SIGMA", "RHO", "SIGMA_COMMON", "RHO_COMMON"),
        help="add a condition with the error process measured on SN-GSR (scripts/error_process.py); "
        "σ in metres per axis, ρ per 25 Hz frame",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    cfg = load_kpi_config(args.config)
    conditions = build_conditions(cfg)
    if args.measured:
        sigma, rho, sigma_c, rho_c = args.measured
        measured = ErrorModel("measured", sigma=sigma, rho=rho, sigma_common=sigma_c, rho_common=rho_c)
        label = f"measured on SN-GSR (σ {sigma:g}/{sigma_c:g} m, ρ {rho:g}/{rho_c:g})"
        conditions.append(Condition("measured", label, "noise", cfg, cfg.output_hz, measured))
    kpis = [k for k in kpi_names(cfg) if k != "minutes"]
    per90 = [k for k in kpis if k != "psv99_kmh"]
    sha, dirty = git_info()
    run_dir = REPO_ROOT / "runs" / f"{dt.date.today():%Y-%m-%d}_{sha}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rosters = []
    n_dropped = n_samples = 0
    for match_id in args.matches:
        t0 = time.perf_counter()
        idsse.prepare_match(match_id)
        periods, dropped, total = period_arrays(idsse.load_players(match_id), cfg)
        n_dropped += dropped
        n_samples += total
        rosters.append(idsse.load_roster(match_id).with_columns(pl.lit(match_id).alias("match_id")))
        for cond in conditions:
            rows.extend(condition_kpis(match_id, periods, cond, args.seed))
        print(f"{match_id}: {len(conditions)} conditions in {time.perf_counter() - t0:.1f} s", flush=True)

    table = pl.DataFrame(rows)
    roster = pl.concat(rosters).select("match_id", "player_id", "team", "position")
    ref_minutes = table.filter(pl.col("condition") == "reference").select("match_id", "player_id", "minutes")
    table = (
        table.drop("minutes")
        .join(ref_minutes, on=["match_id", "player_id"])
        .join(roster, on=["match_id", "player_id"], how="left")
    )
    table.write_parquet(run_dir / "kpis.parquet")
    population = table.filter((pl.col("position") != GOALKEEPER) & (pl.col("minutes") >= args.min_minutes))
    ref = population.filter(pl.col("condition") == "reference")
    n_player_matches = ref.height

    def compare(base: str, test: str) -> pl.DataFrame:
        return compare_to_reference(
            population.filter(pl.col("condition") == base),
            population.filter(pl.col("condition") == test),
            kpis,
            per90=per90,
            n_boot=args.n_boot,
            seed=args.seed,
        ).with_columns(pl.lit(test).alias("condition"), pl.lit(base).alias("baseline"))

    main_conds = [c for c in conditions if c.group in ("control", "noise")]
    comps = [compare("reference", c.name) for c in main_conds]
    # The same noisy conditions against the clean 10 fps control: the effect of the noise alone.
    comps += [compare("control", c.name) for c in main_conds if c.group == "noise"]
    # Smoothing: noise effect against clean data with the same window; clean variants against the reference.
    smooth_rows = []
    for tag in ("0.5s", "none", "2s"):
        smooth_rows.append(compare("reference", f"smooth_{tag}"))
        for model in ("iid_0.2", "ar1_0.99_0.2"):
            smooth_rows.append(compare(f"smooth_{tag}", f"smooth_{tag}_{model}"))
    defs = [compare("control", c.name) for c in conditions if c.group == "definition"]
    comp = pl.concat(comps + smooth_rows + defs)
    comp.write_parquet(run_dir / "comparisons.parquet")

    ghost = ghost_table(cfg, args.seed)
    raw_files = []
    for match_id in args.matches:
        p = idsse.match_paths(match_id)
        raw_files += [(f.name, sha256(f)) for f in (p.match_information, p.positions)]
    shutil.copy(args.config, run_dir / "kpi.yaml")
    command = "uv run python scripts/kpi_sensitivity.py " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    meta = {
        "command": command.strip(),
        "git_sha": sha,
        "git_dirty": dirty,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "matches": args.matches,
        "n_player_matches": n_player_matches,
        "glitch_samples_dropped": n_dropped,
        "samples": n_samples,
        "min_minutes": args.min_minutes,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "config_sha256": sha256(args.config),
        "raw_files_sha256": dict(raw_files),
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = render_report(meta, cfg, conditions, comp, ghost, kpis, raw_files)
    args.report.write_text(report, encoding="utf-8")
    print(f"report: {args.report}\nrun: {run_dir}\n{meta['runtime_s']} s")


def cell(comp: pl.DataFrame, baseline: str, condition: str, kpi: str, metric: str) -> float:
    sub = comp.filter(
        (pl.col("baseline") == baseline) & (pl.col("condition") == condition) & (pl.col("kpi") == kpi)
    )
    return float(sub[metric][0])


def summary_lines(comp: pl.DataFrame) -> list[str]:
    """Key numbers, read from the comparison table (no hand-written values)."""

    def m(cond: str, kpi: str, metric: str = "mape", base: str = "reference") -> str:
        signed = metric == "bias"
        return f"{fmt(cell(comp, base, cond, kpi, metric), 2 if metric != 'spearman' else 3, signed)}"

    worse = cell(comp, "reference", "ar1_0.9_0.2", "distance_m", "mape") > cell(
        comp, "reference", "iid_0.2", "distance_m", "mape"
    )
    mid_word = "more" if worse else "less"
    control = comp.filter((pl.col("baseline") == "reference") & (pl.col("condition") == "control"))
    passed = [KPI_LABELS[r["kpi"]] for r in control.iter_rows(named=True) if r["mape"] < G4_MAPE_PCT]
    explosive = ("explosive_to_hsr", "explosive_to_sprint")
    max_bias = max(abs(r["bias"]) for r in control.iter_rows(named=True) if r["kpi"] not in explosive)
    mean_expl = cell(comp, "reference", "control", "explosive_to_hsr", "ref_mean")
    failed = [
        f"{KPI_LABELS[r['kpi']]} {fmt(r['mape'])} %"
        for r in control.iter_rows(named=True)
        if r["mape"] >= G4_MAPE_PCT
    ]
    return [
        f"- **Gate G4** (clean data, 10 fps vs 25 Hz, MAPE < {G4_MAPE_PCT:g} %): passed by "
        f"{', '.join(passed)}; "
        f"not passed by {', '.join(failed) if failed else 'none'}. Totals stay close (bias within "
        f"±{max_bias:.2f} % for every KPI except explosive accelerations): the misses come from "
        "discretisation at "
        "thresholds, where an effort near the minimum duration or an acceleration near 3 m/s² flips "
        "between the "
        f"two pipelines, on small per-player counts ({mean_expl:.1f} explosive accelerations to HSR per "
        "player-match on average).",
        f"- **Smoothing** is the main lever: with i.i.d. σ = 0.2 m, the distance error is "
        f"{m('iid_0.2', 'distance_m', base='control')} % with the 1 s window and "
        f"{m('smooth_2s_iid_0.2', 'distance_m', base='smooth_2s')} % with 2 s, but on clean data the 2 s "
        "window "
        f"already moves explosive accelerations to HSR by {m('smooth_2s', 'explosive_to_hsr', 'bias')} % "
        "(definitions and noise robustness pull in opposite directions).",
        f"- **Distance**: MAPE {m('iid_0.1', 'distance_m')} % for i.i.d. σ = 0.1 m and "
        f"{m('iid_0.2', 'distance_m')} % for σ = 0.2 m, an overestimate (bias "
        f"{m('iid_0.2', 'distance_m', 'bias')} % at 0.2 m). At σ = 0.2 m, slow errors (AR(1) ρ = 0.99, about "
        f"4 s) give {m('ar1_0.99_0.2', 'distance_m')} %, and errors correlated over about 0.4 s (ρ = 0.9) "
        "give "
        f"{m('ar1_0.9_0.2', 'distance_m')} %, {mid_word} than white noise: what matters is how much error "
        "power "
        "falls in the band the 1 s smoothing lets through, not correlation as such.",
        f"- **High-speed KPIs** react more: at i.i.d. σ = 0.1 m, distance above 25 km/h has MAPE "
        f"{m('iid_0.1', 'dist_gt25_m', base='control')} % and sprint efforts "
        f"{m('iid_0.1', 'sprint_efforts', base='control')} % "
        "against the clean 10 fps control; PSV-99 stays within "
        f"{m('iid_0.1', 'psv99_kmh', base='control')} % (bias "
        f"{m('iid_0.1', 'psv99_kmh', 'bias', 'control')} %).",
        f"- **Explosive accelerations** are the most fragile (a threshold on a second derivative): "
        f"{m('iid_0.1', 'explosive_to_hsr', 'bias', 'control')} % of the total at i.i.d. σ = 0.1 m and "
        f"{m('iid_0.2', 'explosive_to_hsr', 'bias', 'control')} % at 0.2 m, because noise breaks the runs of "
        "more than 0.7 s above 3 m/s².",
        "- **Rankings** of players at i.i.d. σ = 0.2 m: Spearman "
        f"{m('iid_0.2', 'hi_distance_m', 'spearman')} "
        f"for HI distance, {m('iid_0.2', 'sprint_efforts', 'spearman')} for sprint efforts and "
        f"{m('iid_0.2', 'explosive_to_hsr', 'spearman')} for explosive accelerations to HSR.",
    ]


def render_report(
    meta: dict,
    cfg: KpiConfig,
    conditions: list[Condition],
    comp: pl.DataFrame,
    ghost: list[dict],
    kpis: list[str],
    raw_files: list[tuple[str, str]],
) -> str:
    from_ref = comp.filter(pl.col("baseline") == "reference")
    control = from_ref.filter(pl.col("condition") == "control")
    control_rows = []
    for r in control.iter_rows(named=True):
        verdict = "pass" if r["mape"] < G4_MAPE_PCT else "**fail**"
        control_rows.append(
            [
                KPI_LABELS[r["kpi"]],
                fmt(r["ref_mean"], 1),
                ci(r, "mape"),
                ci(r, "bias", signed=True),
                ci(r, "spearman", 3),
                verdict,
            ]
        )
    noise_conds = [c for c in conditions if c.group in ("control", "noise")]
    dirty = " (with uncommitted changes)" if meta["git_dirty"] else ""
    n_matches = len(meta["matches"])

    lines = [
        "# KPI error budget: first sensitivity study (T4.1)",
        "",
        "> **This is a sensitivity analysis on simulated position errors.** None of these numbers is a "
        "measured",
        "> error of predfoot: the real error process (variance, temporal correlation, common-mode part, "
        "dependence",
        "> on the predicted uncertainty) is measured on SN-GSR later (ticket T4.2) and injected in T4.3. KPI",
        "> definitions marked *assumption* below are ours, not SkillCorner's.",
        "",
        *(
            [
                "> **Measured condition.** The row *measured on SN-GSR* injects the error process",
                "> measured by `scripts/error_process.py` (per-frame calibration, no temporal smoothing",
                "> of the camera). Its σ is measured against the SN-GSR labels, which carry their own",
                "> error, so it is an upper bound; its 25 Hz correlations come from 5 fps measurements",
                "> under an exponential-decay assumption, which overstates the correlation of per-frame",
                "> jitter. Read it as the KPI cost of calibrating each frame independently.",
                "",
            ]
            if any(c.name == "measured" for c in conditions)
            else []
        ),
        f"- **Command**: `{meta['command']}`",
        f"- **Git commit**: `{meta['git_sha']}`{dirty}; date {meta['date']}; runtime {meta['runtime_s']} s",
        "- **Data**: IDSSE (Bassek, Rein, Weber & Memmert 2025, CC BY 4.0, "
        "doi:10.6084/m9.figshare.28196177), "
        f"{n_matches} matches ({', '.join(meta['matches'])}), DFL tracking at 25 Hz, from the Hugging Face "
        "mirror "
        "`pysport/idsse-data` (SHA-256 of the raw files at the end).",
        f"- **Population**: {meta['n_player_matches']} player-matches: outfield players with at least "
        f"{meta['min_minutes']:g} minutes on the pitch (goalkeepers excluded).",
        f"- **Config**: `configs/kpi.yaml` (SHA-256 `{meta['config_sha256'][:12]}`), seed {meta['seed']}, "
        f"{meta['n_boot']} bootstrap resamples.",
        "",
        "## Summary",
        "",
        *summary_lines(comp),
        "",
        "## Set-up",
        "",
        "- **Cleaning**: the raw IDSSE feed holds the last position of a player who is not observed, then "
        "jumps "
        "to the re-acquired position; there are also a few identity swaps. Steps faster than "
        f"{cfg.glitch_speed_kmh:g} km/h are cut out of the source data before anything else (held samples, "
        "or "
        f"the sample after the jump): {meta['glitch_samples_dropped']} of {meta['samples']} player-samples "
        f"({100 * meta['glitch_samples_dropped'] / meta['samples']:.3f} %). Without this step each glitch "
        "becomes "
        "a fake sprint after smoothing, and the 25 Hz and 10 fps pipelines disagree on explosive "
        "accelerations "
        "by tens of percent.",
        "- **Reference**: clean IDSSE positions at 25 Hz through the standard pipeline: Savitzky-Golay "
        "smoothing "
        f"of positions over {cfg.smoothing.window_s:g} s (order {cfg.smoothing.polyorder}, 25 samples), "
        "velocity "
        "and acceleration by central differences that never cross a gap, then the KPIs.",
        "- **Control (gate G4)**: the same clean positions resampled to 10 fps (linear interpolation "
        "between the "
        "two neighbouring 25 Hz samples) and smoothed over 11 samples (1.1 s). Gate G4 asks for MAPE < 0.5 "
        "%.",
        "- **Injected errors**: Gaussian, isotropic, added to the 25 Hz positions before the 10 fps "
        "pipeline; σ is "
        "the marginal standard deviation per axis. AR(1) correlation ρ is per 25 Hz frame: ρ = 0.9 is a "
        "correlation time of 0.38 s, ρ = 0.99 of 4.0 s. The common-mode condition gives every player of a "
        "frame "
        "the same slow error (ρ = 0.99) plus an individual one (ρ = 0.9), each with half of the variance. "
        "Linear resampling averages two 25 Hz samples at every other 10 fps sample, which halves i.i.d. "
        "noise "
        "variance there; this is part of the pipeline being tested.",
        "- **Statistics**: MAPE over player-matches with a positive reference; bias = relative error of "
        "the total "
        "over all player-matches; Spearman correlation of player rankings, on per-90-minute values (PSV-99 "
        "as "
        f"is). 95 % intervals: percentile bootstrap resampling whole matches. With {n_matches} matches the "
        "intervals are rough, and they do not include noise-seed variability (one noise draw per condition).",
        "",
        "## KPI definitions",
        "",
        md_table(
            ["KPI", "Definition", "Status"],
            [
                ["Distance", "Sum of steps between consecutive smoothed 10 fps positions", "standard"],
                [
                    "Speed bands",
                    "Distance of the steps whose speed is in < 15, 15-20, 20-25, > 25 km/h",
                    "20 and 25 km/h published",
                ],
                ["HI distance", "Distance at or above 20 km/h", "published threshold"],
                [
                    "HSR / sprint efforts",
                    f"Runs at or above 20 / 25 km/h lasting at least {cfg.min_effort_s:g} s (crossings "
                    "interpolated)",
                    "thresholds published; duration *assumption*",
                ],
                [
                    "PSV-99",
                    f"{cfg.psv99.percentile:g}th percentile of speed samples in runs above "
                    f"{cfg.psv99.activity_kmh:g} km/h lasting at least {cfg.psv99.min_activity_s:g} s",
                    "published wording; segmentation *assumption*",
                ],
                [
                    "Explosive acc. to HSR / sprint",
                    "Acceleration phase starting below 9 km/h, reaching 20 / 25 km/h, with more than 0.7 s "
                    "at 3 m/s² or more",
                    "criteria published; segmentation *assumption*",
                ],
                [
                    "Smoothing",
                    f"Savitzky-Golay over {cfg.smoothing.window_s:g} s, order {cfg.smoothing.polyorder}",
                    "*assumption*",
                ],
            ],
        ),
        "",
        'Sources: SkillCorner, "New feature: explosive accelerations" (6 Feb 2025) and "SkillCorner '
        'releases new peak velocity metric" (11 Jun 2026); see `configs/kpi.yaml`.',
        "",
        "## Gate G4: clean data through the 10 fps pipeline",
        "",
        md_table(
            ["KPI", "Ref. mean", "MAPE % [95 % CI]", "Bias % [95 % CI]", "Spearman [95 % CI]", "G4"],
            control_rows,
        ),
        "",
        "## Sensitivity to position errors (against the 25 Hz reference)",
        "",
        "### MAPE (%)",
        "",
        metric_matrix(from_ref, noise_conds, kpis, "mape", 2, False),
        "",
        "### Bias (%, relative error of the total)",
        "",
        metric_matrix(from_ref, noise_conds, kpis, "bias", 2, True),
        "",
        "### Spearman correlation of player rankings",
        "",
        metric_matrix(from_ref, noise_conds, kpis, "spearman", 3, False),
        "",
        "Confidence intervals for every cell are in the appendix.",
        "",
        "### MAPE (%) against the clean 10 fps control: the effect of the noise alone",
        "",
        "Same noisy conditions, compared with clean data through the same 10 fps pipeline, so that the "
        "difference between the pipelines (gate G4 above) is not counted.",
        "",
        metric_matrix(
            comp.filter(pl.col("baseline") == "control"),
            [c for c in noise_conds if c.group == "noise"],
            kpis,
            "mape",
            2,
            False,
        ),
        "",
        "## Ghost distance of a motionless player",
        "",
        "Closed form 60·f·σ·sqrt(π(1 - ρ)) m/min for raw positions (60·f·σ·√π for i.i.d. noise), checked by "
        "simulating 30 minutes at 25 Hz, and the same noise through the 10 fps pipeline.",
        "",
        md_table(
            ["ρ", "σ (m)", "Closed form (m/min)", "Simulated, raw 25 Hz", "Through the 10 fps pipeline"],
            [
                [
                    fmt(g["rho"], 2),
                    fmt(g["sigma"], 2),
                    fmt(g["closed_form"], 1),
                    fmt(g["raw_25hz"], 1),
                    fmt(g["pipeline_10fps"], 1),
                ]
                for g in ghost
            ],
        ),
        "",
    ]

    # Smoothing table.
    smooth_kpis = ["distance_m", "hi_distance_m", "sprint_efforts", "psv99_kmh", "explosive_to_hsr"]
    srows = []
    for tag, label in (
        ("0.5s", "0.5 s"),
        ("default", f"{cfg.smoothing.window_s:g} s (default)"),
        ("2s", "2 s"),
        ("none", "none"),
    ):
        clean = "control" if tag == "default" else f"smooth_{tag}"
        r_clean = comp.filter((pl.col("baseline") == "reference") & (pl.col("condition") == clean))
        cells = {r["kpi"]: r for r in r_clean.iter_rows(named=True)}
        srows.append(
            [
                label,
                "clean vs reference",
                *[f"{fmt(cells[k]['mape'])} / {fmt(cells[k]['bias'], signed=True)}" for k in smooth_kpis],
            ]
        )
        for model in ("iid_0.2", "ar1_0.99_0.2"):
            test = model if tag == "default" else f"smooth_{tag}_{model}"
            r_noise = comp.filter((pl.col("baseline") == clean) & (pl.col("condition") == test))
            cells = {r["kpi"]: r for r in r_noise.iter_rows(named=True)}
            pretty = "i.i.d. σ=0.2 m" if model == "iid_0.2" else "AR(1) ρ=0.99, σ=0.2 m"
            srows.append(
                [
                    label,
                    f"{pretty} vs clean (same window)",
                    *[f"{fmt(cells[k]['mape'])} / {fmt(cells[k]['bias'], signed=True)}" for k in smooth_kpis],
                ]
            )
    lines += [
        "## Smoothing window",
        "",
        'MAPE % / bias % at 10 fps. Rows "clean vs reference" show the cost of the window on clean data; '
        "rows with noise compare noisy and clean data through the same window, which isolates the effect "
        "of the noise (σ = 0.2 m).",
        "",
        md_table(["Window", "Comparison", *[KPI_LABELS[k] for k in smooth_kpis]], srows),
        "",
    ]

    # Definitions table.
    def_rows = []
    for cond in [c for c in conditions if c.group == "definition"]:
        sub = comp.filter((pl.col("baseline") == "control") & (pl.col("condition") == cond.name))
        changed = [
            r
            for r in sub.iter_rows(named=True)
            if not (abs(r["mape"] or 0.0) < 1e-12 and abs(r["bias"] or 0.0) < 1e-12)
        ]
        text = "; ".join(
            f"{KPI_LABELS[r['kpi']]}: {fmt(r['bias'], 1, signed=True)} % (MAPE {fmt(r['mape'], 1)} %)"
            for r in changed
        )
        def_rows.append([cond.label, text or "no change"])
    lines += [
        "## Unpublished definitions (clean data, 10 fps)",
        "",
        "Change of each KPI when one assumption changes, against the default configuration (bias % of the "
        "total, "
        "MAPE % across player-matches). These are the choices SkillCorner does not publish; they move the "
        "KPI "
        "values themselves, before any position error.",
        "",
        md_table(["Variant", "KPIs that change"], def_rows),
        "",
    ]

    lines += [
        "## Reading these numbers",
        "",
        "- Errors here are stationary, isotropic and independent of the camera; real broadcast errors grow "
        "with "
        "the distance to the camera, are larger along the viewing direction and depend on the calibration "
        "of each "
        "frame (T4.2 measures them, T4.3 injects errors driven by the predicted uncertainty σ̂).",
        "- The reference is IDSSE's own tracking, not ground truth; injected errors are added on top of it.",
        "- Distance is robust to noise once positions are smoothed: for a moving player, noise lengthens "
        "each "
        "step only at second order. Speed-threshold KPIs (counts, PSV-99, explosive accelerations) react at "
        "first order, because noise moves speed across thresholds.",
        "",
        "## Appendix: all comparisons with 95 % intervals",
        "",
    ]
    app_rows = []
    for cond in noise_conds:
        sub = from_ref.filter(pl.col("condition") == cond.name)
        for r in sub.iter_rows(named=True):
            app_rows.append(
                [
                    cond.label,
                    KPI_LABELS[r["kpi"]],
                    str(r["n"]),
                    ci(r, "mape"),
                    ci(r, "bias", signed=True),
                    ci(r, "spearman", 3),
                ]
            )
    lines += [md_table(["Condition", "KPI", "n", "MAPE %", "Bias %", "Spearman"], app_rows), ""]
    lines += [
        "## Appendix: raw data files",
        "",
        md_table(["File", "SHA-256"], [[f"`{n}`", f"`{h}`"] for n, h in raw_files]),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
