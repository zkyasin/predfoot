"""KPI uncertainty card on IDSSE: bias-corrected nominal 90 % intervals and their coverage.

    uv run python scripts/kpi_card.py                                   # seven matches, both cameras
    uv run python scripts/kpi_card.py --matches J03WMX J03WN1 --n-draws 30 --n-outer 8 --n-inner 8

**A consistency check, not a validation.** For each camera, the error model is the one chosen in
``scripts/kpi_spectral.py`` (fitted on SN-GSR dev). Each outfield player-match with at least 45 minutes gets
one "observed" track: clean IDSSE positions plus one draw of that model (the common part shared by the
players of a frame). The card (``predfoot.kpi.card``) is computed from that track and the model only; its
coverage is measured against the clean track through the same 10 fps pipeline. The observed errors come from
the model the card assumes, so this tests the card's arithmetic and its noise-on-noise approximation, not the
error model. Each card uses the nested bootstrap of ``predfoot.kpi.card`` for the variance of its estimate
(``--n-outer`` pseudo-observations with ``--n-inner`` draws each); the previous half-width, from the spread of
one dose, is reported from the same draws for comparison.

Leave-one-match-out calibration: for each held-out match, every KPI's half-width is multiplied by the factor
that gives 90 % coverage on the other matches (the 90 % quantile of |estimate − truth| / half-width there).
Per-player tables go to ``runs/<date>_<sha>_card/``; the report to ``reports/kpi_card.md`` and the figure to
``reports/figures/kpi_card.png``.
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
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import beta

from predfoot.io import idsse
from predfoot.kpi.card import CardLine, card_line, kpi_draws, nested_estimates
from predfoot.kpi.config import load_kpi_config
from predfoot.kpi.noise import KERNELS, fit_error_model, sample_errors
from predfoot.kpi.report import bootstrap_by_group
from predfoot.kpi.simulate import period_tracks, player_kpis, seeded_rng
from predfoot.paths import REPO_ROOT, reports_dir

GOALKEEPER = "TW"
CARD_KPIS = {
    "distance_m": ("Distance", "m", 0),
    "hi_distance_m": ("HI distance (≥ 20 km/h)", "m", 0),
    "dist_gt25_m": ("Sprint distance (> 25 km/h)", "m", 0),
    "hsr_efforts": ("HSR efforts", "", 1),
    "sprint_efforts": ("Sprint efforts", "", 1),
    "psv99_kmh": ("PSV-99", "km/h", 1),
}
LEVEL = 0.9


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


def lomo_coverage(d: pl.DataFrame) -> tuple[float, float]:
    """Leave-one-match-out calibrated coverage of one camera and KPI, and the median scale factor. Raw
    (unprojected) estimates: for a truth inside the support, projection does not change coverage."""
    covered, factors = [], []
    ratio = (d["raw_estimate"] - d["truth"]).abs() / d["half_width"]
    for match_id in d["match_id"].unique().to_list():
        train = ratio.filter(d["match_id"] != match_id).drop_nans().drop_nulls().to_numpy()
        test = d.filter(pl.col("match_id") == match_id)
        k = float(np.quantile(train, LEVEL, method="higher")) if train.size else math.nan
        factors.append(k)
        covered += ((test["raw_estimate"] - test["truth"]).abs() <= k * test["half_width"]).to_list()
    return float(np.mean(covered)), float(np.median(factors))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval for k successes out of n."""
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def coverage_ci(d: pl.DataFrame, column: str, n_boot: int, seed: int) -> tuple[float, float, float, bool]:
    """Coverage and its 95 % interval from a bootstrap over matches. When no resample varies (every interval
    covers, or none does, or there is one match), the bootstrap interval collapses to a point: a
    Clopper-Pearson interval over player-matches is returned instead (it ignores the clustering by match),
    flagged by the last value."""
    hits = d[column].cast(pl.Float64).to_numpy()
    est = bootstrap_by_group(d["match_id"].to_numpy(), lambda rows: float(np.mean(hits[rows])), n_boot, seed)
    if est.lo == est.hi:
        lo, hi = clopper_pearson(int(hits.sum()), hits.size)
        return est.value, lo, hi, True
    return est.value, est.lo, est.hi, False


def ci_cell(value: float, lo: float, hi: float, exact: bool) -> str:
    return f"{100 * value:.0f} % [{100 * lo:.0f}, {100 * hi:.0f}]" + ("†" if exact else "")


def card_cell(line: CardLine, unit: str, digits: int) -> str:
    """``estimate ± half-width unit``, or ``estimate unit [lo, hi]`` when the support cut the interval."""
    lo, hi = line.interval
    u = f" {unit}" if unit else ""
    if lo > line.raw_estimate - line.half_width:
        body = f"{fmt(line.estimate, digits)}{u} [{fmt(lo, digits)}, {fmt(hi, digits)}]"
    else:
        body = f"{fmt(line.estimate, digits)} ± {fmt(line.half_width, digits)}{u}"
    return f"{body} (observed {fmt(line.observed, digits)})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stats", type=Path, default=reports_dir() / "error_process_temporal_dev.json")
    parser.add_argument("--reference", default="sm")
    parser.add_argument("--family", default="matern32", choices=KERNELS)
    parser.add_argument("--marginal", default="copula", choices=("gauss", "copula"))
    parser.add_argument("--matches", nargs="*", default=sorted(idsse.IDSSE_MATCHES))
    parser.add_argument("--n-draws", type=int, default=40, help="error draws per card (each at two doses)")
    parser.add_argument("--n-outer", type=int, default=8, help="pseudo-observations of the nested bootstrap")
    parser.add_argument("--n-inner", type=int, default=8, help="draws per pseudo-observation")
    parser.add_argument("--min-minutes", type=float, default=45.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--report", type=Path, default=reports_dir() / "kpi_card.md")
    parser.add_argument("--figure", type=Path, default=reports_dir() / "figures" / "kpi_card.png")
    args = parser.parse_args()

    t_start = time.perf_counter()
    cfg = load_kpi_config()
    stats = json.loads(args.stats.read_text())
    cameras = list(stats["cameras"])
    models = {
        c: fit_error_model(stats, c, family=args.family, marginal=args.marginal, reference=args.reference)[0]
        for c in cameras
    }
    sha, dirty = git_info()
    run_dir = REPO_ROOT / "runs" / f"{dt.date.today():%Y-%m-%d}_{sha}_card"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    examples: dict[str, dict[str, Any]] = {}
    for match_id in args.matches:
        t0 = time.perf_counter()
        idsse.prepare_match(match_id)
        tracks, _, _ = period_tracks(idsse.load_players(match_id), cfg)
        roster = idsse.load_roster(match_id)
        position = dict(zip(roster["player_id"].to_list(), roster["position"].to_list(), strict=True))
        shirt = dict(zip(roster["player_id"].to_list(), roster["shirt"].to_list(), strict=True))
        team = dict(zip(roster["player_id"].to_list(), roster["team"].to_list(), strict=True))
        truth = player_kpis(tracks, cfg, resample_to=cfg.output_hz)
        population = [
            pid
            for pid, k in truth.items()
            if position.get(pid) != GOALKEEPER and k["minutes"] >= args.min_minutes
        ]
        for camera, model in models.items():
            errors = [
                sample_errors(model, t.xy.shape[0], t.xy.shape[1], seeded_rng(args.seed, match_id, camera, p))
                for p, t in enumerate(tracks)
            ]
            for pid in population:
                parts = []
                for p, t in enumerate(tracks):
                    if pid not in t.player_ids:
                        continue
                    i = t.player_ids.index(pid)
                    span = t.span(i)
                    if span is not None:
                        parts.append((t.xy + errors[p])[span[0] : span[1], i])
                rng = seeded_rng(args.seed, match_id, camera, pid, "bootstrap")
                observed, draws = kpi_draws(parts, model, cfg, rng, n_draws=args.n_draws, doses=(1.0, 2.0))
                nested = (
                    nested_estimates(parts, model, cfg, rng, n_outer=args.n_outer, n_inner=args.n_inner)
                    if args.n_outer > 1
                    else None
                )
                card: dict[str, CardLine] = {}
                for kpi in CARD_KPIS:
                    d1, d2 = draws[1.0][kpi], draws[2.0][kpi]
                    lines = {
                        "linear": card_line(
                            kpi, observed[kpi], d1, LEVEL, outer=nested["linear"][kpi] if nested else None
                        ),
                        "quadratic": card_line(
                            kpi,
                            observed[kpi],
                            d1,
                            LEVEL,
                            d2,
                            outer=nested["quadratic"][kpi] if nested else None,
                        ),
                        "quadratic_spread": card_line(kpi, observed[kpi], d1, LEVEL, d2),
                    }
                    card[kpi] = lines["quadratic"]
                    for method, line in lines.items():
                        (lo, hi), (naive_lo, naive_hi) = line.interval, line.naive_interval
                        rows.append(
                            {
                                "match_id": match_id,
                                "player_id": pid,
                                "team": team.get(pid, ""),
                                "shirt": shirt.get(pid, -1),
                                "camera": camera,
                                "kpi": kpi,
                                "method": method,
                                "truth": truth[pid][kpi],
                                "observed": line.observed,
                                "raw_estimate": line.raw_estimate,
                                "estimate": line.estimate,
                                "truncated": line.truncated,
                                "half_width": line.half_width,
                                "lo": lo,
                                "hi": hi,
                                "naive_lo": naive_lo,
                                "naive_hi": naive_hi,
                                "sd": line.sd,
                            }
                        )
                examples.setdefault(f"{match_id}:{pid}", {})[camera] = card
        print(f"{match_id}: {len(population)} player-matches in {time.perf_counter() - t0:.1f} s", flush=True)

    table = pl.DataFrame(rows).with_columns(
        covered=(pl.col("lo") <= pl.col("truth")) & (pl.col("truth") <= pl.col("hi")),
        naive_covered=(pl.col("naive_lo") <= pl.col("truth")) & (pl.col("truth") <= pl.col("naive_hi")),
    )
    table.write_parquet(run_dir / "cards.parquet")
    meta = {
        "command": "uv run python scripts/kpi_card.py " + " ".join(shlex.quote(a) for a in sys.argv[1:]),
        "git_sha": sha,
        "git_dirty": dirty,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "model": f"{args.family}:{args.marginal}:white",
        "doses": [1.0, 2.0],
        "matches": args.matches,
        "n_draws": args.n_draws,
        "n_outer": args.n_outer,
        "n_inner": args.n_inner,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    plot(table.filter(pl.col("method") == "quadratic"), cameras, args.figure)
    args.report.write_text(render(meta, table, cameras, examples, args), encoding="utf-8")
    print(f"report: {args.report}\nfigure: {args.figure}\nrun: {run_dir}\n{meta['runtime_s']} s")


def fmt(v: float, digits: int = 1, signed: bool = False) -> str:
    if v is None or not math.isfinite(v):
        return "n/a"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render(
    meta: dict[str, Any],
    table: pl.DataFrame,
    cameras: list[str],
    examples: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    dirty = " (with uncommitted changes)" if meta["git_dirty"] else ""
    n_pm = table.filter(pl.col("camera") == cameras[0]).select("match_id", "player_id").unique().height
    quad = table.filter(pl.col("method") == "quadratic")
    cov_rows = []
    for camera in cameras:
        for kpi, (label, unit, digits) in CARD_KPIS.items():
            sel = (pl.col("camera") == camera) & (pl.col("kpi") == kpi) & pl.col("half_width").is_finite()
            q = table.filter(sel & (pl.col("method") == "quadratic"))
            lin = table.filter(sel & (pl.col("method") == "linear"))
            spread = table.filter(sel & (pl.col("method") == "quadratic_spread"))
            lomo, factor = lomo_coverage(q)
            truth_sum = float(q["truth"].sum())
            positive = q.filter(pl.col("truth") > 0)
            rel_hw = (
                float((positive["half_width"] / positive["truth"]).median()) if positive.height else math.nan
            )

            def bias(d: pl.DataFrame, col: str, total: float = truth_sum) -> str:
                return f"{100 * float((d[col] - d['truth']).sum()) / total:+.1f} %"

            cov_rows.append(
                [
                    camera,
                    label,
                    ci_cell(*coverage_ci(q, "covered", args.n_boot, args.seed)),
                    ci_cell(*coverage_ci(spread, "covered", args.n_boot, args.seed)),
                    ci_cell(*coverage_ci(lin, "covered", args.n_boot, args.seed)),
                    ci_cell(*coverage_ci(q, "naive_covered", args.n_boot, args.seed)),
                    f"{100 * lomo:.0f} % (×{factor:.2f})",
                    f"{fmt(float(q['half_width'].median()), digits)} {unit}".strip()
                    + f" ({100 * rel_hw:.0f} %)",
                    f"{100 * float(q['truncated'].mean()):.0f} %",
                    bias(q, "observed"),
                    bias(lin, "estimate"),
                    bias(q, "estimate"),
                ]
            )
    lines = [
        "# KPI uncertainty card: a consistency check on IDSSE",
        "",
        '> **Consistency check, not validation.** The "observed" tracks are clean IDSSE tracking plus errors',
        "> simulated by the same model the card assumes (fitted on SN-GSR dev,"
        " `reports/kpi_budget_spectral.md`).",
        "> Coverage here checks the card's arithmetic and its noise-on-noise"
        " approximation; it says nothing about",
        "> whether the error model is right for real broadcast tracking.",
        "",
        f"- **Command**: `{meta['command']}`",
        f"- **Git commit**: `{meta['git_sha']}`{dirty}; date {meta['date']}; runtime {meta['runtime_s']} s",
        f"- **Error model** per camera: `{meta['model']}` from `reports/error_process_temporal_dev.json`"
        " (`predfoot.kpi.noise.fit_error_model`).",
        f"- **Data**: IDSSE (Bassek, Rein, Weber & Memmert 2025, CC BY 4.0), {len(meta['matches'])} matches"
        f" ({', '.join(meta['matches'])}), {n_pm} outfield player-matches with ≥ 45 min; {meta['n_draws']}"
        f" error draws per card, each at two doses; nested bootstrap with {meta['n_outer']}"
        f" pseudo-observations of {meta['n_inner']} draws each; coverage intervals: {meta['n_boot']}"
        f" bootstrap resamples over matches; seed {meta['seed']}.",
        "",
        "## The card",
        "",
        "From the observed track and the camera's error model only (`predfoot.kpi.card`): fresh errors are",
        "drawn from the model and added to the observed track at one and two extra doses of error variance,",
        "and the 10 fps KPI pipeline runs on each copy (SIMEX, Cook & Stefanski 1994). The **linear**",
        "estimate removes the change caused by one extra dose (noise on noise); the **quadratic** one",
        "extrapolates the parabola through zero, one and two extra doses back to no error. Estimates and",
        "intervals that leave the KPI's support (all these KPIs are ≥ 0) are projected onto it. The",
        "interval is estimate ± t·s: s² is the sampling variance of the estimate over the observed error,",
        "from a nested parametric bootstrap (the same extrapolation on pseudo-observations, the observed",
        "track plus one more model draw), plus the Monte Carlo variance of the draws, combined draw by draw",
        "since both doses reuse each draw. The spread of one dose's effect, the previous choice of s,",
        "understates the sampling variance of KPIs that jump with the track: the extrapolation amplifies",
        "their noise. The intervals are model-based and uncalibrated, hence *nominal* 90 %. The **naive**",
        "interval is the observed value ± t·(spread of one dose), without bias correction. The truth is the",
        "clean track through the same 10 fps pipeline.",
        "",
        "## Coverage of the clean truth",
        "",
        "Coverage of nominal 90 % intervals across player-matches [95 % CI over matches]: quadratic and",
        "linear with the nested-bootstrap variance, quadratic with the spread of one dose (previous",
        "half-width), and naive. LOMO: the quadratic interval with a leave-one-match-out calibrated",
        "half-width (median scale factor in brackets). Truncated: share of quadratic estimates projected",
        "onto the support. Bias: relative error of the KPI summed over player-matches, before (observed)",
        "and after correction (estimate). The observed bias comes from errors simulated by the model of",
        "`reports/kpi_budget_spectral.md`, whose 5 fps check against SN-GSR fails (section 2 there): it",
        "overstates the KPI bias of predfoot's real tracking and is not predfoot's KPI error.",
        "",
        md_table(
            [
                "Camera",
                "KPI",
                "Quadratic",
                "Quadratic, one-dose spread",
                "Linear",
                "Naive",
                "Quadratic, LOMO-calibrated",
                "Median half-width (rel.)",
                "Truncated",
                "Bias, observed",
                "Bias after linear",
                "Bias after quadratic",
            ],
            cov_rows,
        ),
        "",
        "† The bootstrap over matches collapsed to a point (every interval covers, or none does):",
        "Clopper-Pearson interval over player-matches instead, which ignores the clustering by match.",
        "",
        "![Cards for HI distance](figures/kpi_card.png)",
        "",
        "## Example cards",
        "",
    ]
    picks = pick_examples(quad, cameras[-1])
    for key in picks:
        match_id, pid = key.split(":")
        info = table.filter((pl.col("match_id") == match_id) & (pl.col("player_id") == pid)).row(
            0, named=True
        )
        lines += [f"**{match_id}, {info['team']} #{info['shirt']}**", ""]
        rows = []
        for kpi, (label, unit, digits) in CARD_KPIS.items():
            cells = [label]
            for camera in cameras:
                line: CardLine = examples[key][camera][kpi]
                cells.append(card_cell(line, unit, digits))
            truth = quad.filter(
                (pl.col("match_id") == match_id) & (pl.col("player_id") == pid) & (pl.col("kpi") == kpi)
            )["truth"][0]
            cells.append(f"{fmt(truth, digits)} {unit}".strip())
            rows.append(cells)
        lines += [
            md_table(["KPI", *[f"Card, {c} camera (nominal 90 %)" for c in cameras], "Clean truth"], rows),
            "",
        ]
    lines += [
        "## Reading it",
        "",
        "- A naive interval around the observed value inherits the whole error-induced bias, which for",
        "  distance KPIs is many half-widths (column *Naive*): the bias correction is the core of the card.",
        "- The extrapolation assumes a smooth response of each KPI to the error variance. The linear step",
        "  cannot follow a curved response (distance of slow phases grows like σ, threshold KPIs respond",
        "  convexly); the quadratic one follows curvature, at the price of a larger sampling variance: it",
        "  multiplies the observed value by 3, and for KPIs that jump with the track the averaged doses do",
        "  not cancel that noise. Compare the columns *Quadratic* and *Quadratic, one-dose spread*.",
        "- Where coverage stays below 90 % with the nested-bootstrap variance, the remaining extrapolation",
        "  bias is the cause (compare *Bias after quadratic* with the half-width). The leave-one-match-out",
        "  factor shows how much the half-width must grow to reach 90 %; it is fitted on the other matches",
        "  only, on simulated errors, so it does not carry over to real tracking as such.",
        "- Everything is conditional on the error model: on real tracking the card is only as good as the",
        "  error process measured on SN-GSR (stationary, independent of motion, 5 fps, smoothed-label",
        "  reference; see `reports/kpi_budget_spectral.md`).",
        "",
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def pick_examples(table: pl.DataFrame, camera: str) -> list[str]:
    """Three player-matches at the 25th, 50th and 90th percentiles of clean HI distance."""
    d = table.filter((pl.col("camera") == camera) & (pl.col("kpi") == "hi_distance_m")).sort("truth")
    idx = [int(q * (d.height - 1)) for q in (0.25, 0.5, 0.9)]
    return [f"{d['match_id'][i]}:{d['player_id'][i]}" for i in idx]


def plot(table: pl.DataFrame, cameras: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grey, blue, surface = "#0b0b0b", "#52514e", "#b7b5ae", "#2a78d6", "#fcfcfb"
    fig, axes = plt.subplots(1, len(cameras), figsize=(11, 4.6), sharey=True, facecolor=surface)
    for ax, camera in zip(np.atleast_1d(axes), cameras, strict=True):
        d = table.filter((pl.col("camera") == camera) & (pl.col("kpi") == "hi_distance_m")).sort("truth")
        x = np.arange(d.height)
        truth, lo, hi, naive_lo, naive_hi = (
            d[c].to_numpy() for c in ("truth", "lo", "hi", "naive_lo", "naive_hi")
        )
        ax.set_facecolor(surface)
        ax.vlines(x - 0.18, naive_lo, naive_hi, color=grey, lw=1.2, label="naive interval (observed value)")
        ax.vlines(
            x + 0.18,
            lo,
            hi,
            color=blue,
            lw=1.2,
            label="corrected nominal 90 % interval (quadratic SIMEX, nested bootstrap)",
        )
        ax.plot(x, truth, "o", ms=2.6, color=ink, label="clean truth")
        cov = float(d["covered"].mean())
        ax.set_title(f"{camera} camera: corrected interval covers {100 * cov:.0f} %", color=ink, fontsize=10)
        ax.set_xlabel("player-matches, sorted by clean HI distance", color=muted, fontsize=9)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("HI distance ≥ 20 km/h (m)", color=muted, fontsize=9)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "KPI card on IDSSE (simulated errors, consistency check): HI distance per player-match",
        color=ink,
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
