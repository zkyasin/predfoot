"""Figures of the development results, and the numbers behind them in reports/dev_figures.md.

Reads the per-player tables of two scripts/eval_dev.py runs on the SN-GSR dev games, one with the per-frame
camera (``--smooth none``) and one with the PTZ-smoothed camera (``--smooth ptz``), refits the
leave-one-game-out (LOGO) conformal regions at the nominal levels 50, 80, 90 and 95 % with
``eval_dev.logo``, and reads the model-free KPI proxy from reports/error_process_temporal_dev.json. Writes
to reports/figures/:

- ``reliability.png``: held-out coverage of the Mondrian ellipses minus the nominal level, both cameras,
  with player-frames pooled and with every clip weighted equally, against the SN-GSR raw labels and against
  the trusted line oracle (there the regions are calibrated on oracle errors);
- ``area.png``: median area of the Mondrian ellipses and of the constant circle at each level;
- ``coverage_by_distance.png``: coverage at 90 % by tercile of metres per pixel at the feet;
- ``kpi_proxy.png``: distance and high-speed-running distance biases of the KPI proxy against the smoothed
  labels: per-frame camera, smoothed camera, and the raw labels' own bias (the noise floor of the reference).

Each held-out game's quantiles come from the five other games, at every level; intervals resample clips.
Coverage is reported two ways: pooled over the held-out player-frames (eval_dev.py's headline) and with
every clip weighted equally (the mean of the clips' own coverages), which is how the grouped conformal
quantile weighs its calibration clips. The two differ when a clip's coverage depends on its number of
player-frames, as it does against the line oracle; the report breaks coverage down by clip size.

The PTZ table keeps each clip smoothed with the label-free noise parameters of its own held-out fold only.
eval_dev.py fits game g's quantiles on the other games re-smoothed with fold g's parameters, which the table
does not keep, so here they are fitted on the other games' rows as saved: per-game numbers of the smoothed
camera can differ slightly from reports/m2_temporal_dev.md. The per-frame camera's 90 % fits, intervals
included, reproduce reports/m2_logo_dev.md: they draw from one generator, labels first, as eval_dev.py does.

Usage:
    uv run python scripts/make_figures.py --none-run runs/<per-frame run> --ptz-run runs/<ptz run>
The figures hold metrics only, no SoccerNet pixels (ADR-0002).
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import polars as pl
import typer
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from predfoot.io.files import atomic_write_bytes, atomic_write_text
from predfoot.paths import REPO_ROOT


def load_script(name: str) -> ModuleType:
    """A sibling script as a module (scripts/ is not a package), shared with any earlier import of it."""
    key = f"scripts_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


eval_dev = load_script("eval_dev")

app = typer.Typer(add_completion=False)
ALPHAS = (0.5, 0.2, 0.1, 0.05)  # nominal levels 50, 80, 90, 95 %
TERCILE_ALPHA = 0.1
REFS = {"lab": "SN-GSR raw labels", "or": "trusted line oracle"}
CAMERAS = {"none": "per-frame camera", "ptz": "PTZ-smoothed camera"}
# Reference palette (blue / orange, checked for colour-vision deficiencies), as predfoot.viz.minimap.
COLOURS = {"none": "#2a78d6", "ptz": "#eb6834"}
SURFACE, INK, MUTED, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b7b5ae"
DPI = 110
DEFAULT_KPI_JSON = Path("reports/error_process_temporal_dev.json")
DEFAULT_OUT = Path("reports/figures")
DEFAULT_REPORT = Path("reports/dev_figures.md")


@dataclass
class Level:
    """LOGO coverage and median region area of one camera, reference and nominal level."""

    camera: str
    ref: str
    alpha: float
    n: int
    clips: int
    cov_mond: float
    cov_mond_ci: tuple[float, float]
    cov_const: float
    cov_const_ci: tuple[float, float]
    area_mond: float  # median over held-out player-frames, m²
    area_const: float
    cov_clip: float = float("nan")  # every held-out clip weighted equally
    cov_clip_ci: tuple[float, float] = (float("nan"), float("nan"))
    terciles: list[dict[str, Any]] = field(default_factory=list)  # at TERCILE_ALPHA only
    size_terciles: list[dict[str, Any]] = field(default_factory=list)  # at TERCILE_ALPHA only

    @property
    def nominal(self) -> float:
        return 1.0 - self.alpha


def load_run(run_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Headline rows and filled rows (raw calibration failed, camera from the smoother) of a run's table."""
    players = pl.read_parquet(run_dir / "players.parquet")
    if (
        "fold" in players.columns
        and players.filter(pl.col("fold").is_not_null() & (pl.col("fold") != pl.col("game"))).height
    ):
        raise ValueError(f"{run_dir}: expected the held-out rows that eval_dev.py saves (fold == game)")
    return players.filter(~pl.col("filled")), players.filter(pl.col("filled"))


def tercile_rows(held: pl.DataFrame, rng: np.random.Generator, n_boot: int) -> list[dict[str, Any]]:
    """Coverage by tercile of metres per pixel at the feet (edges as in ``eval_dev.logo``), with clip CIs."""
    mpp = held["mpp"].to_numpy()
    edges = np.quantile(mpp, [1 / 3, 2 / 3])
    bins = np.searchsorted(edges, mpp)
    clips = held["clip"].to_numpy()
    out = []
    for b in range(3):
        sel = bins == b
        row: dict[str, Any] = {"bin": b, "n": int(sel.sum()), "edges": edges.tolist()}
        for kind in ("mond", "const"):
            hit = held[f"hit_{kind}"].to_numpy()[sel]
            row[f"cov_{kind}"] = float(hit.mean())
            row[f"cov_{kind}_ci"] = eval_dev._clip_bootstrap(hit, clips[sel], rng, n_boot)
        out.append(row)
    return out


def _per_clip(held: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Each held-out clip's coverage by the Mondrian ellipses and its number of player-frames."""
    per_clip = held.group_by("clip").agg(pl.col("hit_mond").mean().alias("cov"), pl.len().alias("n"))
    per_clip = per_clip.sort("clip")
    return per_clip["cov"].to_numpy(), per_clip["n"].to_numpy()


def clip_weighted(
    held: pl.DataFrame, rng: np.random.Generator, n_boot: int
) -> tuple[float, tuple[float, float]]:
    """Coverage with every held-out clip weighted equally (the mean of the clips' own coverages), as the
    grouped quantile weighs its calibration clips, with a 95 % interval resampling clips."""
    cov, _ = _per_clip(held)
    boot = [cov[rng.integers(0, len(cov), len(cov))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(cov.mean()), (float(lo), float(hi))


def size_tercile_rows(held: pl.DataFrame) -> list[dict[str, Any]]:
    """Mean coverage of the held-out clips in each tercile of their number of player-frames (NaN for an empty
    tercile, when many clips have the same size)."""
    cov, n = _per_clip(held)
    edges = np.quantile(n, [1 / 3, 2 / 3])
    bins = np.searchsorted(edges, n)
    out = []
    for b in range(3):
        sel = bins == b
        mean = float(cov[sel].mean()) if sel.any() else float("nan")
        out.append({"bin": b, "clips": int(sel.sum()), "edges": edges.tolist(), "cov": mean})
    return out


def levels(
    head: pl.DataFrame,
    filled: pl.DataFrame,
    camera: str,
    alphas: tuple[float, ...] = ALPHAS,
    n_boot: int = 300,
    seed: int = 0,
) -> list[Level]:
    """LOGO fits of ``eval_dev.logo`` at each level and reference.

    The fits at eval_dev.py's level (α = 0.1) share one generator seeded with ``seed``, labels first, then
    the oracle: eval_dev.py's own sequence, so their intervals are its intervals. Every other fit gets a fresh
    generator with the same seed, and the extra intervals (clip-weighted, distance terciles) their own.
    """
    out = []
    headline = np.random.default_rng(seed)
    for ref in REFS:
        for alpha in alphas:
            rng = headline if alpha == eval_dev.ALPHA else np.random.default_rng(seed)
            _, s, held, _ = eval_dev.logo(head, ref, rng, n_boot=n_boot, extra=filled, alpha=alpha)
            extra_rng = np.random.default_rng(seed + 1)
            cov_clip, cov_clip_ci = clip_weighted(held, extra_rng, n_boot)
            level = Level(
                camera=camera,
                ref=ref,
                alpha=alpha,
                n=s["n"],
                clips=s["clips"],
                cov_mond=s["cov_mond"],
                cov_mond_ci=s["cov_mond_ci"],
                cov_const=s["cov_const"],
                cov_const_ci=s["cov_const_ci"],
                area_mond=s["a_mond"],
                area_const=s["a_const"],
                cov_clip=cov_clip,
                cov_clip_ci=cov_clip_ci,
            )
            if alpha == TERCILE_ALPHA:
                level.terciles = tercile_rows(held, extra_rng, n_boot)
                level.size_terciles = size_tercile_rows(held)
            out.append(level)
    return out


def kpi_proxy(path: Path) -> dict[str, dict[str, float]]:
    """Distance and HSR-distance biases of the KPI proxy per camera, plus the raw labels' own bias of the
    first camera's sample (the noise floor that scripts/error_process.py reports)."""
    cameras = json.loads(path.read_text())["cameras"]
    out = {
        name: {"distance": c["kpi_proxy"]["distance_bias"], "hsr": c["kpi_proxy"]["hsr_bias"]}
        for name, c in cameras.items()
        if c.get("kpi_proxy")
    }
    first = next(c["kpi_proxy"] for c in cameras.values() if c.get("kpi_proxy"))
    out["raw labels"] = {"distance": first["raw_distance_bias"], "hsr": first["raw_hsr_bias"]}
    return out


# --- figures --------------------------------------------------------------------------------------------


def _figure(width: float, height: float, ncols: int = 1) -> tuple[Figure, list[Axes]]:
    fig = Figure(figsize=(width, height), dpi=DPI, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = list(fig.subplots(1, ncols, squeeze=False)[0])
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GREY)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return fig, axes


def _labels(ax: Axes, title: str, xlabel: str, ylabel: str | None) -> None:
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)


def _save(fig: Figure, path: Path) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=SURFACE, metadata={"Software": None})
    atomic_write_bytes(path, buffer.getvalue())


def _series(rows: list[Level], camera: str, ref: str) -> list[Level]:
    return sorted((r for r in rows if r.camera == camera and r.ref == ref), key=lambda r: r.nominal)


def _pct_err(values: list[float], cis: list[tuple[float, float]]) -> np.ndarray:
    v = np.array(values)
    return 100 * np.array([v - np.array([c[0] for c in cis]), np.array([c[1] for c in cis]) - v])


def plot_reliability(rows: list[Level], path: Path) -> None:
    """Coverage minus the nominal level against the nominal level: a horizontal zero line is the target, so
    offsetting the series sideways does not bias how far each sits from it."""
    fig, axes = _figure(7.4, 3.7, ncols=2)
    offsets = {
        ("none", "pooled"): -0.21,
        ("none", "clip"): -0.07,
        ("ptz", "pooled"): 0.07,
        ("ptz", "clip"): 0.21,
    }
    nominal_levels = sorted(1 - a for a in ALPHAS)
    for ax, ref in zip(axes, REFS, strict=True):
        ax.axhline(0, color=GREY, lw=1.0, zorder=1)
        for camera in CAMERAS:
            rs = _series(rows, camera, ref)
            if not rs:
                continue
            x = np.array([nominal_levels.index(r.nominal) for r in rs], dtype=float)
            nominal = np.array([r.nominal for r in rs])
            for weight in ("pooled", "clip"):
                pooled = weight == "pooled"
                y = [r.cov_mond if pooled else r.cov_clip for r in rs]
                ax.errorbar(
                    x + offsets[(camera, weight)],
                    100 * (np.array(y) - nominal),
                    yerr=_pct_err(y, [r.cov_mond_ci if pooled else r.cov_clip_ci for r in rs]),
                    fmt="o-" if pooled else "o--",
                    color=COLOURS[camera],
                    mfc=COLOURS[camera] if pooled else SURFACE,
                    ms=5,
                    lw=1.2,
                    elinewidth=1.1,
                    capsize=0,
                    mew=1.2,
                    zorder=3,
                )
        ax.set_xticks(range(len(nominal_levels)), [f"{100 * v:.0f} %" for v in nominal_levels])
        title = "SN-GSR raw labels" if ref == "lab" else "line oracle, regions calibrated on it"
        _labels(
            ax,
            f"Reference: {title}",
            "nominal level",
            "held-out coverage minus nominal (points)" if ref == "lab" else None,
        )
    low = min(ax.get_ylim()[0] for ax in axes)
    high = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(low, high)
    handles = []
    for c in CAMERAS:
        handles.append(
            Line2D([], [], color=COLOURS[c], marker="o", lw=1.2, label=f"{CAMERAS[c]}, player-frames pooled")
        )
        handles.append(
            Line2D(
                [],
                [],
                color=COLOURS[c],
                marker="o",
                lw=1.2,
                ls="--",
                mfc=SURFACE,
                label=f"{CAMERAS[c]}, clips weighted equally",
            )
        )
    fig.legend(handles=handles, loc="outside lower center", ncols=2, fontsize=7.5, frameon=False)
    fig.suptitle(
        "Mondrian ellipses, leave one game out (SN-GSR dev, 6 games); bars: 95 % CI over clips",
        fontsize=9,
        color=INK,
    )
    _save(fig, path)


def plot_area(rows: list[Level], path: Path) -> None:
    fig, axes = _figure(7.4, 3.4, ncols=2)
    for ax, ref in zip(axes, REFS, strict=True):
        for camera in CAMERAS:
            rs = _series(rows, camera, ref)
            if not rs:
                continue
            x = [100 * r.nominal for r in rs]
            ax.plot(x, [r.area_mond for r in rs], "o-", color=COLOURS[camera], ms=5, lw=1.5, mec=SURFACE)
            ax.plot(x, [r.area_const for r in rs], "o--", color=COLOURS[camera], ms=5, lw=1.2, mfc=SURFACE)
        ax.set(xlim=(45, 100), xticks=[50, 80, 90, 95])
        ax.set_yscale(
            "log"
        )  # the ratio of the two regions is what matters: a constant ratio is a constant gap
        ticks = [
            t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]
        ]
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        _labels(
            ax,
            f"Reference: {REFS[ref]}",
            "nominal level (%)",
            "median region area (m², log scale)" if ref == "lab" else None,
        )
    handles = []
    for c in CAMERAS:
        handles.append(
            Line2D([], [], color=COLOURS[c], marker="o", lw=1.5, mec=SURFACE, label=f"{CAMERAS[c]}, ellipses")
        )
        handles.append(
            Line2D(
                [],
                [],
                color=COLOURS[c],
                marker="o",
                lw=1.2,
                ls="--",
                mfc=SURFACE,
                label=f"{CAMERAS[c]}, constant circle",
            )
        )
    axes[0].legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False, handlelength=3.2)
    fig.suptitle(
        "Region size at the same nominal level: Mondrian ellipses against a constant circle (held-out games)",
        fontsize=9,
        color=INK,
    )
    _save(fig, path)


def plot_terciles(rows: list[Level], path: Path) -> None:
    fig, axes = _figure(7.4, 3.4, ncols=2)
    offsets = {
        ("none", "mond"): -0.27,
        ("none", "const"): -0.09,
        ("ptz", "mond"): 0.09,
        ("ptz", "const"): 0.27,
    }
    for ax, ref in zip(axes, REFS, strict=True):
        ax.axhline(100 * (1 - TERCILE_ALPHA), color=GREY, lw=1.0, ls="--", zorder=1)
        for camera in CAMERAS:
            at = [r for r in _series(rows, camera, ref) if r.alpha == TERCILE_ALPHA and r.terciles]
            if not at:
                continue
            terciles = at[0].terciles
            for kind in ("mond", "const"):
                y = [t[f"cov_{kind}"] for t in terciles]
                ax.errorbar(
                    np.arange(3) + offsets[(camera, kind)],
                    100 * np.array(y),
                    yerr=_pct_err(y, [t[f"cov_{kind}_ci"] for t in terciles]),
                    fmt="o",
                    color=COLOURS[camera],
                    mfc=COLOURS[camera] if kind == "mond" else SURFACE,
                    ms=5,
                    elinewidth=1.2,
                    capsize=0,
                    mew=1.2,
                    zorder=3,
                )
        ax.set_xticks(range(3), ["near third", "middle third", "far third"])
        ax.set_ylim(75, 100)
        _labels(
            ax,
            f"Reference: {REFS[ref]}",
            "distance tercile (metres per pixel at the feet)",
            "held-out coverage at 90 % (%)" if ref == "lab" else None,
        )
    handles = []
    for c in CAMERAS:
        handles.append(
            Line2D([], [], color=COLOURS[c], marker="o", ls="none", label=f"{CAMERAS[c]}, ellipses")
        )
        handles.append(
            Line2D(
                [],
                [],
                color=COLOURS[c],
                marker="o",
                ls="none",
                mfc=SURFACE,
                label=f"{CAMERAS[c]}, constant circle",
            )
        )
    axes[0].legend(handles=handles, loc="lower left", fontsize=7.5, frameon=False)
    fig.suptitle("Coverage by distance at the 90 % level; bars: 95 % CI over clips", fontsize=9, color=INK)
    _save(fig, path)


def plot_kpi(kpi: dict[str, dict[str, float]], path: Path) -> None:
    fig, (ax,) = _figure(6.4, 3.2)
    groups = [("distance", "Distance"), ("hsr", "High-speed-running distance (≥ 5.56 m/s)")]
    bars = [
        ("none", CAMERAS["none"], COLOURS["none"]),
        ("ptz", CAMERAS["ptz"], COLOURS["ptz"]),
        ("raw labels", "raw labels' own bias (noise floor of the reference)", GREY),
    ]
    width = 0.26
    top = max(100 * kpi[key][g] for key, _, _ in bars for g, _ in groups if key in kpi)
    for j, (key, label, colour) in enumerate(bars):
        if key not in kpi:
            continue
        values = [100 * kpi[key][g] for g, _ in groups]
        x = np.arange(len(groups)) + (j - 1) * (width + 0.03)
        ax.bar(x, values, width, color=colour, label=label, zorder=2)
        for xi, v in zip(x, values, strict=True):
            ax.text(xi, v + 0.015 * top, f"{v:+.1f} %", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(range(len(groups)), [t for _, t in groups])
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.15 * top)
    ax.grid(axis="x", visible=False)
    _labels(ax, "", "", "bias against the smoothed labels (%)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        "Model-free KPI proxy on SN-GSR dev (5 fps tracks, Savitzky-Golay 5/2)", fontsize=9, color=INK
    )
    _save(fig, path)


# --- report ---------------------------------------------------------------------------------------------


def _pct(v: float) -> str:
    return f"{100 * v:.1f} %"


def _ci(ci: tuple[float, float]) -> str:
    return f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def report(
    rows: list[Level], kpi: dict[str, dict[str, float]], command: str, sha: str, sources: str, n_boot: int
) -> str:
    md = [
        "# Development figures: coverage at four levels, region size, distance terciles, KPI proxy",
        "",
        f"`{command}` · commit `{sha}` · {sources}",
        "",
        "Exploratory numbers on the six SN-GSR dev games (2, 3, 4, 5, 6, 9; frames one in five; ground-truth"
        " boxes), not the preregistered test (`docs/preregistration.md`). For every level, each held-out"
        " game's conformal quantiles are fitted on the five other games (`eval_dev.logo`: Mondrian bins over"
        " metres per pixel at the feet × RANSAC inliers, within each covariance source); intervals resample"
        f" clips ({n_boot} draws). Each reference calibrates its own regions: the line-oracle rows are"
        " regions fitted on oracle errors, not the label-calibrated regions that `predfoot run` ships"
        " (`scripts/fit_conformal.py`).",
        "",
        "Coverage is given pooled over the held-out player-frames, as in eval_dev.py, and with every clip"
        " weighted equally (the mean of the clips' own coverages), as the grouped quantile weighs its"
        " calibration clips.",
        "",
        "## Coverage and median region area at each nominal level",
        "",
        "![Reliability](figures/reliability.png)",
        "",
        "![Region area](figures/area.png)",
        "",
        "| Reference | Camera | Nominal | Player-frames | Mondrian ellipses [95 % CI] | Clips weighted"
        " equally [95 % CI] | Median area (m²) | Constant circle [95 % CI] | Area (m²) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ref in REFS:
        for camera in CAMERAS:
            for r in _series(rows, camera, ref):
                md.append(
                    f"| {REFS[ref]} | {CAMERAS[camera]} | {r.nominal:.0%} | {r.n} | {_pct(r.cov_mond)}"
                    f" {_ci(r.cov_mond_ci)} | {_pct(r.cov_clip)} {_ci(r.cov_clip_ci)} | {r.area_mond:.2f} |"
                    f" {_pct(r.cov_const)} {_ci(r.cov_const_ci)} | {r.area_const:.2f} |"
                )
    md += [
        "",
        f"## Coverage by clip size at {1 - TERCILE_ALPHA:.0%}",
        "",
        "Mean coverage of the held-out clips in each tercile of their number of held-out player-frames. Where"
        " it rises with clip size, pooling player-frames weighs the better-covered clips more and lifts"
        " coverage above the clip-weighted figure. Against the line oracle a clip's player-frames are its"
        " trusted ones, frames whose annotated lines fit a homography within 3 px.",
        "",
        "| Reference | Camera | Clip size (player-frames) | Clips | Mean clip coverage |",
        "|---|---|---|---|---|",
    ]
    for ref in REFS:
        for camera in CAMERAS:
            for r in _series(rows, camera, ref):
                for t in r.size_terciles:
                    e1, e2 = t["edges"]
                    span = (f"≤ {e1:.0f}", f"{e1:.0f}-{e2:.0f}", f"> {e2:.0f}")[t["bin"]]
                    cov = _pct(t["cov"]) if t["clips"] else "–"
                    md.append(f"| {REFS[ref]} | {CAMERAS[camera]} | {span} | {t['clips']} | {cov} |")
    md += [
        "",
        f"## Coverage by distance at {1 - TERCILE_ALPHA:.0%}",
        "",
        "![Coverage by distance](figures/coverage_by_distance.png)",
        "",
        "Terciles of metres per pixel at the feet over the held-out player-frames of each camera and"
        " reference.",
        "",
        "| Reference | Camera | Tercile (m/px) | Player-frames | Mondrian ellipses [95 % CI] |"
        " Constant circle [95 % CI] |",
        "|---|---|---|---|---|---|",
    ]
    for ref in REFS:
        for camera in CAMERAS:
            for r in _series(rows, camera, ref):
                for t in r.terciles:
                    e1, e2 = t["edges"]
                    span = ("< " + f"{e1:.3f}", f"{e1:.3f}-{e2:.3f}", "> " + f"{e2:.3f}")[t["bin"]]
                    md.append(
                        f"| {REFS[ref]} | {CAMERAS[camera]} | {span} | {t['n']} | {_pct(t['cov_mond'])}"
                        f" {_ci(t['cov_mond_ci'])} | {_pct(t['cov_const'])} {_ci(t['cov_const_ci'])} |"
                    )
    md += [
        "",
        "## KPI proxy",
        "",
        "![KPI proxy](figures/kpi_proxy.png)",
        "",
        "From `reports/error_process_temporal_dev.json` (`scripts/error_process.py`, the same runs): runs"
        " of at least 8 consecutive cached frames per track, Savitzky-Golay positions and speeds, compared"
        " with the smoothed labels. The raw labels' own bias against the same reference is its noise floor.",
        "",
        "| Track | Distance bias | HSR-distance bias |",
        "|---|---|---|",
    ]
    names = {**CAMERAS, "raw labels": "raw labels (noise floor)"}
    for key, v in kpi.items():
        md.append(f"| {names.get(key, key)} | {v['distance']:+.1%} | {v['hsr']:+.1%} |")
    md += [
        "",
        "Notes. Frames are one in five and correlated within a clip: the effective sample is closer to the"
        " 115 clips than to the player-frames. Filled frames (raw calibration failed, camera from the"
        " smoother) calibrate their own source but are not in these coverages, as in eval_dev.py. The"
        " grouped quantile weighs every calibration clip equally at the level (1 − α)(1 + 1/G) for G"
        " calibration clips (about 96 per fold, fewer in some Mondrian bins): with similar clips, a clip's"
        " coverage lands about (1 − α)/G above nominal, 90.9 % at the 90 % level and 96.0 % at the 95 % level"
        " for G = 96, a little more in bins with fewer clips; the constant circle, one quantile over all"
        " calibration clips, shows it. The correction cannot explain more: where the pooled coverage exceeds"
        " the clip-weighted one, the weighting does (clip-size table). The PTZ table keeps each clip"
        " smoothed with the label-free noise parameters of its own held-out fold; eval_dev.py fits game g's"
        " quantiles on the other games re-smoothed with fold g's parameters (not saved), so here they are"
        " fitted on the other games' saved rows, and the smoothed camera's numbers can differ slightly from"
        " `reports/m2_temporal_dev.md`.",
    ]
    return "\n".join(md) + "\n"


def shown(path: Path) -> str:
    """``path`` relative to the repository when it is inside it, else its name: no local paths in reports."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


@app.command()
def main(
    none_run: Path = typer.Option(..., help="eval_dev.py run folder of the per-frame camera."),  # noqa: B008
    ptz_run: Path = typer.Option(..., help="eval_dev.py run folder of the PTZ-smoothed camera."),  # noqa: B008
    kpi_json: Path = typer.Option(DEFAULT_KPI_JSON, help="error_process.py --json output."),  # noqa: B008
    out_dir: Path = typer.Option(DEFAULT_OUT, help="Folder of the PNG figures."),  # noqa: B008
    report_path: Path = typer.Option(DEFAULT_REPORT, "--report", help="Markdown report."),  # noqa: B008
    n_boot: int = typer.Option(300, help="Bootstrap draws over clips."),
) -> None:
    rows: list[Level] = []
    for camera, run in (("none", none_run), ("ptz", ptz_run)):
        head, filled = load_run(run)
        rows += levels(head, filled, camera, n_boot=n_boot)
        typer.echo(f"{camera}: {len(rows)} fits")
    kpi = kpi_proxy(kpi_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability(rows, out_dir / "reliability.png")
    plot_area(rows, out_dir / "area.png")
    plot_terciles(rows, out_dir / "coverage_by_distance.png")
    plot_kpi(kpi, out_dir / "kpi_proxy.png")
    command = (
        f"uv run python scripts/make_figures.py --none-run runs/{none_run.name} --ptz-run runs/{ptz_run.name}"
    )
    sources = (
        f"per-player tables `runs/{none_run.name}` (per-frame camera) and `runs/{ptz_run.name}`"
        f" (PTZ-smoothed camera), not in git; KPI proxy from `{shown(kpi_json)}`."
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = report(rows, kpi, command, git_sha(), sources, n_boot)
    atomic_write_text(report_path, text)
    typer.echo(text)


if __name__ == "__main__":
    app()
