"""Side-by-side minimap animation of one dev clip: per-frame camera (left), PTZ-smoothed camera (right).

Reads the per-player and per-frame tables of two scripts/eval_dev.py runs on the SN-GSR dev games
(``--smooth none`` and ``--smooth ptz``). Every predicted position is drawn with its calibrated 90 % region
{p : (p − x̂)ᵀ Σp⁻¹ (p − x̂) ≤ q̂}, where q̂ is the Mondrian conformal quantile of the player's bin, fitted by
``eval_dev.logo`` on the five games other than the clip's (leave one game out), within each covariance
source. Labelled positions (SN-GSR raw labels, the reference of the headline coverage) are small dark dots.
A panel is greyed when its camera publishes no position (calibration failed, nothing filled it), and hatched
when the camera published positions that cannot be checked: the frame has no usable SN-GSR position label
(``has_labeled_camera`` false, truncated boxes, labels from a failed annotators' camera), so players.parquet
holds no row for it. Such a frame is not evaluated, and its line reprojection is printed instead, since the
camera may well be wrong there. Only pitch coordinates are drawn, never frame pixels (ADR-0002). For the
smoothed camera the quantiles are fitted on the other games' saved rows (see scripts/make_figures.py for why
this differs slightly from eval_dev.py).

Clip (``--clip auto``): among the clips whose smoothed camera moves fastest (top third of the median image
speed of a ground grid, a label-free measure), the one where smoothing cuts the frame-to-frame jitter of the
common-mode error most: the ratio smoothed / per-frame of the robust common-mode step RMS against the smoothed
labels (``eval_dev.step_stats``, errors of 10 m or more left out). Window (``--start auto``): the
``--seconds`` window where that jitter is largest for the per-frame camera. The numbers behind both choices
are printed next to the dev medians: the clip is chosen to show the smoothing, it is not a typical clip.

Usage:
    uv run python scripts/render_minimap_demo.py --none-run runs/<per-frame run> --ptz-run runs/<ptz run>
writes assets/minimap_demo.gif (5 fps) and assets/minimap_demo.png (the frame of the window where the
per-frame camera's median error is largest).

``--style plain`` draws the same clip, window and frames for a general audience (the showcase page and the
README): panels titled "Frame by frame" and "Smoothed over time", a three-word legend, no counts, areas or
subtitles, larger type; the two honest states stay (greyed "no position", hatched "not checked"). It writes
assets/showcase/smoothing.gif, .mp4 (H.264, needs ffmpeg) and .jpg (the same still) and no report.
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import polars as pl
import typer
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Rectangle
from PIL import Image

from predfoot.io.files import atomic_write_bytes, atomic_write_text
from predfoot.viz.minimap import PITCH_LINE, SURFACE, TEXT, TEXT_SECONDARY, VEIL, draw_pitch, ellipse_axes


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
render_demo = load_script("render_demo")

app = typer.Typer(add_completion=False)
CAMERAS = {"none": "Per-frame camera", "ptz": "PTZ-smoothed camera"}
NAMES = {"none": "per-frame camera", "ptz": "PTZ-smoothed camera"}
COLOURS = {"none": "#2a78d6", "ptz": "#eb6834"}  # as scripts/make_figures.py
NATIVE_FPS = 25.0
STEP_FRAMES = 5  # cached frames are one in five
MARGIN_M = 4.0
PANEL_ASPECT = 1.3  # width / height of the pitch window of each panel
DEFAULT_GIF = Path("assets/minimap_demo.gif")
DEFAULT_PNG = Path("assets/minimap_demo.png")
DEFAULT_REPORT = Path("reports/minimap_demo.md")
#: Panel states: positions with their regions and labels; no position published; positions published on a
#: frame without usable labels (not evaluated).
SHOWN, ABSTAINS, UNLABELLED = "shown", "abstains", "unlabelled"
#: The plain style (--style plain): titles, words on the veils, outputs.
PLAIN_TITLES = {"none": "Frame by frame", "ptz": "Smoothed over time"}
PLAIN_VEILS = {ABSTAINS: "no position", UNLABELLED: "not checked"}
PLAIN_GIF = Path("assets/showcase/smoothing.gif")
PLAIN_MP4 = Path("assets/showcase/smoothing.mp4")
PLAIN_POSTER = Path("assets/showcase/smoothing.jpg")


@dataclass
class Run:
    """One camera's held-out rows (with their quantiles) and per-frame outcomes."""

    players: pl.DataFrame  # held-out rows, filled rows included, with q_mond and hit_mond
    frames: pl.DataFrame  # clip, frame, ok_out, line_px_out, cam_speed_px_s (+ filled when smoothed)


def load_run(run_dir: Path, n_boot: int = 10) -> Run:
    """Held-out rows of a run, each with the Mondrian quantile of its LOGO fold (labels reference)."""
    players = pl.read_parquet(run_dir / "players.parquet")
    head, filled = players.filter(~pl.col("filled")), players.filter(pl.col("filled"))
    _, _, held, extra = eval_dev.logo(head, "lab", np.random.default_rng(0), n_boot=n_boot, extra=filled)
    rows = pl.concat([held, extra], how="diagonal_relaxed") if extra is not None and extra.height else held
    frames = pl.read_parquet(run_dir / "frames.parquet")
    if "filled" not in frames.columns:
        frames = frames.with_columns(pl.lit(False).alias("filled"))
    return Run(rows, frames)


def clip_table(runs: dict[str, Run]) -> pl.DataFrame:
    """Per clip: camera speed of the smoothed camera (median, px/s), robust common-mode step RMS of both
    cameras against the smoothed labels and their ratio, line reprojection p90 and share above 10 px."""
    out = []
    for clip in sorted(runs["none"].players["clip"].unique().to_list()):
        row: dict[str, float | str | int] = {"clip": clip}
        for cam, run in runs.items():
            p = run.players.filter((pl.col("clip") == clip) & ~pl.col("filled"))
            f = run.frames.filter(pl.col("clip") == clip)
            line = f["line_px_out"].cast(pl.Float64).fill_null(np.nan).to_numpy()
            ok = np.isfinite(line)
            row |= {
                "game": int(p["game"][0]),
                f"step_{cam}": eval_dev.step_stats(p)[1],
                f"line_p90_{cam}": float(np.quantile(line[ok], 0.9)) if ok.any() else np.nan,
                f"line_gt10_{cam}": float(np.mean(line[ok] > 10.0)) if ok.any() else np.nan,
                f"speed_{cam}": float(np.nanmedian(f["cam_speed_px_s"].cast(pl.Float64).fill_null(np.nan))),
            }
        out.append(row)
    table = pl.DataFrame(out)
    return table.with_columns((pl.col("step_ptz") / pl.col("step_none")).alias("step_ratio"))


def choose_clip(table: pl.DataFrame) -> str:
    """Among the top third of clips by smoothed-camera speed, the one with the lowest step ratio."""
    fast = table.filter(pl.col("speed_ptz") >= pl.col("speed_ptz").quantile(2 / 3, "linear"))
    return str(fast.sort("step_ratio")["clip"][0])


def choose_start(players: pl.DataFrame, frames: np.ndarray, n: int) -> int:
    """Index of the first cached frame of the ``n``-frame window with the largest robust common-mode step RMS
    of ``players`` (one clip, one camera)."""
    if len(frames) <= n:
        return 0
    head = players.filter(~pl.col("filled"))
    scores = []
    for k in range(len(frames) - n + 1):
        sel = head.filter(pl.col("frame").is_in(frames[k : k + n].tolist()))
        rms = eval_dev.step_stats(sel)[1] if sel.height else np.nan
        scores.append(-np.inf if not np.isfinite(rms) else rms)
    return int(np.argmax(scores))


def crop(runs: dict[str, Run], clip: str, frames: np.ndarray) -> tuple[float, float, float, float]:
    """Pitch window (x0, x1, y0, y1) holding every labelled position of the window, at PANEL_ASPECT."""
    p = runs["none"].players.filter((pl.col("clip") == clip) & pl.col("frame").is_in(frames.tolist()))
    x = (p["x_pred"] - p["ex_lab"]).to_numpy()
    y = (p["y_pred"] - p["ey_lab"]).to_numpy()
    x0, x1 = max(x.min() - MARGIN_M, -57.5), min(x.max() + MARGIN_M, 57.5)
    y0, y1 = max(y.min() - MARGIN_M, -39.0), min(y.max() + MARGIN_M, 39.0)
    w, h = x1 - x0, y1 - y0
    if w / h < PANEL_ASPECT:
        cx, w = (x0 + x1) / 2, h * PANEL_ASPECT
        x0, x1 = cx - w / 2, cx + w / 2
    else:
        cy, h = (y0 + y1) / 2, w / PANEL_ASPECT
        y0, y1 = cy - h / 2, cy + h / 2
    return x0, x1, y0, y1


@dataclass
class View:
    """One camera at one frame: its held-out rows, what its panel shows, and its frame-table entries."""

    rows: pl.DataFrame
    state: str  # SHOWN, ABSTAINS or UNLABELLED
    filled: bool
    line_px: float  # line reprojection of the output camera (NaN without one)


def frame_view(run: Run, clip: str, frame: int) -> View:
    """What one camera's panel shows at ``frame``.

    ABSTAINS when the frame table says the camera published nothing (``ok_out`` false); otherwise SHOWN when
    the run has player rows for the frame, UNLABELLED when it has none: eval_dev.py writes no row for a frame
    without usable position labels, although the camera published positions there.
    """
    rows = run.players.filter((pl.col("clip") == clip) & (pl.col("frame") == frame))
    f = run.frames.filter((pl.col("clip") == clip) & (pl.col("frame") == frame))
    published = not f.height or bool(f["ok_out"][0])
    state = (SHOWN if rows.height else UNLABELLED) if published else ABSTAINS
    line = f["line_px_out"][0] if f.height else None
    return View(
        rows=rows,
        state=state,
        filled=bool(f["filled"][0]) if f.height else False,
        line_px=float("nan") if line is None else float(line),
    )


def frame_stats(rows: pl.DataFrame) -> tuple[int, int, float]:
    """Labelled positions inside their region, players, and median region area (m²) of one camera's frame."""
    det = rows["c00"].to_numpy() * rows["c11"].to_numpy() - rows["c01"].to_numpy() ** 2
    area = np.pi * rows["q_mond"].to_numpy() * np.sqrt(np.clip(det, 0, None))
    return int(rows["hit_mond"].sum()), rows.height, float(np.median(area)) if rows.height else np.nan


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def draw_panel(ax: Axes, cam: str, view: View, window: tuple[float, ...], plain: bool = False) -> None:
    """One camera at one frame: pitch, calibrated 90 % regions, predictions, labels; a grey veil if it
    abstains, a hatched one if its positions cannot be checked. ``plain``: larger marks, a plain title and
    plain words on the veils, no status line."""
    ax.set_facecolor(SURFACE)
    draw_pitch(ax)
    colour = COLOURS[cam]
    rows = view.rows
    dot, label_dot = (6.0, 3.2) if plain else (3.6, 1.8)
    if view.state == SHOWN:
        for r in rows.iter_rows(named=True):
            cov = np.array([[r["c00"], r["c01"]], [r["c01"], r["c11"]]])
            width, height, angle = ellipse_axes(cov, quantile=r["q_mond"])
            centre = (r["x_pred"], r["y_pred"])
            ax.add_patch(
                Ellipse(
                    centre,
                    width,
                    height,
                    angle=angle,
                    facecolor=colour,
                    alpha=0.16,
                    edgecolor="none",
                    zorder=2,
                )
            )
            ax.add_patch(
                Ellipse(centre, width, height, angle=angle, fill=False, edgecolor=colour, lw=0.7, zorder=2)
            )
        ax.plot(rows["x_pred"], rows["y_pred"], "o", ms=dot, color=colour, mec=SURFACE, mew=0.6, zorder=3)
        ax.plot(
            rows["x_pred"] - rows["ex_lab"],
            rows["y_pred"] - rows["ey_lab"],
            "o",
            ms=label_dot,
            color=TEXT,
            zorder=4,
        )
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(VEIL)
    if plain:
        if view.state != SHOWN:
            draw_veil(ax, view.state, PLAIN_VEILS[view.state], fontsize=20)
        ax.set_title(PLAIN_TITLES[cam], loc="left", fontsize=23, color=TEXT, fontweight="bold", pad=7)
        return
    if view.state == SHOWN:
        inside, n, area = frame_stats(rows)
        status = f"{inside} / {n} labels inside their region · median area {area:.1f} m²"
        if view.filled:
            status += " · camera filled"
    else:
        if view.state == ABSTAINS:
            status, word = "abstains: no calibrated camera", "abstains"
        else:
            status, word = "published, no labels to check", "not evaluated"
            if np.isfinite(view.line_px):
                status += f" · line reprojection {view.line_px:.0f} px"
        draw_veil(ax, view.state, word)
    ax.set_title(f"{CAMERAS[cam]}\n{status}", loc="left", fontsize=8.5, color=TEXT, linespacing=1.4)


def draw_veil(ax: Axes, state: str, word: str, fontsize: float = 11) -> None:
    """Grey veil (no position published) or hatched veil (positions published, nothing to check them
    against) over a panel, with ``word`` in its middle."""
    if state == ABSTAINS:
        veil = Rectangle((0, 0), 1, 1, facecolor=VEIL, alpha=0.85, edgecolor="none")
    else:
        veil = Rectangle((0, 0), 1, 1, facecolor=SURFACE, alpha=0.8, hatch="////", edgecolor=PITCH_LINE)
    veil.set(transform=ax.transAxes, zorder=5)
    ax.add_patch(veil)
    ax.text(
        0.5,
        0.5,
        word,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT_SECONDARY,
        zorder=6,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2.0} if state != ABSTAINS else None,
    )


def render_frame(
    fig: Figure,
    runs: dict[str, Run],
    clip: str,
    game: int,
    frame: int,
    window: tuple[float, ...],
    plain: bool = False,
) -> Image.Image:
    fig.clear()
    axes = fig.subplots(1, 2)
    for ax, (cam, run) in zip(axes, runs.items(), strict=True):
        draw_panel(ax, cam, frame_view(run, clip, frame), window, plain=plain)
    if plain:
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=9,
                color=TEXT_SECONDARY,
                mec=SURFACE,
                label="estimated position",
            ),
            Patch(facecolor=TEXT_SECONDARY, alpha=0.16, edgecolor=TEXT_SECONDARY, label="90 % zone"),
            Line2D([], [], marker="o", ls="none", ms=4.5, color=TEXT, label="true position"),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False, fontsize=18,
                   handletextpad=0.3, columnspacing=1.2)  # fmt: skip
        return _image(fig)
    t = (frame - 1) / NATIVE_FPS
    fig.suptitle(
        f"SN-GSR dev clip {clip} (game {game}, quantiles fitted on the five other games) · t = {t:4.1f} s",
        fontsize=9.5,
        color=TEXT,
    )
    handles = [
        Line2D([], [], marker="o", ls="none", color=TEXT_SECONDARY, mec=SURFACE, label="predicted position"),
        Patch(facecolor=TEXT_SECONDARY, alpha=0.16, edgecolor=TEXT_SECONDARY, label="calibrated 90 % region"),
        Line2D([], [], marker="o", ls="none", ms=3, color=TEXT, label="labelled position (SN-GSR)"),
        Patch(facecolor=VEIL, edgecolor=PITCH_LINE, label="no position published"),
        Patch(facecolor=SURFACE, edgecolor=PITCH_LINE, hatch="////", label="not evaluated (no labels)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=5, frameon=False, fontsize=7.5)
    return _image(fig)


def _image(fig: Figure) -> Image.Image:
    canvas = fig.canvas
    assert isinstance(canvas, FigureCanvasAgg)
    canvas.draw()
    return Image.frombuffer("RGBA", canvas.get_width_height(), canvas.buffer_rgba()).convert("RGB")


def save_gif(images: list[Image.Image], path: Path, fps: float, colours: int = 64) -> None:
    """One shared palette (no dithering: flat colours compress well), every frame at 1 / fps."""
    sample = images[:: max(1, len(images) // 6)]
    montage = Image.new("RGB", (images[0].width, images[0].height * len(sample)))
    for k, im in enumerate(sample):
        montage.paste(im, (0, k * im.height))
    palette = montage.quantize(colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    buffer = io.BytesIO()
    frames[0].save(
        buffer, format="GIF", save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0
    )
    atomic_write_bytes(path, buffer.getvalue())


def write_plain(
    images: list[Image.Image], still: Image.Image, gif: Path, mp4: Path, poster: Path, ffmpeg: str | None
) -> None:
    """The plain style's MP4 (H.264, the GIF's 5 fps) and JPEG poster; the GIF is already written."""
    frames = [np.asarray(im)[..., ::-1] for im in images]  # RGB → BGR
    render_demo.write_mp4(frames, mp4, NATIVE_FPS / STEP_FRAMES, crf=24, tune="animation", ffmpeg=ffmpeg)
    buffer = io.BytesIO()
    still.save(buffer, format="JPEG", quality=88, optimize=True)
    atomic_write_bytes(poster, buffer.getvalue())
    for path in (gif, mp4, poster):
        typer.echo(f"{path} ({path.stat().st_size / 1e3:.0f} kB)")


SELECTION = (
    ("speed_ptz", "Image speed of the smoothed camera (px/s, median)", "{:.0f}"),
    ("step_none", "Common-mode step RMS, per-frame camera (m)", "{:.2f}"),
    ("step_ptz", "Common-mode step RMS, smoothed camera (m)", "{:.2f}"),
    ("step_ratio", "Step RMS ratio, smoothed / per-frame", "{:.2f}"),
    ("line_p90_none", "Line reprojection p90, per-frame camera (px)", "{:.1f}"),
    ("line_p90_ptz", "Line reprojection p90, smoothed camera (px)", "{:.1f}"),
    ("line_gt10_none", "Frames above 10 px, per-frame camera", "{:.1%}"),
    ("line_gt10_ptz", "Frames above 10 px, smoothed camera", "{:.1%}"),
)


@app.command()
def main(
    none_run: Path = typer.Option(..., help="eval_dev.py run folder of the per-frame camera."),  # noqa: B008
    ptz_run: Path = typer.Option(..., help="eval_dev.py run folder of the PTZ-smoothed camera."),  # noqa: B008
    clip: str = typer.Option("auto", help="Dev clip id, or auto (see the module docstring)."),
    start: str = typer.Option("auto", help="First second of the window, or auto."),
    seconds: float = typer.Option(15.0, help="Length of the window (s)."),
    gif: Path | None = typer.Option(None, help="Animation path (default by style)."),  # noqa: B008
    png: Path | None = typer.Option(None, help="Still path, report style (PNG)."),  # noqa: B008
    report: Path = typer.Option(DEFAULT_REPORT, help="Markdown report of the choices (report style)."),  # noqa: B008
    width: float = typer.Option(8.8, help="Figure width (inches, 100 dpi)."),
    style: str = typer.Option("report", help="report (the dev figure) or plain (showcase, no numbers)."),
    mp4: Path | None = typer.Option(None, help="H.264 animation path, plain style (default by style)."),  # noqa: B008
    poster: Path | None = typer.Option(None, help="Still path, plain style (JPEG; default by style)."),  # noqa: B008
    ffmpeg: str | None = typer.Option(None, help="ffmpeg executable for the MP4 (default: on PATH)."),
) -> None:
    if style not in ("report", "plain"):
        raise typer.BadParameter(f"--style {style}: expected report or plain")
    plain = style == "plain"
    runs = {"none": load_run(none_run), "ptz": load_run(ptz_run)}
    table = clip_table(runs)
    chosen = choose_clip(table) if clip == "auto" else clip
    if chosen not in table["clip"].to_list():
        raise typer.BadParameter(f"{chosen} is not a clip of {none_run}")
    row = table.filter(pl.col("clip") == chosen).row(0, named=True)
    med = table.select(pl.exclude("clip", "game").median()).row(0, named=True)
    game = int(row["game"])
    frames = np.sort(runs["none"].frames.filter(pl.col("clip") == chosen)["frame"].unique().to_numpy())
    n = max(2, round(seconds * NATIVE_FPS / STEP_FRAMES))
    clip_none = runs["none"].players.filter(pl.col("clip") == chosen)
    k0 = (
        choose_start(clip_none, frames, n)
        if start == "auto"
        else int(np.searchsorted(frames, 1 + float(start) * NATIVE_FPS))
    )
    window_frames = frames[k0 : k0 + n]
    t0, t1 = (window_frames[0] - 1) / NATIVE_FPS, (window_frames[-1] - 1) / NATIVE_FPS
    height = round(width * 0.5 * 50) / 50 if plain else width * 0.47  # plain: even pixel sizes, for H.264
    fig = Figure(figsize=(width, height), dpi=100, facecolor=SURFACE, layout="constrained")
    if plain:
        fig.get_layout_engine().set(rect=(0.005, 0.0, 0.99, 0.975))  # air above the large titles
    FigureCanvasAgg(fig)
    window = crop(runs, chosen, window_frames)
    images = [render_frame(fig, runs, chosen, game, int(fr), window, plain=plain) for fr in window_frames]
    gif = gif or (PLAIN_GIF if plain else DEFAULT_GIF)
    save_gif(images, gif, fps=NATIVE_FPS / STEP_FRAMES)
    # The still: the frame of the window where the per-frame camera's median error is largest.
    err = (
        clip_none.filter(pl.col("frame").is_in(window_frames.tolist()))
        .group_by("frame")
        .agg((pl.col("ex_lab") ** 2 + pl.col("ey_lab") ** 2).sqrt().median().alias("e"))
        .sort("e", descending=True)
    )
    still = int(err["frame"][0]) if err.height else int(window_frames[len(window_frames) // 2])
    still_image = images[int(np.flatnonzero(window_frames == still)[0])]
    if plain:
        write_plain(images, still_image, gif, mp4 or PLAIN_MP4, poster or PLAIN_POSTER, ffmpeg)
        typer.echo(
            f"plain style: clip {chosen} (game {game}), window {t0:.1f}-{t1:.1f} s, {len(images)} frames"
        )
        return
    png = png or DEFAULT_PNG
    buffer = io.BytesIO()
    still_image.save(buffer, format="PNG", optimize=True)
    atomic_write_bytes(png, buffer.getvalue())

    command = (
        "uv run python scripts/render_minimap_demo.py"
        f" --none-run runs/{none_run.name} --ptz-run runs/{ptz_run.name}"
        + (f" --clip {clip}" if clip != "auto" else "")
        + (f" --start {start}" if start != "auto" else "")
        + (f" --seconds {seconds:g}" if seconds != 15.0 else "")
    )
    md = [
        "# Minimap demo: per-frame against PTZ-smoothed camera on one dev clip",
        "",
        f"`{command}` · commit `{git_sha()}` · writes `{gif.name}` ({gif.stat().st_size / 1e3:.0f} kB,"
        f" {len(images)} frames at 5 fps) and `{png.name}` ({png.stat().st_size / 1e3:.0f} kB).",
        "",
        "Exploratory SN-GSR dev data, ground-truth boxes. Each player's 90 % region uses the Mondrian"
        " quantile of its bin fitted on the five games other than the clip's (leave one game out). Only"
        " pitch coordinates are drawn (ADR-0002).",
        "",
        f"**Clip {chosen} (game {game}).** Chosen by rule, to show the smoothing: among the third of the"
        f" {table.height} dev clips whose smoothed camera moves fastest, the one where smoothing cuts the"
        " frame-to-frame jitter of the common-mode error most (robust step RMS against the smoothed labels,"
        " errors of 10 m or more left out). It is not a typical clip:",
        "",
        f"| Quantity | {chosen} | Median over the dev clips |",
        "|---|---|---|",
    ]
    md += [f"| {label} | {fmt.format(row[key])} | {fmt.format(med[key])} |" for key, label, fmt in SELECTION]
    views = {cam: [frame_view(run, chosen, int(fr)) for fr in window_frames] for cam, run in runs.items()}
    md += [
        "",
        f"**Window {t0:.1f}-{t1:.1f} s**: the {seconds:g} s where the per-frame camera's common-mode error"
        " jitters most.",
        "",
        "| Camera | Common-mode step RMS (m) | Labelled positions inside their 90 % region |"
        " Frames without positions (grey) | Frames not evaluated (hatched) |",
        "|---|---|---|---|---|",
    ]
    for cam, run in runs.items():
        sel = run.players.filter((pl.col("clip") == chosen) & pl.col("frame").is_in(window_frames.tolist()))
        step = eval_dev.step_stats(sel.filter(~pl.col("filled")))[1]
        states = [v.state for v in views[cam]]
        md.append(
            f"| {NAMES[cam]} | {step:.2f} | {sel['hit_mond'].mean():.1%} of {sel.height} |"
            f" {states.count(ABSTAINS)} of {len(states)} | {states.count(UNLABELLED)} of {len(states)} |"
        )
    unchecked = []
    for cam in runs:
        items = [
            f"t = {(int(fr) - 1) / NATIVE_FPS:.1f} s, "
            + (f"{v.line_px:.1f} px" if np.isfinite(v.line_px) else "no line annotations")
            for fr, v in zip(window_frames, views[cam], strict=True)
            if v.state == UNLABELLED
        ]
        if items:
            unchecked.append(f"{NAMES[cam]} {'; '.join(items)}")
    if unchecked:
        md += [
            "",
            "Not evaluated: the camera published positions, but SN-GSR has no usable position label on the"
            " frame, so eval_dev.py wrote no player row and nothing checks them. Line reprojection of those"
            f" frames (annotated lines to the projected pitch): {'. '.join(unchecked)}.",
        ]
    stills = []
    for cam, run in runs.items():
        rows = run.players.filter((pl.col("clip") == chosen) & (pl.col("frame") == still))
        inside, n_players, area = frame_stats(rows)
        stills.append(f"{NAMES[cam]} {inside} / {n_players} inside, median area {area:.1f} m²")
    md += [
        "",
        f"**Still** (t = {(still - 1) / NATIVE_FPS:.1f} s, the frame of the window where the per-frame"
        f" camera's median error is largest): {'; '.join(stills)}.",
    ]
    text = "\n".join(md) + "\n"
    atomic_write_text(report, text)
    typer.echo(text)


if __name__ == "__main__":
    app()
