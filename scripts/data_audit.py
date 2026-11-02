"""Audit the SN-GSR index and write reports/data_audit.md (aggregate numbers only).

    uv run python scripts/build_index.py
    uv run python scripts/data_audit.py

Dev clips (train + valid) get per-clip statistics. The locked test split only gets its structure (clip counts
and game ids): no label-noise floor and no per-clip number, so that nothing about it can steer development.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from predfoot.io import sngsr
from predfoot.paths import REPO_ROOT, data_dir, reports_dir

#: Game ids per split expected before the audit; the audit reports what it actually finds.
EXPECTED_GAMES = {"train": [4, 6, 9], "valid": [2, 3, 5], "test": [7, 8, 11]}
PITCH_HALF_LENGTH, PITCH_HALF_WIDTH = 52.5, 34.0  # SN-GSR assumes 105 x 68 m
OUTSIDE_MARGIN_M = 5.0
FLOOR_OUTLIER_M = 20.0

app = typer.Typer(add_completion=False)


def git_commit() -> str:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    sha = run("rev-parse", "--short", "HEAD") or "unknown"
    return f"{sha} (uncommitted changes)" if run("status", "--porcelain", "--untracked-files=no") else sha


def _table(df: pl.DataFrame, floats: int = 2) -> str:
    """Markdown table; floats rounded."""
    df = df.with_columns(pl.selectors.float().round(floats))
    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join([header, rule, *rows])


def frame_views(lines: pl.DataFrame) -> pl.DataFrame:
    """Per (clip, frame): which parts of the pitch the annotated lines show."""
    cls = pl.col("line_class")
    return lines.group_by("clip_id", "frame").agg(
        box_left=cls.str.contains(r"^(Big|Small) rect\. left").any(),
        box_right=cls.str.contains(r"^(Big|Small) rect\. right").any(),
        middle=cls.is_in(["Middle line", "Circle central"]).any(),
        n_planar=cls.filter(pl.col("is_planar")).n_unique(),
    )


def dev_clip_table(clips: pl.DataFrame, persons: pl.DataFrame, lines: pl.DataFrame) -> pl.DataFrame:
    """One row per dev clip: identity, crowding, truncation, zoom, view and label-noise floor."""
    per_clip = persons.group_by("clip_id").agg(
        persons_per_frame=pl.len() / pl.col("frame").n_unique(),
        truncated_pct=pl.col("truncated").mean() * 100,
        box_h_median_px=pl.col("bbox_h").median(),
    )
    views = (
        frame_views(lines)
        .group_by("clip_id")
        .agg(
            box_pct=(pl.col("box_left") | pl.col("box_right")).mean() * 100,
            middle_pct=pl.col("middle").mean() * 100,
            side=pl.when(pl.col("box_left").sum() > pl.col("box_right").sum())
            .then(pl.lit("left"))
            .otherwise(pl.lit("right")),
            planar_lines_per_frame=pl.col("n_planar").mean(),
        )
    )
    floor = sngsr.label_noise_floor(persons).rename(
        {"p50": "floor_p50_m", "p90": "floor_p90_m", "p99": "floor_p99_m", "max": "floor_max_m"}
    )
    return (
        clips.select(
            "clip_id",
            "split",
            "game_id",
            "action_class",
            pl.col("n_frames").alias("frames"),
            camera_pct=pl.col("n_frames_labeled_camera") / pl.col("n_frames") * 100,
        )
        .join(per_clip, on="clip_id", how="left")
        .join(views, on="clip_id", how="left")
        .join(floor.drop("n"), on="clip_id", how="left")
        .sort("game_id", "clip_id")
    )


def floor_outliers(persons: pl.DataFrame, threshold_m: float) -> str:
    """Describe the person-frames whose smoothed and raw positions disagree by more than ``threshold_m``."""
    dist = ((pl.col("x") - pl.col("x_raw")) ** 2 + (pl.col("y") - pl.col("y_raw")) ** 2).sqrt()
    per_frame = (
        persons.filter(pl.col("x").is_not_null() & pl.col("x_raw").is_not_null())
        .group_by("clip_id", "frame")
        .agg(n=pl.len(), n_far=(dist > threshold_m).sum())
        .filter(pl.col("n_far") > 0)
    )
    whole = int((per_frame["n_far"] * 2 >= per_frame["n"]).sum())
    return (
        f"- Gaps above {threshold_m:.0f} m: {int(per_frame['n_far'].sum()):,} person-frames in "
        f"{per_frame.height} frames of {per_frame['clip_id'].n_unique()} clips; in {whole} of these frames "
        "at least half of the persons exceed it, i.e. the raw camera of the whole frame failed. The floor "
        "is therefore quoted as quantiles; the maxima only flag such frames."
    )


def outside_pitch(persons: pl.DataFrame) -> pl.DataFrame:
    """Person-frames whose smoothed or raw ground truth lies more than 5 m outside a 105 x 68 m pitch."""
    excess = []
    for x, y in (("x", "y"), ("x_raw", "y_raw")):
        excess.append(
            pl.max_horizontal(pl.col(x).abs() - PITCH_HALF_LENGTH, pl.col(y).abs() - PITCH_HALF_WIDTH, 0.0)
        )
    return persons.with_columns(excess_m=pl.max_horizontal(*excess)).filter(
        pl.col("excess_m") > OUTSIDE_MARGIN_M
    )


def line_class_frequencies(lines: pl.DataFrame, n_frames: int) -> pl.DataFrame:
    instances = lines.group_by("clip_id", "frame", "line_class", "is_planar").agg(n_points=pl.len())
    known = pl.DataFrame({"line_class": list(sngsr.LINE_CLASSES)})
    freq = instances.group_by("line_class", "is_planar").agg(
        frames_pct=pl.len() / n_frames * 100, points_per_instance=pl.col("n_points").mean()
    )
    return (
        known.join(freq, on="line_class", how="full", coalesce=True)
        .with_columns(
            is_planar=pl.col("line_class").str.strip_chars().is_in(list(sngsr.PLANAR_LINE_CLASSES)),
            frames_pct=pl.col("frames_pct").fill_null(0.0),
            known=pl.col("line_class").is_in(list(sngsr.LINE_CLASSES)),
        )
        .sort("frames_pct", descending=True)
    )


def render(index_dir: Path, command: str) -> str:
    clips = pl.read_parquet(index_dir / "clips.parquet")
    persons = pl.read_parquet(index_dir / "persons.parquet")
    lines = pl.read_parquet(index_dir / "lines.parquet")
    dev_clips = clips.filter(pl.col("split").is_in(sngsr.DEV_SPLITS))
    if set(persons["clip_id"].unique()) - set(dev_clips["clip_id"]):
        raise RuntimeError("persons.parquet holds non-dev clips: rebuild the index")

    out = [
        "# SN-GSR data audit",
        "",
        f"- Generated {dt.date.today().isoformat()} by `{command}` at commit `{git_commit()}`.",
        "- Source: SN-GSR-2025 labels (v1.3) downloaded with `scripts/download_data.py sn-gsr-labels`, "
        "indexed with `scripts/build_index.py` (checksums in `MANIFEST.sha256`).",
        "- Aggregate numbers only. The test split is locked (ADR-0002): only its structure is reported.",
        "- Pitch positions are in the internal frame (y towards the far touchline), pixels in 1920 × 1080.",
        "",
        "## Structure per split",
        "",
    ]
    structure = (
        clips.group_by("split")
        .agg(
            clips=pl.len(),
            games=pl.col("game_id").unique().sort(),
            clips_per_game=pl.col("game_id").value_counts(sort=False),
            all_750_frames=(pl.col("n_frames") == sngsr.N_FRAMES).all(),
            image_size=pl.format("{}x{}", "width", "height").unique(),
            labels_version=pl.col("labels_version").unique(),
        )
        .with_columns(
            clips_per_game=pl.col("clips_per_game")
            .list.eval(pl.format("{}: {}", pl.element().struct[0], pl.element().struct[1]))
            .list.sort()
            .list.join(", "),
            games=pl.col("games").cast(pl.List(pl.String)).list.join(", "),
            image_size=pl.col("image_size").list.join(", "),
            labels_version=pl.col("labels_version").list.join(", "),
        )
        .with_columns(
            expected_games=pl.col("split").replace_strict(
                {k: ", ".join(map(str, v)) for k, v in EXPECTED_GAMES.items()}, default=None
            )
        )
        .with_columns(matches_research=pl.col("games") == pl.col("expected_games"))
        .sort(pl.col("split").replace_strict({"train": 0, "valid": 1, "test": 2}, default=3))
    )
    out += [_table(structure), ""]
    games_ok = structure["matches_research"].all()
    out.append(
        "Games per split match the research split (dev = train + valid, test locked)."
        if games_ok
        else "**Games per split differ from the research split: the folds must be rebuilt.**"
    )
    flags = dev_clips.select(
        pl.col("n_frames_labeled_person", "n_frames_labeled_pitch", "n_frames_labeled_camera").sum()
    ).row(0)
    n_dev_frames = int(dev_clips["n_frames"].sum())
    out += [
        "",
        f"Dev frames: {n_dev_frames:,}; with labelled persons {flags[0]:,}, pitch {flags[1]:,}, "
        f"camera {flags[2]:,}. Frames without a labelled camera keep their boxes and pitch lines but have "
        "no ground-truth pitch positions (`x`, `y` null): evaluations must drop them (`frames.parquet` "
        "keeps the flags). They concentrate in a few clips (`camera_pct` below).",
        "",
    ]

    table = dev_clip_table(dev_clips, persons, lines)
    all_floor = sngsr.label_noise_floor(persons.with_columns(all=pl.lit("dev")), by=("all",)).row(
        0, named=True
    )
    out += [
        "## Label-noise floor (dev)",
        "",
        "Distance between the smoothed (`bbox_pitch`) and raw (`bbox_pitch_raw`) ground-truth foot position "
        "of the same box, over persons (ball excluded). It measures the frame-to-frame jitter of the "
        "annotators' camera, not its systematic error: a lower bound on ground-truth error.",
        "",
        f"- All dev person-frames (n = {all_floor['n']:,}): p50 {all_floor['p50']:.3f} m · "
        f"p90 {all_floor['p90']:.3f} m · p99 {all_floor['p99']:.3f} m · max {all_floor['max']:.2f} m.",
        f"- Per-clip p90: min {table['floor_p90_m'].min():.3f} m · "
        f"median {table['floor_p90_m'].median():.3f} m · max {table['floor_p90_m'].max():.3f} m.",
        floor_outliers(persons, FLOOR_OUTLIER_M),
    ]
    per_game = (
        table.group_by("game_id")
        .agg(
            clips=pl.len(),
            floor_p90_median_m=pl.col("floor_p90_m").median(),
            floor_p90_max_m=pl.col("floor_p90_m").max(),
            truncated_pct_median=pl.col("truncated_pct").median(),
        )
        .sort("game_id")
    )
    out += ["", _table(per_game, floats=3), ""]

    out += [
        "## Dev clips",
        "",
        "`box_pct`: frames showing a penalty or goal area; `middle_pct`: frames showing the halfway line or "
        "centre circle; `side`: the goal more often in view; `box_h_median_px`: median person box height "
        "(a zoom proxy); `truncated_pct`: boxes within 2 px of an image border; `camera_pct`: frames with a "
        "labelled camera (hence ground-truth positions).",
        "",
        _table(table.rename({"persons_per_frame": "persons/frame"}), floats=2),
        "",
        f"All dev clips have {sngsr.N_FRAMES} frames: {bool((table['frames'] == sngsr.N_FRAMES).all())}. "
        f"Action classes: "
        + ", ".join(
            f"{a} {n}" for a, n in table["action_class"].value_counts(sort=True).iter_rows() if a is not None
        )
        + ".",
        "",
    ]

    freq = line_class_frequencies(lines, n_dev_frames)
    out += [
        "## Line classes (dev)",
        "",
        "`frames_pct`: share of dev frames where the class is annotated. Non-planar classes (goal posts and "
        "crossbars) never enter a homography; the circles are planar arcs.",
        "",
        _table(
            freq.select("line_class", "is_planar", "frames_pct", "points_per_instance", "known"), floats=1
        ),
        "",
    ]
    unknown = freq.filter(~pl.col("known"))
    if unknown.height:
        out += [f"**Unknown line classes:** {', '.join(unknown['line_class'])}", ""]

    outside = outside_pitch(persons)
    dupes = persons.filter(pl.col("track_id").is_not_null()).group_by("clip_id", "frame", "track_id").len()
    dupes = dupes.filter(pl.col("len") > 1)
    null_pos = persons.filter(pl.col("x").is_null() | pl.col("x_raw").is_null())
    off_image = lines.filter(
        (pl.col("u") < -2) | (pl.col("u") > 1922) | (pl.col("v") < -2) | (pl.col("v") > 1082)
    )
    out += [
        "## Anomalies (dev)",
        "",
        f"- Person-frames more than {OUTSIDE_MARGIN_M:.0f} m outside a 105 × 68 m pitch (smoothed or raw): "
        f"{outside.height:,} in {outside['clip_id'].n_unique()} clip(s)"
        + (f", largest excess {outside['excess_m'].max():.1f} m." if outside.height else "."),
    ]
    if outside.height:
        worst = (
            outside.group_by("clip_id")
            .agg(
                person_frames=pl.len(),
                tracks=pl.col("track_id").n_unique(),
                max_excess_m=pl.col("excess_m").max(),
            )
            .sort("person_frames", descending=True)
            .head(10)
        )
        out += ["", _table(worst, floats=1), ""]
    out += [
        f"- Person-frames without a pitch position: {null_pos.height:,}.",
        f"- Duplicate (clip, frame, track_id): {dupes.height:,}.",
        f"- Line points more than 2 px outside the image: {off_image.height:,}.",
        "- Roles: "
        + ", ".join(f"{r} {n:,}" for r, n in persons["role"].value_counts(sort=True).iter_rows())
        + ".",
        "",
    ]
    return "\n".join(out)


@app.command()
def main(
    out: Annotated[Path | None, typer.Option(help="Report path (default reports/data_audit.md).")] = None,
) -> None:
    """Write the audit of PREDFOOT_DATA_DIR/index to reports/data_audit.md."""
    text = render(data_dir() / "index", "uv run python scripts/data_audit.py")
    path = out or reports_dir() / "data_audit.md"
    path.write_text(text)
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
