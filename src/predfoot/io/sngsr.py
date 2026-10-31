"""SoccerNet Game State Reconstruction (SN-GSR-2025): file layout, label parsing and the Parquet index.

Labels v1.3, checked on 2026-10-31 against SNGS-056 (valid):

- ``info``: ``name`` (clip id), ``game_id`` (a string), ``action_class``, ``frame_rate``, ``seq_length``...;
- ``images``: ``image_id``, ``file_name`` (``000001.jpg``), ``width``, ``height``, ``has_labeled_*`` flags;
- ``annotations`` of categories 1-4 and 7 (player, goalkeeper, referee, ball, other): ``track_id``,
  ``attributes`` {role, team, jersey}, ``bbox_image`` {x, y, x_center, y_center, w, h} in pixels, and
  ``bbox_pitch`` / ``bbox_pitch_raw`` (bottom-left, -middle and -right points) in SoccerNet pitch coordinates;
- one annotation of category 5 per frame: ``lines`` = {line class: [{x, y}, ...]} in image coordinates
  normalised to [0, 1].

Tables produced here (ADR-0001), with pixels in the original frame and metres in the internal pitch frame:

- ``clips``: clip_id, split, game_id, action_class, n_frames, width, height, frame_rate, seq_length,
  game_time_start, game_time_stop, visibility, labels_version, n_frames_labeled_{person,pitch,camera};
- ``persons`` (players, goalkeepers, referees, others; never the ball): clip_id, frame, track_id, role, team,
  jersey, bbox_x, bbox_y, bbox_w, bbox_h, foot_u, foot_v, x, y (smoothed ground truth), x_raw, y_raw (raw
  ground truth), truncated;
- ``lines``: clip_id, frame, line_class, is_planar, point_idx, u, v;
- ``frames``: clip_id, frame, has_labeled_person, has_labeled_pitch, has_labeled_camera. Frames without a
  labelled camera have no ground-truth pitch positions (checked on SNGS-035), so evaluations drop them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from predfoot.io.conventions import sn_normalised_to_pixels, sn_to_internal_xy
from predfoot.paths import data_dir

# ------------------------------------------------------------------------------------------------- layout

HF_REPO = "SoccerNet/SN-GSR-2025"
LABELED_SPLITS: tuple[str, ...] = ("train", "valid", "test")
DEV_SPLITS: tuple[str, ...] = ("train", "valid")
#: Locked until the preregistration is frozen (ADR-0002): never tune or report per-clip numbers on it.
LOCKED_SPLITS: tuple[str, ...] = ("test",)
LABELS_NAME = "Labels-GameState.json"
SEQUENCES_INFO_NAME = "sequences_info.json"
N_FRAMES = 750

_CLIP = r"SNGS-\d{3,4}"
CLIP_ID = re.compile(rf"^{_CLIP}$")
LABELS_MEMBER = re.compile(rf"^({_CLIP})/Labels-GameState\.json$")
FRAME_MEMBER = re.compile(rf"^({_CLIP})/img1/(\d{{6}})\.jpg$")


def split_url(split: str) -> str:
    """Download URL of one split's zip on the Hugging Face Hub."""
    if split not in (*LABELED_SPLITS, "challenge"):
        raise ValueError(f"unknown SN-GSR split: {split!r}")
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{split}.zip"


def sngsr_dir(root: Path | None = None) -> Path:
    return (root if root is not None else data_dir()) / "sn-gsr"


def labels_path(split: str, clip_id: str, root: Path | None = None) -> Path:
    return sngsr_dir(root) / split / _checked_clip(clip_id) / LABELS_NAME


def frame_path(split: str, clip_id: str, frame: int, root: Path | None = None) -> Path:
    return sngsr_dir(root) / split / _checked_clip(clip_id) / "img1" / f"{frame:06d}.jpg"


def sequences_info_path(split: str, root: Path | None = None) -> Path:
    return sngsr_dir(root) / split / SEQUENCES_INFO_NAME


def _checked_clip(clip_id: str) -> str:
    if not CLIP_ID.match(clip_id):
        raise ValueError(f"not an SN-GSR clip id: {clip_id!r}")
    return clip_id


def labels_members(names: Iterable[str]) -> dict[str, str]:
    """Map clip id to the archive member holding its labels."""
    out: dict[str, str] = {}
    for name in names:
        match = LABELS_MEMBER.match(name)
        if match:
            out[match.group(1)] = name
    return out


def frame_members(names: Iterable[str], clips: Iterable[str], every: int = 1) -> dict[tuple[str, int], str]:
    """Map (clip id, frame) to archive members for frames 1, 1 + every, 1 + 2 every... of the given clips."""
    if every < 1:
        raise ValueError("every must be >= 1")
    wanted = set(clips)
    out: dict[tuple[str, int], str] = {}
    for name in names:
        match = FRAME_MEMBER.match(name)
        if match and match.group(1) in wanted:
            frame = int(match.group(2))
            if (frame - 1) % every == 0:
                out[(match.group(1), frame)] = name
    return out


def local_clips(root: Path | None = None, splits: Sequence[str] = LABELED_SPLITS) -> dict[str, str]:
    """Clip id -> split for every clip whose labels are on disk."""
    out: dict[str, str] = {}
    for split in splits:
        base = sngsr_dir(root) / split
        if not base.is_dir():
            continue
        for path in sorted(base.glob(f"*/{LABELS_NAME}")):
            if CLIP_ID.match(path.parent.name):
                out[path.parent.name] = split
    return out


# ---------------------------------------------------------------------------------------------- line classes

#: The 26 line classes of categories[4]["lines"].
LINE_CLASSES: tuple[str, ...] = (
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Circle central",
    "Circle left",
    "Circle right",
    "Goal left crossbar",
    "Goal left post left",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
)
#: Goal posts and crossbars are 3D (not on the ground plane): they must never enter a homography.
NON_PLANAR_LINE_CLASSES = frozenset(c for c in LINE_CLASSES if c.startswith("Goal "))
#: Whitelist used for ``is_planar``: an unknown class is treated as non-planar.
PLANAR_LINE_CLASSES = frozenset(LINE_CLASSES) - NON_PLANAR_LINE_CLASSES
#: Planar but curved: fit them as arcs, not straight lines.
ARC_LINE_CLASSES = frozenset(c for c in LINE_CLASSES if c.startswith("Circle "))


def is_planar_line(line_class: str) -> bool:
    """True for the ground-plane line classes (a whitelist, so unknown names are False)."""
    return line_class.strip() in PLANAR_LINE_CLASSES


# --------------------------------------------------------------------------------------------------- parsing

PERSON_CATEGORIES: Mapping[int, str] = {1: "player", 2: "goalkeeper", 3: "referee", 7: "other"}
BALL_CATEGORY = 4
PITCH_CATEGORY = 5
#: A box closer than this to an image border is flagged ``truncated`` (its feet may be cut off).
TRUNCATION_MARGIN_PX = 2.0

CLIPS_SCHEMA: dict[str, Any] = {
    "clip_id": pl.String,
    "split": pl.String,
    "game_id": pl.Int64,
    "action_class": pl.String,
    "n_frames": pl.Int64,
    "width": pl.Int64,
    "height": pl.Int64,
    "frame_rate": pl.Float64,
    "seq_length": pl.Int64,
    "game_time_start": pl.String,
    "game_time_stop": pl.String,
    "visibility": pl.String,
    "labels_version": pl.String,
    "n_frames_labeled_person": pl.Int64,
    "n_frames_labeled_pitch": pl.Int64,
    "n_frames_labeled_camera": pl.Int64,
}
PERSONS_SCHEMA: dict[str, Any] = {
    "clip_id": pl.String,
    "frame": pl.Int32,
    "track_id": pl.Int64,
    "role": pl.String,
    "team": pl.String,
    "jersey": pl.String,
    "bbox_x": pl.Float64,
    "bbox_y": pl.Float64,
    "bbox_w": pl.Float64,
    "bbox_h": pl.Float64,
    "foot_u": pl.Float64,
    "foot_v": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "x_raw": pl.Float64,
    "y_raw": pl.Float64,
    "truncated": pl.Boolean,
}
LINES_SCHEMA: dict[str, Any] = {
    "clip_id": pl.String,
    "frame": pl.Int32,
    "line_class": pl.String,
    "is_planar": pl.Boolean,
    "point_idx": pl.Int32,
    "u": pl.Float64,
    "v": pl.Float64,
}
FRAMES_SCHEMA: dict[str, Any] = {
    "clip_id": pl.String,
    "frame": pl.Int32,
    "has_labeled_person": pl.Boolean,
    "has_labeled_pitch": pl.Boolean,
    "has_labeled_camera": pl.Boolean,
}


@dataclass(frozen=True)
class ClipLabels:
    """The tables of one clip."""

    clip: pl.DataFrame
    persons: pl.DataFrame
    lines: pl.DataFrame
    frames: pl.DataFrame


class LabelsError(ValueError):
    """A labels file does not have the expected structure."""


def _frame_of(file_name: str, n_frames: int) -> int:
    stem = Path(file_name).stem
    if not stem.isdigit():
        raise LabelsError(f"unexpected image file name {file_name!r}")
    frame = int(stem)
    if not 1 <= frame <= n_frames:
        raise LabelsError(f"frame {frame} outside 1..{n_frames}")
    return frame


def _bottom_middle(box: Mapping[str, Any] | None) -> tuple[float, float]:
    if not box or box.get("x_bottom_middle") is None or box.get("y_bottom_middle") is None:
        return (np.nan, np.nan)
    return (float(box["x_bottom_middle"]), float(box["y_bottom_middle"]))


def parse_labels(
    data: Mapping[str, Any],
    *,
    split: str,
    clip_id: str | None = None,
    truncation_margin_px: float = TRUNCATION_MARGIN_PX,
) -> ClipLabels:
    """Turn one parsed ``Labels-GameState.json`` into the ``clips``, ``persons`` and ``lines`` tables.

    Pitch positions go from SoccerNet coordinates to the internal frame (y ↦ -y); line points from normalised
    coordinates to pixels; ``frame`` is the number in the image file name (1-based).
    """
    info = data["info"]
    name = str(info.get("name") or "")
    clip_id = clip_id or name
    if not CLIP_ID.match(clip_id) or (name and name != clip_id):
        raise LabelsError(f"clip id {clip_id!r} does not match info.name {name!r}")
    images = data["images"]
    if not images:
        raise LabelsError(f"{clip_id}: no images")
    sizes = {(int(im["width"]), int(im["height"])) for im in images}
    if len(sizes) != 1:
        raise LabelsError(f"{clip_id}: several image sizes {sorted(sizes)}")
    ((width, height),) = sizes
    seq_length = int(info.get("seq_length") or len(images))
    last_frame = max(seq_length, len(images))
    frame_of = {str(im["image_id"]): _frame_of(im["file_name"], last_frame) for im in images}

    clip = pl.DataFrame(
        {
            "clip_id": [clip_id],
            "split": [split],
            "game_id": [int(info["game_id"])],
            "action_class": [info.get("action_class")],
            "n_frames": [len(images)],
            "width": [width],
            "height": [height],
            "frame_rate": [float(info["frame_rate"]) if info.get("frame_rate") is not None else None],
            "seq_length": [seq_length],
            "game_time_start": [info.get("game_time_start")],
            "game_time_stop": [info.get("game_time_stop")],
            "visibility": [info.get("visibility")],
            "labels_version": [info.get("version")],
            "n_frames_labeled_person": [sum(bool(im.get("has_labeled_person")) for im in images)],
            "n_frames_labeled_pitch": [sum(bool(im.get("has_labeled_pitch")) for im in images)],
            "n_frames_labeled_camera": [sum(bool(im.get("has_labeled_camera")) for im in images)],
        },
        schema=CLIPS_SCHEMA,
    )

    p_frame: list[int] = []
    p_track: list[int | None] = []
    p_role: list[str | None] = []
    p_team: list[str | None] = []
    p_jersey: list[str | None] = []
    p_box: list[tuple[float, float, float, float]] = []
    p_xy: list[tuple[float, float]] = []
    p_xy_raw: list[tuple[float, float]] = []
    l_frame: list[int] = []
    l_class: list[str] = []
    l_idx: list[int] = []
    l_xy: list[tuple[float, float]] = []

    for ann in data["annotations"]:
        category = ann.get("category_id")
        frame = frame_of.get(str(ann.get("image_id")))
        if frame is None:
            raise LabelsError(f"{clip_id}: annotation {ann.get('id')} refers to an unknown image")
        if category == PITCH_CATEGORY:
            for line_class, points in (ann.get("lines") or {}).items():
                for k, point in enumerate(points):
                    l_frame.append(frame)
                    l_class.append(line_class.strip())
                    l_idx.append(k)
                    l_xy.append((float(point["x"]), float(point["y"])))
            continue
        if category not in PERSON_CATEGORIES:
            continue  # ball (4) and camera (6)
        attributes = ann.get("attributes") or {}
        box = ann["bbox_image"]
        jersey = attributes.get("jersey")
        p_frame.append(frame)
        p_track.append(int(ann["track_id"]) if ann.get("track_id") is not None else None)
        p_role.append(attributes.get("role") or PERSON_CATEGORIES[category])
        p_team.append(attributes.get("team"))
        p_jersey.append(str(jersey) if jersey is not None else None)
        p_box.append((float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])))
        p_xy.append(_bottom_middle(ann.get("bbox_pitch")))
        p_xy_raw.append(_bottom_middle(ann.get("bbox_pitch_raw")))

    boxes = np.array(p_box, dtype=np.float64).reshape(-1, 4)
    bx, by, bw, bh = boxes.T
    margin = truncation_margin_px
    truncated = (bx <= margin) | (by <= margin) | (bx + bw >= width - margin) | (by + bh >= height - margin)
    xy = sn_to_internal_xy(np.array(p_xy, dtype=np.float64).reshape(-1, 2))
    xy_raw = sn_to_internal_xy(np.array(p_xy_raw, dtype=np.float64).reshape(-1, 2))
    persons = pl.DataFrame(
        {
            "clip_id": [clip_id] * len(p_frame),
            "frame": p_frame,
            "track_id": p_track,
            "role": p_role,
            "team": p_team,
            "jersey": p_jersey,
            "bbox_x": bx,
            "bbox_y": by,
            "bbox_w": bw,
            "bbox_h": bh,
            "foot_u": bx + bw / 2.0,  # bottom-middle of the box
            "foot_v": by + bh,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "x_raw": xy_raw[:, 0],
            "y_raw": xy_raw[:, 1],
            "truncated": truncated,
        },
        schema=PERSONS_SCHEMA,
    ).with_columns(pl.col("x", "y", "x_raw", "y_raw").fill_nan(None))

    uv = sn_normalised_to_pixels(np.array(l_xy, dtype=np.float64).reshape(-1, 2), width, height)
    lines = pl.DataFrame(
        {
            "clip_id": [clip_id] * len(l_frame),
            "frame": l_frame,
            "line_class": l_class,
            "is_planar": [is_planar_line(c) for c in l_class],
            "point_idx": l_idx,
            "u": uv[:, 0],
            "v": uv[:, 1],
        },
        schema=LINES_SCHEMA,
    )
    frames = pl.DataFrame(
        {
            "clip_id": [clip_id] * len(images),
            "frame": [frame_of[str(im["image_id"])] for im in images],
            "has_labeled_person": [bool(im.get("has_labeled_person")) for im in images],
            "has_labeled_pitch": [bool(im.get("has_labeled_pitch")) for im in images],
            "has_labeled_camera": [bool(im.get("has_labeled_camera")) for im in images],
        },
        schema=FRAMES_SCHEMA,
    )
    return ClipLabels(
        clip=clip,
        persons=persons.sort("frame", "track_id"),
        lines=lines.sort("frame", "line_class", "point_idx"),
        frames=frames.sort("frame"),
    )


def load_labels(path: Path, *, split: str) -> ClipLabels:
    """Parse ``<split>/<clip>/Labels-GameState.json``; the clip id is the parent directory's name."""
    with path.open("rb") as f:
        data = json.load(f)
    return parse_labels(data, split=split, clip_id=path.parent.name)


# ----------------------------------------------------------------------------------------------------- index

INDEX_TABLES = ("clips", "persons", "lines", "frames")


def _load_for_index(args: tuple[str, str]) -> ClipLabels:
    path, split = args
    return load_labels(Path(path), split=split)


def build_index(
    root: Path | None = None,
    *,
    splits: Sequence[str] = LABELED_SPLITS,
    out_dir: Path | None = None,
    workers: int = 4,
) -> dict[str, Path]:
    """Parse every downloaded labels file into Parquet tables under ``index/``.

    ``clips.parquet`` lists the clips of every split. ``persons``, ``lines`` and ``frames`` hold the dev
    splits only; the locked split's tables go to ``index/test/``, so that reading the default tables can never
    touch test labels by accident. Returns the paths written.
    """
    root = root if root is not None else data_dir()
    out_dir = out_dir if out_dir is not None else root / "index"
    jobs = [(str(labels_path(split, clip, root)), split) for clip, split in local_clips(root, splits).items()]
    if not jobs:
        raise FileNotFoundError(f"no {LABELS_NAME} under {sngsr_dir(root)}")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parsed = list(pool.map(_load_for_index, jobs))
    else:
        parsed = [_load_for_index(job) for job in jobs]

    written: dict[str, Path] = {}
    clips = pl.concat([p.clip for p in parsed]).sort("split", "clip_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    written["clips"] = out_dir / "clips.parquet"
    clips.write_parquet(written["clips"])
    groups = {"": [p for p in parsed if p.clip["split"][0] not in LOCKED_SPLITS]}
    groups.update({s: [p for p in parsed if p.clip["split"][0] == s] for s in LOCKED_SPLITS})
    for subdir, items in groups.items():
        if not items:
            continue
        target = out_dir / subdir if subdir else out_dir
        target.mkdir(parents=True, exist_ok=True)
        key = f"{subdir}/" if subdir else ""
        tables = {
            "persons": pl.concat([p.persons for p in items]).sort("clip_id", "frame", "track_id"),
            "lines": pl.concat([p.lines for p in items]).sort("clip_id", "frame", "line_class", "point_idx"),
            "frames": pl.concat([p.frames for p in items]).sort("clip_id", "frame"),
        }
        for name, table in tables.items():
            written[f"{key}{name}"] = target / f"{name}.parquet"
            table.write_parquet(written[f"{key}{name}"])
    return written


def label_noise_floor(persons: pl.DataFrame, by: Sequence[str] = ("clip_id",)) -> pl.DataFrame:
    """Distribution of ``|smoothed − raw|`` ground-truth positions (metres) per group.

    It measures the frame-to-frame jitter of the annotators' camera, not its systematic error, so it is a
    lower bound on the ground-truth error, not an estimate of it.
    """
    dist = ((pl.col("x") - pl.col("x_raw")) ** 2 + (pl.col("y") - pl.col("y_raw")) ** 2).sqrt()
    return (
        persons.filter(pl.all_horizontal(pl.col("x", "y", "x_raw", "y_raw").is_not_null()))
        .with_columns(dist=dist)
        .group_by(list(by))
        .agg(
            n=pl.len(),
            p50=pl.col("dist").quantile(0.5, interpolation="linear"),
            p90=pl.col("dist").quantile(0.9, interpolation="linear"),
            p99=pl.col("dist").quantile(0.99, interpolation="linear"),
            max=pl.col("dist").max(),
        )
        .sort(list(by))
    )


# ------------------------------------------------------------------------------------------- clip selections


class SelectedClip(BaseModel):
    """One dev clip picked for an experiment, with the reason it was picked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clip_id: str = Field(pattern=r"^SNGS-\d{3,4}$")
    split: Literal["train", "valid"]  # never the locked test split
    game_id: int
    action_class: str
    reason: str = Field(min_length=10)


class ClipSelection(BaseModel):
    """A list of dev clips and the frame stride to use (for example ``configs/spike_clips.yaml``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    purpose: str
    every: int = Field(ge=1)
    clips: list[SelectedClip] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_clips(self) -> ClipSelection:
        ids = [c.clip_id for c in self.clips]
        if len(set(ids)) != len(ids):
            raise ValueError("a clip is listed twice")
        return self

    @property
    def clip_ids(self) -> list[str]:
        return [c.clip_id for c in self.clips]


def load_clip_selection(path: Path) -> ClipSelection:
    """Read and validate a clip selection YAML file."""
    with path.open() as f:
        return ClipSelection.model_validate(yaml.safe_load(f))
