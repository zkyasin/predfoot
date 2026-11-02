"""Input of ``predfoot run``: a small JSON file describing a clip (format ``predfoot-clip/1``).

It describes a clip without its frames: per native frame, the pitch → image
homography (or null when calibration failed) and the foot points of the visible people, in pixels of the
original image (OpenCV convention). Example::

    {"format": "predfoot-clip/1", "clip_id": "synthetic-001", "fps": 25, "image_size": [1920, 1080],
     "pitch": {"length": 105, "width": 68}, "teams": {"1": "home", "12": "away"},
     "frames": [{"frame": 0, "homography": [[...], [...], [...]],
                 "feet": [{"track_id": 1, "u": 812.4, "v": 640.2}]}, ...]}

Frames must be listed in order, numbered 0 … n-1 (native frame indices). ``teams`` is optional.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CLIP_FORMAT = "predfoot-clip/1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class FootPoint(_Strict):
    """Bottom centre of one person's box, in pixels."""

    track_id: int = Field(ge=0)
    u: float
    v: float


class ClipFrame(_Strict):
    """One native frame: its homography (pitch → image, None if calibration failed) and foot points."""

    frame: int = Field(ge=0)
    homography: list[list[float]] | None = None
    feet: list[FootPoint] = Field(default_factory=list)

    @field_validator("homography")
    @classmethod
    def _three_by_three(cls, h: list[list[float]] | None) -> list[list[float]] | None:
        if h is not None and (len(h) != 3 or any(len(row) != 3 for row in h)):
            raise ValueError("homography must be a 3×3 matrix")
        return h

    @model_validator(mode="after")
    def _unique_tracks(self) -> ClipFrame:
        ids = [p.track_id for p in self.feet]
        if len(ids) != len(set(ids)):
            raise ValueError(f"frame {self.frame}: a track id appears twice")
        return self


class Pitch(_Strict):
    length: float = Field(default=105.0, gt=0)
    width: float = Field(default=68.0, gt=0)


class ClipInput(_Strict):
    """A whole clip."""

    format: Literal["predfoot-clip/1"]
    clip_id: str = Field(min_length=1)
    fps: float = Field(gt=0, description="Native frame rate.")
    image_size: tuple[int, int] = Field(description="(width, height) in pixels.")
    pitch: Pitch = Pitch()
    teams: dict[int, Literal["home", "away"]] = Field(default_factory=dict)
    frames: list[ClipFrame] = Field(min_length=1)

    @field_validator("image_size")
    @classmethod
    def _positive_size(cls, size: tuple[int, int]) -> tuple[int, int]:
        if min(size) <= 0:
            raise ValueError("image width and height must be positive")
        return size

    @field_validator("fps")
    @classmethod
    def _finite_fps(cls, fps: float) -> float:
        if not math.isfinite(fps):
            raise ValueError("fps must be finite")
        return fps

    @model_validator(mode="after")
    def _contiguous(self) -> ClipInput:
        for i, frame in enumerate(self.frames):
            if frame.frame != i:
                raise ValueError(
                    f"frames must be numbered 0 … n-1 in order; position {i} has frame {frame.frame}"
                )
        return self


def load_clip(path: Path) -> ClipInput:
    """Read and validate a clip file (raises ``pydantic.ValidationError`` with every problem found)."""
    return ClipInput.model_validate_json(Path(path).read_bytes())
