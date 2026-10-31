"""Data contracts shared by every module (ADR-0001).

Image quantities are in pixels of the original frame (OpenCV convention), pitch quantities in metres in the
internal frame (see :mod:`predfoot.io.conventions`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class CalibStatus(StrEnum):
    """Outcome of a per-frame calibration."""

    OK = "ok"
    FEW_POINTS = "few_points"
    DEGENERATE = "degenerate"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class KeypointObs:
    """Pitch keypoints detected in one frame."""

    ids: IntArray  # (K,) template keypoint ids
    uv: FloatArray  # (K, 2) pixels
    conf: FloatArray  # (K,) in [0, 1]


@dataclass(frozen=True)
class LinePointObs:
    """Points sampled on detected pitch lines in one frame."""

    line_ids: IntArray  # (M,) template line ids
    uv: FloatArray  # (M, 2) pixels
    conf: FloatArray  # (M,) in [0, 1]


@dataclass(frozen=True)
class CalibResult:
    """Per-frame calibration.

    ``H`` maps pitch points (x, y, 1) to image points (u, v, 1) up to scale, normalised so that H[2, 2] = 1.
    The parameter vector θ is the first 8 entries of ``H`` in row-major order; ``cov_theta`` is its 8×8
    covariance. H[2, 2] is the image-plane depth of the centre spot, which broadcast cameras always see in
    front of them, so the normalisation is safe.
    """

    status: CalibStatus
    H: FloatArray | None = None  # (3, 3)
    cov_theta: FloatArray | None = None  # (8, 8)
    s2: float | None = None  # residual variance, px²
    rms_px: float | None = None
    n_inliers: int = 0
    cond: float | None = None  # condition number of JᵀWJ
    dof: float | None = None  # degrees of freedom of the noise scale; regions use F quantiles, not χ²

    @property
    def ok(self) -> bool:
        return self.status is CalibStatus.OK


@dataclass(frozen=True)
class FootObs:
    """Ground contact point of one person in one frame: the bottom centre of the box."""

    track_id: int
    uv: FloatArray  # (2,) pixels
    cov_uv: FloatArray | None = None  # (2, 2) px²; None means exact (ground-truth boxes)
    bbox_xywh: FloatArray | None = None  # (4,) pixels
    truncated: bool = False  # the box touches the image border, so the feet may be cut off


@dataclass(frozen=True)
class Localization:
    """Pitch position of one person in one frame, with its calibrated uncertainty."""

    track_id: int
    xy: FloatArray | None  # (2,) metres; None when the frame abstains
    cov_xy: FloatArray | None = None  # (2, 2) m², calibrated
    p90_radius_m: float | None = None
    is_detected: bool = True
