"""Coordinate conventions (ADR-0001).

Internal pitch frame = SkillCorner's: origin at the centre spot, metres, x towards the right-hand goal as seen
from the main camera, y towards the far touchline (the stand opposite the main camera).

SoccerNet (SN-GSR, SN-Calibration) and SoccerTrack v2 use the same x but the opposite y: a player close to the
main camera has y > 0. Checked on 2026-10-30 against SN-GSR clip SNGS-056, the SoccerTrack v2 README and the
``image_corners_projection`` of SkillCorner open-data match 1874553 (top image corners at y = +39).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Linear map from SoccerNet-style pitch coordinates to the internal frame. It is a reflection (det = -1), so
#: ellipse orientations flip and the covariance cross term changes sign.
SN_TO_INTERNAL: FloatArray = np.diag([1.0, -1.0])

#: Half-pixel offset applied when turning SoccerNet normalised coordinates into OpenCV pixel coordinates
#: (pixel centres on integers). Not verified; the possible error is at most 0.5 px.
SN_NORMALISED_OFFSET_PX = 0.0


def sn_to_internal_xy(xy: ArrayLike) -> FloatArray:
    """Convert SoccerNet-style pitch positions of shape (..., 2) to the internal frame."""
    out = np.array(xy, dtype=np.float64, copy=True)
    out[..., 1] *= -1.0
    return out


def internal_to_sn_xy(xy: ArrayLike) -> FloatArray:
    """Inverse of :func:`sn_to_internal_xy` (the map is its own inverse)."""
    return sn_to_internal_xy(xy)


def sn_to_internal_cov(cov: ArrayLike) -> FloatArray:
    """Convert position covariances of shape (..., 2, 2): Σ' = D Σ Dᵀ with D = diag(1, -1)."""
    out = np.array(cov, dtype=np.float64, copy=True)
    out[..., 0, 1] *= -1.0
    out[..., 1, 0] *= -1.0
    return out


def sn_normalised_to_pixels(points: ArrayLike, width: int, height: int) -> FloatArray:
    """Convert SoccerNet normalised image points of shape (..., 2), in [0, 1], to pixel coordinates."""
    p = np.asarray(points, dtype=np.float64)
    return p * np.array([width, height], dtype=np.float64) - SN_NORMALISED_OFFSET_PX
