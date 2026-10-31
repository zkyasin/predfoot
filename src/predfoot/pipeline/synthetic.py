"""Synthetic broadcast cameras for tests and the tracer-bullet fixture.

A pinhole camera without distortion, placed in the internal pitch frame (x towards the right-hand goal, y
towards the far touchline, z up) and aimed at a target point. Its ground-plane homography maps pitch points
(x, y, 1) to pixels (u, v, 1), OpenCV convention (pixel centres on integers, so the principal point is
((W-1)/2, (H-1)/2)). The calibration module will provide real cameras; this one only needs to be plausible
and exact.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from predfoot.contracts import FloatArray


def look_at_homography(
    camera_xyz: ArrayLike, target_xyz: ArrayLike, focal_px: float, image_size: tuple[int, int]
) -> FloatArray:
    """Pitch → image homography of a level camera (no roll) at ``camera_xyz`` aimed at ``target_xyz``.

    Returns H normalised so that H[2, 2] = 1, as in :class:`predfoot.contracts.CalibResult`.
    """
    c = np.asarray(camera_xyz, dtype=np.float64)
    fwd = np.asarray(target_xyz, dtype=np.float64) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])  # image +u
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("a vertical optical axis has no defined image right")
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)  # image +v
    rot = np.stack([right, down, fwd])  # world → camera
    width, height = image_size
    k = np.array([[focal_px, 0.0, (width - 1) / 2], [0.0, focal_px, (height - 1) / 2], [0.0, 0.0, 1.0]])
    h = k @ np.column_stack([rot[:, 0], rot[:, 1], -rot @ c])
    return h / h[2, 2]


def project_ground(H: FloatArray, xy: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Project pitch points of shape (N, 2) to pixels; also return the homogeneous w (> 0 in front)."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H, dtype=np.float64).T
    return q[:, :2] / q[:, 2:3], q[:, 2]
