"""Pinhole cameras for broadcast views, used to simulate frames with exactly known geometry.

World frame = internal pitch frame with z up. Camera frame = OpenCV (x right, y down, z forward).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from predfoot.geometry.homography import normalise

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Camera:
    K: FloatArray  # (3, 3) intrinsics
    R: FloatArray  # (3, 3) world → camera rotation
    C: FloatArray  # (3,) camera centre in the world
    width: int = 1920
    height: int = 1080

    @property
    def H(self) -> FloatArray:
        """Ground-plane homography pitch (x, y, 1) → image, normalised with H[2, 2] = 1."""
        t = -self.R @ self.C
        return normalise(self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], t]))

    def project(self, xyz: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """World points (..., 3) → pixels (..., 2) and depths (...,)."""
        p = (np.asarray(xyz, dtype=np.float64) - self.C) @ self.R.T
        depth = p[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (p @ self.K.T)[..., :2] / depth[..., None]
        return uv, depth

    def in_image(self, uv: ArrayLike, depth: ArrayLike, margin: float = 0.0) -> NDArray[np.bool_]:
        uv = np.asarray(uv)
        return (
            (np.asarray(depth) > 0)
            & (uv[..., 0] >= margin)
            & (uv[..., 0] <= self.width - 1 - margin)
            & (uv[..., 1] >= margin)
            & (uv[..., 1] <= self.height - 1 - margin)
        )


def look_at(
    centre: ArrayLike,
    target: ArrayLike,
    focal_px: float,
    roll_deg: float = 0.0,
    width: int = 1920,
    height: int = 1080,
) -> Camera:
    """Camera at ``centre`` looking at ``target`` (world points); square pixels, centred principal point."""
    c = np.asarray(centre, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - c
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    if roll_deg:
        a = np.deg2rad(roll_deg)
        roll = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
        R = roll @ R
    K = np.array([[focal_px, 0, (width - 1) / 2], [0, focal_px, (height - 1) / 2], [0, 0, 1.0]])
    return Camera(K=K, R=R, C=c, width=width, height=height)


def random_broadcast_camera(rng: np.random.Generator, width: int = 1920, height: int = 1080) -> Camera:
    """A plausible main broadcast camera: behind the near touchline, elevated, aimed at a point of the pitch.

    Ranges: centre x ∈ [−8, 8] m, 15–45 m behind the near touchline, height 12–30 m; target anywhere on the
    pitch; horizontal field of view 12–60° (from tight to wide shots); roll within ±1.5°.
    """
    centre = np.array([rng.uniform(-8, 8), -34.0 - rng.uniform(15, 45), rng.uniform(12, 30)])
    target = np.array([rng.uniform(-50, 50), rng.uniform(-30, 30), 0.0])
    hfov = np.deg2rad(rng.uniform(12, 60))
    focal = (width / 2) / np.tan(hfov / 2)
    return look_at(centre, target, focal, roll_deg=rng.uniform(-1.5, 1.5), width=width, height=height)
