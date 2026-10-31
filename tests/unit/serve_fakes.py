"""Synthetic camera, calibration covariance and keypoint detector for the serving tests (no torch)."""

from __future__ import annotations

import numpy as np

from predfoot.contracts import KeypointObs
from predfoot.geometry import homography as hg
from predfoot.geometry.camera import Camera, look_at
from predfoot.geometry.pitch import pnlcalib_keypoints
from predfoot.keypoints.pnlcalib import LineEnds


def broadcast_camera(width: int = 1920, height: int = 1080) -> Camera:
    """A wide main-camera view of the right half, the same shot at any resolution."""
    return look_at([0.0, -55.0, 20.0], [10.0, 0.0, 0.0], 1800.0 * width / 1920, width=width, height=height)


def visible_keypoints(cam: Camera, margin: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ids, pitch positions and exact pixels of the planar template keypoints inside the image."""
    kps = pnlcalib_keypoints()
    ids = np.array([k for k, (_, _, planar) in kps.items() if planar])
    xy = np.array([kps[k][:2] for k in ids])
    uv, depth = cam.project(np.column_stack([xy, np.zeros(len(xy))]))
    keep = cam.in_image(uv, depth, margin=margin)
    return ids[keep], xy[keep], uv[keep]


def laplace_covariance(cam: Camera, sigma_px: float = 1.0) -> np.ndarray:
    """Σθ = σ² (JᵀJ)⁻¹ of a calibration on the visible template keypoints, at the true camera."""
    _, xy, _ = visible_keypoints(cam)
    j = hg.jacobian_project_theta(cam.H, xy).reshape(-1, hg.N_PARAMS)
    return sigma_px**2 * np.linalg.inv(j.T @ j)


class SyntheticDetector:
    """Stands in for PnLCalibNetworks: the template keypoints that ``camera`` sees, with pixel noise."""

    version = "synthetic@test"

    def __init__(self, camera: Camera, sigma_px: float = 0.5, seed: int = 0) -> None:
        self.camera = camera
        self.sigma_px = sigma_px
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def detect(self, frames_rgb: np.ndarray) -> list[tuple[KeypointObs, LineEnds]]:
        self.calls += 1
        out = []
        for _ in frames_rgb:
            ids, _, uv = visible_keypoints(self.camera)
            noisy = uv + self.rng.normal(scale=self.sigma_px, size=uv.shape)
            kps = KeypointObs(ids=ids.astype(np.int64), uv=noisy, conf=np.full(len(ids), 0.9))
            ends = LineEnds(
                line_ids=np.zeros(0, dtype=np.int64), uv=np.zeros((0, 2, 2)), conf=np.zeros((0, 2))
            )
            out.append((kps, ends))
        return out
