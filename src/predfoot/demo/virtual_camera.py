"""Virtual broadcast cameras inside a fixed panoramic video (SoccerTrack v2 demo).

The rectified panorama is a pinhole view of the pitch from one projection centre C: a pitch point X lands on
the rectified pixel Knew R (X − C). A virtual camera that turns and zooms about the same centre is a PTZ
camera of :mod:`predfoot.geometry.ptz` (square pixels, principal point at the image centre, no roll), so its
pitch → image homography is known by construction:

    H_virt = K_virt R_virt M(C) = (K_virt R_virt Rᵀ Knew⁻¹) H_pano.

The rotation homography in brackets maps every viewing ray, not only those that hit the pitch, so a virtual
view of the whole scene (players included) is rendered from the original video frame by a single ``cv2.remap``
whose table composes virtual pixel → rectified pixel (the inverse rotation homography) → video pixel (the
dataset's remap tables, sampled bilinearly).

The camera is exact for the *model*, the released fisheye calibration. How far the real pitch lies from that
model is a property of the dataset (measured in ``reports/soccertrack_v2.md``); the 65 annotated keypoints,
mapped into a view with :func:`video_to_view`, are true correspondences of that view whatever the model error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d

from predfoot.contracts import FloatArray
from predfoot.geometry import homography as hg
from predfoot.geometry import ptz
from predfoot.io.soccertrack import FisheyeCalibration

BROADCAST_SIZE = (1920, 1080)
_OUTSIDE = -1.0e5  # map value that cv2.remap treats as outside the source image


def _centre_matrix(centre: ArrayLike) -> FloatArray:
    c = np.asarray(centre, dtype=np.float64).reshape(3)
    return np.array([[1.0, 0.0, -c[0]], [0.0, 1.0, -c[1]], [0.0, 0.0, -c[2]]])


def _apply(H: FloatArray, uv: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Homography on points (n, 2), with a mask of positive homogeneous scale (ray in front)."""
    q = np.column_stack([uv, np.ones(len(uv))]) @ H.T
    front = q[:, 2] > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        out = q[:, :2] / q[:, 2:3]
    return np.where(front[:, None], out, np.nan), front


@dataclass(frozen=True)
class PanoramaCamera:
    """The released camera of a SoccerTrack v2 match and its remap tables (rectified → video pixels)."""

    calib: FisheyeCalibration
    map_x: NDArray[np.float32]
    map_y: NDArray[np.float32]

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the rectified image (= the video size in SoccerTrack v2)."""
        return int(self.map_x.shape[1]), int(self.map_x.shape[0])

    def homography(self) -> FloatArray:
        """Internal pitch points → rectified pixels, H = Knew R M(C), normalised (H[2, 2] = 1)."""
        return hg.normalise(self.calib.Knew @ self.calib.R @ _centre_matrix(self.calib.C))

    def video_to_rectified(self, uv: ArrayLike) -> FloatArray:
        """Video pixels (n, 2) → rectified pixels, through the fisheye model (inverse of the remap tables)."""
        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        if not len(pts):
            return np.zeros((0, 2))
        c = self.calib
        out = cv2.fisheye.undistortPoints(pts, c.K, c.D.reshape(4, 1), R=np.eye(3), P=c.Knew)
        return np.asarray(out, dtype=np.float64).reshape(-1, 2)

    def rectified_to_video(self, uv: ArrayLike) -> FloatArray:
        """Rectified pixels (n, 2) → video pixels, by bilinear sampling of the remap tables (NaN outside)."""
        pts = np.asarray(uv, dtype=np.float32).reshape(1, -1, 2)
        if not pts.shape[1]:
            return np.zeros((0, 2))
        px, py = np.ascontiguousarray(pts[..., 0]), np.ascontiguousarray(pts[..., 1])
        sample = [
            cv2.remap(m, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            for m in (self.map_x, self.map_y)
        ]
        return np.column_stack([s.reshape(-1) for s in sample]).astype(np.float64)

    def project_video(self, xy: ArrayLike) -> FloatArray:
        """Internal pitch points (n, 2) → video pixels under the released model."""
        rect = hg.project(self.homography(), np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        return self.rectified_to_video(rect)


@dataclass(frozen=True)
class VirtualView:
    """A PTZ camera φ = (pan, tilt, roll, log f) at the panorama's centre (ptz conventions)."""

    phi: FloatArray
    size: tuple[int, int] = BROADCAST_SIZE

    @property
    def pp(self) -> tuple[float, float]:
        return ((self.size[0] - 1) / 2, (self.size[1] - 1) / 2)

    @property
    def K(self) -> FloatArray:
        f = float(np.exp(self.phi[3]))
        return np.array([[f, 0.0, self.pp[0]], [0.0, f, self.pp[1]], [0.0, 0.0, 1.0]])

    @property
    def R(self) -> FloatArray:
        return ptz.rotation(float(self.phi[0]), float(self.phi[1]), float(self.phi[2]))

    def homography(self, centre: ArrayLike) -> FloatArray:
        """Internal pitch points → pixels of this view (H[2, 2] = 1): the reference camera."""
        return ptz.homography(self.phi, centre, self.pp)


def rectified_from_view(view: VirtualView, pano: PanoramaCamera) -> FloatArray:
    """Rotation homography from virtual pixels to rectified pixels: Knew R_pano R_virtᵀ K_virt⁻¹."""
    c = pano.calib
    return np.asarray(c.Knew @ c.R @ view.R.T @ np.linalg.inv(view.K), dtype=np.float64)


def view_homography(view: VirtualView, pano: PanoramaCamera) -> FloatArray:
    """H_virt = (K_virt R_virt Rᵀ Knew⁻¹) H_pano, normalised; equal to ``view.homography(C)``."""
    return hg.normalise(np.linalg.inv(rectified_from_view(view, pano)) @ pano.homography())


def remap_tables(view: VirtualView, pano: PanoramaCamera) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Tables (height, width) from virtual pixels to video pixels, for one ``cv2.remap`` of a video frame."""
    w, h = view.size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ rectified_from_view(view, pano).T
    front = rays[..., 2] > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        rx = np.where(front, rays[..., 0] / rays[..., 2], _OUTSIDE).astype(np.float32)
        ry = np.where(front, rays[..., 1] / rays[..., 2], _OUTSIDE).astype(np.float32)
    out = [
        cv2.remap(m, rx, ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=_OUTSIDE)
        for m in (pano.map_x, pano.map_y)
    ]
    return np.asarray(out[0], dtype=np.float32), np.asarray(out[1], dtype=np.float32)


def render(
    frame: NDArray[np.uint8], tables: tuple[NDArray[np.float32], NDArray[np.float32]]
) -> NDArray[np.uint8]:
    """The virtual view of a video frame (black where the panorama has no pixels)."""
    out = cv2.remap(
        frame, tables[0], tables[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return np.asarray(out, dtype=np.uint8)


def video_to_view(uv: ArrayLike, view: VirtualView, pano: PanoramaCamera) -> FloatArray:
    """Video pixels (n, 2) → pixels of the view (NaN behind it): where the render puts that content."""
    rect = pano.video_to_rectified(uv)
    out, _ = _apply(np.linalg.inv(rectified_from_view(view, pano)), rect)
    return out


# --- following the action ----------------------------------------------------------------------------------


def aim(target_xy: ArrayLike, centre: ArrayLike) -> tuple[float, float]:
    """(pan, tilt) of a roll-free PTZ camera at ``centre`` aimed at the pitch point ``target_xy``."""
    t = np.asarray(target_xy, dtype=np.float64).reshape(2)
    c = np.asarray(centre, dtype=np.float64).reshape(3)
    dx, dy, dz = t[0] - c[0], t[1] - c[1], -c[2]
    return float(np.arctan2(dx, dy)), float(np.arctan2(-dz, np.hypot(dx, dy)))


@dataclass(frozen=True)
class FollowConfig:
    """How the virtual camera frames the players."""

    size: tuple[int, int] = BROADCAST_SIZE
    smooth_s: float = 1.5  # Gaussian σ applied over time to the aim point and to log f, seconds
    spread_quantiles: tuple[float, float] = (0.1, 0.9)  # robust horizontal extent of the players
    fill: float = 0.75  # fraction of the half-width the extent may use
    hfov_deg: tuple[float, float] = (22.0, 55.0)  # horizontal field of view range (zoomed in, wide)
    aim_offset_m: tuple[float, float] = (0.0, 0.0)  # added to the players' centroid


def follow_action(
    feet_xy: Sequence[FloatArray], centre: ArrayLike, fps: float, config: FollowConfig | None = None
) -> FloatArray:
    """PTZ parameters (n, 4) of a virtual camera that follows the players, one row per frame.

    ``feet_xy[k]`` holds the internal pitch positions (m, 2) of the players of frame k. The camera aims at
    their centroid and zooms so that their robust horizontal extent fills ``fill`` of the half-width, within
    the field-of-view range; aim point and log focal length are smoothed with a Gaussian of ``smooth_s``
    seconds. Frames without players reuse the nearest frame that has some.
    """
    cfg = config or FollowConfig()
    c = np.asarray(centre, dtype=np.float64).reshape(3)
    n = len(feet_xy)
    if n == 0:
        return np.zeros((0, 4))
    has = np.array([len(f) > 0 for f in feet_xy])
    if not has.any():
        raise ValueError("no players in any frame: nothing to follow")
    target = np.full((n, 2), np.nan)
    for k in range(n):
        if has[k]:
            target[k] = np.asarray(feet_xy[k], dtype=np.float64).reshape(-1, 2).mean(axis=0)
    idx = np.arange(n)
    for d in range(2):
        target[:, d] = np.interp(idx, idx[has], target[has, d])
    sigma = cfg.smooth_s * fps
    if sigma > 0:
        target = gaussian_filter1d(target, sigma, axis=0, mode="nearest")
    target = target + np.asarray(cfg.aim_offset_m, dtype=np.float64)
    angles = np.array([aim(t, c) for t in target])
    half_w = cfg.size[0] / 2
    lo, hi = np.radians(cfg.hfov_deg)
    log_f = np.full(n, np.nan)
    for k in range(n):
        if not has[k]:
            continue
        rot = ptz.rotation(angles[k, 0], angles[k, 1], 0.0)
        pts = np.asarray(feet_xy[k], dtype=np.float64).reshape(-1, 2)
        cam = (np.column_stack([pts, np.zeros(len(pts))]) - c) @ rot.T
        ang = np.arctan2(cam[:, 0], cam[:, 2])
        q_lo, q_hi = np.quantile(ang, cfg.spread_quantiles)
        hfov = float(np.clip(2 * max(abs(q_lo), abs(q_hi)) / cfg.fill, lo, hi))
        log_f[k] = np.log(half_w / np.tan(hfov / 2))
    log_f = np.interp(idx, idx[has], log_f[has])
    if sigma > 0:
        log_f = gaussian_filter1d(log_f, sigma, mode="nearest")
    return np.column_stack([angles, np.zeros(n), log_f])
