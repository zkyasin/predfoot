"""Synthetic pan-tilt-zoom camera trajectories with exactly known geometry, for the temporal smoother."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

from predfoot.contracts import CalibResult, CalibStatus
from predfoot.geometry import homography as hg
from predfoot.geometry import ptz
from predfoot.geometry.camera import look_at
from predfoot.geometry.pitch import pnlcalib_keypoints

CENTRE = np.array([2.0, -64.0, 20.0])
WIDTH, HEIGHT = 1920, 1080
PLANAR_KP = np.array([(x, y) for x, y, planar in pnlcalib_keypoints().values() if planar])
#: 25 fixed pixels in the lower 55 % of the image, where the players are.
PROBES = np.array(
    [[u, v] for u in np.linspace(80, 1840, 5) for v in np.linspace(0.45 * HEIGHT + 10, 1060, 5)]
)


def ptz_path(n: int = 150, fps: float = 5.0, whip_at_s: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Times (n,) and φ (n, 4) of a smooth broadcast camera with a whip pan (peak about 25°/s)."""
    t = np.arange(n) / fps
    pan = (
        5
        + 20 * np.sin(2 * np.pi * t / 12)
        + 6 * np.sin(2 * np.pi * t / 5 + 1)
        + 15 * np.tanh((t - whip_at_s) / 0.6)
    )
    tilt = 17 + 2 * np.sin(2 * np.pi * t / 9)
    roll = 0.4 + 0.1 * np.sin(2 * np.pi * t / 20)
    logf = np.log(2500) + 0.4 * np.sin(2 * np.pi * t / 15)
    return t, np.column_stack([np.radians(pan), np.radians(tilt), np.radians(roll), logf])


def visible(h: np.ndarray, xy: np.ndarray, margin: float = 5.0) -> np.ndarray:
    """Mask of ground points in front of the camera and inside the image (with a margin)."""
    q = np.column_stack([xy, np.ones(len(xy))]) @ h.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[:, :2] / q[:, 2:3]
    return (
        (q[:, 2] > 0)
        & (uv[:, 0] >= margin)
        & (uv[:, 0] <= WIDTH - 1 - margin)
        & (uv[:, 1] >= margin)
        & (uv[:, 1] <= HEIGHT - 1 - margin)
    )


def _inv_scaled(a: np.ndarray) -> np.ndarray:
    d = 1.0 / np.sqrt(np.diag(a))
    return np.linalg.inv(a * np.outer(d, d)) * np.outer(d, d)


def laplace_cov(h: np.ndarray, xy: np.ndarray, sigma_px: float) -> np.ndarray:
    """Σθ = σ² (JᵀJ)⁻¹ of a homography fitted to exact keypoints ``xy`` with pixel noise σ."""
    j = hg.jacobian_project_theta(h, xy).reshape(-1, 8)
    return sigma_px**2 * _inv_scaled(j.T @ j)


def _sqrt_psd(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (a + a.T))
    return np.asarray(v * np.sqrt(np.clip(w, 0.0, None)))


def gaussian_calibs(
    phi: np.ndarray,
    centre: np.ndarray,
    rng: np.random.Generator,
    sigma_px: float = 2.0,
    scale: float = 1.0,
    outliers: Collection[int] = (),
    failed: Collection[int] = (),
    other_camera: tuple[Collection[int], np.ndarray] | None = None,
) -> list[CalibResult]:
    """Per-frame calibrations whose θ̂ is drawn from N(θ, scale · Σθ).

    Σθ is the Laplace covariance of the visible planar keypoints (fewer than 6: FEW_POINTS). ``outliers`` are
    drawn around φ + (5°, 0, 0, 0), consistent with the PTZ model; ``other_camera = (frames, C2)`` replaces
    those frames by a camera at C2 looking at the ground point under the image centre with f = 2200 px.
    """
    others = set(other_camera[0]) if other_camera is not None else set()
    out: list[CalibResult] = []
    for k, p in enumerate(phi):
        eps = rng.standard_normal(8)  # drawn for every frame, so outliers do not shift the noise of others
        if k in failed:
            out.append(CalibResult(CalibStatus.FEW_POINTS))
            continue
        pk = p + (np.array([np.radians(5.0), 0.0, 0.0, 0.0]) if k in outliers else 0.0)
        h = ptz.homography(pk, centre)
        if k in others and other_camera is not None:
            ground = hg.backproject(ptz.homography(p, centre), [ptz.PP_1080P])[0][0]
            h = look_at(other_camera[1], [ground[0], ground[1], 0.0], 2200.0).H
        xy = PLANAR_KP[visible(h, PLANAR_KP)]
        if len(xy) < 6:
            out.append(CalibResult(CalibStatus.FEW_POINTS, n_inliers=len(xy)))
            continue
        cov = laplace_cov(h, xy, sigma_px)
        theta = hg.theta_from_h(h) + _sqrt_psd(scale * cov) @ eps
        out.append(
            CalibResult(
                CalibStatus.OK,
                H=hg.h_from_theta(theta),
                cov_theta=cov,
                s2=sigma_px**2,
                rms_px=sigma_px,
                n_inliers=len(xy),
                cond=1.0,
                dof=None,
            )
        )
    return out


def cv_process(
    rng: np.random.Generator,
    n: int,
    dt: float,
    q: Sequence[float],
    phi0: Sequence[float] = (0.0, np.radians(17.0), np.radians(0.4), np.log(2500.0)),
    factors: np.ndarray | None = None,
) -> np.ndarray:
    """φ (n, 4) of a constant-velocity process with spectral densities ``q``, starting at rest at ``phi0``.

    ``factors`` (n,) multiplies the process noise of the step into each frame (same random draws).
    """
    qc = np.diag(np.asarray(q, dtype=np.float64))
    f = np.block([[np.eye(4), dt * np.eye(4)], [np.zeros((4, 4)), np.eye(4)]])
    qq = np.block([[dt**3 / 3 * qc, dt**2 / 2 * qc], [dt**2 / 2 * qc, dt * qc]])
    root = np.linalg.cholesky(qq)
    x = np.concatenate([np.asarray(phi0, dtype=np.float64), np.zeros(4)])
    out = np.empty((n, 4))
    for k in range(n):
        out[k] = x[:4]
        scale = 1.0 if factors is None or k + 1 >= n else float(np.sqrt(factors[k + 1]))
        x = f @ x + scale * root @ rng.standard_normal(8)
    return out


def probe_errors(h_est: Sequence[np.ndarray | None], h_true: Sequence[np.ndarray]) -> np.ndarray:
    """Pitch errors (T, 25, 2) of the probe pixels back-projected by each estimate; NaN without a camera."""
    out = np.full((len(h_true), len(PROBES), 2), np.nan)
    for k, (he, ht) in enumerate(zip(h_est, h_true, strict=True)):
        if he is None:
            continue
        xt, ok_t = hg.backproject(ht, PROBES)
        xe, ok_e = hg.backproject(he, PROBES)
        out[k] = np.where((ok_t & ok_e)[:, None], xe - xt, np.nan)
    return out


def common_step_rms(errors: np.ndarray) -> float:
    """RMS of the frame-to-frame change of the common-mode (median over probes) error."""
    with np.errstate(all="ignore"):
        common = np.nanmedian(errors, axis=1)
    return float(np.sqrt(np.nanmean(np.diff(common, axis=0) ** 2)))
