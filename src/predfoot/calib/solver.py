"""Per-frame calibration: RANSAC on pitch ↔ image correspondences, robust Levenberg-Marquardt refinement,
and a Laplace approximation of the uncertainty of the 8 homography parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from predfoot.contracts import CalibResult, CalibStatus
from predfoot.geometry import homography as hg

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SolverConfig:
    """Calibration settings. Pixel values refer to a 1920 × 1080 frame."""

    ransac_threshold_px: float = 8.0
    ransac_iterations: int = 300
    min_inliers: int = 5  # 4 fit H exactly; 5 leave residual degrees of freedom to estimate the noise
    loss_scale_px: float = 3.0  # soft-L1 scale: residuals beyond ~3 px are progressively down-weighted
    sigma0_px: float = 2.0  # prior keypoint noise, used to shrink the per-frame noise estimate
    prior_dof: float = 10.0  # strength of that prior, in residual degrees of freedom
    max_scaled_condition: float = 1e10  # above this, the configuration is treated as degenerate


def _robust_weights(residuals: FloatArray, scale: float) -> FloatArray:
    """IRLS weights ρ'(z) of the soft-L1 loss ρ(z) = 2(√(1+z) − 1), with z = (r / scale)²."""
    return 1.0 / np.sqrt(1.0 + (residuals / scale) ** 2)


def _non_degenerate(pts: FloatArray, min_area: float) -> bool:
    """No three of the four points are (nearly) collinear."""
    for i in range(4):
        a, b, c = (pts[j] for j in range(4) if j != i)
        if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 2 * min_area:
            return False
    return True


def ransac(
    xy: ArrayLike,
    uv: ArrayLike,
    rng: np.random.Generator,
    config: SolverConfig | None = None,
) -> tuple[FloatArray | None, BoolArray]:
    """Best homography from minimal samples of 4 correspondences, refitted on its inliers by DLT."""
    cfg = config or SolverConfig()
    xy, uv = np.asarray(xy, dtype=np.float64), np.asarray(uv, dtype=np.float64)
    n = len(xy)
    best_h, best_inliers, best_cost = None, np.zeros(n, dtype=bool), np.inf
    if n < 4:
        return None, best_inliers
    thr2 = cfg.ransac_threshold_px**2
    for _ in range(cfg.ransac_iterations):
        idx = rng.choice(n, size=4, replace=False)
        if not (_non_degenerate(xy[idx], 1.0) and _non_degenerate(uv[idx], 25.0)):
            continue
        try:
            h = hg.dlt(xy[idx], uv[idx])
        except np.linalg.LinAlgError:
            continue
        err2 = np.sum((hg.project(h, xy) - uv) ** 2, axis=1)
        err2 = np.where(np.isfinite(err2), err2, np.inf)
        inliers = err2 < thr2
        cost = np.minimum(err2, thr2).sum()  # MSAC: rank hypotheses by truncated squared error
        if inliers.sum() >= 4 and cost < best_cost:
            best_h, best_inliers, best_cost = h, inliers, cost
        if n <= 4:
            break
    if best_h is None:
        return None, best_inliers
    return hg.dlt(xy[best_inliers], uv[best_inliers]), best_inliers


@dataclass(frozen=True)
class LineObservation:
    """Image points lying on a planar template segment from ``a`` to ``b`` (pitch metres).

    Typically the two visible ends of a detected pitch line: their position along the line is unknown, so
    each point only constrains its distance to the projected line (one residual per point).
    """

    a: tuple[float, float]
    b: tuple[float, float]
    uv: FloatArray  # (k, 2) pixels
    weight: float = 1.0


def _projected_line(h: FloatArray, line: LineObservation) -> FloatArray:
    """Image line through the projections of the segment ends, normalised so that a² + b² = 1."""
    pa = h @ np.array([line.a[0], line.a[1], 1.0])
    pb = h @ np.array([line.b[0], line.b[1], 1.0])
    ln = np.cross(pa, pb)
    return ln / np.hypot(ln[0], ln[1])


def _residuals(
    theta: FloatArray, xy: FloatArray, uv: FloatArray, w: FloatArray, lines: list[LineObservation]
) -> FloatArray:
    """Weighted residuals: 2 per point correspondence, then 1 per point on a line (pixels)."""
    h = hg.h_from_theta(theta)
    parts = [(hg.project(h, xy) - uv).reshape(-1) * np.repeat(np.sqrt(w), 2)]
    for line in lines:
        parts.append(
            (np.c_[line.uv, np.ones(len(line.uv))] @ _projected_line(h, line)) * np.sqrt(line.weight)
        )
    return np.concatenate(parts)


def _jacobian(
    theta: FloatArray, xy: FloatArray, uv: FloatArray, w: FloatArray, lines: list[LineObservation]
) -> FloatArray:
    """Analytic Jacobian for the points; central differences for the line residuals."""
    h = hg.h_from_theta(theta)
    jp = hg.jacobian_project_theta(h, xy).reshape(-1, hg.N_PARAMS) * np.repeat(np.sqrt(w), 2)[:, None]
    if not lines:
        return jp
    n_pts = 2 * len(xy)
    cols = []
    for k in range(hg.N_PARAMS):
        step = 1e-6 * max(abs(theta[k]), 1e-6)
        tp, tm = theta.copy(), theta.copy()
        tp[k] += step
        tm[k] -= step
        cols.append(
            (_residuals(tp, xy, uv, w, lines)[n_pts:] - _residuals(tm, xy, uv, w, lines)[n_pts:]) / (2 * step)
        )
    return np.vstack([jp, np.stack(cols, axis=1)])


def refine(
    H0: ArrayLike,
    xy: ArrayLike,
    uv: ArrayLike,
    weights: ArrayLike | None = None,
    config: SolverConfig | None = None,
    lines: list[LineObservation] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Robust (soft-L1) Levenberg-Marquardt on point and line residuals.

    Returns H and the (weighted) residual vector at the optimum.
    """
    cfg = config or SolverConfig()
    xy, uv = np.asarray(xy, dtype=np.float64), np.asarray(uv, dtype=np.float64)
    w = np.ones(len(xy)) if weights is None else np.asarray(weights, dtype=np.float64)
    obs = lines or []
    res = least_squares(
        _residuals,
        hg.theta_from_h(H0),
        jac=_jacobian,
        args=(xy, uv, w, obs),
        loss="soft_l1",
        f_scale=cfg.loss_scale_px,
        x_scale="jac",
    )
    return hg.h_from_theta(res.x), _residuals(res.x, xy, uv, w, obs)


def laplace_covariance(
    J: ArrayLike,
    residuals: ArrayLike,
    config: SolverConfig | None = None,
) -> tuple[FloatArray, float, float, float]:
    """Covariance of θ for the soft-L1 M-estimator: Σθ = s² (Jᵀ J)⁻¹ from weighted residuals and Jacobian.

    The scale uses Huber's correction for M-estimators,
    s² = [Σ ψ(r)² / (m − 8)] / [mean ψ'(r)]², with ψ(r) = r / √(1 + z), ψ'(r) = (1 + z)^(−3/2), z = (r / c)².
    It is consistent under Gaussian noise; the naive IRLS estimate Σ w r² / (m − 8) underestimated σ² by about
    20 % in simulation, which made the 90 % regions cover only 83 %.

    The per-frame s² has few degrees of freedom, so it is shrunk towards ``sigma0_px``:
    s²_post = (ν s² + ν₀ σ₀²) / (ν + ν₀) with ν = m − 8: a scaled-inverse-χ² prior with ν₀ degrees of
    freedom. Regions built from Σθ should therefore use Student/F quantiles with ν + ν₀ degrees of freedom.

    Returns (Σθ, s²_post, scaled condition number, ν + ν₀).
    """
    cfg = config or SolverConfig()
    jac = np.asarray(J, dtype=np.float64)
    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    info = jac.T @ jac
    z = (r / cfg.loss_scale_px) ** 2
    psi, dpsi = r * _robust_weights(r, cfg.loss_scale_px), (1.0 + z) ** -1.5
    dof = max(len(r) - hg.N_PARAMS, 0)
    s2 = float(np.sum(psi**2) / dof / np.mean(dpsi) ** 2) if dof > 0 else cfg.sigma0_px**2
    s2_post = (dof * s2 + cfg.prior_dof * cfg.sigma0_px**2) / (dof + cfg.prior_dof)
    d = 1.0 / np.sqrt(np.clip(np.diag(info), 1e-300, None))
    scaled = info * d[:, None] * d[None, :]
    cond = float(np.linalg.cond(scaled))
    cov_scaled = np.linalg.pinv(scaled)
    return s2_post * cov_scaled * d[:, None] * d[None, :], s2_post, cond, dof + cfg.prior_dof


def calibrate(
    xy: ArrayLike,
    uv: ArrayLike,
    rng: np.random.Generator,
    weights: ArrayLike | None = None,
    config: SolverConfig | None = None,
    lines: list[LineObservation] | None = None,
) -> tuple[CalibResult, BoolArray]:
    """Full per-frame calibration from pitch points ``xy`` (metres), their detections ``uv`` (pixels) and,
    optionally, points observed on pitch lines.

    RANSAC runs on the point correspondences; lines whose points are far from their projection under the
    RANSAC solution are dropped before the joint refinement. Returns the result and the point inlier mask.
    The frame abstains (status ≠ OK) with too few inliers or a degenerate configuration.
    """
    cfg = config or SolverConfig()
    xy, uv = np.asarray(xy, dtype=np.float64), np.asarray(uv, dtype=np.float64)
    w = np.ones(len(xy)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(xy) < 4:
        return CalibResult(CalibStatus.FEW_POINTS, n_inliers=len(xy)), np.zeros(len(xy), dtype=bool)
    h0, inliers = ransac(xy, uv, rng, cfg)
    n_in = int(inliers.sum())
    if h0 is None:
        return CalibResult(CalibStatus.FEW_POINTS, n_inliers=n_in), inliers
    kept = [
        ln
        for ln in (lines or [])
        if np.all(
            np.abs(np.c_[ln.uv, np.ones(len(ln.uv))] @ _projected_line(h0, ln)) < 2 * cfg.ransac_threshold_px
        )
    ]
    n_obs = 2 * n_in + sum(len(ln.uv) for ln in kept)
    if n_obs < 2 * cfg.min_inliers:
        return CalibResult(CalibStatus.FEW_POINTS, n_inliers=n_in), inliers
    xi, ui, wi = xy[inliers], uv[inliers], w[inliers]
    h, r = refine(h0, xi, ui, wi, cfg, kept)
    cov, s2, cond, dof = laplace_covariance(_jacobian(hg.theta_from_h(h), xi, ui, wi, kept), r, cfg)
    point_res = hg.project(h, xi) - ui
    rms = float(np.sqrt(np.mean(np.sum(point_res**2, axis=1))))
    status = CalibStatus.OK
    if not (np.isfinite(h).all() and np.isfinite(cov).all()) or cond > cfg.max_scaled_condition:
        status = CalibStatus.DEGENERATE
    result = CalibResult(status, H=h, cov_theta=cov, s2=s2, rms_px=rms, n_inliers=n_in, cond=cond, dof=dof)
    return result, inliers
