"""Propagate calibration and foot-point uncertainty to pitch positions (delta method)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2, f

from predfoot.geometry import homography as hg

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: χ²₂ quantile at 0.90: the 90 % ellipse of a 2D Gaussian is {e : eᵀ Σ⁻¹ e ≤ 4.605}.
CHI2_2_90 = 4.605170185988092


def region_quantile(dim: int, level: float = 0.9, dof: float | None = None) -> float:
    """Threshold q such that {e : eᵀ Σ⁻¹ e ≤ q} has probability ``level``.

    With a known noise scale this is the χ² quantile. When Σ carries a noise scale estimated with ``dof``
    degrees of freedom, eᵀ Σ⁻¹ e / dim follows an F(dim, dof) law, so the threshold is dim · F⁻¹(level).
    """
    if dof is None or not np.isfinite(dof):
        return float(chi2.ppf(level, dim))
    return float(dim * f.ppf(level, dim, dof))


def position_covariance(
    H: ArrayLike,
    cov_theta: ArrayLike,
    uv: ArrayLike,
    cov_uv: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Pitch positions of pixels ``uv`` (n, 2) with Σp = Jθ Σθ Jθᵀ + Ju Σu Juᵀ, shape (n, 2, 2).

    ``cov_uv`` is (2, 2) or (n, 2, 2) in px²; None means exact pixels (ground-truth boxes).
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    xy, valid = hg.backproject(H, uv)
    j_theta = hg.jacobian_backproject_theta(H, uv)
    cov = j_theta @ np.asarray(cov_theta, dtype=np.float64) @ np.swapaxes(j_theta, -1, -2)
    if cov_uv is not None:
        j_uv = hg.jacobian_backproject_uv(H, uv)
        cov = cov + j_uv @ np.asarray(cov_uv, dtype=np.float64) @ np.swapaxes(j_uv, -1, -2)
    return xy, cov, valid


def mahalanobis2(err: ArrayLike, cov: ArrayLike) -> FloatArray:
    """Squared Mahalanobis distances eᵀ Σ⁻¹ e for errors (n, 2) and covariances (n, 2, 2)."""
    e = np.asarray(err, dtype=np.float64)
    return np.einsum("...i,...ij,...j->...", e, np.linalg.inv(np.asarray(cov, dtype=np.float64)), e)


def ellipse_axes(cov: ArrayLike, q: float = CHI2_2_90) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Semi-major, semi-minor axes (metres) and orientation (radians, from +x) of {e : eᵀΣ⁻¹e ≤ q}."""
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    major, minor = np.sqrt(q * vals[..., 1]), np.sqrt(q * np.clip(vals[..., 0], 0, None))
    angle = np.arctan2(vecs[..., 1, 1], vecs[..., 0, 1])
    return major, minor, angle


def circumscribed_radius(cov: ArrayLike, q: float = CHI2_2_90) -> FloatArray:
    """Radius of the circle that contains the ellipse {e : eᵀΣ⁻¹e ≤ q}: √(q · λmax)."""
    return np.sqrt(q * np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))[..., -1])
