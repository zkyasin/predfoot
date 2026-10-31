"""Pitch positions with calibrated 90 % regions for given foot pixels: the numpy core of ``/v1/localize``.

The definitions are those of the real pipeline stages (:mod:`predfoot.pipeline.real`):
Σp = Jθ Σθ Jθᵀ (+ Ju Σu Juᵀ) from :func:`predfoot.uncertainty.delta.position_covariance`; for a region
threshold q, the calibrated covariance is (q / χ²₂(0.90)) · Σp, so that its χ²₂(0.90) ellipse is the region
{e : eᵀ Σp⁻¹ e ≤ q}, and ``p90_radius_m`` = √(q · λmax(Σp)) is the radius of the circle that contains it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from predfoot.uncertainty.delta import CHI2_2_90, position_covariance, region_quantile

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
QuantileSource = Literal["conformal", "laplace_f", "laplace_chi2"]


@dataclass(frozen=True)
class Positions:
    """Localisation of n feet. Rows where ``valid`` is False hold NaN: above the horizon, or a position, a
    threshold or a region that is not finite (a row is never half null)."""

    xy: FloatArray  # (n, 2) metres, internal pitch frame
    cov: FloatArray  # (n, 2, 2) m², calibrated
    radius: FloatArray  # (n,) metres
    q: FloatArray  # (n,) threshold of each region
    valid: BoolArray  # (n,)


def region_threshold(
    conformal_quantile: float | None = None, dof: float | None = None
) -> tuple[float, QuantileSource]:
    """Threshold q of the 90 % regions {e : eᵀ Σp⁻¹ e ≤ q}, and where it comes from.

    A conformal quantile (fitted on held-out games) wins. Without one, the region is the uncalibrated Laplace
    region: χ²₂(0.90) = 4.605 when the noise scale of Σθ is taken as known, 2 · F⁻¹(0.90; 2, dof) when it was
    estimated with ``dof`` degrees of freedom (``CalibResult.dof``). On the SN-GSR dev games the uncalibrated
    regions cover only about 18 % of the errors, so pass a conformal quantile whenever one is available.
    """
    if conformal_quantile is not None:
        return float(conformal_quantile), "conformal"
    if dof is not None:
        return region_quantile(2, 0.9, dof), "laplace_f"
    return CHI2_2_90, "laplace_chi2"


def localize_feet(
    H: ArrayLike,
    cov_theta: ArrayLike,
    uv: ArrayLike,
    cov_uv: ArrayLike | None = None,
    q: float | ArrayLike = CHI2_2_90,
) -> Positions:
    """Positions, calibrated covariances and 90 % radii of foot pixels ``uv`` (n, 2).

    ``H`` maps the pitch to the image with H[2, 2] = 1 and ``cov_theta`` is the 8 × 8 covariance of its first
    8 entries; ``cov_uv`` is (2, 2) or (n, 2, 2) in px², None for exact pixels; ``q`` is one threshold for
    all feet or one per foot (n,).
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    n = len(uv)
    qs = np.broadcast_to(np.asarray(q, dtype=np.float64), (n,))
    xy_out, cov_out = np.full((n, 2), np.nan), np.full((n, 2, 2), np.nan)
    radius, q_out = np.full(n, np.nan), np.full(n, np.nan)
    if n == 0:
        return Positions(xy_out, cov_out, radius, q_out, np.zeros(0, dtype=bool))
    # Pixels above the horizon divide by a vanishing depth on the way, and extreme inputs overflow: such rows
    # are flagged invalid, not warned about.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        xy, cov, valid = position_covariance(H, cov_theta, uv, cov_uv)
        valid = valid & np.isfinite(xy).all(axis=1) & np.isfinite(cov).all(axis=(1, 2)) & np.isfinite(qs)
        scaled = np.full((n, 2, 2), np.nan)
        scaled[valid] = cov[valid] * (qs[valid] / CHI2_2_90)[:, None, None]
        r = np.full(n, np.nan)  # √(q · λmax(Σp)): circumscribed_radius with one q per foot
        r[valid] = np.sqrt(qs[valid] * np.linalg.eigvalsh(cov[valid])[:, -1])
    valid &= np.isfinite(scaled).all(axis=(1, 2)) & np.isfinite(r)
    xy_out[valid], cov_out[valid], radius[valid], q_out[valid] = xy[valid], scaled[valid], r[valid], qs[valid]
    return Positions(xy_out, cov_out, radius, q_out, valid)
