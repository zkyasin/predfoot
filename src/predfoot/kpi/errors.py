"""Position-error models for the KPI error budget.

The error of player i at frame t, per axis, is ``e[t, i] = c[t] + η[t, i]``: a common-mode part shared by
every player of the frame (for broadcast tracking, mostly the camera calibration) plus an individual part
(detection, foot point). Each part is a stationary Gaussian AR(1) process at the native frame rate:
``e[t] = ρ e[t-1] + sqrt(1 - ρ²) σ ε[t]``, with marginal standard deviation σ and lag-one autocorrelation ρ
(ρ = 0 is i.i.d. noise). The two axes are independent with the same σ.

The real error process of predfoot is measured on SN-GSR by ``scripts/error_process.py``;
:func:`split_common_mode` and :func:`fit_ar1` are estimators for that step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import lfilter

from predfoot.kpi.speed import FloatArray


@dataclass(frozen=True)
class ErrorModel:
    """Common-mode + individual AR(1) position errors, per axis, in metres (ρ per native frame)."""

    name: str
    sigma: float = 0.0  # individual part
    rho: float = 0.0
    sigma_common: float = 0.0
    rho_common: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma < 0 or self.sigma_common < 0:
            raise ValueError("sigma must be non-negative")
        if not (0.0 <= self.rho < 1.0 and 0.0 <= self.rho_common < 1.0):
            raise ValueError("rho must be in [0, 1)")

    @property
    def total_sigma(self) -> float:
        """Marginal standard deviation per axis of the total error."""
        return math.hypot(self.sigma, self.sigma_common)


def ar1_noise(
    n: int, sigma: float, rho: float, rng: np.random.Generator, shape: tuple[int, ...] = ()
) -> FloatArray:
    """Stationary Gaussian AR(1) noise of shape (n, *shape) along the first axis (no burn-in needed)."""
    eps = rng.standard_normal((n, *shape))
    if rho == 0.0 or n == 0:
        return sigma * eps
    w = sigma * math.sqrt(1.0 - rho * rho) * eps
    w[0] = sigma * eps[0]  # start from the stationary distribution
    return np.asarray(lfilter([1.0], [1.0, -rho], w, axis=0), dtype=np.float64)


def sample_errors(model: ErrorModel, n_frames: int, n_players: int, rng: np.random.Generator) -> FloatArray:
    """Errors of shape (n_frames, n_players, 2) drawn from ``model``."""
    e = np.zeros((n_frames, n_players, 2))
    if model.sigma > 0:
        e += ar1_noise(n_frames, model.sigma, model.rho, rng, (n_players, 2))
    if model.sigma_common > 0:
        e += ar1_noise(n_frames, model.sigma_common, model.rho_common, rng, (1, 2))
    return e


def split_common_mode(errors: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Split errors (n_frames, n_players, 2) into a common part and individual remainders.

    The common part is the per-frame median over players; missing players (NaN) are ignored.
    """
    e = np.asarray(errors, dtype=np.float64)
    common = np.nanmedian(e, axis=1)
    return common, e - common[:, None, :]


def fit_ar1(e: ArrayLike) -> tuple[float, float]:
    """Marginal standard deviation and lag-one autocorrelation of a 1-D series (NaN-tolerant)."""
    x = np.asarray(e, dtype=np.float64)
    sigma = float(np.nanstd(x))
    a, b = x[:-1], x[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    rho = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else math.nan
    return sigma, rho


def ghost_distance_per_min(sigma: float, hz: float, rho: float = 0.0) -> float:
    """Expected distance per minute "run" by a motionless player whose position has AR(1) noise.

    Each step between consecutive samples is a 2-D Gaussian vector with per-axis variance 2σ²(1 - ρ), whose
    mean norm is σ·sqrt(π(1 - ρ)); there are 60·hz steps per minute. For i.i.d. noise this is 60·f·σ·√π,
    about 266 m/min for σ = 0.1 m at 25 Hz. It applies to raw positions (no smoothing, no resampling).
    """
    return 60.0 * hz * sigma * math.sqrt(math.pi * (1.0 - rho))
