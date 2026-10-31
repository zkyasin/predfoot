"""Stationary Gaussian position-error processes whose spectrum respects the 5 fps measurement (T4.3).

The AR(1) model of :mod:`predfoot.kpi.errors` extrapolates an exponential autocorrelation from 5 fps
(lags of 0.2 s) to 25 Hz. An exponential has a cusp at lag zero: its paths are not differentiable and its
spectrum decays as 1/f², so the extrapolation puts error power at frequencies that were never measured and
that speed estimates amplify. Here each axis of each part of the error (the common part shared by the players
of a frame, mostly the camera, and the individual part) is a zero-mean stationary Gaussian process with
standard deviation σ and autocorrelation

    ρ(τ) = w_n·ν(τ) + w_m·k(τ/ℓ) + w_s·k(τ/T),     w_n + w_m + w_s = 1,

- ``k`` is the kernel family: ``exp`` (Ornstein-Uhlenbeck, rough), ``matern32`` and ``matern52`` (paths once
  and twice mean-square differentiable) or ``se`` (squared exponential, infinitely smooth), at a main time
  scale ℓ and an optional slow scale T;
- ν is the *nugget*: power that decorrelates within one 5 fps interval. Every lag the measurement sees is a
  multiple of 0.2 s, where the nugget is zero, so its shape at 25 Hz is an assumption, not a fit:
  ``white`` (independent frames, flat spectrum up to 12.5 Hz) or ``bandlimited`` (flat spectrum up to
  ``band_hz`` = 2.5 Hz, ν(τ) = sinc(2·band_hz·τ): the smoothest process still uncorrelated at 0.2 s lags).

:func:`fit_axis` fits the weights and time scales by weighted least squares to the measured ACF (σ is matched
exactly). :func:`sample_axis` draws exact samples on a regular grid by circulant embedding (Dietrich & Newsam
1997): the covariance is embedded in a circulant matrix whose eigenvalues come from one FFT; the band-limited
nugget is defined directly by its eigenvalues (a flat spectrum on the frequency bins up to ``band_hz``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import scipy.fft
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from predfoot.kpi.speed import FloatArray

KERNELS = ("exp", "matern32", "matern52", "se")
NUGGETS = ("white", "bandlimited")
AXES = ("x", "y")

#: Embedded covariances whose most negative eigenvalue exceeds this fraction of the largest are refused.
_NEG_TOL = 1e-6
_MAX_EMBED_DOUBLINGS = 4
#: Complex rows per FFT batch when sampling (bounds memory).
_BATCH = 8


def kernel_corr(kernel: str, lag_s: ArrayLike, length_s: float) -> FloatArray:
    """Unit-variance correlation of ``kernel`` at time scale ``length_s`` (seconds), at lags ``lag_s``."""
    t = np.abs(np.asarray(lag_s, dtype=np.float64)) / length_s
    if kernel == "exp":
        return np.exp(-t)
    if kernel == "matern32":
        a = math.sqrt(3.0) * t
        return (1.0 + a) * np.exp(-a)
    if kernel == "matern52":
        a = math.sqrt(5.0) * t
        return (1.0 + a + a * a / 3.0) * np.exp(-a)
    if kernel == "se":
        return np.exp(-0.5 * t * t)
    raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")


def kernel_curvature(kernel: str, length_s: float) -> float:
    """−k''(0): the variance of the time derivative of a unit-variance process (infinite for ``exp``)."""
    if kernel == "exp":
        return math.inf
    factor = {"matern32": 3.0, "matern52": 5.0 / 3.0, "se": 1.0}
    if kernel not in factor:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")
    return factor[kernel] / length_s**2


@dataclass(frozen=True)
class AxisProcess:
    """One axis of one error part: σ (metres) and the autocorrelation of the module docstring."""

    sigma: float
    kernel: str = "matern32"
    length_s: float = 1.0
    w_nugget: float = 0.0
    w_slow: float = 0.0
    slow_s: float = 10.0
    nugget: str = "white"
    band_hz: float = 2.5

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.kernel not in KERNELS:
            raise ValueError(f"unknown kernel {self.kernel!r}; expected one of {KERNELS}")
        if self.nugget not in NUGGETS:
            raise ValueError(f"unknown nugget {self.nugget!r}; expected one of {NUGGETS}")
        if not (0.0 <= self.w_nugget <= 1.0 and 0.0 <= self.w_slow <= 1.0):
            raise ValueError("weights must be in [0, 1]")
        if self.w_nugget + self.w_slow > 1.0 + 1e-12:
            raise ValueError("w_nugget + w_slow must not exceed 1")
        if self.length_s <= 0 or self.slow_s <= 0 or self.band_hz <= 0:
            raise ValueError("time scales and band must be positive")

    @property
    def w_main(self) -> float:
        return max(0.0, 1.0 - self.w_nugget - self.w_slow)

    def smooth_acf(self, lag_s: ArrayLike) -> FloatArray:
        """The kernel part of ρ (main + slow terms), without the nugget."""
        lag = np.asarray(lag_s, dtype=np.float64)
        out = self.w_main * kernel_corr(self.kernel, lag, self.length_s)
        if self.w_slow > 0:
            out = out + self.w_slow * kernel_corr(self.kernel, lag, self.slow_s)
        return np.asarray(out, dtype=np.float64)

    def acf(self, lag_s: ArrayLike) -> FloatArray:
        """Autocorrelation ρ at lags in seconds (continuous-time model)."""
        lag = np.abs(np.asarray(lag_s, dtype=np.float64))
        white = self.nugget == "white"
        nug = (lag == 0).astype(np.float64) if white else np.sinc(2.0 * self.band_hz * lag)
        return np.asarray(self.smooth_acf(lag) + self.w_nugget * nug, dtype=np.float64)

    def derivative_variance(self) -> float:
        """Variance of the time derivative (m²/s²): infinite when any part of ρ has a cusp at zero."""
        if self.sigma == 0:
            return 0.0
        total = 0.0
        if self.w_nugget > 0:
            if self.nugget == "white":
                return math.inf
            total += self.w_nugget * (2.0 * math.pi * self.band_hz) ** 2 / 3.0
        for w, scale in ((self.w_main, self.length_s), (self.w_slow, self.slow_s)):
            if w > 0:
                total += w * kernel_curvature(self.kernel, scale)
        return self.sigma**2 * total

    def with_nugget(self, nugget: str) -> AxisProcess:
        return replace(self, nugget=nugget)


def ou_through_first_lag(sigma: float, rho: float, lag_s: float = 0.2) -> AxisProcess:
    """The single Ornstein-Uhlenbeck process through one measured lag: the previous AR(1) injection.

    At 25 Hz its lag-one correlation is ρ^(1/(25·lag_s)), the lower end of the bracket that
    ``scripts/error_process.py`` prints.
    """
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1)")
    return AxisProcess(sigma=sigma, kernel="exp", length_s=-lag_s / math.log(rho))


# --- exact sampling ------------------------------------------------------------------------------------


def _embedding_size(n: int) -> int:
    return int(scipy.fft.next_fast_len(max(2 * (n - 1), 2)))


@lru_cache(maxsize=64)
def _sqrt_eigenvalues(procs: tuple[AxisProcess, ...], n: int, hz: float) -> FloatArray:
    """sqrt(λ/m) of the circulant embedding of the summed covariance of ``procs`` on n samples at ``hz``."""
    m = _embedding_size(n)
    for _ in range(_MAX_EMBED_DOUBLINGS + 1):
        k = np.arange(m)
        lag = np.minimum(k, m - k) / hz
        c = np.zeros(m)
        white = 0.0
        for p in procs:
            c += p.sigma**2 * p.smooth_acf(lag)
            if p.nugget == "white":
                white += p.sigma**2 * p.w_nugget
        lam = np.fft.fft(c).real + white
        freqs = np.abs(np.fft.fftfreq(m, d=1.0 / hz))
        for p in procs:
            if p.nugget == "bandlimited" and p.w_nugget > 0:
                band = freqs <= p.band_hz
                lam[band] += p.sigma**2 * p.w_nugget * m / band.sum()
        if lam.min() >= -_NEG_TOL * lam.max():
            out = np.sqrt(np.clip(lam, 0.0, None) / m)
            out.setflags(write=False)
            return out
        m *= 2
    raise ValueError("circulant embedding is not non-negative definite; the time scale is too long")


def sample_axis(
    procs: Sequence[AxisProcess], n: int, n_paths: int, rng: np.random.Generator, hz: float = 25.0
) -> FloatArray:
    """Exact samples (n, n_paths) on a grid at ``hz`` of the sum of the independent processes ``procs``.

    Every path is stationary from its first sample. Real and imaginary parts of one complex FFT give two
    independent paths.
    """
    active = tuple(p for p in procs if p.sigma > 0)
    out = np.zeros((n, n_paths))
    if not active or n == 0 or n_paths == 0:
        return out
    if n == 1:
        return rng.standard_normal((1, n_paths)) * math.sqrt(sum(p.sigma**2 for p in active))
    s = _sqrt_eigenvalues(active, n, float(hz))
    m = s.size
    done = 0
    while done < n_paths:
        rows = min(_BATCH, (n_paths - done + 1) // 2)
        z = rng.standard_normal((rows, m)) + 1j * rng.standard_normal((rows, m))
        y = np.fft.fft(z * s, axis=1)[:, :n]
        paths = np.concatenate([y.real, y.imag], axis=0)[: n_paths - done]
        out[:, done : done + paths.shape[0]] = paths.T
        done += paths.shape[0]
    return out


@dataclass(frozen=True)
class PartProcess:
    """The two axes of one error part (pitch frame: x along the touchlines, y across)."""

    x: AxisProcess
    y: AxisProcess

    def axis(self, name: str) -> AxisProcess:
        return self.x if name == "x" else self.y

    def with_nugget(self, nugget: str) -> PartProcess:
        return PartProcess(self.x.with_nugget(nugget), self.y.with_nugget(nugget))


@dataclass(frozen=True)
class SpectralErrorModel:
    """Common part (one draw per frame, shared by all players) plus an individual part per player."""

    name: str
    common: PartProcess | None = None
    individual: PartProcess | None = None

    def with_nugget(self, nugget: str, name: str | None = None) -> SpectralErrorModel:
        return SpectralErrorModel(
            name or self.name,
            self.common.with_nugget(nugget) if self.common else None,
            self.individual.with_nugget(nugget) if self.individual else None,
        )

    def player_processes(self, axis: str) -> tuple[AxisProcess, ...]:
        """The independent processes whose sum is one player's error on ``axis`` (common + individual)."""
        return tuple(p.axis(axis) for p in (self.common, self.individual) if p is not None)


def sample_errors(
    model: SpectralErrorModel, n_frames: int, n_players: int, rng: np.random.Generator, hz: float = 25.0
) -> FloatArray:
    """Errors (n_frames, n_players, 2): the common part is shared by every player of a frame."""
    e = np.zeros((n_frames, n_players, 2))
    for a, name in enumerate(AXES):
        if model.individual is not None:
            e[:, :, a] += sample_axis([model.individual.axis(name)], n_frames, n_players, rng, hz)
        if model.common is not None:
            e[:, :, a] += sample_axis([model.common.axis(name)], n_frames, 1, rng, hz)
    return e


def sample_player_errors(
    model: SpectralErrorModel, n_frames: int, n_draws: int, rng: np.random.Generator, hz: float = 25.0
) -> FloatArray:
    """``n_draws`` independent error tracks (n_frames, n_draws, 2) of one player (common + individual)."""
    e = np.empty((n_frames, n_draws, 2))
    for a, name in enumerate(AXES):
        e[:, :, a] = sample_axis(model.player_processes(name), n_frames, n_draws, rng, hz)
    return e


# --- fitting -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AcfFit:
    """A fitted axis process and its residuals (model minus measured ACF) at the measured lags."""

    process: AxisProcess
    lags_s: FloatArray
    measured: FloatArray
    residuals: FloatArray

    @property
    def max_abs_residual(self) -> float:
        return float(np.max(np.abs(self.residuals)))


def fit_axis(
    sigma: float,
    lags_s: ArrayLike,
    acf: ArrayLike,
    n_pairs: ArrayLike | None = None,
    *,
    kernel: str = "matern32",
    slow: bool = False,
    nugget: str = "white",
    min_scale_s: float = 0.2,
) -> AcfFit:
    """Weighted least-squares fit of the nugget weight and time scales to a measured ACF.

    Weights are n_pairs / (1 − ρ̂²)², the inverse of the large-sample variance of a correlation estimate up
    to the (unknown) effective-sample-size factor, which only rescales them. With ``slow`` a second term of
    the same kernel is fitted too (T between 1 and 120 s; the two scales are ordered after the fit). σ is the
    measured value. The nugget shape does not enter the fit: it vanishes at every lag that is a multiple of
    the 5 fps interval.

    Kernel time scales are kept at or above ``min_scale_s`` (default: the 0.2 s interval of the
    measurement). A shorter scale is not resolved by the lags and would only act as a nugget with an
    arbitrary 25 Hz shape; power that decorrelates faster goes to the nugget, whose shape is then an explicit
    assumption.
    """
    lags = np.asarray(lags_s, dtype=np.float64)
    r = np.asarray(acf, dtype=np.float64)
    if lags.shape != r.shape or lags.ndim != 1 or lags.size < (4 if slow else 2):
        raise ValueError("need matching 1-D lags and ACF values, at least 2 (4 with a slow term)")
    w = np.ones_like(r) if n_pairs is None else np.asarray(n_pairs, dtype=np.float64)
    w = w / np.clip(1.0 - r**2, 1e-3, None) ** 2
    sw = np.sqrt(w / w.sum())

    def unpack(theta: FloatArray) -> AxisProcess:
        w_n, log_l = float(theta[0]), float(theta[1])
        w_s, slow_s = 0.0, 10.0
        if slow:
            w_s, slow_s = (1.0 - w_n) * float(theta[2]), math.exp(float(theta[3]))
        return AxisProcess(
            sigma=sigma,
            kernel=kernel,
            length_s=math.exp(log_l),
            w_nugget=w_n,
            w_slow=min(w_s, 1.0 - w_n),
            slow_s=slow_s,
            nugget=nugget,
        )

    def resid(theta: FloatArray) -> FloatArray:
        return np.asarray(sw * (unpack(theta).smooth_acf(lags) - r), dtype=np.float64)

    lo = [0.0, math.log(min_scale_s)] + ([0.0, math.log(max(1.0, min_scale_s))] if slow else [])
    hi = [1.0, math.log(60.0)] + ([1.0, math.log(120.0)] if slow else [])
    best = None
    for l0 in (0.3, 1.0, 3.0, 10.0):
        k0 = float(kernel_corr(kernel, lags[0], l0))
        w0 = float(np.clip(1.0 - r[0] / max(k0, 1e-6), 0.0, 0.95))
        starts = [[w0, math.log(l0)]]
        if slow:
            starts = [[w0, math.log(l0), u0, math.log(t0)] for u0 in (0.2, 0.5) for t0 in (5.0, 30.0)]
        for x0 in starts:
            x0c = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            sol = least_squares(resid, x0c, bounds=(lo, hi), x_scale="jac")
            if best is None or sol.cost < best.cost:
                best = sol
    assert best is not None
    proc = unpack(best.x)
    if proc.w_slow < 1e-4 or proc.w_main < 1e-4:  # one scale is enough: keep it as the main term
        main_scale = proc.length_s if proc.w_main >= proc.w_slow else proc.slow_s
        proc = replace(proc, length_s=main_scale, w_slow=0.0)
    elif proc.slow_s < proc.length_s:
        proc = replace(proc, length_s=proc.slow_s, slow_s=proc.length_s, w_slow=proc.w_main)
    return AcfFit(proc, lags, r, proc.smooth_acf(lags) - r)
