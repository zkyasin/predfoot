"""From positions to speed and acceleration, without ever differentiating across a gap.

Positions live on a regular time grid with NaN where a player is missing (not tracked, off the pitch, or a
frame where the pipeline abstains). Every step works on gap-free segments only:

1. resampling (25 → 10 Hz) by linear interpolation between the two bracketing samples, NaN if either is
   missing;
2. Savitzky–Golay smoothing of each segment (an assumption, see ``configs/kpi.yaml``);
3. velocity by central differences inside each segment and one-sided differences at its two ends; an
   isolated sample has no velocity;
4. speed = |velocity|, and acceleration = the same derivative applied to speed (rate of change of speed, the
   usual meaning in physical reports: turning at constant speed is not an acceleration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter

from predfoot.kpi.config import KpiConfig, SmoothingConfig

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

KMH_PER_MS = 3.6


def kmh_to_ms(v: ArrayLike) -> FloatArray:
    """km/h → m/s."""
    return np.asarray(v, dtype=np.float64) / KMH_PER_MS


def ms_to_kmh(v: ArrayLike) -> FloatArray:
    """m/s → km/h."""
    return np.asarray(v, dtype=np.float64) * KMH_PER_MS


def segments(valid: BoolArray) -> tuple[IntArray, IntArray]:
    """Start and stop (exclusive) indices of the runs of True in a 1-D boolean array."""
    padded = np.concatenate([[False], np.asarray(valid, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[::2].astype(np.int64), edges[1::2].astype(np.int64)


def _valid_rows(values: FloatArray) -> BoolArray:
    """Rows of a (n,) or (n, d) array whose entries are all finite."""
    finite = np.isfinite(values)
    return finite if finite.ndim == 1 else finite.all(axis=tuple(range(1, finite.ndim)))


def resample(values: ArrayLike, hz_in: int, hz_out: int) -> FloatArray:
    """Resample a regularly sampled signal of shape (n, ...) from ``hz_in`` to ``hz_out``, both in Hz.

    Output sample j is at time j / hz_out (the first input sample is at time 0). It is the linear
    interpolation of the two input samples around it (a copy when it falls on one), and NaN when either of
    them is NaN. Gaps are never bridged: an output sample that follows a missing input sample unused by its
    predecessor is NaN too. Integer arithmetic keeps the sample times exact.
    """
    x = np.asarray(values, dtype=np.float64)
    if hz_in == hz_out:
        return x.copy()
    n = x.shape[0]
    if n == 0:
        return x.copy()
    m = (n - 1) * hz_out // hz_in + 1
    num = np.arange(m, dtype=np.int64) * hz_in
    k0 = num // hz_out
    rem = num % hz_out
    out = x[k0].copy()
    between = rem > 0
    if between.any():
        w = (rem[between] / hz_out).reshape((-1,) + (1,) * (x.ndim - 1))
        out[between] = (1.0 - w) * x[k0[between]] + w * x[k0[between] + 1]
    # When downsampling, two consecutive output samples can straddle a missing input sample that neither of
    # them uses; the later one is dropped so that no output step spans the gap.
    invalid = ~_valid_rows(x)
    if invalid.any() and m > 1:
        before = np.concatenate([[0], np.cumsum(invalid)])  # invalid input samples in x[:k]
        last_used = k0 + between
        hidden = before[k0[1:]] - before[last_used[:-1] + 1] > 0
        out[1:][hidden] = np.nan
    return out


def drop_glitches(xy: ArrayLike, hz: float, max_speed_ms: float) -> tuple[FloatArray, int]:
    """Cut tracking glitches out of positions (n, 2): return a copy with NaN in their place, and their count.

    A step faster than ``max_speed_ms`` between consecutive samples is a glitch, not a movement. Raw tracking
    feeds typically hold the last position while a player is not observed and then jump to the re-acquired
    one: the held samples are dropped, except the first (the last real observation). A jump with no held
    position before it (an identity swap, a single wild sample) drops the sample after the jump. Either way a
    gap separates the two sides, so no derivative ever spans the jump. Apply this to source data before any
    simulated noise, which would otherwise be mistaken for glitches.
    """
    p = np.array(xy, dtype=np.float64, copy=True)
    if p.shape[0] < 2:
        return p, 0
    d = np.diff(p, axis=0)
    step = np.hypot(d[:, 0], d[:, 1]) * hz  # NaN across gaps
    with np.errstate(invalid="ignore"):
        jumps = np.flatnonzero(step > max_speed_ms)
    if jumps.size == 0:
        return p, 0
    # Runs of zero steps a..b-1 mean samples a..b repeat the same position; key them by their last sample b.
    held_start, held_stop = segments(step == 0.0)
    held_from = dict(zip(held_stop.tolist(), held_start.tolist(), strict=True))
    drop = np.zeros(p.shape[0], dtype=bool)
    for k in jumps.tolist():  # the jump is between samples k and k + 1
        if drop[k]:
            continue  # it starts at a sample already cut out (the way back from a wild sample)
        if k in held_from:
            drop[held_from[k] + 1 : k + 1] = True
        else:
            drop[k + 1] = True
    p[drop] = np.nan
    return p, int(drop.sum())


def window_samples(window_s: float, hz: float) -> int:
    """Odd number of samples closest to ``window_s`` seconds (rounded up to odd)."""
    n = max(1, round(window_s * hz))
    return n if n % 2 == 1 else n + 1


def smooth(values: ArrayLike, hz: float, cfg: SmoothingConfig) -> FloatArray:
    """Smooth each gap-free segment of a (n,) or (n, d) signal separately.

    Segments shorter than the window use the longest odd window that fits; segments too short for the
    polynomial order are left as they are.
    """
    x = np.array(values, dtype=np.float64, copy=True)
    if cfg.method == "none":
        return x
    window = window_samples(cfg.window_s, hz)
    starts, stops = segments(_valid_rows(x))
    for s, e in zip(starts, stops, strict=True):
        length = int(e - s)
        w = min(window, length if length % 2 == 1 else length - 1)
        if w <= cfg.polyorder:
            continue
        x[s:e] = savgol_filter(x[s:e], w, cfg.polyorder, axis=0, mode="interp")
    return x


def derivative(values: ArrayLike, hz: float) -> FloatArray:
    """Time derivative of a (n,) or (n, d) signal, never across NaN.

    Central differences where both neighbours are valid, one-sided differences at the ends of a segment,
    NaN for isolated samples and invalid rows.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[0]
    out = np.full_like(x, np.nan)
    if n < 2:
        return out
    valid = _valid_rows(x)
    prev_ok = np.zeros(n, dtype=bool)
    next_ok = np.zeros(n, dtype=bool)
    prev_ok[1:] = valid[:-1] & valid[1:]
    next_ok[:-1] = valid[:-1] & valid[1:]

    central = prev_ok & next_ok
    forward = next_ok & ~prev_ok
    backward = prev_ok & ~next_ok
    idx = np.flatnonzero(central)
    out[idx] = (x[idx + 1] - x[idx - 1]) * (hz / 2.0)
    idx = np.flatnonzero(forward)
    out[idx] = (x[idx + 1] - x[idx]) * hz
    idx = np.flatnonzero(backward)
    out[idx] = (x[idx] - x[idx - 1]) * hz
    return out


@dataclass(frozen=True)
class Kinematics:
    """Smoothed positions, speed and acceleration of one player on a regular grid (NaN where missing)."""

    hz: float
    xy: FloatArray  # (n, 2) metres
    speed: FloatArray  # (n,) m/s
    accel: FloatArray  # (n,) m/s², rate of change of speed

    @property
    def dt(self) -> float:
        return 1.0 / self.hz


def kinematics(xy: ArrayLike, hz: float, smoothing: SmoothingConfig) -> Kinematics:
    """Smooth positions (n, 2) sampled at ``hz``, then differentiate them."""
    p = smooth(np.asarray(xy, dtype=np.float64), hz, smoothing)
    v = derivative(p, hz)
    speed = np.hypot(v[:, 0], v[:, 1])
    return Kinematics(hz=float(hz), xy=p, speed=speed, accel=derivative(speed, hz))


def pipeline(xy_native: ArrayLike, cfg: KpiConfig, *, resample_to: int | None) -> Kinematics:
    """The standard pipeline on positions sampled at ``cfg.native_hz``.

    ``resample_to=None`` works at the native rate (the reference); ``resample_to=cfg.output_hz`` first
    resamples to the export rate (10 fps), as for tracking data delivered at SkillCorner's frame rate.
    """
    xy = np.asarray(xy_native, dtype=np.float64)
    hz = cfg.native_hz
    if resample_to is not None and resample_to != hz:
        xy = resample(xy, hz, resample_to)
        hz = resample_to
    return kinematics(xy, hz, cfg.smoothing)
