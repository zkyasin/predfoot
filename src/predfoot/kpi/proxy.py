"""Model-free KPI proxy at 5 fps, shared by the SN-GSR measurement and its simulation on IDSSE.

SN-GSR dev positions are evaluated one frame in five (5 fps), in runs of consecutive frames. On each run, the
proxy smooths positions with a Savitzky-Golay filter (window 5, order 2) and takes velocities from the same
filter (first derivative); distance is the sum of the steps between smoothed positions, and high-speed-running
(HSR) distance the sum of the steps whose two end speeds average at least 5.56 m/s (20 km/h).
``scripts/error_process.py`` applies it to predfoot's positions and to the smoothed labels;
``scripts/kpi_spectral.py`` applies it to clean IDSSE tracks subsampled at 5 fps, with and without simulated
errors, so that both sides use exactly the same arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import savgol_filter

from predfoot.kpi.speed import FloatArray

#: Sampling interval of the SN-GSR evaluation (one native 25 fps frame in five).
PROXY_DT_S = 0.2
PROXY_SG_WINDOW = 5
PROXY_SG_ORDER = 2
#: 20 km/h in m/s, rounded as in the SN-GSR measurement.
PROXY_HSR_MS = 5.56
#: Shortest run the proxy uses (samples).
PROXY_MIN_RUN = 8


@dataclass(frozen=True)
class RunKinematics:
    """Proxy kinematics of one run: total distance, HSR distance and per-sample speed."""

    distance_m: float
    hsr_distance_m: float
    speed_ms: FloatArray


def run_kinematics(xy: ArrayLike, dt_s: float = PROXY_DT_S, hsr_ms: float = PROXY_HSR_MS) -> RunKinematics:
    """Savitzky-Golay distance, HSR distance and speed of one gap-free run of positions (n, 2).

    ``n`` must be at least the filter window. The filter uses scipy's default ``interp`` edges, as the
    measurement on SN-GSR does.
    """
    p = np.asarray(xy, dtype=np.float64)
    if p.shape[0] < PROXY_SG_WINDOW:
        raise ValueError(f"a run needs at least {PROXY_SG_WINDOW} samples, got {p.shape[0]}")
    pos = savgol_filter(p, PROXY_SG_WINDOW, PROXY_SG_ORDER, axis=0)
    vel = savgol_filter(p, PROXY_SG_WINDOW, PROXY_SG_ORDER, deriv=1, delta=dt_s, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    fast = 0.5 * (speed[1:] + speed[:-1]) >= hsr_ms
    return RunKinematics(float(steps.sum()), float(steps[fast].sum()), speed)


@dataclass
class ProxyTotals:
    """Accumulates the proxy of a test track against its reference over many runs."""

    distance_test: float = 0.0
    distance_ref: float = 0.0
    hsr_test: float = 0.0
    hsr_ref: float = 0.0
    runs: int = 0
    samples: int = 0
    duration_s: float = 0.0
    speed_err: list[FloatArray] = field(default_factory=list)

    def add(self, test_xy: ArrayLike, ref_xy: ArrayLike, dt_s: float = PROXY_DT_S) -> None:
        """Add one run seen through the test positions and the reference positions (same samples)."""
        t = run_kinematics(test_xy, dt_s)
        r = run_kinematics(ref_xy, dt_s)
        self.distance_test += t.distance_m
        self.distance_ref += r.distance_m
        self.hsr_test += t.hsr_distance_m
        self.hsr_ref += r.hsr_distance_m
        self.runs += 1
        self.samples += t.speed_ms.size
        self.duration_s += (t.speed_ms.size - 1) * dt_s
        self.speed_err.append(t.speed_ms - r.speed_ms)

    def summary(self) -> dict[str, float]:
        """Distance and HSR-distance biases (relative errors of the totals) and speed-error statistics."""
        err = np.concatenate(self.speed_err) if self.speed_err else np.empty(0)
        nan = float("nan")
        return {
            "runs": float(self.runs),
            "samples": float(self.samples),
            "distance_bias": self.distance_test / self.distance_ref - 1 if self.distance_ref > 0 else nan,
            "hsr_bias": self.hsr_test / self.hsr_ref - 1 if self.hsr_ref > 0 else nan,
            "speed_rms": float(np.sqrt(np.mean(err**2))) if err.size else nan,
            "speed_p90": float(np.quantile(np.abs(err), 0.9)) if err.size else nan,
            "ref_distance_m": self.distance_ref,
            "ref_hsr_m": self.hsr_ref,
            "duration_s": self.duration_s,
            "ref_mean_speed_ms": self.distance_ref / self.duration_s if self.duration_s > 0 else nan,
        }
