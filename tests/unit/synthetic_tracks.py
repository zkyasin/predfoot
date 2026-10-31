"""Synthetic trajectories with known kinematics for the KPI tests.

A player moves along the x axis through phases of constant acceleration; positions are exact at every sample.
"""

from __future__ import annotations

import numpy as np


def straight_line(phases: list[tuple[float, float]], hz: float, v0: float = 0.0) -> np.ndarray:
    """Positions (n, 2) sampled at ``hz`` from t = 0 to the end of the last phase, both included.

    ``phases`` lists ``(duration_s, acceleration_ms2)``; the speed starts at ``v0`` m/s.
    """
    durations = np.array([d for d, _ in phases], dtype=float)
    accels = np.array([a for _, a in phases], dtype=float)
    t_knots = np.concatenate([[0.0], np.cumsum(durations)])
    v_knots = v0 + np.concatenate([[0.0], np.cumsum(accels * durations)])
    if np.any(v_knots < -1e-12):
        raise ValueError("speed must stay non-negative")
    x_knots = np.concatenate([[0.0], np.cumsum(v_knots[:-1] * durations + 0.5 * accels * durations**2)])

    n = round(t_knots[-1] * hz) + 1
    t = np.arange(n) / hz
    k = np.clip(np.searchsorted(t_knots, t, side="right") - 1, 0, len(phases) - 1)
    tau = t - t_knots[k]
    x = x_knots[k] + v_knots[k] * tau + 0.5 * accels[k] * tau**2
    return np.column_stack([x, np.zeros(n)])


def run_with_ramps(
    speed_ms: float, plateau_s: float, accel: float = 4.0, rest_s: float = 2.0
) -> list[tuple[float, float]]:
    """Rest, accelerate at ``accel`` to ``speed_ms``, hold it for ``plateau_s``, decelerate, rest."""
    ramp = speed_ms / accel
    return [(rest_s, 0.0), (ramp, accel), (plateau_s, 0.0), (ramp, -accel), (rest_s, 0.0)]


def from_step_speeds(step_speeds: list[float] | np.ndarray, hz: float) -> np.ndarray:
    """Positions (n + 1, 2) along x whose n consecutive steps are covered at the given speeds (m/s).

    Speed changes are instantaneous, so central differences give the mean of the two speeds at a change.
    """
    steps = np.asarray(step_speeds, dtype=float) / hz
    x = np.concatenate([[0.0], np.cumsum(steps)])
    return np.column_stack([x, np.zeros_like(x)])
