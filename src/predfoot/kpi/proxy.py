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

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
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
#: Native frames between consecutive SN-GSR evaluation frames (one in five of 25 fps).
PROXY_STEP_FRAMES = 5
#: Errors against the labels this large are calibration failures, left to abstention: runs break there.
PROXY_MAX_ERROR_M = 10.0
#: Keys of a player row in the tables of scripts/eval_dev.py.
PLAYER_KEYS = ["clip", "frame", "track_id"]
_DT_S = PROXY_STEP_FRAMES / 25.0


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


# --- SN-GSR player tables ----------------------------------------------------------------------------------


def table_references(players: pl.DataFrame) -> list[str]:
    """References with error columns in a players table: labels, line oracle, smoothed labels."""
    return [r for r in ("lab", "or", "sm") if f"ex_{r}" in players.columns]


def players_kpi_proxy(players: pl.DataFrame) -> dict[str, Any] | None:
    """Distance, HSR distance and speed of the prediction against the smoothed labels (5 fps, SG filters).

    Per track, runs of at least :data:`PROXY_MIN_RUN` consecutive evaluation frames with every error against
    the labels below :data:`PROXY_MAX_ERROR_M`; the raw labels' own biases against the smoothed labels are the
    noise floor of that reference. None without a usable run.
    """
    if not {"x_pred", "y_pred", "ex_sm", "ey_sm"} <= set(players.columns):
        return None
    d = players.filter(pl.col("ex_lab").is_not_nan() & pl.col("ex_sm").is_not_nan()).sort(PLAYER_KEYS)
    pred_vs_sm, raw_vs_sm = ProxyTotals(), ProxyTotals()
    run_lengths: Counter[int] = Counter()
    for (_clip, _track), g in d.partition_by("clip", "track_id", as_dict=True).items():
        frames = g["frame"].to_numpy()
        pred = g.select("x_pred", "y_pred").to_numpy()
        sm = pred - g.select("ex_sm", "ey_sm").to_numpy()
        raw = pred - g.select("ex_lab", "ey_lab").to_numpy()
        good = np.hypot(*(pred - raw).T) < PROXY_MAX_ERROR_M
        breaks = np.flatnonzero((np.diff(frames) != PROXY_STEP_FRAMES) | ~good[1:] | ~good[:-1]) + 1
        for idx in np.split(np.arange(len(frames)), breaks):
            idx = idx[good[idx]]
            if len(idx) < PROXY_MIN_RUN:
                continue
            run_lengths[len(idx)] += 1
            pred_vs_sm.add(pred[idx], sm[idx], _DT_S)
            raw_vs_sm.add(raw[idx], sm[idx], _DT_S)
    if not pred_vs_sm.runs:
        return None
    p, r = pred_vs_sm.summary(), raw_vs_sm.summary()
    return {
        "runs": pred_vs_sm.runs,
        "samples": pred_vs_sm.samples,
        "distance_bias": p["distance_bias"],
        "raw_distance_bias": r["distance_bias"],
        "hsr_bias": p["hsr_bias"],
        "raw_hsr_bias": r["hsr_bias"],
        "hsr_sm_m": p["ref_hsr_m"],
        "sm_distance_m": p["ref_distance_m"],
        "sm_duration_s": p["duration_s"],
        "sm_mean_speed_ms": p["ref_mean_speed_ms"],
        "speed_rms": p["speed_rms"],
        "speed_p90": p["speed_p90"],
        "run_lengths": dict(sorted(run_lengths.items())),
    }


def restrict_shared(a: pl.DataFrame, b: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Both tables on the player-frames where both have finite errors for every shared reference (the line
    oracle excepted: it exists on part of the frames only)."""
    refs = [r for r in table_references(a) if r in table_references(b)]
    finite = [pl.col(f"ex_{r}").is_not_nan() for r in refs if r != "or"]
    ka = a.filter(*finite).select(PLAYER_KEYS)
    kb = b.filter(*finite).select(PLAYER_KEYS)
    both = ka.join(kb, on=PLAYER_KEYS)
    return a.join(both, on=PLAYER_KEYS), b.join(both, on=PLAYER_KEYS)
