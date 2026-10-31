"""Per-player physical KPIs from :class:`~predfoot.kpi.speed.Kinematics`.

Definitions (thresholds come from ``configs/kpi.yaml``, which says what is published and what is assumed):

- **distance**: sum of the distances between consecutive valid (smoothed) positions. A step belongs to the
  speed band of its own speed, distance / Δt, so band distances add up to the total exactly. HI distance is
  the distance at or above the HSR threshold. Bands are closed below and open above: [15, 20) km/h, etc.
- **efforts**: maximal runs of samples whose speed is at or above a threshold. The duration of a run is the
  time between its two threshold crossings, interpolated linearly between samples (half a sample is added at
  a gap or at the end of the data), which keeps durations comparable between 25 Hz and 10 Hz. A run counts
  when it lasts at least ``min_effort_s``. HSR efforts are runs above the HSR threshold, so a sprint is
  usually an HSR effort too.
- **PSV-99**: the chosen percentile of the speed over activities (runs above ``psv99.activity_kmh`` lasting
  at least ``psv99.min_activity_s``), taken over their samples or over each activity's peak.
- **explosive accelerations**: acceleration phases (consecutive samples where speed increases) that start
  below ``start_below_kmh``, end at or above the HSR (or sprint) threshold and contain a run of more than
  ``min_duration_s`` at or above ``accel_ms2``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike

from predfoot.kpi.config import KpiConfig
from predfoot.kpi.speed import BoolArray, FloatArray, IntArray, Kinematics, kmh_to_ms, ms_to_kmh, segments

#: Tolerance on duration comparisons, so that 3 × 0.1 s counts as 0.3 s.
_EPS_S = 1e-9


@dataclass(frozen=True)
class Runs:
    """Maximal runs of samples at or above a threshold."""

    start: IntArray  # first sample of each run
    stop: IntArray  # one past its last sample
    duration_s: FloatArray


def runs_above(values: ArrayLike, threshold: float, hz: float) -> Runs:
    """Runs of consecutive samples with ``values >= threshold``; NaN never belongs to a run.

    The duration runs from the interpolated up-crossing to the interpolated down-crossing. When the sample
    before (after) a run is missing or outside the array, the run is extended by half a sample on that side.
    """
    v = np.asarray(values, dtype=np.float64)
    start, stop = segments(v >= threshold)
    lead = np.full(start.shape, 0.5)
    trail = np.full(start.shape, 0.5)
    # The neighbour of a run, when it exists and is finite, is below the threshold: interpolate the crossing.
    idx = np.flatnonzero(start > 0)
    idx = idx[np.isfinite(v[start[idx] - 1])]
    s = start[idx]
    lead[idx] = (v[s] - threshold) / (v[s] - v[s - 1])
    idx = np.flatnonzero(stop < v.size)
    idx = idx[np.isfinite(v[stop[idx]])]
    e = stop[idx]
    trail[idx] = (v[e - 1] - threshold) / (v[e - 1] - v[e])
    duration = ((stop - start - 1) + lead + trail) / hz
    return Runs(start=start, stop=stop, duration_s=duration)


def step_lengths(xy: ArrayLike) -> FloatArray:
    """Distance between consecutive positions of an (n, 2) array; NaN when either end is missing."""
    p = np.asarray(xy, dtype=np.float64)
    d = np.diff(p, axis=0)
    return np.hypot(d[:, 0], d[:, 1])


def path_length(xy: ArrayLike) -> float:
    """Total distance along an (n, 2) trajectory, skipping steps across gaps."""
    return float(np.nansum(step_lengths(xy)))


def _fmt(v: float) -> str:
    return f"{v:g}"


def band_names(edges_kmh: Sequence[float]) -> list[str]:
    """Names of the distance-per-speed-band KPIs, e.g. ``dist_lt15_m``, ``dist_15_20_m``, ``dist_gt25_m``."""
    names = [f"dist_lt{_fmt(edges_kmh[0])}_m"]
    names += [f"dist_{_fmt(a)}_{_fmt(b)}_m" for a, b in pairwise(edges_kmh)]
    names.append(f"dist_gt{_fmt(edges_kmh[-1])}_m")
    return names


def kpi_names(cfg: KpiConfig) -> list[str]:
    """Names of the KPIs returned by :func:`compute_kpis`, in report order."""
    return [
        "minutes",
        "distance_m",
        *band_names(cfg.band_edges_kmh),
        "hi_distance_m",
        "hsr_efforts",
        "sprint_efforts",
        "psv99_kmh",
        "explosive_to_hsr",
        "explosive_to_sprint",
    ]


def _mask_from_runs(n: int, start: IntArray, stop: IntArray) -> BoolArray:
    marks = np.zeros(n + 1, dtype=np.int64)
    np.add.at(marks, start, 1)
    np.add.at(marks, stop, -1)
    return np.cumsum(marks[:-1]) > 0


def explosive_accelerations(kin: Kinematics, cfg: KpiConfig) -> tuple[int, int]:
    """Number of explosive accelerations reaching HSR speed, and reaching sprint speed."""
    ea = cfg.explosive_acceleration
    high = runs_above(kin.accel, ea.accel_ms2, kin.hz)
    keep = high.duration_s > ea.min_duration_s + _EPS_S
    if not keep.any():
        return 0, 0
    with np.errstate(invalid="ignore"):
        p_start, p_stop = segments(kin.accel > 0.0)
    # A run at or above accel_ms2 > 0 lies inside exactly one acceleration phase.
    phases = np.unique(np.searchsorted(p_start, high.start[keep], side="right") - 1)
    start_below = float(kmh_to_ms(ea.start_below_kmh))
    hsr, sprint = float(kmh_to_ms(cfg.hsr_kmh)), float(kmh_to_ms(cfg.sprint_kmh))
    to_hsr = to_sprint = 0
    n = kin.speed.size
    for k in phases:
        s, e = int(p_start[k]), int(p_stop[k])
        v_start = np.nanmin(kin.speed[max(s - 1, 0) : e])
        v_end = np.nanmax(kin.speed[s : min(e + 1, n)])
        if v_start < start_below:
            to_hsr += int(v_end >= hsr)
            to_sprint += int(v_end >= sprint)
    return to_hsr, to_sprint


def compute_kpis(parts: Sequence[Kinematics], cfg: KpiConfig) -> dict[str, float]:
    """KPIs of one player over several parts of a match (periods), which are never joined.

    Distances, counts and minutes add up over the parts; PSV-99 pools the activities of all parts.
    """
    bands = band_names(cfg.band_edges_kmh)
    edges = kmh_to_ms(cfg.band_edges_kmh)
    hsr, sprint = float(kmh_to_ms(cfg.hsr_kmh)), float(kmh_to_ms(cfg.sprint_kmh))
    activity = float(kmh_to_ms(cfg.psv99.activity_kmh))
    out = dict.fromkeys(kpi_names(cfg), 0.0)
    psv_values: list[FloatArray] = []

    for kin in parts:
        out["minutes"] += float(np.isfinite(kin.xy).all(axis=1).sum()) * kin.dt / 60.0

        steps = step_lengths(kin.xy)
        steps = steps[np.isfinite(steps)]
        step_speed = steps * kin.hz
        out["distance_m"] += float(steps.sum())
        band = np.searchsorted(edges, step_speed, side="right")
        for b, name in enumerate(bands):
            out[name] += float(steps[band == b].sum())
        out["hi_distance_m"] += float(steps[step_speed >= hsr].sum())

        for name, threshold in (("hsr_efforts", hsr), ("sprint_efforts", sprint)):
            runs = runs_above(kin.speed, threshold, kin.hz)
            out[name] += float(np.count_nonzero(runs.duration_s >= cfg.min_effort_s - _EPS_S))

        act = runs_above(kin.speed, activity, kin.hz)
        keep = act.duration_s >= cfg.psv99.min_activity_s - _EPS_S
        if keep.any():
            if cfg.psv99.statistic == "samples":
                psv_values.append(kin.speed[_mask_from_runs(kin.speed.size, act.start[keep], act.stop[keep])])
            else:
                peaks = [kin.speed[s:e].max() for s, e in zip(act.start[keep], act.stop[keep], strict=True)]
                psv_values.append(np.array(peaks))

        to_hsr, to_sprint = explosive_accelerations(kin, cfg)
        out["explosive_to_hsr"] += to_hsr
        out["explosive_to_sprint"] += to_sprint

    pooled = np.concatenate(psv_values) if psv_values else np.empty(0)
    psv = np.percentile(pooled, cfg.psv99.percentile) if pooled.size else np.nan
    out["psv99_kmh"] = float(ms_to_kmh(psv))
    return out
