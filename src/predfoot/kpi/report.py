"""Agreement between KPIs computed under a test condition and under the reference condition.

- **MAPE**: mean of |test - ref| / ref over player-matches with ref > 0, in %.
- **bias**: sum(test - ref) / sum(ref), in %: the relative error of the KPI summed over all players.
- **Spearman**: rank correlation between reference and test values across player-matches (do the rankings of
  players survive?). KPIs listed in ``per90`` are divided by minutes played first, so that playing time does
  not dominate the ranking.
- Confidence intervals come from a bootstrap that resamples whole matches (the players of a match share its
  conditions), with percentile intervals.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from scipy.stats import rankdata

from predfoot.kpi.speed import IntArray


@dataclass(frozen=True)
class Estimate:
    """A statistic with its bootstrap confidence interval."""

    value: float
    lo: float
    hi: float


def _pair(ref: ArrayLike, test: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(ref, dtype=np.float64)
    t = np.asarray(test, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(t)
    return r[ok], t[ok]


def mape(ref: ArrayLike, test: ArrayLike) -> float:
    """Mean absolute percentage error over pairs with a positive reference (NaN when there is none)."""
    r, t = _pair(ref, test)
    pos = r > 0
    if not pos.any():
        return math.nan
    return float(np.mean(np.abs(t[pos] - r[pos]) / r[pos]) * 100.0)


def bias_pct(ref: ArrayLike, test: ArrayLike) -> float:
    """Relative error of the total, in % (NaN when the reference total is not positive)."""
    r, t = _pair(ref, test)
    total = r.sum()
    return float((t.sum() - total) / total * 100.0) if total > 0 else math.nan


def spearman(ref: ArrayLike, test: ArrayLike) -> float:
    """Spearman rank correlation (average ranks for ties); NaN when either side is constant."""
    r, t = _pair(ref, test)
    if r.size < 3 or np.ptp(r) == 0 or np.ptp(t) == 0:
        return math.nan
    return float(np.corrcoef(rankdata(r), rankdata(t))[0, 1])


def bootstrap_by_group(
    groups: ArrayLike,
    statistic: Callable[[IntArray], float],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Estimate:
    """Percentile bootstrap of ``statistic(row_indices)``, resampling whole groups with replacement."""
    g = np.asarray(groups)
    labels = np.unique(g)
    rows = [np.flatnonzero(g == label) for label in labels]
    value = statistic(np.arange(g.size))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, labels.size, labels.size)
        stats[b] = statistic(np.concatenate([rows[k] for k in pick]))
    alpha = (1.0 - level) / 2.0
    if np.all(np.isnan(stats)):
        return Estimate(value, math.nan, math.nan)
    lo, hi = np.nanquantile(stats, [alpha, 1.0 - alpha])
    return Estimate(float(value), float(lo), float(hi))


def compare_to_reference(
    ref: pl.DataFrame,
    test: pl.DataFrame,
    kpis: Sequence[str],
    *,
    group: str = "match_id",
    key: Sequence[str] = ("match_id", "player_id"),
    per90: Collection[str] = (),
    n_boot: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """One row per KPI: MAPE, bias and Spearman with match-bootstrap intervals.

    ``ref`` and ``test`` hold one row per player-match with the ``key`` columns, ``minutes`` (only the
    reference minutes are used) and the KPI columns.
    """
    joined = ref.select(*key, "minutes", *kpis).join(
        test.select(*key, *kpis), on=list(key), how="inner", suffix="_test"
    )
    groups = joined[group].to_numpy()
    minutes = joined["minutes"].to_numpy()
    rows = []
    for kpi in kpis:
        r = joined[kpi].to_numpy().astype(np.float64)
        t = joined[f"{kpi}_test"].to_numpy().astype(np.float64)
        rr, tt = (r / minutes * 90.0, t / minutes * 90.0) if kpi in per90 else (r, t)
        stats = {
            "mape": bootstrap_by_group(groups, _on_rows(mape, r, t), n_boot, seed),
            "bias": bootstrap_by_group(groups, _on_rows(bias_pct, r, t), n_boot, seed),
            "spearman": bootstrap_by_group(groups, _on_rows(spearman, rr, tt), n_boot, seed),
        }
        row: dict[str, float | int | str] = {
            "kpi": kpi,
            "n": int(np.isfinite(r).sum()),
            "ref_mean": float(np.nanmean(r)),
        }
        for name, est in stats.items():
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = est.value, est.lo, est.hi
        rows.append(row)
    return pl.DataFrame(rows)


def _on_rows(
    fn: Callable[[np.ndarray, np.ndarray], float], ref: np.ndarray, test: np.ndarray
) -> Callable[[IntArray], float]:
    """``fn`` restricted to a subset of rows, for :func:`bootstrap_by_group`."""

    def statistic(rows: IntArray) -> float:
        return fn(ref[rows], test[rows])

    return statistic
