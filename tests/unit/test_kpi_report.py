import math

import numpy as np
import polars as pl
import pytest

from predfoot.kpi.report import bias_pct, bootstrap_by_group, compare_to_reference, mape, spearman


def test_mape_ignores_zero_references_and_bias_is_relative_to_the_total():
    ref = np.array([100.0, 200.0, 0.0])
    test = np.array([110.0, 180.0, 3.0])
    assert mape(ref, test) == pytest.approx(10.0)
    assert bias_pct(ref, test) == pytest.approx((10.0 - 20.0 + 3.0) / 300.0 * 100)
    assert math.isnan(mape(np.zeros(3), np.ones(3)))


def test_spearman_of_rankings():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(ref, ref * 10) == pytest.approx(1.0)
    assert spearman(ref, -ref) == pytest.approx(-1.0)
    assert math.isnan(spearman(ref, np.ones(4)))


def test_bootstrap_by_group_resamples_whole_groups():
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    values = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])
    est = bootstrap_by_group(groups, lambda idx: float(values[idx].mean()), n_boot=500, seed=3)
    assert est.value == pytest.approx(5.0)
    assert 1.0 <= est.lo < est.value < est.hi <= 9.0
    # A statistic that does not vary between groups has a degenerate interval.
    flat = bootstrap_by_group(groups, lambda idx: 2.0, n_boot=50, seed=3)
    assert (flat.lo, flat.value, flat.hi) == (2.0, 2.0, 2.0)
    again = bootstrap_by_group(groups, lambda idx: float(values[idx].mean()), n_boot=500, seed=3)
    assert (again.lo, again.hi) == (est.lo, est.hi)


def test_compare_to_reference_joins_players_and_reports_each_kpi():
    ref = pl.DataFrame(
        {
            "match_id": ["m1", "m1", "m2", "m2"],
            "player_id": ["p1", "p2", "p3", "p4"],
            "minutes": [90.0, 90.0, 45.0, 90.0],
            "distance_m": [10_000.0, 11_000.0, 5_000.0, 9_000.0],
        }
    )
    test = ref.with_columns(pl.col("distance_m") * 1.1).reverse()
    table = compare_to_reference(ref, test, ["distance_m"], n_boot=200, seed=0)
    row = table.row(0, named=True)
    assert row["kpi"] == "distance_m"
    assert row["n"] == 4
    assert row["mape"] == pytest.approx(10.0)
    assert row["bias"] == pytest.approx(10.0)
    assert row["spearman"] == pytest.approx(1.0)
    assert row["mape_lo"] == pytest.approx(10.0)
    assert row["mape_hi"] == pytest.approx(10.0)
