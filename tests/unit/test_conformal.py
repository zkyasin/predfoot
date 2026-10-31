import numpy as np
import pytest

from predfoot.uncertainty.conformal import (
    MondrianConformal,
    conformal_quantile,
    frame_scores,
    group_split,
    grouped_quantile,
)


def test_conformal_quantile_is_the_corrected_order_statistic():
    s = np.arange(1, 10, dtype=float)  # n = 9
    assert conformal_quantile(s, 0.1) == 9.0  # ⌈10 × 0.9⌉ = 9th smallest
    assert conformal_quantile(s, 0.5) == 5.0
    assert conformal_quantile(s[:5], 0.1) == np.inf  # ⌈6 × 0.9⌉ = 6 > 5


def test_grouped_quantile_gives_each_group_the_same_weight():
    # One huge clip of small scores and nine small clips of large scores: the big clip must not dominate.
    scores = np.concatenate([np.zeros(10_000), np.ones(90) * 10])
    groups = np.concatenate([np.zeros(10_000), np.repeat(np.arange(1, 10), 10)])
    assert grouped_quantile(scores, groups, 0.1) == 10.0
    assert conformal_quantile(scores, 0.1) == 0.0


def test_group_split_never_shares_a_match():
    games = np.array([2, 3, 4, 5, 6, 9] * 10)
    cal, test = group_split(games, held_out=[4, 9])
    assert not (set(games[cal]) & set(games[test]))
    assert cal.sum() + test.sum() == len(games)


def simulate_clips(rng, n_clips, per_clip=60, scale_by_clip=True):
    """Chi-square scores with a clip-level random scale: dependent within clips, exchangeable across clips."""
    scales = rng.lognormal(0.0, 0.4, size=n_clips) if scale_by_clip else np.ones(n_clips)
    groups = np.repeat(np.arange(n_clips), per_clip)
    scores = rng.chisquare(2, size=n_clips * per_clip) * scales[groups]
    return scores, groups


def test_grouped_conformal_covers_new_clips(rng):
    coverages = []
    for _ in range(300):
        s_cal, g_cal = simulate_clips(rng, 60)
        s_test, _ = simulate_clips(rng, 60)
        q = grouped_quantile(s_cal, g_cal, 0.1)
        coverages.append(np.mean(s_test <= q))
    assert np.mean(coverages) == pytest.approx(0.90, abs=0.015)
    assert np.mean(coverages) >= 0.89


def test_mondrian_fixes_bin_wise_coverage(rng):
    n = 40_000
    zoom = rng.uniform(0, 1, n)
    groups = rng.integers(0, 80, n)
    scores = rng.chisquare(2, n) * np.where(zoom > 0.5, 4.0, 1.0)  # wide shots are four times worse
    model = MondrianConformal.fit(scores, groups, {"zoom": zoom}, {"zoom": 2}, alpha=0.1)
    zoom_t = rng.uniform(0, 1, n)
    scores_t = rng.chisquare(2, n) * np.where(zoom_t > 0.5, 4.0, 1.0)
    q = model.quantile_for({"zoom": zoom_t})
    for sel in (zoom_t <= 0.5, zoom_t > 0.5):
        assert np.mean(scores_t[sel] <= q[sel]) == pytest.approx(0.90, abs=0.02)
    pooled_cov_wide = np.mean(scores_t[zoom_t > 0.5] <= model.pooled)
    assert pooled_cov_wide < 0.85  # one global quantile under-covers the hard bin


def test_small_bins_fall_back_to_the_pooled_quantile(rng):
    scores = rng.chisquare(2, 1000)
    groups = rng.integers(0, 20, 1000)
    feat = np.where(np.arange(1000) < 50, 0.0, 1.0)  # one bin with only 50 scores
    model = MondrianConformal.fit(scores, groups, {"f": feat}, {"f": 2}, min_count=200)
    assert model.quantile_for({"f": np.array([0.0])})[0] == model.pooled


def test_json_round_trip(rng):
    model = MondrianConformal.fit(
        rng.chisquare(2, 2000), rng.integers(0, 30, 2000), {"z": rng.uniform(size=2000)}, {"z": 3}
    )
    again = MondrianConformal.from_json(model.to_json())
    probe = {"z": np.linspace(0, 1, 7)}
    np.testing.assert_allclose(again.quantile_for(probe), model.quantile_for(probe))


def test_frame_scores_take_the_worst_player():
    scores = np.array([1.0, 5.0, 2.0, 0.5, 0.7])
    frames = np.array([10, 10, 11, 11, 12])
    s, ids = frame_scores(scores, frames)
    np.testing.assert_array_equal(ids, [10, 11, 12])
    np.testing.assert_array_equal(s, [5.0, 2.0, 0.7])
