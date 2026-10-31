import numpy as np
import pytest

from predfoot.kpi.config import KpiConfig, SmoothingConfig
from predfoot.kpi.speed import (
    derivative,
    drop_glitches,
    kinematics,
    kmh_to_ms,
    ms_to_kmh,
    pipeline,
    resample,
    segments,
    smooth,
    window_samples,
)
from tests.unit.synthetic_tracks import straight_line

NO_SMOOTHING = SmoothingConfig(method="none")
SAVGOL_1S = SmoothingConfig(method="savgol", window_s=1.0, polyorder=2)


def test_unit_conversions():
    assert kmh_to_ms(36.0) == pytest.approx(10.0)
    assert ms_to_kmh(10.0) == pytest.approx(36.0)
    assert kmh_to_ms(25.0) == pytest.approx(6.9444444444)
    np.testing.assert_allclose(ms_to_kmh(kmh_to_ms(np.array([9.0, 15.0, 20.0]))), [9.0, 15.0, 20.0])


def test_segments():
    starts, stops = segments(np.array([False, True, True, False, True]))
    assert starts.tolist() == [1, 4]
    assert stops.tolist() == [3, 5]
    starts, stops = segments(np.array([True, True]))
    assert (starts.tolist(), stops.tolist()) == ([0], [2])
    starts, stops = segments(np.array([False, False]))
    assert starts.size == 0
    assert stops.size == 0


@pytest.mark.parametrize(
    ("window_s", "hz", "expected"), [(1.0, 10, 11), (1.0, 25, 25), (0.5, 10, 5), (2.0, 10, 21)]
)
def test_window_is_odd_and_close_to_the_requested_duration(window_s, hz, expected):
    assert window_samples(window_s, hz) == expected


def test_resampling_25_to_10_hz_is_exact_on_linear_motion():
    t = np.arange(26) / 25  # one second at 25 Hz
    xy = np.column_stack([2.0 * t, -1.0 * t])
    out = resample(xy, 25, 10)
    assert out.shape == (11, 2)
    np.testing.assert_allclose(out[:, 0], 2.0 * np.arange(11) / 10, atol=1e-12)
    np.testing.assert_allclose(out[:, 1], -1.0 * np.arange(11) / 10, atol=1e-12)
    # 10 Hz samples at even indices fall exactly on 25 Hz samples (t = 0, 0.2, 0.4 s...).
    np.testing.assert_array_equal(out[::2], xy[::5])


def test_resampling_never_interpolates_across_a_gap():
    t = np.arange(26) / 25
    xy = np.column_stack([t, t])
    xy[3] = np.nan  # t = 0.12 s: brackets the 10 Hz sample at t = 0.1 s (between 25 Hz samples 2 and 3)
    xy[10] = np.nan  # t = 0.4 s: coincides with the 10 Hz sample 4
    out = resample(xy, 25, 10)
    assert np.isnan(out[1]).all()
    assert np.isnan(out[4]).all()
    assert np.isfinite(out[[0, 2, 3, 5, 6, 7, 8, 9, 10]]).all()


def test_resampling_at_the_same_rate_is_the_identity():
    xy = np.random.default_rng(0).normal(size=(50, 2))
    np.testing.assert_array_equal(resample(xy, 25, 25), xy)


def test_savgol_preserves_quadratic_motion_and_never_crosses_a_gap():
    t = np.arange(40) / 10
    xy = np.column_stack([1.0 + 2.0 * t + 0.5 * t**2, 3.0 - t])
    xy[20:] += 100.0  # the player "jumps" across the gap: smoothing must not mix the two sides
    xy[19] = np.nan
    out = smooth(xy, 10, SAVGOL_1S)
    np.testing.assert_allclose(out[:19], xy[:19], atol=1e-9)
    np.testing.assert_allclose(out[20:], xy[20:], atol=1e-9)
    assert np.isnan(out[19]).all()


def test_short_segments_are_kept_and_smoothing_can_be_disabled():
    xy = np.array([[0.0, 0.0], [1.0, 0.5], [np.nan, np.nan], [5.0, 5.0], [np.nan, np.nan]])
    np.testing.assert_array_equal(smooth(xy, 10, SAVGOL_1S), xy)
    noisy = np.random.default_rng(1).normal(size=(30, 2))
    np.testing.assert_array_equal(smooth(noisy, 10, NO_SMOOTHING), noisy)


def test_derivative_uses_central_differences_inside_segments_and_one_sided_at_their_ends():
    x = np.array([0.0, 1.0, 4.0, 9.0, np.nan, 100.0, 101.0, 102.0, np.nan, 7.0])
    d = derivative(x, hz=1.0)
    np.testing.assert_allclose(d[:4], [1.0, 2.0, 4.0, 5.0])  # forward, central, central, backward
    assert np.isnan(d[4])
    np.testing.assert_allclose(d[5:8], [1.0, 1.0, 1.0])
    assert np.isnan(d[8])
    assert np.isnan(d[9])  # an isolated sample has no derivative


def test_derivative_of_two_dimensional_positions():
    t = np.arange(5) / 10
    xy = np.column_stack([3.0 * t, -4.0 * t])
    np.testing.assert_allclose(derivative(xy, hz=10), np.tile([3.0, -4.0], (5, 1)))


@pytest.mark.parametrize("smoothing", [NO_SMOOTHING, SAVGOL_1S])
@pytest.mark.parametrize("resample_to", [None, 10])
def test_gaps_never_create_speed_spikes(smoothing, resample_to):
    # Walk at 1.5 m/s, vanish for 2 s, reappear 30 m further on, keep walking.
    xy = straight_line([(20.0, 0.0)], hz=25, v0=1.5)
    xy[250:300] = np.nan
    xy[300:, 0] += 30.0
    cfg = KpiConfig(smoothing=smoothing)
    kin = pipeline(xy, cfg, resample_to=resample_to)
    assert np.nanmax(kin.speed) == pytest.approx(1.5, abs=1e-9)
    assert np.nanmax(np.abs(kin.accel)) == pytest.approx(0.0, abs=1e-6)


def test_kinematics_of_constant_acceleration():
    xy = straight_line([(3.0, 2.0)], hz=10, v0=1.0)  # speed 1 + 2t
    kin = kinematics(xy, 10, NO_SMOOTHING)
    t = np.arange(31) / 10
    # Central differences are exact on quadratics; the one-sided end values are off by a·h/2.
    np.testing.assert_allclose(kin.speed[1:-1], 1.0 + 2.0 * t[1:-1], atol=1e-9)
    np.testing.assert_allclose(kin.accel[2:-2], 2.0, atol=1e-9)
    assert kin.dt == pytest.approx(0.1)


def _walk(n: int, step: float = 0.06) -> np.ndarray:
    return np.column_stack([np.arange(n) * step, np.zeros(n)])  # 1.5 m/s at 25 Hz


def test_held_positions_before_a_jump_are_dropped():
    xy = _walk(40)
    xy[10:20] = xy[10]  # the tracker holds the last position while the player is not observed...
    xy[20:] += 5.0  # ...then jumps to where the player is
    clean, dropped = drop_glitches(xy, hz=25, max_speed_ms=12.0)
    assert dropped == 9
    assert np.isnan(clean[11:20]).all()
    np.testing.assert_array_equal(clean[:11], xy[:11])  # the last real observation is kept
    np.testing.assert_array_equal(clean[20:], xy[20:])


def test_teleports_and_isolated_outliers_are_cut_out():
    xy = _walk(30)
    xy[15:] += np.array([20.0, 10.0])  # teleport while moving: the sample after the jump goes
    clean, dropped = drop_glitches(xy, hz=25, max_speed_ms=12.0)
    assert dropped == 1
    assert np.isnan(clean[15]).all()
    xy = _walk(30)
    xy[8] += 3.0  # one wild sample: jumps away and back
    clean, dropped = drop_glitches(xy, hz=25, max_speed_ms=12.0)
    assert dropped == 1
    assert np.isnan(clean[8]).all()
    assert np.isfinite(np.delete(clean, 8, axis=0)).all()


def test_plausible_tracks_and_existing_gaps_are_left_alone():
    xy = straight_line([(5.0, 2.0), (5.0, 0.0)], hz=25, v0=0.0)  # up to 10 m/s, under the threshold
    xy[40:45] = np.nan
    clean, dropped = drop_glitches(xy, hz=25, max_speed_ms=12.0)
    assert dropped == 0
    np.testing.assert_array_equal(clean, xy)


def test_resampling_never_bridges_a_gap_that_falls_between_output_samples():
    # 10 fps samples use 25 Hz samples (12, 13) and (15): sample 14 is used by neither. A one-sample gap
    # there, followed by a teleport, must still separate the two sides.
    xy = _walk(40)
    xy[14] = np.nan
    xy[15:] += 20.0
    out = resample(xy, 25, 10)
    assert np.isfinite(out[5]).all()  # t = 0.5 s uses samples 12 and 13
    assert np.isnan(out[6]).all()  # t = 0.6 s would be the first sample after the hidden gap
    kin = pipeline(xy, KpiConfig(), resample_to=10)
    assert np.nanmax(kin.speed) == pytest.approx(1.5, abs=1e-6)
