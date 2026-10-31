import math

import numpy as np
import pytest

from predfoot.kpi.config import KpiConfig, Psv99Config, SmoothingConfig
from predfoot.kpi.errors import ghost_distance_per_min
from predfoot.kpi.metrics import compute_kpis, kpi_names, path_length, runs_above
from predfoot.kpi.speed import kinematics, kmh_to_ms, pipeline
from tests.unit.synthetic_tracks import from_step_speeds, run_with_ramps, straight_line

NO_SMOOTHING = SmoothingConfig(method="none")
SAVGOL_1S = SmoothingConfig(method="savgol", window_s=1.0, polyorder=2)


def kpis(xy, hz=10, smoothing=NO_SMOOTHING, **overrides):
    cfg = KpiConfig(smoothing=smoothing, **overrides)
    return compute_kpis([kinematics(xy, hz, cfg.smoothing)], cfg)


# --- runs above a threshold


def test_run_duration_uses_interpolated_threshold_crossings():
    runs = runs_above(np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]), 2.0, hz=1.0)
    assert runs.start.tolist() == [2]
    assert runs.stop.tolist() == [5]
    assert runs.duration_s.tolist() == pytest.approx([2.0])  # crossings at t = 2 and t = 4

    runs = runs_above(np.array([0.0, 3.0, 3.0, 0.0]), 1.5, hz=1.0)
    assert runs.duration_s.tolist() == pytest.approx([2.0])  # crossings at t = 0.5 and t = 2.5


def test_runs_stop_at_nan_and_extend_half_a_sample_at_edges():
    runs = runs_above(np.array([3.0, 3.0, 0.0]), 1.5, hz=10.0)
    assert runs.duration_s.tolist() == pytest.approx([0.2])
    runs = runs_above(np.array([np.nan, 3.0, 3.0, np.nan, 3.0]), 1.5, hz=1.0)
    assert runs.start.tolist() == [1, 4]
    assert runs.duration_s.tolist() == pytest.approx([2.0, 1.0])
    assert runs_above(np.full(4, np.nan), 1.0, hz=1.0).start.size == 0


# --- distance and speed bands


@pytest.mark.parametrize("smoothing", [NO_SMOOTHING, SAVGOL_1S])
def test_constant_18_kmh_for_60_s(smoothing):
    xy = straight_line([(60.0, 0.0)], hz=10, v0=kmh_to_ms(18.0))
    k = kpis(xy, smoothing=smoothing)
    assert k["distance_m"] == pytest.approx(300.0, abs=1e-9)
    assert k["dist_15_20_m"] == pytest.approx(300.0, abs=1e-9)
    for name in ("dist_lt15_m", "dist_20_25_m", "dist_gt25_m", "hi_distance_m"):
        assert k[name] == 0.0
    assert k["hsr_efforts"] == 0
    assert k["sprint_efforts"] == 0
    assert k["explosive_to_hsr"] == 0
    assert k["psv99_kmh"] == pytest.approx(18.0, abs=1e-9)


def test_constant_18_kmh_through_the_25_to_10_hz_pipeline():
    xy = straight_line([(60.0, 0.0)], hz=25, v0=5.0)
    cfg = KpiConfig()
    for resample_to in (None, 10):
        k = compute_kpis([pipeline(xy, cfg, resample_to=resample_to)], cfg)
        assert k["distance_m"] == pytest.approx(300.0, abs=1e-9)
        assert k["hsr_efforts"] == 0


def test_constant_27_kmh_counts_as_high_intensity_distance_and_one_sprint():
    xy = straight_line([(10.0, 0.0)], hz=10, v0=7.5)
    k = kpis(xy)
    assert k["distance_m"] == pytest.approx(75.0)
    assert k["dist_gt25_m"] == pytest.approx(75.0)
    assert k["hi_distance_m"] == pytest.approx(75.0)
    assert (k["hsr_efforts"], k["sprint_efforts"]) == (1, 1)
    assert k["psv99_kmh"] == pytest.approx(27.0)


def test_band_distances_partition_the_total_distance(rng):
    xy = np.cumsum(rng.normal(scale=0.4, size=(3000, 2)), axis=0)
    k = kpis(xy, smoothing=SAVGOL_1S)
    bands = k["dist_lt15_m"] + k["dist_15_20_m"] + k["dist_20_25_m"] + k["dist_gt25_m"]
    assert bands == pytest.approx(k["distance_m"])
    assert k["hi_distance_m"] == pytest.approx(k["dist_20_25_m"] + k["dist_gt25_m"])


# --- efforts


@pytest.mark.parametrize(("min_effort_s", "expected"), [(1.0, 1), (2.0, 1), (3.0, 1), (4.0, 0)])
def test_three_second_run_at_27_kmh(min_effort_s, expected):
    # 3 s at 27 km/h between 4 m/s² ramps: 3.28 s above 25 km/h, 3.97 s above 20 km/h.
    xy = straight_line(run_with_ramps(kmh_to_ms(27.0), plateau_s=3.0), hz=10)
    k = kpis(xy, min_effort_s=min_effort_s)
    assert k["sprint_efforts"] == expected
    if min_effort_s <= 3.0:
        assert k["hsr_efforts"] == 1


@pytest.mark.parametrize(("min_effort_s", "expected"), [(1.0, 1), (2.0, 1), (4.0, 0)])
def test_three_second_run_with_instantaneous_start_and_stop(min_effort_s, expected):
    xy = from_step_speeds([0.0] * 20 + [7.5] * 30 + [0.0] * 20, hz=10)
    assert kpis(xy, min_effort_s=min_effort_s)["sprint_efforts"] == expected


def test_separate_runs_are_counted_separately():
    walk, sprint = [1.5] * 30, [7.5] * 20
    xy = from_step_speeds(walk + sprint + walk + sprint + walk, hz=10)
    k = kpis(xy)
    assert (k["sprint_efforts"], k["hsr_efforts"]) == (2, 2)


# --- PSV-99


def test_psv99_is_a_percentile_of_speed_samples_during_activities():
    xy = from_step_speeds([kmh_to_ms(16.0)] * 100 + [kmh_to_ms(30.0)] * 100, hz=10)
    assert kpis(xy)["psv99_kmh"] == pytest.approx(30.0)


def test_psv99_run_peaks_variant_and_minimum_activity_duration():
    walk = [kmh_to_ms(5.0)] * 50
    xy = from_step_speeds(walk + [kmh_to_ms(20.0)] * 50 + walk + [kmh_to_ms(30.0)] * 50 + walk, hz=10)
    peaks = Psv99Config(statistic="run_peaks")
    assert kpis(xy, psv99=peaks)["psv99_kmh"] == pytest.approx(np.percentile([20.0, 30.0], 99))
    assert kpis(xy)["psv99_kmh"] == pytest.approx(30.0)
    short = from_step_speeds(walk + [kmh_to_ms(20.0)] * 5 + walk, hz=10)  # 0.5 s above 15 km/h
    assert math.isnan(kpis(short)["psv99_kmh"])


# --- explosive accelerations


@pytest.mark.parametrize(
    ("phases", "v0", "expected"),
    [
        # 0 -> 28.8 km/h at 4 m/s² for 2 s: reaches HSR and sprint speed.
        ([(2.0, 0.0), (2.0, 4.0), (3.0, 0.0), (2.0, -4.0), (2.0, 0.0)], 0.0, (1, 1)),
        # 0 -> 21.6 km/h: reaches HSR only.
        ([(2.0, 0.0), (1.5, 4.0), (3.0, 0.0)], 0.0, (1, 0)),
        # 2.5 m/s² is not a high acceleration.
        ([(2.0, 0.0), (3.0, 2.5), (3.0, 0.0)], 0.0, (0, 0)),
        # Starts from 10.8 km/h, above the 9 km/h walking pace.
        ([(2.0, 0.0), (1.2, 4.0), (3.0, 0.0)], 3.0, (0, 0)),
        # 0.6 s at 4 m/s² (about 0.5 s above 3 m/s² once differentiated), then 1.5 m/s² up to 23 km/h.
        ([(2.0, 0.0), (0.6, 4.0), (2.0, 1.5), (2.0, 0.0)], 1.0, (0, 0)),
        # 1.0 s at 4 m/s², then 1.5 m/s² up to 23.4 km/h.
        ([(2.0, 0.0), (1.0, 4.0), (1.0, 1.5), (2.0, 0.0)], 1.0, (1, 0)),
    ],
)
def test_explosive_accelerations_on_acceleration_ramps(phases, v0, expected):
    xy = straight_line(phases, hz=10, v0=v0)
    k = kpis(xy)
    assert (k["explosive_to_hsr"], k["explosive_to_sprint"]) == expected


def test_explosive_acceleration_survives_smoothing_and_resampling():
    phases = [(2.0, 0.0), (2.0, 4.0), (3.0, 0.0), (2.0, -4.0), (2.0, 0.0)]
    assert kpis(straight_line(phases, hz=10), smoothing=SAVGOL_1S)["explosive_to_sprint"] == 1
    cfg = KpiConfig()
    k = compute_kpis([pipeline(straight_line(phases, hz=25), cfg, resample_to=10)], cfg)
    assert (k["explosive_to_hsr"], k["explosive_to_sprint"]) == (1, 1)


# --- bookkeeping


def test_kpis_add_up_over_periods_and_handle_empty_parts():
    cfg = KpiConfig(smoothing=NO_SMOOTHING)
    a = kinematics(straight_line([(10.0, 0.0)], hz=10, v0=7.5), 10, cfg.smoothing)
    b = kinematics(straight_line([(20.0, 0.0)], hz=10, v0=5.0), 10, cfg.smoothing)
    empty = kinematics(np.full((50, 2), np.nan), 10, cfg.smoothing)
    k = compute_kpis([a, b, empty], cfg)
    assert k["distance_m"] == pytest.approx(175.0)
    assert k["sprint_efforts"] == 1
    assert k["minutes"] == pytest.approx((101 + 201) / 10 / 60)
    assert set(k) == set(kpi_names(cfg))
    nothing = compute_kpis([empty], cfg)
    assert nothing["distance_m"] == 0.0
    assert nothing["minutes"] == 0.0
    assert math.isnan(nothing["psv99_kmh"])


def test_ghost_distance_of_a_motionless_player_matches_the_closed_form(rng):
    hz, sigma, minutes = 25, 0.1, 10
    noise = sigma * rng.standard_normal((minutes * 60 * hz + 1, 2))
    assert ghost_distance_per_min(0.1, 25) == pytest.approx(265.9, abs=0.05)  # 60·f·σ·√π
    assert path_length(noise) / minutes == pytest.approx(ghost_distance_per_min(sigma, hz), rel=0.01)
