import math

import numpy as np
import pytest
from scipy.stats import t as student_t

from predfoot.kpi.card import card_line, extrapolate, kpi_card, kpi_draws, nested_estimates
from predfoot.kpi.config import KpiConfig
from predfoot.kpi.metrics import compute_kpis
from predfoot.kpi.noise import AxisProcess, PartProcess, SpectralErrorModel, sample_player_errors
from predfoot.kpi.speed import pipeline


def smooth_model(sigma: float) -> SpectralErrorModel:
    proc = AxisProcess(sigma=sigma, kernel="matern32", length_s=0.5, w_nugget=0.2)
    return SpectralErrorModel("test", individual=PartProcess(proc, proc))


def line_track(speed_ms: float, seconds: float, hz: int = 25) -> np.ndarray:
    n = round(seconds * hz) + 1
    return np.column_stack([speed_ms * np.arange(n) / hz, np.zeros(n)])


def test_card_line_arithmetic():
    line = card_line("distance_m", 10.0, [11.0, 13.0, 12.0, 12.0])
    sd = float(np.std([1.0, 3.0, 2.0, 2.0], ddof=1))
    t3 = float(student_t.ppf(0.95, 3))
    half = t3 * sd * math.sqrt(1.25)
    assert line.bias == pytest.approx(2.0)
    assert line.estimate == pytest.approx(8.0)
    assert line.half_width == pytest.approx(half)
    assert line.interval == pytest.approx((8.0 - half, 8.0 + half))
    assert line.covers(8.0)
    assert not line.covers(8.0 + 1.01 * half)
    assert line.naive_half_width == pytest.approx(t3 * sd)  # K(y) has no Monte Carlo error
    assert line.covers(10.0 + 0.99 * t3 * sd, naive=True)
    assert not line.covers(10.0 + 1.01 * t3 * sd, naive=True)
    assert line.text("Distance", "m") == f"Distance = 8 ± {half:.0f} m (nominal 90 %)"
    assert math.isnan(card_line("x", 1.0, [2.0]).half_width)  # one draw is not enough


def test_quadratic_extrapolation_arithmetic():
    """K(0) = 10, mean K(1) = 12, mean K(2) = 15: the parabola gives K(−1) = 3·10 − 3·12 + 15 = 9. Both doses
    reuse each draw, so the Monte Carlo variance is that of −3·D₁ + D₂ draw by draw."""
    d1, d2 = np.array([1.0, 3.0, 2.0, 2.0]), np.array([5.0, 5.0, 4.0, 6.0])
    line = card_line("distance_m", 10.0, 10.0 + d1, draws2=10.0 + d2)
    assert line.estimate == pytest.approx(9.0)
    sd1, mc = float(np.std(d1, ddof=1)), float(np.var(-3.0 * d1 + d2, ddof=1)) / 4
    want = float(student_t.ppf(0.95, 3)) * math.sqrt(sd1**2 + mc)
    assert line.half_width == pytest.approx(want)
    with pytest.raises(ValueError, match="method"):
        kpi_card(
            [np.zeros((10, 2))],
            SpectralErrorModel("none"),
            KpiConfig(),
            np.random.default_rng(0),
            method="cubic",
        )


def test_draws_are_kept_in_pairs():
    """A draw whose dose-2 value is missing is dropped at both doses."""
    full = extrapolate(10.0, [11.0, 13.0, 12.0], [15.0, 15.0, 14.0])
    gappy = extrapolate(10.0, [11.0, 13.0, 12.0, 50.0], [15.0, 15.0, 14.0, np.nan])
    assert gappy == pytest.approx(full)
    assert gappy[3] == 3


def test_estimates_are_projected_onto_the_support():
    """An extrapolation below zero (a count cannot be negative) is moved to zero, and so is the interval."""
    d1, d2 = np.array([0.0, 1.0, 1.0, 0.0, 1.0]), np.array([2.0, 3.0, 2.0, 2.0, 3.0])
    line = card_line("sprint_efforts", 1.0, 1.0 + d1, draws2=1.0 + d2)
    assert line.raw_estimate == pytest.approx(1.0 - 3.0 * 0.6 + 2.4)  # 1.6: inside the support
    assert not line.truncated
    low = card_line("sprint_efforts", 1.0, 1.0 + d1 + 1.0, draws2=1.0 + d2)  # raw 1.6 − 3 = −1.4
    assert low.truncated
    assert low.estimate == 0.0
    lo, hi = low.interval
    assert lo == 0.0
    assert hi == pytest.approx(max(low.raw_estimate + low.half_width, 0.0))
    assert low.text("Sprint efforts", digits=1).startswith("Sprint efforts = 0.0 [0.0, ")
    assert low.covers(0.0)
    free = card_line("sprint_efforts", 1.0, 1.0 + d1 + 1.0, draws2=1.0 + d2, lower=None)
    assert free.estimate == pytest.approx(low.raw_estimate)
    assert not free.truncated


def test_calibrated_lines_drop_the_nominal_label():
    line = card_line("distance_m", 10.0, [11.0, 13.0, 12.0, 12.0]).calibrated(1.5)
    assert line.calibration == 1.5
    assert line.text("Distance", "m") == f"Distance = 8 ± {line.half_width:.0f} m (90 %)"
    assert line.half_width == pytest.approx(1.5 * card_line("d", 10.0, [11.0, 13.0, 12.0, 12.0]).half_width)


def test_nested_estimates_need_the_quadratic_doses(rng):
    with pytest.raises(ValueError, match="doses"):
        nested_estimates([np.zeros((10, 2))], smooth_model(0.1), KpiConfig(), rng, doses=(1.0, 3.0))


def test_card_without_error_returns_the_exact_kpi(rng):
    cfg = KpiConfig()
    track = line_track(6.5, 60.0)  # 6.5 m/s is above the 20 km/h HSR threshold
    card = kpi_card([track], SpectralErrorModel("none"), cfg, rng, n_draws=5)
    assert card["distance_m"].estimate == pytest.approx(6.5 * 60.0, rel=1e-9)
    assert card["hi_distance_m"].estimate == pytest.approx(6.5 * 60.0, rel=1e-9)
    assert card["distance_m"].half_width == 0.0
    assert "minutes" not in card


def test_bias_correction_moves_a_ghost_distance_towards_zero_but_not_all_the_way(rng):
    """A motionless player: noise creates distance; one extra dose estimates only part of it (σ regime)."""
    cfg = KpiConfig()
    model = smooth_model(0.3)
    observed = np.zeros((25 * 300 + 1, 2)) + sample_player_errors(model, 25 * 300 + 1, 1, rng)[:, 0]
    card = kpi_card([observed], model, cfg, np.random.default_rng(1), n_draws=20)
    linear = kpi_card([observed], model, cfg, np.random.default_rng(1), n_draws=20, method="linear")
    d = card["distance_m"]
    assert d.observed > 50.0
    assert 0.0 < d.estimate < linear["distance_m"].estimate < 0.8 * d.observed


def test_cards_on_synthetic_runs_with_known_kpis(rng):
    """Players along curved paths at varying speeds (clean KPIs known).

    Distance: the corrected interval covers the clean value and the naive one does not (the bias is several
    standard deviations). Distance and HI distance: the bias correction removes most of the bias. The error
    (σ = 0.3 m) is large enough for the HI-distance bias to stand out from the noise of a mean over 16
    players; the card draws from its own generator, so the tracks do not depend on its cost settings.
    """
    cfg = KpiConfig()
    model = smooth_model(0.3)
    card_rng = rng.spawn(1)[0]
    rows = []
    for _ in range(16):
        n = 25 * 120 + 1
        knots = rng.uniform(1.0, 7.5, size=13)
        speed = np.interp(np.arange(n) / 25, np.linspace(0, 120, 13), knots)
        heading = np.cumsum(rng.normal(0, 0.02, n))
        step = speed / 25
        clean = np.column_stack([np.cumsum(step * np.cos(heading)), np.cumsum(step * np.sin(heading))])
        truth = compute_kpis([pipeline(clean, cfg, resample_to=cfg.output_hz)], cfg)
        observed = clean + sample_player_errors(model, n, 1, rng)[:, 0]
        card = kpi_card([observed], model, cfg, card_rng, n_draws=30)
        rows.append({k: (truth[k], card[k]) for k in ("distance_m", "hi_distance_m")})
    dist = [r["distance_m"] for r in rows]
    assert np.mean([c.covers(t) for t, c in dist]) >= 0.8
    assert np.mean([c.covers(t, naive=True) for t, c in dist]) <= 0.2
    for k in ("distance_m", "hi_distance_m"):
        obs_err = np.mean([c.observed - t for t, c in (r[k] for r in rows)])
        est_err = np.mean([c.estimate - t for t, c in (r[k] for r in rows)])
        assert obs_err > 0
        assert abs(est_err) < 0.5 * obs_err


def test_half_width_follows_the_spread_of_the_estimate_for_threshold_kpis(rng):
    """Repeated observed tracks of one clean track whose speed crosses the 20 km/h threshold every 7.5 s.
    The SD of the quadratic estimate across them is what s must reflect. The spread of one dose (the previous
    s) is about half of it for these KPIs, because the extrapolation amplifies the observed noise of a KPI
    that jumps with the track; the nested bootstrap is close or conservative."""
    cfg = KpiConfig()
    model = smooth_model(0.3)
    n = 25 * 120 + 1
    speed = 5.56 + 1.2 * np.sin(2.0 * np.pi * np.arange(n) / 25 / 15.0)
    clean = np.column_stack([np.cumsum(speed) / 25, np.zeros(n)])
    kpis = ("hsr_efforts", "hi_distance_m")
    est: dict[str, list[float]] = {k: [] for k in kpis}
    s_nested: dict[str, list[float]] = {k: [] for k in kpis}
    s_spread: dict[str, list[float]] = {k: [] for k in kpis}
    for _ in range(48):
        observed = clean + sample_player_errors(model, n, 1, rng)[:, 0]
        obs, draws = kpi_draws([observed], model, cfg, rng, n_draws=10)
        nested = nested_estimates([observed], model, cfg, rng, n_outer=6, n_inner=6)
        for k in kpis:
            line = card_line(k, obs[k], draws[1.0][k], draws2=draws[2.0][k], outer=nested["quadratic"][k])
            spread = card_line(k, obs[k], draws[1.0][k], draws2=draws[2.0][k])
            est[k].append(line.raw_estimate)
            s_nested[k].append(line.sd)
            s_spread[k].append(spread.sd)
    for k in kpis:
        sd = float(np.std(est[k], ddof=1))
        assert np.mean(s_spread[k]) < 0.8 * sd
        assert 0.75 * sd < np.mean(s_nested[k]) < 2.0 * sd
