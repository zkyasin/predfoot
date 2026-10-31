import math

import numpy as np
import pytest
from scipy.stats import t as student_t

from predfoot.kpi.card import card_line, kpi_card
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
    half = float(student_t.ppf(0.95, 3)) * sd * math.sqrt(1.25)
    assert line.bias == pytest.approx(2.0)
    assert line.estimate == pytest.approx(8.0)
    assert line.half_width == pytest.approx(half)
    assert line.interval == pytest.approx((8.0 - half, 8.0 + half))
    assert line.covers(8.0)
    assert not line.covers(8.0 + 1.01 * half)
    assert line.covers(10.0 + 0.99 * half, naive=True)
    assert line.text("Distance", "m") == f"Distance = 8 ± {half:.0f} m (90 %)"
    assert math.isnan(card_line("x", 1.0, [2.0]).half_width)  # one draw is not enough


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
    card = kpi_card([observed], model, cfg, rng, n_draws=20)
    d = card["distance_m"]
    assert d.observed > 50.0
    assert 0.0 < d.estimate < 0.8 * d.observed


def test_cards_on_synthetic_runs_with_known_kpis(rng):
    """Players along curved paths at varying speeds (clean KPIs known).

    Distance: the corrected interval covers the clean value and the naive one does not (the bias is several
    standard deviations). HI distance: the bias correction removes most of the bias, but the bootstrap
    spread underestimates the error of a threshold KPI, so its raw interval is too narrow; this is why
    scripts/kpi_card.py also reports a leave-one-match-out width calibration.
    """
    cfg = KpiConfig()
    model = smooth_model(0.15)
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
        card = kpi_card([observed], model, cfg, rng, n_draws=30)
        rows.append({k: (truth[k], card[k]) for k in ("distance_m", "hi_distance_m")})
    dist = [r["distance_m"] for r in rows]
    assert np.mean([c.covers(t) for t, c in dist]) >= 0.8
    assert np.mean([c.covers(t, naive=True) for t, c in dist]) <= 0.2
    for k in ("distance_m", "hi_distance_m"):
        obs_err = np.mean([c.observed - t for t, c in (r[k] for r in rows)])
        est_err = np.mean([c.estimate - t for t, c in (r[k] for r in rows)])
        assert obs_err > 0
        assert abs(est_err) < 0.5 * obs_err
