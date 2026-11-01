import numpy as np
import polars as pl
import pytest

from predfoot.kpi.config import KpiConfig
from predfoot.kpi.simulate import period_tracks, player_kpis, seeded_rng


def table() -> pl.DataFrame:
    """Two periods; player b joins late in period 1; a runs at 3 m/s along x."""
    rows = []
    for period, f0 in ((1, 10_000), (2, 100_000)):
        for k in range(250):
            rows.append({"frame": f0 + k, "period": period, "player_id": "a", "x": 3.0 * k / 25, "y": 0.0})
            if period == 2 or k >= 100:
                rows.append({"frame": f0 + k, "period": period, "player_id": "b", "x": 0.0, "y": 1.0})
    return pl.DataFrame(rows)


def test_period_tracks_put_players_on_the_period_grid():
    tracks, dropped, total = period_tracks(table(), KpiConfig())
    assert [t.period for t in tracks] == [1, 2]
    assert [t.frame0 for t in tracks] == [10_000, 100_000]
    assert tracks[0].player_ids == ("a", "b")
    assert tracks[0].xy.shape == (250, 2, 2)
    assert tracks[0].span(1) == (100, 250)
    assert np.isnan(tracks[0].xy[:100, 1]).all()
    assert (dropped, total) == (0, 2 * 250 + 150 + 250)


def test_player_kpis_with_and_without_errors(rng):
    cfg = KpiConfig()
    tracks, _, _ = period_tracks(table(), cfg)
    clean = player_kpis(tracks, cfg, resample_to=cfg.output_hz)
    assert set(clean) == {"a", "b"}
    # 250 native frames give 100 samples at 10 fps, the last one at 9.9 s.
    assert clean["a"]["distance_m"] == pytest.approx(2 * 3.0 * 9.9, rel=1e-9)
    assert clean["b"]["distance_m"] == pytest.approx(0.0, abs=1e-9)
    noise = [0.2 * rng.standard_normal(t.xy.shape) for t in tracks]
    noisy = player_kpis(tracks, cfg, resample_to=cfg.output_hz, errors=noise, players={"b"})
    assert set(noisy) == {"b"}
    assert noisy["b"]["distance_m"] > 1.0  # noise makes a motionless player "run"


def test_seeded_rng_is_stable_and_key_dependent():
    a = seeded_rng(1, "J03WMX", "ptz", 1).standard_normal(3)
    np.testing.assert_array_equal(a, seeded_rng(1, "J03WMX", "ptz", 1).standard_normal(3))
    assert not np.allclose(a, seeded_rng(1, "J03WMX", "ptz", 2).standard_normal(3))
