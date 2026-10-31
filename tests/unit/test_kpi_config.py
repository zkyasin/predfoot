import pydantic
import pytest

from predfoot.kpi.config import KpiConfig, load_kpi_config


def test_the_shipped_config_holds_the_published_thresholds():
    cfg = load_kpi_config()
    assert (cfg.hsr_kmh, cfg.sprint_kmh) == (20.0, 25.0)
    ea = cfg.explosive_acceleration
    assert (ea.accel_ms2, ea.min_duration_s, ea.start_below_kmh) == (3.0, 0.7, 9.0)
    assert (cfg.psv99.activity_kmh, cfg.psv99.percentile) == (15.0, 99.0)
    assert (cfg.native_hz, cfg.output_hz) == (25, 10)
    # The assumptions default to the values documented in the planning (EXECUTION.md §8).
    assert cfg.min_effort_s == 1.0
    assert (cfg.smoothing.method, cfg.smoothing.window_s) == ("savgol", 1.0)
    assert cfg.psv99.statistic == "samples"
    assert cfg == KpiConfig()


def test_invalid_configs_are_rejected(tmp_path):
    with pytest.raises(pydantic.ValidationError, match="sprint_kmh"):
        KpiConfig(hsr_kmh=25.0, sprint_kmh=20.0)
    with pytest.raises(pydantic.ValidationError, match="band_edges_kmh"):
        KpiConfig(band_edges_kmh=(20.0, 15.0))
    bad = tmp_path / "kpi.yaml"
    bad.write_text("hsr_kmh: 20\nunknown_key: 1\n", encoding="utf-8")
    with pytest.raises(pydantic.ValidationError, match="unknown_key"):
        load_kpi_config(bad)
