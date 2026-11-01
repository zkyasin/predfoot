import numpy as np
import pytest

from predfoot.kpi.proxy import PROXY_DT_S, ProxyTotals, run_kinematics


def line(speed_ms: float, n: int, dt: float = PROXY_DT_S) -> np.ndarray:
    return np.column_stack([speed_ms * dt * np.arange(n), np.zeros(n)])


def test_run_kinematics_is_exact_on_a_constant_speed_line():
    """Savitzky-Golay of order 2 reproduces a straight line, so distance and speed are exact."""
    k = run_kinematics(line(6.0, 30))
    assert k.distance_m == pytest.approx(6.0 * PROXY_DT_S * 29, rel=1e-12)
    assert k.hsr_distance_m == pytest.approx(k.distance_m, rel=1e-12)  # 6 m/s is above 5.56 m/s
    np.testing.assert_allclose(k.speed_ms, 6.0, rtol=1e-10)
    slow = run_kinematics(line(5.0, 30))
    assert slow.hsr_distance_m == 0.0


def test_run_kinematics_refuses_runs_shorter_than_the_window():
    with pytest.raises(ValueError, match="at least"):
        run_kinematics(line(1.0, 4))


def test_proxy_totals_measure_the_bias_of_a_noisy_track(rng):
    """White noise inflates the distance of a slow walker; a perfect track has no bias."""
    perfect, noisy = ProxyTotals(), ProxyTotals()
    for _ in range(50):
        ref = line(1.5, 40)
        perfect.add(ref, ref)
        noisy.add(ref + 0.3 * rng.standard_normal(ref.shape), ref)
    p, q = perfect.summary(), noisy.summary()
    assert p["distance_bias"] == pytest.approx(0.0, abs=1e-12)
    assert p["speed_rms"] == pytest.approx(0.0, abs=1e-9)
    assert p["ref_mean_speed_ms"] == pytest.approx(1.5, rel=1e-9)
    assert p["runs"] == 50
    assert p["samples"] == 50 * 40
    assert q["distance_bias"] > 0.2
    assert q["speed_rms"] > 0.5
