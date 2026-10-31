import numpy as np
import pytest

from predfoot.kpi.errors import (
    ErrorModel,
    ar1_noise,
    fit_ar1,
    ghost_distance_per_min,
    sample_errors,
    split_common_mode,
)
from predfoot.kpi.metrics import path_length


def lag1(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def test_iid_noise_has_the_requested_sigma_and_no_memory(rng):
    e = ar1_noise(200_000, sigma=0.3, rho=0.0, rng=rng)
    assert e.std() == pytest.approx(0.3, rel=0.01)
    assert abs(lag1(e)) < 0.01


@pytest.mark.parametrize("rho", [0.9, 0.99])
def test_ar1_noise_is_stationary_with_the_requested_marginal_sigma_and_correlation(rng, rho):
    e = ar1_noise(1_000_000, sigma=0.2, rho=rho, rng=rng)
    assert e.std() == pytest.approx(0.2, rel=0.03)
    assert lag1(e) == pytest.approx(rho, abs=0.005)
    # Stationary from the first sample: no burn-in needed.
    starts = ar1_noise(2, sigma=0.2, rho=rho, rng=rng, shape=(20_000,))[0]
    assert starts.std() == pytest.approx(0.2, rel=0.03)


def test_fit_ar1_recovers_the_parameters(rng):
    e = ar1_noise(300_000, sigma=0.15, rho=0.95, rng=rng)
    sigma, rho = fit_ar1(e)
    assert sigma == pytest.approx(0.15, rel=0.03)
    assert rho == pytest.approx(0.95, abs=0.005)


def test_common_mode_is_shared_by_all_players_and_variances_add(rng):
    model = ErrorModel("mix", sigma=0.1, rho=0.5, sigma_common=0.2, rho_common=0.99)
    assert model.total_sigma == pytest.approx(np.hypot(0.1, 0.2))
    e = sample_errors(model, n_frames=50_000, n_players=40, rng=rng)
    assert e.shape == (50_000, 40, 2)
    assert e.std() == pytest.approx(model.total_sigma, rel=0.05)
    common, individual = split_common_mode(e)
    assert common.shape == (50_000, 2)
    assert individual.std() == pytest.approx(0.1, rel=0.05)
    assert fit_ar1(common[:, 0])[1] == pytest.approx(0.99, abs=0.01)


def test_errors_are_reproducible_from_the_seed():
    model = ErrorModel("ar1", sigma=0.1, rho=0.9)
    a = sample_errors(model, 100, 3, np.random.default_rng(7))
    b = sample_errors(model, 100, 3, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert not np.any(sample_errors(ErrorModel("none"), 10, 2, np.random.default_rng(7)))


@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_ghost_distance_closed_form_for_correlated_noise(rng, rho):
    hz, sigma = 25, 0.1
    noise = ar1_noise(15_001, sigma=sigma, rho=rho, rng=rng, shape=(2,))
    assert path_length(noise) / 10 == pytest.approx(ghost_distance_per_min(sigma, hz, rho), rel=0.03)
