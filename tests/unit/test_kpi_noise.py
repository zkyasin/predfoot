import math

import numpy as np
import pytest

from predfoot.kpi.errors import split_common_mode
from predfoot.kpi.noise import (
    KERNELS,
    AxisProcess,
    PartProcess,
    SpectralErrorModel,
    fit_axis,
    kernel_corr,
    kernel_curvature,
    ou_through_first_lag,
    sample_axis,
    sample_errors,
    sample_player_errors,
)

LAGS_5FPS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])


def empirical_acf(e: np.ndarray, lag: int) -> float:
    """Pooled lag correlation of (n, paths) samples around the known zero mean."""
    return float(np.mean(e[:-lag] * e[lag:]) / np.mean(e**2))


@pytest.mark.parametrize("kernel", ["matern32", "matern52", "se"])
def test_kernel_curvature_matches_finite_differences(kernel):
    h = 1e-4
    numeric = 2.0 * (1.0 - float(kernel_corr(kernel, h, 0.7))) / h**2
    assert numeric == pytest.approx(kernel_curvature(kernel, 0.7), rel=1e-3)
    assert math.isinf(kernel_curvature("exp", 0.7))


@pytest.mark.parametrize("kernel", KERNELS)
def test_sampled_acf_matches_the_model(rng, kernel):
    proc = AxisProcess(sigma=0.4, kernel=kernel, length_s=0.8, w_nugget=0.3, w_slow=0.2, slow_s=4.0)
    e = sample_axis([proc], n=6000, n_paths=64, rng=rng, hz=25.0)
    assert e.std() == pytest.approx(0.4, rel=0.03)
    for lag in (1, 2, 5, 10, 25):
        assert empirical_acf(e, lag) == pytest.approx(float(proc.acf(lag / 25.0)), abs=0.03)


def test_bandlimited_nugget_is_uncorrelated_at_5fps_lags_but_smooth_at_25hz(rng):
    proc = AxisProcess(sigma=1.0, w_nugget=1.0, nugget="bandlimited", band_hz=2.5)
    e = sample_axis([proc], n=8000, n_paths=32, rng=rng, hz=25.0)
    assert e.std() == pytest.approx(1.0, rel=0.03)
    for lag in (5, 10, 15):  # 0.2, 0.4, 0.6 s
        assert abs(empirical_acf(e, lag)) < 0.03
    assert empirical_acf(e, 1) == pytest.approx(float(np.sinc(0.2)), abs=0.03)
    white = AxisProcess(sigma=1.0, w_nugget=1.0)
    assert abs(empirical_acf(sample_axis([white], 8000, 32, rng), 1)) < 0.02


def test_circulant_embedding_is_exact_from_the_first_sample(rng):
    """Covariance of short series equals the model covariance at every pair of times."""
    proc = AxisProcess(sigma=1.0, kernel="matern52", length_s=0.15, w_nugget=0.2)
    e = sample_axis([proc], n=6, n_paths=200_000, rng=rng, hz=25.0)
    cov = e @ e.T / e.shape[1]
    lags = np.abs(np.subtract.outer(np.arange(6), np.arange(6))) / 25.0
    np.testing.assert_allclose(cov, proc.acf(lags), atol=0.012)


def test_sum_of_processes_adds_covariances(rng):
    a = AxisProcess(sigma=0.3, kernel="matern32", length_s=1.0, w_nugget=0.5)
    b = AxisProcess(sigma=0.4, kernel="exp", length_s=0.2)
    e = sample_axis([a, b], n=5000, n_paths=64, rng=rng)
    assert e.var() == pytest.approx(0.25, rel=0.04)
    want = (0.09 * float(a.acf(0.2)) + 0.16 * float(b.acf(0.2))) / 0.25
    assert empirical_acf(e, 5) == pytest.approx(want, abs=0.03)


def test_matern_paths_have_a_finite_derivative_variance_while_ou_paths_do_not(rng):
    """Variance of finite-difference velocities as the sampling rate grows, over the same 20 s."""
    matern = AxisProcess(sigma=0.5, kernel="matern32", length_s=1.0)
    ou = AxisProcess(sigma=0.5, kernel="exp", length_s=1.0)
    var = {}
    for name, proc in (("matern", matern), ("ou", ou)):
        for hz in (25.0, 100.0, 400.0):
            e = sample_axis([proc], n=int(20 * hz), n_paths=48, rng=rng, hz=hz)
            var[name, hz] = float(np.var(np.diff(e, axis=0) * hz))
    assert matern.derivative_variance() == pytest.approx(3 * 0.25, rel=1e-12)
    assert var["matern", 400.0] == pytest.approx(matern.derivative_variance(), rel=0.1)
    assert var["matern", 400.0] / var["matern", 100.0] == pytest.approx(1.0, abs=0.1)
    assert math.isinf(ou.derivative_variance())
    # OU: Var(Δe/Δt) = 2σ²(1 − e^(−Δt/ℓ))/Δt² ≈ 2σ²/(ℓ Δt), proportional to the sampling rate.
    assert var["ou", 400.0] / var["ou", 100.0] == pytest.approx(4.0, rel=0.1)
    assert math.isinf(AxisProcess(sigma=0.5, w_nugget=0.1).derivative_variance())
    banded = AxisProcess(sigma=0.5, w_nugget=0.1, nugget="bandlimited")
    assert math.isfinite(banded.derivative_variance())


@pytest.mark.parametrize("kernel", ["exp", "matern32", "matern52", "se"])
def test_fit_recovers_an_exact_acf(kernel):
    truth = AxisProcess(sigma=0.5, kernel=kernel, length_s=0.9, w_nugget=0.35)
    fit = fit_axis(0.5, LAGS_5FPS, truth.acf(LAGS_5FPS), np.full(5, 1000), kernel=kernel)
    assert fit.process.w_nugget == pytest.approx(0.35, abs=1e-3)
    assert fit.process.length_s == pytest.approx(0.9, rel=1e-3)
    assert fit.max_abs_residual < 1e-6
    assert fit.process.sigma == 0.5


def test_fit_with_a_slow_term_matches_a_two_scale_acf():
    lags = np.arange(1, 11) * 0.2
    truth = AxisProcess(sigma=0.4, kernel="matern32", length_s=0.4, w_nugget=0.2, w_slow=0.4, slow_s=8.0)
    fit = fit_axis(0.4, lags, truth.acf(lags), kernel="matern32", slow=True)
    assert fit.max_abs_residual < 2e-3
    assert fit.process.w_nugget == pytest.approx(0.2, abs=0.02)


def test_ou_through_first_lag_is_the_legacy_ar1():
    proc = ou_through_first_lag(0.4, 0.57)
    assert float(proc.acf(0.2)) == pytest.approx(0.57)
    assert float(proc.acf(0.04)) == pytest.approx(0.57 ** (1 / 5))
    with pytest.raises(ValueError, match="rho"):
        ou_through_first_lag(0.4, 1.0)


def test_invalid_processes_are_refused():
    with pytest.raises(ValueError, match="kernel"):
        AxisProcess(sigma=0.1, kernel="cauchy")
    with pytest.raises(ValueError, match="exceed"):
        AxisProcess(sigma=0.1, w_nugget=0.7, w_slow=0.5)
    with pytest.raises(ValueError, match="nugget"):
        AxisProcess(sigma=0.1, nugget="pink")


def test_common_part_is_shared_and_player_draws_have_the_summed_variance(rng):
    common = PartProcess(AxisProcess(0.3, length_s=2.0), AxisProcess(0.4, length_s=2.0))
    indiv = PartProcess(AxisProcess(0.1, w_nugget=0.5), AxisProcess(0.2, w_nugget=0.5))
    model = SpectralErrorModel("m", common=common, individual=indiv)
    e = sample_errors(model, n_frames=4000, n_players=30, rng=rng)
    assert e.shape == (4000, 30, 2)
    c, ind = split_common_mode(e)
    assert ind[..., 1].std() == pytest.approx(0.2, rel=0.08)
    assert c[:, 1].std() == pytest.approx(0.4, rel=0.15)
    p = sample_player_errors(model, n_frames=4000, n_draws=40, rng=rng)
    assert p.shape == (4000, 40, 2)
    assert p[..., 0].std() == pytest.approx(math.hypot(0.3, 0.1), rel=0.1)
    a = sample_errors(model, 50, 3, np.random.default_rng(3))
    b = sample_errors(model, 50, 3, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert sample_errors(SpectralErrorModel("none"), 10, 2, rng).any() is np.False_


def test_copula_process_has_the_target_marginal_and_normal_score_acf(rng):
    """Laplace marginal (kurtosis 6): quantiles reproduced, and the normal scores keep the Gaussian ACF."""
    from scipy.special import ndtri
    from scipy.stats import laplace, rankdata

    probs = np.linspace(0.001, 0.999, 201)
    quantiles = laplace.ppf(probs, scale=0.2)
    proc = AxisProcess(
        sigma=float(laplace.std(scale=0.2)),
        kernel="matern32",
        length_s=0.6,
        w_nugget=0.3,
        probs=tuple(probs),
        quantiles=tuple(quantiles),
    )
    assert proc.is_copula
    e = sample_axis([proc], n=5000, n_paths=40, rng=rng)
    np.testing.assert_allclose(
        np.quantile(e, [0.1, 0.5, 0.9, 0.99]), laplace.ppf([0.1, 0.5, 0.9, 0.99], scale=0.2), atol=0.02
    )
    scores = ndtri((rankdata(e.ravel()) - 0.5) / e.size).reshape(e.shape)
    for lag in (1, 5, 10):
        assert empirical_acf(scores, lag) == pytest.approx(float(proc.acf(lag / 25.0)), abs=0.03)
    # A copula process sums with a Gaussian one: variances add.
    g = AxisProcess(sigma=0.1, w_nugget=1.0)
    total = sample_axis([proc, g], n=5000, n_paths=40, rng=rng)
    assert total.var() == pytest.approx(proc.sigma**2 + 0.01, rel=0.08)
    with pytest.raises(ValueError, match="probs"):
        AxisProcess(sigma=0.1, probs=(0.5, 0.2), quantiles=(0.0, 1.0))
