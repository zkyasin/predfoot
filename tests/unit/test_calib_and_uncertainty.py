import numpy as np
import pytest
from scipy.stats import chi2

from predfoot.calib.solver import SolverConfig, calibrate
from predfoot.contracts import CalibStatus
from predfoot.geometry import homography as hg
from predfoot.geometry.camera import look_at, random_broadcast_camera
from predfoot.geometry.pitch import pnlcalib_keypoints
from predfoot.uncertainty.delta import (
    CHI2_2_90,
    circumscribed_radius,
    ellipse_axes,
    mahalanobis2,
    position_covariance,
    region_quantile,
)

PLANAR_KP = np.array([(x, y) for x, y, planar in pnlcalib_keypoints().values() if planar])
FAST = SolverConfig(ransac_iterations=60)


def simulated_frame(cam, rng, sigma_px, n_extra=12, min_points=8):
    """Visible template keypoints plus extra ground points (like line samples), with Gaussian pixel noise."""
    pts = np.vstack([PLANAR_KP, rng.uniform([-52, -34], [52, 34], size=(n_extra, 2))])
    uv, depth = cam.project(np.column_stack([pts, np.zeros(len(pts))]))
    keep = cam.in_image(uv, depth, margin=5)
    if keep.sum() < min_points:
        return None
    return pts[keep], uv[keep] + rng.normal(scale=sigma_px, size=(keep.sum(), 2))


def test_exact_recovery_without_noise(rng):
    cam = look_at([0.0, -60.0, 20.0], [5.0, 0.0, 0.0], 1800.0)
    xy, uv = simulated_frame(cam, rng, sigma_px=0.0)
    result, inliers = calibrate(xy, uv, rng, config=FAST)
    assert result.status is CalibStatus.OK
    assert inliers.all()
    np.testing.assert_allclose(result.H, cam.H, rtol=1e-6, atol=1e-9)


def test_robust_to_twenty_percent_outliers(rng):
    cam = look_at([0.0, -55.0, 18.0], [-10.0, 5.0, 0.0], 1600.0)
    xy, uv = simulated_frame(cam, rng, sigma_px=1.0, n_extra=30)
    bad = rng.random(len(uv)) < 0.2
    uv_bad = uv.copy()
    uv_bad[bad] += rng.uniform(40, 200, size=(bad.sum(), 2)) * rng.choice([-1, 1], size=(bad.sum(), 2))
    result, inliers = calibrate(xy, uv_bad, rng)
    assert result.ok
    assert not (inliers & bad).any()
    probe = rng.uniform([-40, -25], [40, 25], size=(50, 2))
    err = np.linalg.norm(hg.project(result.H, probe) - hg.project(cam.H, probe), axis=1)
    assert np.median(err) < 1.5


def test_too_few_points_abstain(rng):
    cam = look_at([0.0, -60.0, 20.0], [5.0, 0.0, 0.0], 1800.0)
    xy, uv = simulated_frame(cam, rng, sigma_px=1.0)
    result, _ = calibrate(xy[:4], uv[:4], rng)
    assert result.status is CalibStatus.FEW_POINTS
    assert result.H is None


def test_collinear_points_never_return_nan(rng):
    xy = np.column_stack([np.linspace(-30, 30, 8), np.zeros(8)])  # all on the halfway... on y = 0
    uv = np.column_stack([np.linspace(100, 1800, 8), np.full(8, 500.0)])
    result, _ = calibrate(xy, uv, rng)
    assert result.status is not CalibStatus.OK
    if result.H is not None:
        assert np.isfinite(result.H).all()


def _coverage(n_frames, rng, sigma_px=1.5):
    theta_hits, pos_hits = [], []
    cfg = SolverConfig(ransac_iterations=60, sigma0_px=sigma_px)
    while len(theta_hits) < n_frames:
        cam = random_broadcast_camera(rng)
        frame = simulated_frame(cam, rng, sigma_px)
        if frame is None:
            continue
        result, inliers = calibrate(*frame, rng, config=cfg)
        if not result.ok or not inliers.all():
            continue
        err = hg.theta_from_h(result.H) - hg.theta_from_h(cam.H)
        theta_hits.append(err @ np.linalg.solve(result.cov_theta, err) <= region_quantile(8, 0.9, result.dof))
        players = rng.uniform([-50, -32], [50, 32], size=(40, 2))
        uv, depth = cam.project(np.column_stack([players, np.zeros(len(players))]))
        seen = cam.in_image(uv, depth)
        if seen.sum() == 0:
            continue
        xy_hat, cov, valid = position_covariance(result.H, result.cov_theta, uv[seen])
        d2 = mahalanobis2(xy_hat[valid] - players[seen][valid], cov[valid])
        pos_hits.extend(d2 <= region_quantile(2, 0.9, result.dof))
    return np.mean(theta_hits), np.mean(pos_hits)


def test_laplace_covariance_is_calibrated_quick(rng):
    theta_cov, pos_cov = _coverage(150, rng)
    assert theta_cov == pytest.approx(0.90, abs=0.07)
    assert pos_cov == pytest.approx(0.90, abs=0.07)


@pytest.mark.slow
def test_laplace_covariance_is_calibrated_long(rng):
    theta_cov, pos_cov = _coverage(1000, rng)
    assert theta_cov == pytest.approx(0.90, abs=0.025)
    assert pos_cov == pytest.approx(0.90, abs=0.025)


def test_ellipse_helpers():
    cov = np.array([[4.0, 0.0], [0.0, 1.0]])
    major, minor, angle = ellipse_axes(cov)
    assert major == pytest.approx(2 * np.sqrt(CHI2_2_90))
    assert minor == pytest.approx(np.sqrt(CHI2_2_90))
    assert abs(np.cos(angle)) == pytest.approx(1.0)
    assert circumscribed_radius(cov) == pytest.approx(major)
    assert mahalanobis2([2.0, 0.0], cov) == pytest.approx(1.0)


def test_foot_point_noise_adds_to_the_covariance(rng):
    cam = look_at([0.0, -60.0, 20.0], [5.0, 0.0, 0.0], 1800.0)
    xy, uv = simulated_frame(cam, rng, sigma_px=1.0)
    result, _ = calibrate(xy, uv, rng, config=FAST)
    feet = hg.project(cam.H, [[0.0, 20.0]])
    _, cov_gt, _ = position_covariance(result.H, result.cov_theta, feet)
    _, cov_det, _ = position_covariance(result.H, result.cov_theta, feet, cov_uv=np.diag([4.0, 9.0]))
    assert np.trace(cov_det[0]) > np.trace(cov_gt[0])


def test_region_quantile():
    assert region_quantile(2, 0.9) == pytest.approx(CHI2_2_90)
    assert region_quantile(2, 0.9, dof=1e9) == pytest.approx(CHI2_2_90, rel=1e-4)
    assert region_quantile(2, 0.9, dof=20) > CHI2_2_90  # an estimated scale widens the region
    assert region_quantile(8, 0.9) == pytest.approx(chi2.ppf(0.9, 8))


def test_lines_constrain_a_sparse_frame(rng):
    from predfoot.calib.solver import LineObservation
    from predfoot.geometry.pitch import pnlcalib_lines

    cam = look_at([0.0, -50.0, 18.0], [-5.0, 10.0, 0.0], 2200.0)
    xy, uv = simulated_frame(cam, rng, sigma_px=1.0, n_extra=0, min_points=4)
    keep = slice(0, 5)  # only five keypoints: a weak frame
    lines = []
    for seg in pnlcalib_lines().values():
        if not seg.planar:
            continue
        t = np.linspace(0.05, 0.95, 12)[:, None]
        pts = np.add(seg.a[:2], t * np.subtract(seg.b[:2], seg.a[:2]))
        puv, depth = cam.project(np.column_stack([pts, np.zeros(len(pts))]))
        vis = cam.in_image(puv, depth)
        if vis.sum() >= 2:
            ends = puv[vis][[0, -1]] + rng.normal(scale=1.0, size=(2, 2))
            lines.append(LineObservation(seg.a[:2], seg.b[:2], ends))
    assert len(lines) >= 3
    probe = rng.uniform([-40, -25], [30, 25], size=(200, 2))
    truth = hg.project(cam.H, probe)

    def mean_err(res):
        return np.median(np.linalg.norm(hg.project(res.H, probe) - truth, axis=1))

    without, _ = calibrate(xy[keep], uv[keep], np.random.default_rng(1), config=FAST)
    with_lines, _ = calibrate(xy[keep], uv[keep], np.random.default_rng(1), config=FAST, lines=lines)
    assert with_lines.ok
    assert with_lines.dof > (without.dof or 0)
    if without.ok:
        assert mean_err(with_lines) < mean_err(without)
