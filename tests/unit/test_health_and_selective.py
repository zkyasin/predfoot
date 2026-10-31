import numpy as np
import pytest

from predfoot.calib.solver import SolverConfig, calibrate
from predfoot.eval.selective import aurc, removed_at, risk_coverage
from predfoot.geometry.camera import look_at
from predfoot.geometry.pitch import pnlcalib_keypoints
from predfoot.health.features import FEATURES, _hull_area, calibration_features, calibration_jump_px

PLANAR = np.array([(x, y) for x, y, planar in pnlcalib_keypoints().values() if planar])


def frame(cam, rng, sigma=1.0):
    uv, depth = cam.project(np.column_stack([PLANAR, np.zeros(len(PLANAR))]))
    keep = cam.in_image(uv, depth, margin=5)
    return PLANAR[keep], uv[keep] + rng.normal(scale=sigma, size=(keep.sum(), 2))


def test_hull_area():
    assert _hull_area(np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]])) == pytest.approx(4.0)
    assert _hull_area(np.array([[0, 0], [1, 1]])) == 0.0


def test_jump_is_zero_for_the_same_camera_and_grows_with_motion():
    cam = look_at([0.0, -55.0, 20.0], [0.0, 0.0, 0.0], 1800.0)
    moved = look_at([0.0, -55.0, 20.0], [3.0, 0.0, 0.0], 1800.0)
    assert calibration_jump_px(cam.H, cam.H, (1920, 1080)) == pytest.approx(0.0, abs=1e-6)
    assert calibration_jump_px(moved.H, cam.H, (1920, 1080)) > 20


def test_features_of_a_good_and_a_failed_frame(rng):
    cam = look_at([0.0, -55.0, 20.0], [5.0, 0.0, 0.0], 1800.0)
    xy, uv = frame(cam, rng)
    good, _ = calibrate(xy, uv, rng, config=SolverConfig(ransac_iterations=60))
    f = calibration_features(good, uv, n_kp=len(uv), n_lines=0, kp_conf=np.full(len(uv), 0.9), H_prev=cam.H)
    assert set(f) == set(FEATURES)
    assert np.isfinite([f[k] for k in FEATURES]).all()
    failed, _ = calibrate(xy[:3], uv[:3], rng)
    g = calibration_features(failed, uv[:3], n_kp=3, n_lines=0, kp_conf=np.full(3, 0.5))
    assert np.isnan(g["rms_px"])
    assert g["n_kp"] == 3


def test_risk_coverage_and_aurc():
    risk = np.array([0.1, 0.9, 0.2, 0.8])
    loss = np.array([0.0, 1.0, 0.0, 1.0])
    cov, kept = risk_coverage(risk, loss)
    np.testing.assert_allclose(cov, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(kept, [0.0, 0.0, 1 / 3, 0.5])
    assert aurc(risk, loss) < aurc(-risk, loss)  # a good ranking has a lower area
    assert removed_at(risk, loss.astype(bool), 0.5) == 1.0
    assert removed_at(-risk, loss.astype(bool), 0.5) == 0.0
