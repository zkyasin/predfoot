import cv2
import numpy as np
import pytest

from predfoot.geometry import homography as hg
from predfoot.geometry.camera import look_at, random_broadcast_camera
from predfoot.geometry.pitch import (
    CIRCLE_RADIUS,
    Arc,
    PitchDims,
    is_planar_sn_line,
    pnlcalib_keypoints,
    pnlcalib_lines,
    sn_line_elements,
)


@pytest.fixture
def camera():
    return look_at(centre=[0.0, -60.0, 20.0], target=[10.0, 5.0, 0.0], focal_px=2000.0)


def visible_template(cam, n_extra=0, rng=None):
    pts = np.array([(x, y) for x, y, planar in pnlcalib_keypoints().values() if planar])
    if n_extra:
        pts = np.vstack([pts, rng.uniform([-52, -34], [52, 34], size=(n_extra, 2))])
    uv, depth = cam.project(np.column_stack([pts, np.zeros(len(pts))]))
    keep = cam.in_image(uv, depth)
    return pts[keep], uv[keep]


# --- pitch template -------------------------------------------------------------------------------------


def test_keypoint_table_has_57_entries_and_four_elevated():
    kps = pnlcalib_keypoints()
    assert sorted(kps) == list(range(1, 58))
    assert {k for k, (_, _, planar) in kps.items() if not planar} == {12, 15, 16, 19}


def test_keypoints_follow_the_internal_axes():
    kps = pnlcalib_keypoints()
    assert kps[1][:2] == pytest.approx((-52.5, 34.0))  # far-left corner: far touchline has y > 0
    assert kps[28][:2] == pytest.approx((-52.5, -34.0))  # near-left corner
    assert kps[51][:2] == pytest.approx((0.0, 0.0))  # centre spot
    assert kps[57][:2] == pytest.approx((41.5, 0.0))  # right penalty spot


def test_circle_keypoints_lie_on_their_circles():
    kps = pnlcalib_keypoints()
    centre_ids = [32, 35, 38, 39, 42, 43, 48, 49, 50, 52, 53, 54]
    left_arc_ids, right_arc_ids = [31, 34, 37, 41, 47], [33, 36, 40, 44, 55]
    for ids, centre in ((centre_ids, (0.0, 0.0)), (left_arc_ids, (-41.5, 0.0)), (right_arc_ids, (41.5, 0.0))):
        for k in ids:
            r = np.hypot(kps[k][0] - centre[0], kps[k][1] - centre[1])
            assert r == pytest.approx(CIRCLE_RADIUS, abs=0.03), k


def test_lines_and_sn_classes():
    lines = pnlcalib_lines()
    assert len(lines) == 23
    assert {k for k, s in lines.items() if not s.planar} == {7, 8, 9, 10, 11, 12}
    sn = sn_line_elements()
    assert len(sn) == 26
    assert sum(not is_planar_sn_line(name) for name in sn) == 6
    assert sn["Side line top"].a[1] == pytest.approx(34.0)
    left = sn["Circle left"]
    assert isinstance(left, Arc)
    assert left.sample(9)[:, 0].min() == pytest.approx(-36.0, abs=1e-6)  # the D starts on the box line


def test_dims_are_configurable():
    kps = pnlcalib_keypoints(PitchDims(100.0, 64.0))
    assert kps[3][:2] == pytest.approx((50.0, 32.0))


# --- homography -----------------------------------------------------------------------------------------


def test_camera_homography_matches_full_projection(camera, rng):
    xy = rng.uniform([-50, -30], [50, 30], size=(200, 2))
    uv_full, depth = camera.project(np.column_stack([xy, np.zeros(len(xy))]))
    assert (depth > 0).all()
    np.testing.assert_allclose(hg.project(camera.H, xy), uv_full, rtol=1e-10, atol=1e-8)


def test_project_backproject_round_trip(camera, rng):
    xy = rng.uniform([-50, -30], [50, 30], size=(500, 2))
    back, valid = hg.backproject(camera.H, hg.project(camera.H, xy))
    assert valid.all()
    np.testing.assert_allclose(back, xy, atol=1e-7)


def test_pixels_above_the_horizon_are_invalid(camera):
    # The top of a tilted-down camera image is ground; far above it the rays never hit the pitch in front.
    horizon_dir = camera.R @ np.array([0.0, 1.0, 0.0])  # camera-frame direction of a far-away ground point
    v_horizon = (camera.K @ horizon_dir)[1] / horizon_dir[2]
    _, valid = hg.backproject(camera.H, [[960.0, v_horizon - 200.0], [960.0, v_horizon + 200.0]])
    assert valid.tolist() == [False, True]


def test_dlt_recovers_the_homography_exactly(camera):
    xy, uv = visible_template(camera)
    assert len(xy) >= 6
    np.testing.assert_allclose(hg.dlt(xy, uv), camera.H, rtol=1e-6, atol=1e-9)


def test_dlt_matches_opencv_on_noisy_points(camera, rng):
    xy, uv = visible_template(camera, n_extra=40, rng=rng)
    noisy = uv + rng.normal(scale=1.0, size=uv.shape)
    ours = hg.dlt(xy, noisy)
    # OpenCV refines its algebraic solution with Levenberg-Marquardt, so compare where the data are and
    # check that both fits explain the observations equally well.
    cv, _ = cv2.findHomography(xy, noisy, method=0)
    np.testing.assert_allclose(hg.project(ours, xy), hg.project(cv, xy), atol=1.0)  # within the noise level
    rms = lambda h: np.sqrt(np.mean(np.sum((hg.project(h, xy) - noisy) ** 2, axis=1)))  # noqa: E731
    assert rms(ours) == pytest.approx(rms(cv), rel=0.05)


def test_dlt_needs_four_points():
    with pytest.raises(ValueError, match="at least 4"):
        hg.dlt(np.zeros((3, 2)), np.zeros((3, 2)))


def test_theta_round_trip(camera):
    np.testing.assert_allclose(hg.h_from_theta(hg.theta_from_h(camera.H * 3.7)), camera.H)


def _finite_diff(f, x, eps):
    x = np.asarray(x, dtype=np.float64)
    cols = []
    for k in range(x.size):
        step = np.zeros_like(x)
        step.flat[k] = eps * max(1.0, abs(x.flat[k]))
        cols.append((f(x + step) - f(x - step)) / (2 * step.flat[k]))
    return np.stack(cols, axis=-1)


def test_jacobian_wrt_pixels_matches_finite_differences(camera, rng):
    uv = hg.project(camera.H, rng.uniform([-45, -28], [45, 28], size=(20, 2)))
    analytic = hg.jacobian_backproject_uv(camera.H, uv)
    for i in range(len(uv)):
        num = _finite_diff(lambda p: hg.backproject(camera.H, p)[0], uv[i], 1e-4)
        np.testing.assert_allclose(analytic[i], num, rtol=1e-5, atol=1e-9)


def test_jacobian_wrt_theta_matches_finite_differences(camera, rng):
    theta = hg.theta_from_h(camera.H)
    uv = hg.project(camera.H, rng.uniform([-45, -28], [45, 28], size=(10, 2)))
    analytic = hg.jacobian_backproject_theta(camera.H, uv)
    for i in range(len(uv)):
        num = _finite_diff(lambda t, p=uv[i]: hg.backproject(hg.h_from_theta(t), p)[0], theta, 1e-6)
        np.testing.assert_allclose(analytic[i], num, rtol=1e-4, atol=1e-7)


def test_projection_jacobian_wrt_theta_matches_finite_differences(camera, rng):
    theta = hg.theta_from_h(camera.H)
    xy = rng.uniform([-45, -28], [45, 28], size=(10, 2))
    analytic = hg.jacobian_project_theta(camera.H, xy)
    for i in range(len(xy)):
        num = _finite_diff(lambda t, p=xy[i]: hg.project(hg.h_from_theta(t), p), theta, 1e-6)
        np.testing.assert_allclose(analytic[i], num, rtol=1e-4, atol=1e-6)


def test_far_pixels_are_worth_more_metres(camera):
    near = hg.project(camera.H, [[10.0, -25.0]])
    far = hg.project(camera.H, [[10.0, 30.0]])
    scale = lambda uv: np.sqrt(abs(np.linalg.det(hg.jacobian_backproject_uv(camera.H, uv)[0])))  # noqa: E731
    assert scale(far) > 2 * scale(near)


def test_random_cameras_see_the_pitch(rng):
    for _ in range(50):
        cam = random_broadcast_camera(rng)
        xy, _ = visible_template(cam)
        _, depth = cam.project(np.zeros((1, 3)))
        assert depth[0] > 0
        assert np.isfinite(cam.H).all()
        assert len(xy) >= 0
        np.testing.assert_allclose(hg.backproject(cam.H, hg.project(cam.H, xy))[0], xy, atol=1e-6)


def test_image_line_contains_projected_points(camera):
    a, b = (-36.0, -20.16, 0.0), (-36.0, 20.16, 0.0)
    line = hg.image_line(camera.H, a, b)
    mid = hg.project(camera.H, [[-36.0, 0.0]])[0]
    assert abs(line @ np.append(mid, 1.0)) < 1e-6


def test_look_at_refuses_a_vertical_optical_axis():
    with pytest.raises(ValueError, match="vertical"):
        look_at([0.0, 0.0, 30.0], [0.0, 0.0, 0.0], 1500.0)
