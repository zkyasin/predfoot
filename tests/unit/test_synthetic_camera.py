import numpy as np
import pytest

from predfoot.pipeline.synthetic import look_at_homography, project_ground

CAMERA = (0.0, -45.0, 18.0)
SIZE = (1920, 1080)


def _pinhole(camera, target, focal, size, xyz):
    """Independent projection of 3D points, written out by hand."""
    c, t = np.asarray(camera, float), np.asarray(target, float)
    fwd = (t - c) / np.linalg.norm(t - c)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    d = np.asarray(xyz, float) - c
    depth = d @ fwd
    u = (size[0] - 1) / 2 + focal * (d @ right) / depth
    v = (size[1] - 1) / 2 + focal * (d @ down) / depth
    return np.stack([u, v], axis=-1), depth


def test_homography_matches_a_pinhole_projection_of_ground_points(rng):
    target = (5.0, 3.0, 0.0)
    H = look_at_homography(CAMERA, target, focal_px=1800.0, image_size=SIZE)
    xy = rng.uniform([-52.5, -34.0], [52.5, 34.0], size=(200, 2))
    uv, w = project_ground(H, xy)
    expected, depth = _pinhole(CAMERA, target, 1800.0, SIZE, np.column_stack([xy, np.zeros(len(xy))]))
    np.testing.assert_allclose(uv, expected, atol=1e-8)
    assert np.all((w > 0) == (depth > 0))


def test_homography_is_normalised_and_sees_the_centre_spot_in_front():
    H = look_at_homography(CAMERA, (0.0, 0.0, 0.0), focal_px=1800.0, image_size=SIZE)
    assert H[2, 2] == pytest.approx(1.0)
    uv, w = project_ground(H, [[0.0, 0.0]])
    np.testing.assert_allclose(uv[0], [959.5, 539.5], atol=1e-9)  # the target lands on the image centre
    assert w[0] > 0


def test_right_hand_goal_is_on_the_right_and_far_side_is_up():
    H = look_at_homography(CAMERA, (0.0, 0.0, 0.0), focal_px=1800.0, image_size=SIZE)
    uv, _ = project_ground(H, [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
    assert uv[1, 0] > uv[0, 0]  # +x → image right
    assert uv[2, 1] < uv[0, 1]  # +y (far touchline) → image up


def test_camera_cannot_look_straight_down():
    with pytest.raises(ValueError, match="vertical"):
        look_at_homography((0.0, 0.0, 20.0), (0.0, 0.0, 0.0), focal_px=1800.0, image_size=SIZE)
