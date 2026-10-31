import numpy as np
import pytest

from predfoot.io.conventions import (
    SN_TO_INTERNAL,
    internal_to_sn_xy,
    sn_normalised_to_pixels,
    sn_to_internal_cov,
    sn_to_internal_xy,
)


def test_player_near_main_camera_ends_up_on_the_near_side():
    # SNGS-056, frame 1: a player low in the image (close to the camera) has SoccerNet y = +16.9.
    xy = sn_to_internal_xy([34.61, 16.91])
    assert xy[0] == pytest.approx(34.61)
    assert xy[1] == pytest.approx(-16.91)


def test_conversion_is_an_involution():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-50, 50, size=(100, 2))
    np.testing.assert_allclose(internal_to_sn_xy(sn_to_internal_xy(xy)), xy)


def test_covariance_follows_the_reflection():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(50, 2, 2))
    cov = a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(2)
    expected = SN_TO_INTERNAL @ cov @ SN_TO_INTERNAL.T
    out = sn_to_internal_cov(cov)
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(np.linalg.eigvalsh(out), np.linalg.eigvalsh(cov))


def test_ellipse_orientation_flips():
    cov = np.array([[2.0, 0.8], [0.8, 1.0]])
    tilt = lambda c: np.arctan2(*np.linalg.eigh(c)[1][::-1, -1])  # noqa: E731
    assert np.sign(tilt(sn_to_internal_cov(cov))) == -np.sign(tilt(cov))


def test_inputs_are_not_modified():
    xy = np.array([[1.0, 2.0]])
    sn_to_internal_xy(xy)
    np.testing.assert_array_equal(xy, [[1.0, 2.0]])


def test_normalised_points_scale_to_the_frame():
    pts = sn_normalised_to_pixels([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25]], width=1920, height=1080)
    np.testing.assert_allclose(pts, [[0, 0], [1920, 1080], [960, 270]])
