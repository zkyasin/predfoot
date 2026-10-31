import numpy as np

from predfoot.geometry import homography as hg
from predfoot.geometry import ptz
from predfoot.geometry.camera import look_at, random_broadcast_camera


def _phi_of(cam):
    return np.array([*ptz.angles(cam.R), np.log(cam.K[0, 0])])


def assert_same_h(a, b, rtol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * np.abs(b).max())


def test_rotation_round_trips_through_angles(rng):
    for _ in range(20):
        pan, tilt, roll = rng.uniform(-1.2, 1.2), rng.uniform(-1.0, 1.0), rng.uniform(-0.3, 0.3)
        r = ptz.rotation(pan, tilt, roll)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1.0)
        np.testing.assert_allclose(ptz.angles(r), [pan, tilt, roll], atol=1e-12)


def test_angle_signs_follow_the_convention():
    forward = lambda r: r[2]  # noqa: E731  (third row: optical axis in world coordinates)
    np.testing.assert_allclose(forward(ptz.rotation(0.0, 0.0, 0.0)), [0.0, 1.0, 0.0], atol=1e-15)
    assert forward(ptz.rotation(0.2, 0.0, 0.0))[0] > 0  # pan > 0 turns right (towards +x)
    assert forward(ptz.rotation(0.0, 0.2, 0.0))[2] < 0  # tilt > 0 looks down


def test_homography_matches_look_at_for_random_cameras(rng):
    for _ in range(20):
        centre = np.array([rng.uniform(-10, 10), rng.uniform(-90, -45), rng.uniform(8, 35)])
        target = np.array([rng.uniform(-50, 50), rng.uniform(-30, 30), 0.0])
        f = rng.uniform(900, 5000)
        cam = look_at(centre, target, f, roll_deg=rng.uniform(-2, 2))
        assert_same_h(ptz.homography(_phi_of(cam), centre), cam.H)
        assert_same_h(hg.h_from_theta(ptz.theta(_phi_of(cam), centre)), cam.H)


def test_unnormalised_homography_sees_the_centre_spot_in_front(rng):
    cam = random_broadcast_camera(rng)
    phi = _phi_of(cam)
    k = np.diag([np.exp(phi[3]), np.exp(phi[3]), 1.0])
    k[:2, 2] = ptz.PP_1080P
    raw = k @ ptz.rotation(*phi[:3]) @ np.array([[1, 0, -cam.C[0]], [0, 1, -cam.C[1]], [0, 0, -cam.C[2]]])
    assert raw[2, 2] > 0
    assert np.isclose(raw[2, 2], (cam.R @ (np.zeros(3) - cam.C))[2])


def test_decompose_recovers_pan_tilt_roll_zoom_and_centre(rng):
    for _ in range(50):
        cam = random_broadcast_camera(rng)
        out = ptz.decompose(cam.H)
        assert out is not None
        phi, centre = out
        np.testing.assert_allclose(phi, _phi_of(cam), atol=1e-6)
        np.testing.assert_allclose(centre, cam.C, atol=1e-6)


def test_phi_given_centre_inverts_the_homography(rng):
    for _ in range(20):
        cam = random_broadcast_camera(rng)
        phi = _phi_of(cam)
        np.testing.assert_allclose(ptz.phi_given_centre(ptz.homography(phi, cam.C), cam.C), phi, atol=1e-8)


def test_batched_thetas_match_single_frames(rng):
    cam = random_broadcast_camera(rng)
    phis = _phi_of(cam) + rng.normal(scale=0.05, size=(6, 4))
    batch = ptz.thetas(phis, cam.C)
    for p, t in zip(phis, batch, strict=True):
        np.testing.assert_allclose(t, ptz.theta(p, cam.C), rtol=1e-12, atol=1e-15)


def test_jacobian_agrees_with_a_smaller_step(rng):
    for _ in range(5):
        cam = random_broadcast_camera(rng)
        phi = _phi_of(cam)
        a = ptz.jacobian_theta_phi(phi, cam.C)
        b = ptz.jacobian_theta_phi(phi, cam.C, step=1e-7)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5 * np.abs(a).max())


def test_an_aspect_error_is_off_model_once_the_centre_is_fixed(rng):
    """A 5 % aspect error is not a PTZ camera of the clip's centre.

    With a free centre one frame cannot tell: the decomposition absorbs the aspect error by moving C by about
    a metre (exactly so at zero pan and roll) and reproduces the homography within ~2 px. With the centre
    fixed, the 4 PTZ parameters leave residuals above 5 px, which the clip-level misfit gate sees.
    """
    from scipy.optimize import least_squares

    grid = np.array([[u, v] for u in np.linspace(100, 1820, 6) for v in np.linspace(500, 1060, 4)])
    for _ in range(10):
        cam = random_broadcast_camera(rng)
        k = cam.K.copy()
        k[0, 0] *= 1.05
        h = hg.normalise(k @ np.column_stack([cam.R[:, 0], cam.R[:, 1], -cam.R @ cam.C]))
        ground, valid = hg.backproject(h, grid)
        target = grid[valid]

        def residuals(p, ground=ground[valid], target=target, centre=cam.C):
            return (hg.project(ptz.homography(p, centre), ground) - target).ravel()

        fit = least_squares(residuals, ptz.phi_given_centre(h, cam.C))
        assert np.abs(fit.fun).max() > 5.0
        exact = least_squares(residuals, _phi_of(cam) + 0.01)
        assert np.abs(exact.fun).max() > 5.0  # the true φ is no better: the aspect error is the cause
        out = ptz.decompose(h)
        if out is not None:  # a free centre absorbs it
            moved = hg.project(ptz.homography(*out), ground[valid])
            assert np.abs(moved - target).max() < np.abs(fit.fun).max()
