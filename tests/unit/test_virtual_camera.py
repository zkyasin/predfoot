import cv2
import numpy as np
import pytest

from predfoot.demo import virtual_camera as vc
from predfoot.geometry import homography as hg
from predfoot.geometry import ptz
from predfoot.io.soccertrack import FisheyeCalibration

VIDEO = (960, 480)  # (width, height): a small synthetic panorama


@pytest.fixture(scope="module")
def pano() -> vc.PanoramaCamera:
    """A fisheye 'video' with anisotropic pixels, its rectified pinhole image and the camera of a stand."""
    K = np.array([[340.0, 0, 492], [0, 236, 228], [0, 0, 1]])
    D = np.array([-0.13, 0.06, 0.0, 0.0])
    Knew = np.array([[150.0, 0, 480], [0, 104, 236], [0, 0, 1]])
    mx, my = cv2.fisheye.initUndistortRectifyMap(K, D.reshape(4, 1), np.eye(3), Knew, VIDEO, cv2.CV_32FC1)
    calib = FisheyeCalibration(
        K=K, D=D, Knew=Knew, R=ptz.rotation(0.0, 0.33, 0.01), C=np.array([0.4, -40.5, 13.5]), rms_px=0.0
    )
    return vc.PanoramaCamera(calib, mx.astype(np.float32), my.astype(np.float32))


def view(pan=0.15, tilt=0.3, hfov_deg=40.0, size=(320, 180)) -> vc.VirtualView:
    f = (size[0] / 2) / np.tan(np.radians(hfov_deg) / 2)
    return vc.VirtualView(np.array([pan, tilt, 0.0, np.log(f)]), size)


def test_view_homography_is_the_ptz_camera_at_the_panorama_centre(pano):
    v = view()
    np.testing.assert_allclose(vc.view_homography(v, pano), v.homography(pano.calib.C), rtol=1e-9, atol=1e-9)


def test_rectified_and_video_pixels_round_trip(pano, rng):
    rect = rng.uniform([300, 150], [660, 330], size=(50, 2))
    video = pano.rectified_to_video(rect)
    np.testing.assert_allclose(pano.video_to_rectified(video), rect, atol=0.05)


def test_remap_tables_and_video_to_view_are_inverse(pano, rng):
    v = view()
    tables = vc.remap_tables(v, pano)
    assert tables[0].shape == (180, 320)
    pix = rng.integers([10, 10], [310, 170], size=(40, 2))
    video = np.column_stack([tables[0][pix[:, 1], pix[:, 0]], tables[1][pix[:, 1], pix[:, 0]]])
    inside = (
        (video[:, 0] > 0) & (video[:, 0] < VIDEO[0] - 1) & (video[:, 1] > 0) & (video[:, 1] < VIDEO[1] - 1)
    )
    assert inside.mean() > 0.9
    np.testing.assert_allclose(vc.video_to_view(video[inside], v, pano), pix[inside], atol=0.05)


def test_a_projected_pitch_point_lands_where_the_render_puts_it(pano):
    v = view()
    target = np.array([[6.0, 4.0]])
    expected = hg.project(v.homography(pano.calib.C), target)[0]
    assert 40 < expected[0] < 280
    assert 30 < expected[1] < 150
    q = pano.project_video(target)[0]
    yy, xx = np.mgrid[0 : VIDEO[1], 0 : VIDEO[0]]
    blob = np.exp(-((xx - q[0]) ** 2 + (yy - q[1]) ** 2) / (2 * 1.5**2))
    frame = np.repeat((blob * 255).astype(np.uint8)[..., None], 3, axis=2)
    img = vc.render(frame, vc.remap_tables(v, pano))[..., 0].astype(np.float64)
    win = np.zeros_like(img)
    r = 25
    cx, cy = (round(float(c)) for c in expected)
    win[cy - r : cy + r, cx - r : cx + r] = img[cy - r : cy + r, cx - r : cx + r]
    assert win.max() > 50
    ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    centroid = np.array([(xs * win).sum(), (ys * win).sum()]) / win.sum()
    np.testing.assert_allclose(centroid, expected, atol=0.35)


def test_aim_puts_the_target_on_the_principal_point():
    centre = np.array([0.4, -40.5, 13.5])
    for target in ([0.0, 0.0], [-30.0, 20.0], [45.0, -25.0]):
        pan, tilt = vc.aim(target, centre)
        v = vc.VirtualView(np.array([pan, tilt, 0.0, np.log(2000.0)]))
        np.testing.assert_allclose(hg.project(v.homography(centre), [target]), [v.pp], atol=1e-6)
        assert tilt > 0


def test_follow_action_tracks_the_centroid_and_zooms_out_for_spread_players():
    centre = np.array([0.4, -40.5, 13.5])
    tight = [np.array([[10.0, 5.0], [12.0, 6.0], [11.0, 3.0]])] * 5
    wide = [np.array([[-30.0, 5.0], [30.0, 6.0], [0.0, 3.0]])] * 5
    cfg = vc.FollowConfig(smooth_s=0.0)
    phi_t = vc.follow_action(tight, centre, fps=25, config=cfg)
    phi_w = vc.follow_action(wide, centre, fps=25, config=cfg)
    assert phi_t.shape == (5, 4)
    np.testing.assert_allclose(phi_t[:, 2], 0.0)
    pan, tilt = vc.aim(np.mean(tight[0], axis=0), centre)
    np.testing.assert_allclose(phi_t[:, :2], [[pan, tilt]] * 5)
    assert np.all(phi_t[:, 3] > phi_w[:, 3])  # tighter group → longer focal length
    hfov = np.degrees(2 * np.arctan(960 / np.exp(phi_t[:, 3])))
    np.testing.assert_allclose(hfov, cfg.hfov_deg[0])  # clamped at the narrowest view
    gap = [tight[0], np.zeros((0, 2)), tight[0]]
    assert np.isfinite(vc.follow_action(gap, centre, fps=25)).all()
