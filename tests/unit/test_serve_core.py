"""Numpy cores of the HTTP API: localisation, upload checks, calibration of detections, lazy model loading."""

import io
import threading
import time

import cv2
import numpy as np
import pytest
from scipy.stats import f

from predfoot.calib.solver import SolverConfig
from predfoot.contracts import CalibResult, CalibStatus, FootObs, KeypointObs
from predfoot.geometry import homography as hg
from predfoot.io.skillcorner import image_corners_projection
from predfoot.keypoints.pnlcalib import LineEnds
from predfoot.pipeline.clip import ClipFrame
from predfoot.pipeline.real import real_stages
from predfoot.serve.calibrate import (
    LazyDetector,
    ModelUnavailable,
    calibrate_detections,
    load_pnlcalib,
    solver_config_for,
)
from predfoot.serve.images import ImageRejected, decode_rgb, inspect_image
from predfoot.serve.localize import localize_feet, region_threshold
from predfoot.uncertainty.delta import CHI2_2_90, position_covariance
from tests.unit.serve_fakes import (
    SyntheticDetector,
    broadcast_camera,
    jpeg_with_orientation,
    laplace_covariance,
)

PLAYERS = np.array([[5.0, -10.0], [20.0, 5.0], [30.0, 15.0], [45.0, -30.0]])


def horizon_v(H, u):
    """Row of the horizon at column u: where the back-projected homogeneous scale vanishes."""
    g = np.linalg.inv(H)[2]
    return -(g[0] * u + g[2]) / g[1]


def test_positions_follow_the_geometry():
    cam = broadcast_camera()
    out = localize_feet(cam.H, laplace_covariance(cam), hg.project(cam.H, PLAYERS))
    assert out.valid.all()
    np.testing.assert_allclose(out.xy, PLAYERS, atol=1e-6)


def test_calibrated_covariance_and_radius_describe_the_same_region():
    cam = broadcast_camera()
    cov_theta = laplace_covariance(cam)
    uv = hg.project(cam.H, PLAYERS)
    q = 11.3  # a conformal quantile
    out = localize_feet(cam.H, cov_theta, uv, q=q)
    _, sigma_p, _ = position_covariance(cam.H, cov_theta, uv)
    np.testing.assert_allclose(out.cov, sigma_p * q / CHI2_2_90, rtol=1e-12)
    # the χ²₂(0.90) circle of the calibrated covariance is the q circle of Σp
    np.testing.assert_allclose(out.radius, np.sqrt(CHI2_2_90 * np.linalg.eigvalsh(out.cov)[:, -1]))
    np.testing.assert_allclose(out.radius, np.sqrt(q * np.linalg.eigvalsh(sigma_p)[:, -1]))
    wide = localize_feet(cam.H, cov_theta, uv, q=4 * q)
    np.testing.assert_allclose(wide.radius, 2 * out.radius)


def test_same_definitions_as_the_pipeline_stage():
    cam = broadcast_camera()
    cov_theta = laplace_covariance(cam)
    uv = hg.project(cam.H, PLAYERS)
    stages = real_stages({}, conformal=None, version="test")
    calib = CalibResult(CalibStatus.OK, H=cam.H, cov_theta=cov_theta, n_inliers=25)
    locs = stages.localize(calib, [FootObs(track_id=i, uv=p) for i, p in enumerate(uv)])
    out = localize_feet(cam.H, cov_theta, uv)
    for i, loc in enumerate(locs):
        np.testing.assert_allclose(out.xy[i], loc.xy, rtol=1e-12)
        np.testing.assert_allclose(out.cov[i], loc.cov_xy, rtol=1e-12)
        assert out.radius[i] == pytest.approx(loc.p90_radius_m, rel=1e-12)
    assert stages.calibrate(ClipFrame(frame=0)).status is CalibStatus.FEW_POINTS  # no detections, no camera


def test_pixels_above_the_horizon_have_no_position():
    cam = broadcast_camera()
    v0 = horizon_v(cam.H, 960.0)
    uv = np.array([[960.0, v0 - 50.0], [960.0, v0 + 50.0], [100.0, 900.0]])
    out = localize_feet(cam.H, laplace_covariance(cam), uv)
    np.testing.assert_array_equal(out.valid, [False, True, True])
    assert np.isnan(out.xy[0]).all()
    assert np.isnan(out.cov[0]).all()
    assert np.isnan(out.radius[0])
    assert np.isfinite(out.radius[1:]).all()


def test_foot_pixel_noise_widens_the_regions():
    cam = broadcast_camera()
    cov_theta = laplace_covariance(cam)
    uv = hg.project(cam.H, PLAYERS)
    cov_uv = np.diag([4.0, 9.0])
    exact = localize_feet(cam.H, cov_theta, uv)
    noisy = localize_feet(cam.H, cov_theta, uv, cov_uv=cov_uv)
    j_uv = hg.jacobian_backproject_uv(cam.H, uv)
    np.testing.assert_allclose(noisy.cov - exact.cov, j_uv @ cov_uv @ np.swapaxes(j_uv, 1, 2), rtol=1e-9)
    assert (noisy.radius > exact.radius).all()


def test_one_threshold_per_foot_and_non_finite_ones_invalidate_their_row():
    cam = broadcast_camera()
    cov_theta = laplace_covariance(cam)
    uv = hg.project(cam.H, PLAYERS)
    q = np.array([CHI2_2_90, 11.3, np.inf, 20.0])
    out = localize_feet(cam.H, cov_theta, uv, q=q)
    np.testing.assert_array_equal(out.valid, [True, True, False, True])
    np.testing.assert_array_equal(np.isnan(out.q), [False, False, True, False])
    for i in (0, 1, 3):
        alone = localize_feet(cam.H, cov_theta, uv[i : i + 1], q=q[i])
        np.testing.assert_allclose(out.cov[i], alone.cov[0], rtol=1e-12)
        assert out.radius[i] == pytest.approx(alone.radius[0], rel=1e-12)
        assert out.q[i] == q[i]
    assert np.isnan(out.xy[2]).all()  # a row is never half null


def test_no_feet_no_positions():
    cam = broadcast_camera()
    out = localize_feet(cam.H, laplace_covariance(cam), np.zeros((0, 2)))
    assert out.xy.shape == (0, 2)
    assert out.valid.shape == (0,)


def test_region_threshold_prefers_a_conformal_quantile():
    assert region_threshold() == (CHI2_2_90, "laplace_chi2")
    q, source = region_threshold(dof=12.0)
    assert source == "laplace_f"
    assert q == pytest.approx(2 * f.ppf(0.9, 2, 12.0))
    assert q > CHI2_2_90  # an estimated noise scale widens the region
    assert region_threshold(7.5, dof=12.0) == (7.5, "conformal")


def encode(ext, image):
    ok, buf = cv2.imencode(ext, image)
    assert ok
    return buf.tobytes()


def test_uploads_are_identified_from_their_header():
    img = np.zeros((54, 96, 3), dtype=np.uint8)
    assert inspect_image(encode(".jpg", img), "image/jpeg") == ("JPEG", (96, 54))
    assert inspect_image(encode(".png", img), "image/png; charset=binary") == ("PNG", (96, 54))


@pytest.mark.parametrize(
    ("data", "declared", "status"),
    [
        (b"hello", "text/plain", 415),
        (b"GIF89a not an image we take", "image/gif", 415),
        (b"not a jpeg at all", "image/jpeg", 415),
        (b"", "image/png", 415),
        ("png", "image/jpeg", 415),  # the content does not match the declared type
        ("png", None, 415),
    ],
)
def test_non_images_are_rejected(data, declared, status):
    if data == "png":
        data = encode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ImageRejected) as err:
        inspect_image(data, declared)
    assert err.value.status_code == status


def test_images_with_too_many_pixels_are_rejected_before_decoding():
    data = encode(".png", np.zeros((20, 30, 3), dtype=np.uint8))
    with pytest.raises(ImageRejected) as err:
        inspect_image(data, "image/png", max_pixels=500)
    assert err.value.status_code == 413


def test_the_exif_orientation_is_ignored():
    data = jpeg_with_orientation(96, 54, 6)
    assert inspect_image(data, "image/jpeg") == ("JPEG", (96, 54))
    assert decode_rgb(data, (96, 54)).shape == (54, 96, 3)  # the stored grid, as the header says


def test_a_decoded_size_that_contradicts_the_header_is_rejected():
    with pytest.raises(ImageRejected) as err:
        decode_rgb(encode(".png", np.zeros((54, 96, 3), dtype=np.uint8)), (54, 96))
    assert err.value.status_code == 415


def test_multi_picture_jpegs_are_jpegs():
    from PIL import Image

    img = Image.fromarray(np.zeros((54, 96, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, "MPO", save_all=True, append_images=[img])  # a JPEG with an MPF segment, as phones write
    data = buf.getvalue()
    assert data[:2] == b"\xff\xd8"
    assert inspect_image(data, "image/jpeg") == ("MPO", (96, 54))
    assert decode_rgb(data, (96, 54)).shape == (54, 96, 3)
    with pytest.raises(ImageRejected):
        inspect_image(data, "image/png")


def test_only_the_jpeg_and_png_parsers_read_uploads():
    data = encode(".bmp", np.zeros((8, 8, 3), dtype=np.uint8))  # Pillow reads BMP, but not for uploads
    with pytest.raises(ImageRejected) as err:
        inspect_image(data, "image/png")
    assert err.value.status_code == 415
    assert "not a readable JPEG or PNG image" in str(err.value)


def test_decoded_pixels_are_rgb():
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue in OpenCV's channel order
    rgb = decode_rgb(encode(".png", bgr))
    assert rgb.shape == (4, 6, 3)
    assert (rgb[..., 2] == 255).all()
    assert (rgb[..., :2] == 0).all()


def test_solver_pixel_settings_scale_with_the_frame():
    assert solver_config_for((1920, 1080)) == SolverConfig()
    half = solver_config_for((960, 540))
    assert half.ransac_threshold_px == pytest.approx(SolverConfig().ransac_threshold_px / 2)
    assert half.loss_scale_px == pytest.approx(SolverConfig().loss_scale_px / 2)
    assert half.sigma0_px == pytest.approx(SolverConfig().sigma0_px / 2)
    assert half.min_inliers == SolverConfig().min_inliers


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720)])
def test_detections_calibrate_to_the_true_camera(size):
    cam = broadcast_camera(*size)
    kps, ends = SyntheticDetector(cam, sigma_px=0.5 * size[0] / 1920).detect(np.zeros((1, 1, 1, 3)))[0]
    result, corners = calibrate_detections(kps, ends, size)
    assert result.ok
    assert result.H is not None
    assert result.dof is not None
    got, valid = hg.backproject(result.H, hg.project(cam.H, PLAYERS))
    assert valid.all()
    assert np.abs(got - PLAYERS).max() < 0.1  # 2 cm measured with 0.5 px noise
    truth = image_corners_projection(cam.H, size)
    assert corners is not None
    assert truth is not None
    assert np.abs(corners - truth).max() < 0.2


def test_too_few_keypoints_abstain():
    kps = KeypointObs(
        ids=np.array([2, 3, 32]),
        uv=np.array([[100.0, 100.0], [900.0, 120.0], [500.0, 600.0]]),
        conf=np.ones(3),
    )
    ends = LineEnds(line_ids=np.zeros(0, dtype=np.int64), uv=np.zeros((0, 2, 2)), conf=np.zeros((0, 2)))
    result, corners = calibrate_detections(kps, ends, (1920, 1080))
    assert result.status is CalibStatus.FEW_POINTS
    assert corners is None


def test_networks_load_once_under_concurrent_requests():
    loads = []

    def slow_load():
        loads.append(1)
        time.sleep(0.05)
        return SyntheticDetector(broadcast_camera())

    lazy = LazyDetector(slow_load)
    assert lazy.version is None
    got = []
    threads = [threading.Thread(target=lambda: got.append(lazy.get())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(loads) == 1
    assert all(d is got[0] for d in got)
    assert lazy.version == "synthetic@test"


def test_a_failed_load_is_retried():
    attempts = []

    def flaky_load():
        attempts.append(1)
        if len(attempts) == 1:
            raise ModelUnavailable("weights still downloading")
        return SyntheticDetector(broadcast_camera())

    lazy = LazyDetector(flaky_load)
    with pytest.raises(ModelUnavailable):
        lazy.get()
    assert lazy.get().version == "synthetic@test"


def test_missing_weights_make_the_networks_unavailable(tmp_path):
    with pytest.raises(ModelUnavailable) as err:
        load_pnlcalib(tmp_path / "nowhere")
    message = str(err.value)
    assert "weights" in message or "torch" in message  # torch is absent in CI
    assert str(tmp_path) not in message  # no server paths in what clients see
