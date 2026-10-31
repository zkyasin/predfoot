import numpy as np
import pytest

from predfoot.calib.lines import (
    fit_image_line,
    intersection_correspondences,
    line_oracle,
    line_reprojection_px,
)
from predfoot.geometry import homography as hg
from predfoot.geometry.camera import look_at
from predfoot.geometry.pitch import Segment, sn_line_elements


@pytest.fixture
def camera():
    return look_at(centre=[0.0, -55.0, 20.0], target=[35.0, 0.0, 0.0], focal_px=1500.0)


def annotate(cam, rng=None, noise=0.0, n=6):
    """Simulated line annotations: points along each visible straight planar line, in pixels."""
    out = {}
    for name, el in sn_line_elements().items():
        if not isinstance(el, Segment) or not el.planar:
            continue
        t = np.linspace(0, 1, 40)[:, None]
        pts = np.add(el.a[:2], t * np.subtract(el.b[:2], el.a[:2]))
        uv, depth = cam.project(np.column_stack([pts, np.zeros(len(pts))]))
        keep = cam.in_image(uv, depth)
        if keep.sum() >= 2:
            sel = uv[keep][np.linspace(0, keep.sum() - 1, min(n, keep.sum())).astype(int)]
            if noise:
                sel = sel + rng.normal(scale=noise, size=sel.shape)
            out[name] = sel
    return out


def test_fit_image_line():
    line = fit_image_line([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    assert abs(line @ [1.0, 2.0, 1.0]) < 1e-9
    assert np.hypot(line[0], line[1]) == pytest.approx(1.0)


def test_intersections_are_exact_without_noise(camera):
    xy, uv = intersection_correspondences(annotate(camera))
    assert len(xy) >= 6
    np.testing.assert_allclose(hg.project(camera.H, xy), uv, atol=1e-6)


def test_line_oracle_recovers_the_camera(camera, rng):
    h = line_oracle(annotate(camera, rng, noise=0.5))
    assert h is not None
    probe = rng.uniform([0, -30], [50, 30], size=(100, 2))
    err = np.linalg.norm(hg.project(h, probe) - hg.project(camera.H, probe), axis=1)
    assert np.median(err) < 1.5


def test_reprojection_error_is_zero_for_the_true_camera(camera):
    lines = annotate(camera)
    assert line_reprojection_px(camera.H, lines) == pytest.approx(0.0, abs=1e-6)
    shifted = camera.H.copy()
    shifted[0, 2] += 10.0  # move the projection 10 px to the right (in homogeneous units)
    assert line_reprojection_px(shifted, lines) > 1.0
    assert np.isnan(line_reprojection_px(None, lines))


def test_goal_frames_and_arcs_are_ignored(camera):
    lines = annotate(camera)
    lines["Goal right crossbar"] = np.array([[100.0, 100.0], [300.0, 120.0]])
    lines["Circle right"] = np.array([[500.0, 500.0], [600.0, 520.0], [700.0, 560.0]])
    xy, _ = intersection_correspondences(lines)
    xy_clean, _ = intersection_correspondences(annotate(camera))
    assert len(xy) == len(xy_clean)
