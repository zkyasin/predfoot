"""scripts/render_hero.py without data or torch: the camera between keyframes, the map and the frame."""

import importlib.util
import sys

import numpy as np
import pytest

from predfoot.calib import temporal as tm
from predfoot.contracts import CalibResult, CalibStatus
from predfoot.demo import calibration as dc
from predfoot.demo import virtual_camera as vc
from predfoot.geometry import homography as hg
from predfoot.geometry import ptz
from predfoot.paths import REPO_ROOT
from predfoot.uncertainty.delta import CHI2_2_90

from .synthetic_camera_track import CENTRE, gaussian_calibs, ptz_path


def _load(name):
    key = f"scripts_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


hero = _load("render_hero")
PP = tm.SmootherConfig().principal_point
PARAMS = tm.NoiseParams(kappa2=1.0, q_pan=5e-3, q_tilt=1e-4, q_logf=1e-3)


def test_hermite_keeps_the_end_states_and_reproduces_a_cubic(rng):
    a = rng.standard_normal((4, 4))  # four cubics t ↦ a0 + a1 t + a2 t² + a3 t³
    dt = 0.2

    def value(t):
        return a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3

    def rate(t):
        return a[1] + 2 * a[2] * t + 3 * a[3] * t**2

    for s in (0.0, 0.3, 0.5, 0.8, 1.0):
        got = hero.hermite(value(0.0), rate(0.0), value(dt), rate(dt), dt, s)
        np.testing.assert_allclose(got, value(s * dt), atol=1e-12)


def smoothed_clip(n=3, step=5, fps=25.0, flags=None, ok=None):
    """Keyframes 0, 5, 10… of a camera panning at a constant rate, smoothed in one segment."""
    keys = step * np.arange(n)
    rate = np.array([0.05, 0.01, 0.0, 0.1])  # rad/s and 1/s
    phi0 = np.array([0.0, 0.35, 0.0, np.log(1500.0)])
    phi = phi0 + np.outer(keys / fps, rate)
    mean = np.concatenate([phi, np.tile(rate, (n, 1))], axis=1)
    cov = np.tile(np.diag([1e-6, 1e-6, 1e-8, 1e-6, 1e-4, 1e-4, 1e-6, 1e-4]), (n, 1, 1))
    track = tm.SmoothedTrack(
        mean=mean, cov=cov, R=cov[:, :4, :4], segment=np.zeros(n, int), used=np.ones(n, bool),
        rejected=np.zeros(n, bool), short=np.zeros(n, bool), manoeuvre=np.zeros(n, bool),
        loose=np.zeros(n, bool), nis=np.full(n, np.nan), deletion=np.full(n, np.nan), nll=0.0, n_iter=1,
    )  # fmt: skip
    fl = flags or tm.FrameFlags.empty(n)
    if flags is None:
        fl.smoothed[:] = True
        fl.segment[:] = 0
    ok = np.ones(n, bool) if ok is None else ok
    results = [
        CalibResult(
            CalibStatus.OK, H=ptz.homography(p, CENTRE, PP), cov_theta=np.eye(8) * 1e-8, n_inliers=9 + k
        )
        if ok[k]
        else CalibResult(CalibStatus.UNHEALTHY)
        for k, p in enumerate(phi)
    ]
    return keys, hero.Smoothed(results, fl, track, CENTRE.copy()), phi0, rate


def test_a_frame_between_keyframes_gets_the_smoothed_state_at_its_time():
    keys, sm, phi0, rate = smoothed_clip()
    sources = hero.keyframe_sources(sm.flags)
    cam = hero.camera_at(2, keys, sm, sources)
    assert cam is not None
    assert (cam.how, cam.source, cam.n_inliers) == ("interpolated", "smoother", 9)
    np.testing.assert_allclose(cam.H, ptz.homography(phi0 + rate * 2 / 25, CENTRE, PP), atol=1e-9)
    eig = np.linalg.eigvalsh(cam.cov_theta)
    assert eig[0] > -1e-9 * eig[-1]  # positive semi-definite up to rounding
    key = hero.camera_at(5, keys, sm, sources)
    assert key is not None
    assert key.how == "keyframe"
    assert key.n_inliers == 10
    with pytest.raises(ValueError, match="outside the keyframes"):
        hero.camera_at(11, keys, sm, sources)


def test_without_a_smoothed_pair_a_frame_holds_the_nearest_camera_or_abstains():
    flags = tm.FrameFlags.empty(3)
    flags.smoothed[[0, 2]] = True
    flags.passthrough[1] = True
    flags.segment[:] = [0, -1, 0]
    keys, sm, _, _ = smoothed_clip(flags=flags)
    sources = hero.keyframe_sources(flags)
    held = hero.camera_at(4, keys, sm, sources)
    assert held is not None
    assert held.how == "held"
    np.testing.assert_array_equal(held.H, sm.results[1].H)
    flags = tm.FrameFlags.empty(3)  # the middle raw calibration failed and the smoother filled it
    flags.smoothed[[0, 2]] = True
    flags.filled[1] = True
    flags.segment[:] = 0
    keys, sm, _, _ = smoothed_clip(flags=flags)
    assert hero.camera_at(4, keys, sm, hero.keyframe_sources(flags)).how == "interpolated"
    flags = tm.FrameFlags.empty(3)
    flags.smoothed[0] = True
    keys, sm, _, _ = smoothed_clip(flags=flags, ok=np.array([True, False, False]))
    assert hero.camera_at(4, keys, sm, hero.keyframe_sources(flags)) is None  # nearest keyframe abstains


def test_bridged_keyframes_take_the_smoother_model():
    flags = tm.FrameFlags.empty(4)
    flags.smoothed[[0, 2]] = True
    flags.bridged[1] = True
    flags.passthrough[3] = True
    assert dc.cov_sources(flags)[1] == "frame"
    assert hero.keyframe_sources(flags) == ["smoother", "smoother", "smoother", "frame"]


def test_smooth_keyframes_is_smooth_results_with_its_centre(rng):
    _, phi = ptz_path(n=60)
    calibs = gaussian_calibs(phi, CENTRE, rng)
    frames, dup = 5 * np.arange(60), np.zeros(60, bool)
    sm = hero.smooth_keyframes(frames, calibs, dup, PARAMS)
    ref, flags, track = tm.smooth_results(frames, calibs, dup, tm.SmootherConfig(), PARAMS)
    assert sm.centre is not None
    assert np.linalg.norm(sm.centre - CENTRE) < 2.0
    for a, b in zip(sm.results, ref, strict=True):
        assert a.ok == b.ok
        if a.ok:
            np.testing.assert_array_equal(a.H, b.H)
    np.testing.assert_array_equal(sm.flags.smoothed, flags.smoothed)
    np.testing.assert_array_equal(sm.track.mean, track.mean)


def test_keyframes_bracket_every_frame_shown():
    keys = hero.key_frames(100, 200)
    assert keys[0] == 100
    assert keys[-1] >= 299
    assert len(keys) == 41
    assert set(np.diff(keys)) == {5}


def test_pitch_map_pixels_and_zone_axes():
    pmap = hero.PitchMap(720)
    assert pmap.height == round(73 * 720 / 110)
    corners = pmap.to_px(np.array([[-52.5, 34.0], [52.5, -34.0]]))
    np.testing.assert_allclose(corners[0], [2.5 * pmap.scale - 0.5, 2.5 * pmap.scale - 0.5])
    grid = pmap.ground()
    np.testing.assert_allclose(pmap.to_px(grid[10, 20]), [[20.0, 10.0]], atol=1e-9)
    (major, minor), angle = pmap.ellipse(np.diag([4.0, 1.0]))  # 2 m sd along x, 1 m along y
    np.testing.assert_allclose([major, minor], np.sqrt(CHI2_2_90) * np.array([2.0, 1.0]) * pmap.scale)
    assert abs(angle) in (0.0, 180.0)
    bg = pmap.background()
    assert bg.shape == (pmap.height, 720, 3)
    assert bg.dtype == np.uint8


def test_footprint_is_the_ground_the_view_sees():
    H = ptz.homography([0.0, 0.35, 0.0, np.log(1500.0)], CENTRE)
    pmap = hero.PitchMap(360)
    grid = np.concatenate([pmap.ground(), np.ones((pmap.height, 360, 1))], axis=-1)
    mask = hero.footprint(grid, H, (1920, 1080))
    assert 0.02 < mask.mean() < 0.9
    seen = hg.backproject(H, np.array([[960.0, 900.0]]))[0][0]  # under the lower middle of the image
    u, v = np.round(pmap.to_px(seen)[0]).astype(int)
    assert mask[v, u]
    behind = ptz.homography([np.pi, 0.35, 0.0, np.log(1500.0)], CENTRE)
    assert not hero.footprint(grid, behind, (1920, 1080))[v, u]


def test_display_transform_follows_the_crop_and_the_resize():
    to_disp = hero.view_to_display(0.5, 100)
    np.testing.assert_allclose(hg.project(to_disp, np.array([[0.0, 100.0], [1919.0, 1079.0]])),
                               [[-0.25, -0.25], [959.25, 489.25]])  # fmt: skip


def test_place_scales_the_covariance_by_the_conformal_quantile():
    model = dc.load()["ptz"].model_for("smoother")
    H = ptz.homography([0.0, 0.35, 0.0, np.log(1500.0)], CENTRE)
    cam = hero.Camera(H, np.eye(8) * 1e-9, "smoother", 12, "keyframe")
    feet = np.array([[960.0, 900.0], [300.0, 700.0]])
    placed = hero.place(cam, feet, model)
    assert placed.valid.all()
    np.testing.assert_allclose(placed.xy, hg.backproject(H, feet)[0], atol=1e-9)
    assert (placed.radius() > 0).all()
    assert hero.place(cam, np.zeros((0, 2)), model).xy.shape == (0, 2)


def test_a_frame_is_the_view_above_the_map_and_an_abstained_frame_has_no_zone():
    pmap = hero.PitchMap(240)
    bg = pmap.background()
    pad = 2
    g = pmap.ground(pad)
    grid = np.concatenate([g, np.ones((*g.shape[:2], 1))], axis=-1)
    view = np.full((120, 240, 3), 90, np.uint8)
    H = hero.view_to_display(240 / 1920, 0) @ ptz.homography([0.0, 0.35, 0.0, np.log(1500.0)], CENTRE)
    placed = hero.Placed(np.array([[0.0, 0.0]]), np.diag([4.0, 1.0])[None], np.array([True]))
    feet = np.array([[120.0, 100.0]])
    top = hero.draw_view(view, H, feet)
    bottom = hero.draw_map(pmap, bg, grid, pad, H, (240, 120), placed)
    frame = hero.stack(top, bottom)
    assert frame.shape[1] == 240
    assert frame.shape[0] % 2 == 0
    amber = np.all(np.abs(bottom.astype(int) - hero.PLAYER_BGR) < 30, axis=2)
    assert amber.any()
    empty = hero.draw_map(pmap, bg, grid, pad, None, (240, 120), None)
    assert not np.all(np.abs(empty.astype(int) - hero.PLAYER_BGR) < 30, axis=2).any()
    ring = hero.draw_view(view, None, feet)
    assert not np.array_equal(ring, view)  # the feet keep a grey ring


def test_saved_detections_are_reused_only_for_the_same_views(tmp_path):
    follow = vc.FollowConfig(hfov_deg=(25.0, 45.0), aim_offset_m=(0.0, -6.0))
    key = hero.view_key("117093", follow, 275, 575)
    det = {
        "kp_ids": np.array([3, 7]),
        "kp_uv": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "kp_conf": np.array([0.5, 0.9]),
        "line_ids": np.array([], int),
        "line_uv": np.zeros((0, 2, 2)),
    }
    path = tmp_path / "detections.json"
    hero.save_detections(path, key, {325: det})
    back = hero.load_detections(path, key)
    assert list(back) == [325]
    for k, v in det.items():
        np.testing.assert_array_equal(back[325][k], v)
    with pytest.raises(ValueError, match="other views"):
        hero.load_detections(path, hero.view_key("117093", follow, 300, 575))
