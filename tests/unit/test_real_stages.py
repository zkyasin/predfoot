import numpy as np

from predfoot.contracts import FootObs
from predfoot.geometry import homography as hg
from predfoot.geometry.camera import look_at
from predfoot.geometry.pitch import pnlcalib_keypoints
from predfoot.pipeline.clip import ClipFrame
from predfoot.pipeline.real import real_stages
from predfoot.uncertainty.conformal import MondrianConformal
from predfoot.uncertainty.delta import CHI2_2_90


def synthetic_detections(cam, rng, sigma=1.0):
    kps = pnlcalib_keypoints()
    ids = np.array([k for k, (_, _, planar) in kps.items() if planar])
    xy = np.array([kps[k][:2] for k in ids])
    uv, depth = cam.project(np.column_stack([xy, np.zeros(len(xy))]))
    keep = cam.in_image(uv, depth, margin=5)
    return {
        "kp_ids": ids[keep],
        "kp_uv": uv[keep] + rng.normal(scale=sigma, size=(keep.sum(), 2)),
        "kp_conf": np.full(keep.sum(), 0.9),
        "line_ids": np.array([], dtype=int),
        "line_uv": np.zeros((0, 2, 2)),
    }


def test_real_stages_locate_players_with_calibrated_radii(rng):
    cam = look_at([0.0, -55.0, 20.0], [10.0, 0.0, 0.0], 1800.0)
    stages = real_stages({0: synthetic_detections(cam, rng)}, conformal=None, version="test")
    calib = stages.calibrate(ClipFrame(frame=0))
    assert calib.ok
    players = np.array([[5.0, -10.0], [20.0, 5.0], [30.0, 15.0]])
    feet = [FootObs(track_id=i, uv=uv) for i, uv in enumerate(hg.project(cam.H, players))]
    locs = stages.localize(calib, feet)
    got = np.array([loc.xy for loc in locs])
    assert np.abs(got - players).max() < 1.0
    for loc in locs:
        assert loc.p90_radius_m is not None
        assert loc.cov_xy is not None
        # the calibrated covariance and the radius describe the same region
        assert np.isclose(loc.p90_radius_m, np.sqrt(CHI2_2_90 * np.linalg.eigvalsh(loc.cov_xy)[-1]))


def test_conformal_quantiles_scale_the_regions(rng):
    cam = look_at([0.0, -55.0, 20.0], [10.0, 0.0, 0.0], 1800.0)
    dets = {0: synthetic_detections(cam, rng)}
    wide = MondrianConformal(alpha=0.1, pooled=4 * CHI2_2_90)
    narrow = real_stages(dets, conformal=None, version="a")
    scaled = real_stages(dets, conformal=wide, version="b")
    feet = [FootObs(track_id=0, uv=hg.project(cam.H, [[10.0, 0.0]])[0])]
    r_narrow = narrow.localize(narrow.calibrate(ClipFrame(frame=0)), feet)[0].p90_radius_m
    r_wide = scaled.localize(scaled.calibrate(ClipFrame(frame=0)), feet)[0].p90_radius_m
    assert r_narrow is not None
    assert r_wide is not None
    assert np.isclose(r_wide, 2 * r_narrow)


def test_missing_detections_abstain():
    stages = real_stages({}, conformal=None, version="test")
    calib = stages.calibrate(ClipFrame(frame=3))
    assert not calib.ok
    assert stages.health(calib) == (None, True)
    assert stages.localize(calib, [FootObs(track_id=1, uv=np.array([10.0, 10.0]))])[0].xy is None


def ptz_detections(h, rng, sigma=1.0):
    """Detections of the planar template keypoints seen by homography ``h``."""
    from .synthetic_camera_track import visible

    kps = pnlcalib_keypoints()
    ids = np.array([k for k, (_, _, planar) in kps.items() if planar])
    xy = np.array([kps[k][:2] for k in ids])
    keep = visible(h, xy)
    return {
        "kp_ids": ids[keep],
        "kp_uv": hg.project(h, xy[keep]) + rng.normal(scale=sigma, size=(keep.sum(), 2)),
        "kp_conf": np.full(keep.sum(), 0.9),
        "line_ids": np.array([], dtype=int),
        "line_uv": np.zeros((0, 2, 2)),
    }


def test_temporal_stages_smooth_a_ptz_clip(rng):
    from predfoot.calib.temporal import NoiseParams, SmootherConfig
    from predfoot.geometry import ptz

    from .synthetic_camera_track import CENTRE, common_step_rms, probe_errors, ptz_path

    _, phi = ptz_path(n=30)
    h_true = [ptz.homography(p, CENTRE) for p in phi]
    dets = {k: ptz_detections(h, rng, sigma=2.0) for k, h in enumerate(h_true)}
    cfg, params = SmootherConfig(fps=5.0), NoiseParams()  # the clip's frame index runs at 5 fps
    per_frame = real_stages(dets, conformal=None, version="v")
    smoothed = real_stages(dets, conformal=None, version="v", temporal=(cfg, params))
    assert smoothed.version == f"v+ptz-{cfg.digest()}-{params.digest()}"
    raw = [per_frame.calibrate(ClipFrame(frame=k)) for k in range(30)]
    out = [smoothed.calibrate(ClipFrame(frame=k)) for k in range(30)]
    assert all(r.ok for r in out)
    assert all(r.dof is None for r in out)
    centres = np.array([ptz.decompose(r.H)[1] for r in out])
    assert np.ptp(centres, axis=0).max() < 1e-6  # one projection centre for the whole clip
    e_raw = probe_errors([r.H for r in raw], h_true)
    e_out = probe_errors([r.H for r in out], h_true)
    assert common_step_rms(e_out) < 0.5 * common_step_rms(e_raw)
    assert not smoothed.calibrate(ClipFrame(frame=99)).ok
    feet = [FootObs(track_id=0, uv=hg.project(h_true[5], [[10.0, 5.0]])[0])]
    loc = smoothed.localize(out[5], feet)[0]
    assert loc.xy is not None
    assert np.abs(loc.xy - [10.0, 5.0]).max() < 1.0


def test_temporal_stages_take_a_game_centre(rng):
    """A clip too short for its own centre is smoothed at the match's centre, as in the dev evaluation."""
    from dataclasses import replace

    from predfoot.calib.temporal import NoiseParams, SmootherConfig, estimate_centre
    from predfoot.geometry import ptz

    from .synthetic_camera_track import CENTRE, ptz_path

    _, phi = ptz_path(n=60)
    dets = {k: ptz_detections(ptz.homography(p, CENTRE), rng, sigma=2.0) for k, p in enumerate(phi)}
    cfg, params = SmootherConfig(fps=5.0), NoiseParams()
    per_frame = real_stages(dets, conformal=None, version="v")
    raw = [per_frame.calibrate(ClipFrame(frame=k)) for k in range(60)]
    own = estimate_centre(raw, np.zeros(60, bool), cfg)
    assert own is not None
    assert own.reliable
    game = replace(own, source="game")
    short = {k: dets[k] for k in range(8)}  # fewer than centre_min_frames: no centre of its own
    alone = real_stages(short, conformal=None, version="v", temporal=(cfg, params))
    with_game = real_stages(short, conformal=None, version="v", temporal=(cfg, params), game_centre=game)
    assert with_game.version == f"{alone.version}-{game.digest()}"
    out_alone = [alone.calibrate(ClipFrame(frame=k)) for k in range(8)]
    out_game = [with_game.calibrate(ClipFrame(frame=k)) for k in range(8)]
    for r_alone, r_raw in zip(out_alone, raw[:8], strict=True):
        np.testing.assert_array_equal(r_alone.H, r_raw.H)  # passed through
    centres = np.array([ptz.decompose(r.H)[1] for r in out_game])
    np.testing.assert_allclose(centres, np.tile(game.centre, (8, 1)), atol=1e-6)
