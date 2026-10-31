import hashlib

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


def test_real_stages_describe_themselves_for_meta_json():
    laplace = real_stages({}, conformal=None, version="test")
    assert laplace.conformal is None
    assert laplace.description is not None
    assert "uncalibrated Laplace" in laplace.description["localization"]
    model = MondrianConformal(alpha=0.1, edges={"mpp": [0.05]}, quantiles={"0": 6.0}, pooled=7.0)
    stages = real_stages({}, conformal=model, version="test")
    assert stages.description is not None
    assert "Mondrian" in stages.description["localization"]
    assert stages.conformal == {
        "kind": "mondrian",
        "sha256": hashlib.sha256(model.to_json().encode("utf-8")).hexdigest(),
        "alpha": 0.1,
        "pooled": 7.0,
        "edges": {"mpp": [0.05]},
        "quantiles": {"0": 6.0},
    }
