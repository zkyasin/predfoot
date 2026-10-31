import json

import numpy as np
import pytest
from pydantic import ValidationError

from predfoot.contracts import CalibResult, CalibStatus, FootObs, Localization
from predfoot.io.skillcorner import CHI2_2_090
from predfoot.io.timebase import ExportSlot
from predfoot.pipeline import stub
from predfoot.pipeline.clip import ClipFrame, load_clip
from predfoot.pipeline.run import NativeResult, export_frame
from predfoot.pipeline.synthetic import look_at_homography, make_broadcast_clip, project_ground
from tests.fixtures import FIXTURES

SIZE = (1920, 1080)
H = look_at_homography((0.0, -45.0, 18.0), (0.0, 20.0, 0.0), focal_px=1800.0, image_size=SIZE)


def _clip_dict(**overrides):
    clip = {
        "format": "predfoot-clip/1",
        "clip_id": "c",
        "fps": 25,
        "image_size": [1920, 1080],
        "frames": [{"frame": 0, "homography": H.tolist(), "feet": [{"track_id": 1, "u": 900.0, "v": 700.0}]}],
    }
    clip.update(overrides)
    return clip


def _write(tmp_path, obj):
    path = tmp_path / "clip.json"
    path.write_text(json.dumps(obj))
    return path


def test_clip_loads_with_defaults(tmp_path):
    clip = load_clip(_write(tmp_path, _clip_dict(teams={"1": "home"})))
    assert clip.pitch.length == 105.0
    assert clip.teams == {1: "home"}
    assert clip.frames[0].feet[0].track_id == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"format": "predfoot-clip/2"}, "format"),
        ({"fps": 0}, "fps"),
        ({"image_size": [1920, 0]}, "positive"),
        ({"frames": [{"frame": 1, "homography": None, "feet": []}]}, "numbered"),
        ({"frames": [{"frame": 0, "homography": [[1.0, 0.0], [0.0, 1.0]], "feet": []}]}, "3×3"),
        ({"teams": {"1": "left"}}, "teams"),
        ({"unknown_key": 1}, "unknown_key"),
        (
            {
                "frames": [
                    {
                        "frame": 0,
                        "feet": [{"track_id": 1, "u": 1.0, "v": 1.0}, {"track_id": 1, "u": 2.0, "v": 2.0}],
                    }
                ]
            },
            "twice",
        ),
    ],
)
def test_bad_clips_are_rejected(tmp_path, overrides, message):
    with pytest.raises(ValidationError, match=message):
        load_clip(_write(tmp_path, _clip_dict(**overrides)))


def test_nan_is_rejected(tmp_path):
    path = tmp_path / "clip.json"
    path.write_text(json.dumps(_clip_dict()).replace("900.0", "NaN"))
    with pytest.raises(ValidationError):
        load_clip(path)


def test_calibrate_normalises_the_homography():
    result = stub.calibrate(ClipFrame(frame=0, homography=(-2.5 * H).tolist()))
    assert isinstance(result, CalibResult)
    assert result.ok
    assert result.H is not None
    np.testing.assert_allclose(result.H, H, rtol=1e-12)
    assert result.H[2, 2] == 1.0


def test_calibrate_reports_failures():
    assert stub.calibrate(ClipFrame(frame=0)).status is CalibStatus.FEW_POINTS
    singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]
    assert stub.calibrate(ClipFrame(frame=0, homography=singular)).status is CalibStatus.DEGENERATE
    no_centre = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    assert stub.calibrate(ClipFrame(frame=0, homography=no_centre)).status is CalibStatus.DEGENERATE


def test_localize_inverts_the_homography(rng):
    xy = rng.uniform([-50.0, -15.0], [50.0, 34.0], size=(50, 2))
    uv, w = project_ground(H, xy)
    assert np.all(w > 0)
    feet = [FootObs(track_id=i, uv=uv[i]) for i in range(len(xy))]
    locs = stub.localize(CalibResult(status=CalibStatus.OK, H=H), feet)
    assert all(isinstance(loc, Localization) for loc in locs)
    np.testing.assert_allclose(np.array([loc.xy for loc in locs]), xy, atol=1e-8)


def test_points_above_the_horizon_get_no_position():
    # The camera aims at y = 20 m, so the horizon is about 41 px below the top of the image.
    feet = [FootObs(track_id=1, uv=np.array([960.0, 10.0])), FootObs(track_id=2, uv=np.array([960.0, 600.0]))]
    above, below = stub.localize(CalibResult(status=CalibStatus.OK, H=H), feet)
    assert above.xy is None
    assert above.cov_xy is None
    assert above.p90_radius_m is None
    assert below.xy is not None


def test_placeholder_uncertainty_is_consistent():
    (loc,) = stub.localize(
        CalibResult(status=CalibStatus.OK, H=H),
        [FootObs(track_id=1, uv=np.array([960.0, 600.0]))],
        radius_m=1.5,
    )
    assert loc.p90_radius_m == 1.5
    assert loc.cov_xy is not None
    # The 90 % ellipse of the covariance is the circle of radius p90_radius_m.
    assert np.sqrt(CHI2_2_090 * np.linalg.eigvalsh(loc.cov_xy).max()) == pytest.approx(1.5)


def test_failed_calibration_abstains_without_positions():
    calib = stub.calibrate(ClipFrame(frame=0))
    locs = stub.localize(calib, [FootObs(track_id=4, uv=np.array([1.0, 2.0]))])
    assert locs[0].xy is None
    assert stub.health(calib) == (None, True)
    assert stub.health(stub.calibrate(ClipFrame(frame=0, homography=H.tolist()))) == (None, False)


def _native(abstain=False, tracks=((1, (0.0, 0.0)),), health=None):
    locs = {
        tid: Localization(
            track_id=tid,
            xy=None if abstain else np.array(xy),
            cov_xy=None if abstain else 0.2 * np.eye(2),
            p90_radius_m=None if abstain else 1.0,
        )
        for tid, xy in tracks
    }
    corners = None if abstain else np.zeros((4, 2))
    return NativeResult(abstain=abstain, calib_health=health, corners=corners, localizations=locs)


def test_export_frame_interpolates_and_keeps_tracks_seen_on_both_sides():
    r0 = _native(tracks=((1, (0.0, 0.0)), (2, (5.0, 5.0))), health=0.9)
    r1 = _native(tracks=((1, (2.0, -2.0)),), health=0.7)
    frame = export_frame(ExportSlot(export_frame=3, i0=7, i1=8, w=0.5), r0, r1, export_fps=10)
    assert frame.timestamp == "00:00:00.30"
    assert [p.track_id for p in frame.players] == [1]
    np.testing.assert_allclose(frame.players[0].xy, [1.0, -1.0])
    assert frame.calib_health == 0.7


def test_export_frame_abstains_if_either_side_does():
    frame = export_frame(ExportSlot(3, 7, 8, 0.5), _native(), _native(abstain=True), export_fps=10)
    assert frame.abstain
    assert frame.corners is None
    assert all(p.xy is None for p in frame.players)
    exact = export_frame(ExportSlot(2, 5, 5, 0.0), _native(), _native(abstain=True), export_fps=10)
    assert not exact.abstain  # w = 0: only native frame 5 counts


def test_committed_fixture_matches_the_generator():
    committed = json.loads((FIXTURES / "synthetic_clip.json").read_text())
    fresh, _ = make_broadcast_clip()
    assert {k: v for k, v in committed.items() if k != "frames"} == {
        k: v for k, v in fresh.items() if k != "frames"
    }
    assert len(committed["frames"]) == len(fresh["frames"])
    for a, b in zip(committed["frames"], fresh["frames"], strict=True):
        assert (a["homography"] is None) == (b["homography"] is None)
        if a["homography"] is not None:
            np.testing.assert_allclose(a["homography"], b["homography"], rtol=1e-9)
        assert [p["track_id"] for p in a["feet"]] == [p["track_id"] for p in b["feet"]]
        np.testing.assert_allclose(
            [[p["u"], p["v"]] for p in a["feet"]], [[p["u"], p["v"]] for p in b["feet"]], atol=2e-3
        )
