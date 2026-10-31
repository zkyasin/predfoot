import json
import math

import numpy as np
import pytest

from predfoot.contracts import Localization
from predfoot.io import skillcorner as sc
from predfoot.pipeline.synthetic import look_at_homography, project_ground
from tests.fixtures import FIXTURES

SIZE = (1920, 1080)
EXCERPT = FIXTURES / "skillcorner_opendata_excerpt.jsonl"


def _loc(tid, xy=(1.0, 2.0), var=0.2, r=1.0, detected=True):
    return Localization(
        track_id=tid,
        xy=None if xy is None else np.array(xy, dtype=float),
        cov_xy=None if var is None else var * np.eye(2),
        p90_radius_m=r,
        is_detected=detected,
    )


def _frame(**kw):
    defaults = dict(frame=3, timestamp="00:00:00.30", players=(_loc(7), _loc(2, xy=(-10.0, 5.5))))
    defaults.update(kw)
    return sc.TrackingFrame(**defaults)


# --- image corners --------------------------------------------------------------------------------


def test_wide_view_top_corners_are_clipped_5_m_beyond_the_far_touchline():
    H = look_at_homography((0.0, -45.0, 18.0), (0.0, 0.0, 0.0), focal_px=1800.0, image_size=SIZE)
    corners = sc.image_corners_projection(H, SIZE)
    assert corners is not None
    tl, bl, br, tr = corners
    assert tl[1] == pytest.approx(39.0)
    assert tr[1] == pytest.approx(39.0)
    assert -35.0 < bl[1] < -15.0  # near side is not clipped
    assert tl[0] == pytest.approx(-tr[0])  # centred camera: symmetric footprint
    assert bl[0] == pytest.approx(-br[0])


@pytest.mark.parametrize("target_y", [0.0, 20.0])  # top corners below / above the horizon
def test_clipped_corner_lies_on_the_image_border(target_y):
    """The clipped corner is where the image's side border crosses y = 39 on the ground."""
    H = look_at_homography((8.0, -45.0, 18.0), (3.0, target_y, 0.0), focal_px=1800.0, image_size=SIZE)
    corners = sc.image_corners_projection(H, SIZE)
    assert corners is not None
    uv, w = project_ground(H, corners)
    assert np.all(w > 0)
    np.testing.assert_allclose(uv[[0, 1], 0], -0.5, atol=1e-6)  # left border
    np.testing.assert_allclose(uv[[2, 3], 0], SIZE[0] - 0.5, atol=1e-6)  # right border
    np.testing.assert_allclose(uv[[1, 2], 1], SIZE[1] - 0.5, atol=1e-6)  # bottom corners are the real ones
    np.testing.assert_allclose(corners[[0, 3], 1], 39.0, atol=1e-9)


def test_horizon_inside_the_image_is_detected():
    # Aimed beyond the far touchline: the top image corners look above the horizon.
    H = look_at_homography((0.0, -45.0, 18.0), (0.0, 20.0, 0.0), focal_px=1800.0, image_size=SIZE)
    q = np.linalg.solve(H, [-0.5, -0.5, 1.0])
    assert q[2] < 0  # the precondition this test relies on
    corners = sc.image_corners_projection(H, SIZE)
    assert corners is not None
    assert np.all(np.isfinite(corners))


def test_zoomed_view_keeps_unclipped_corners():
    H = look_at_homography((0.0, -45.0, 18.0), (20.0, 0.0, 0.0), focal_px=5000.0, image_size=SIZE)
    corners = sc.image_corners_projection(H, SIZE)
    assert corners is not None
    assert np.all(corners[:, 1] < 39.0)
    uv, _ = project_ground(H, corners)
    expected = [[-0.5, -0.5], [-0.5, 1079.5], [1919.5, 1079.5], [1919.5, -0.5]]
    np.testing.assert_allclose(uv, expected, atol=1e-6)


def test_corners_are_null_when_the_camera_sees_no_ground():
    # Optical axis 20° above the horizontal, vertical half field of view 16.7°: no pixel sees the ground.
    H = look_at_homography((0.0, -45.0, 18.0), (0.0, 60.0, 56.2), focal_px=1800.0, image_size=SIZE)
    assert np.linalg.solve(H, [-0.5, 1079.5, 1.0])[2] < 0  # even the bottom corners are above the horizon
    assert sc.image_corners_projection(H, SIZE) is None


def test_corners_are_null_when_the_whole_view_is_beyond_the_clipping_line():
    # Optical axis 10° above the horizontal: the bottom border meets the ground about 150 m away.
    H = look_at_homography((0.0, -45.0, 18.0), (0.0, 60.0, 36.5), focal_px=1800.0, image_size=SIZE)
    assert sc.image_corners_projection(H, SIZE) is None


def test_corners_respect_the_pitch_width():
    H = look_at_homography((0.0, -45.0, 18.0), (0.0, 0.0, 0.0), focal_px=1800.0, image_size=SIZE)
    corners = sc.image_corners_projection(H, SIZE, pitch_width=64.0)
    assert corners is not None
    assert corners[0, 1] == pytest.approx(37.0)


# --- records --------------------------------------------------------------------------------------


def test_record_has_skillcorner_keys_in_order_then_extensions():
    rec = sc.frame_to_record(_frame())
    assert list(rec) == [
        "frame",
        "timestamp",
        "period",
        "ball_data",
        "possession",
        "image_corners_projection",
        "calib_health",
        "abstain",
        "player_data",
    ]
    assert list(rec["player_data"][0]) == ["x", "y", "player_id", "is_detected", "p90_radius_m", "pos_cov_m2"]
    assert rec["ball_data"] == {"x": None, "y": None, "z": None, "is_detected": None}
    assert rec["possession"] == {"player_id": None, "group": None}
    assert list(rec["image_corners_projection"]) == list(sc.CORNER_KEYS)
    assert all(v is None for v in rec["image_corners_projection"].values())


def test_players_are_sorted_by_track_id_and_player_id_is_the_track_id():
    rec = sc.frame_to_record(_frame())
    assert [p["player_id"] for p in rec["player_data"]] == [2, 7]


def test_values_are_rounded_like_skillcorner():
    loc = Localization(
        track_id=1,
        xy=np.array([12.345678, -0.0001]),
        cov_xy=np.array([[0.123456, 0.0123449], [0.0123451, 0.2]]),
        p90_radius_m=0.98765,
    )
    p = sc.frame_to_record(_frame(players=(loc,), calib_health=0.93333))["player_data"][0]
    assert p["x"] == 12.35
    assert p["y"] == 0.0
    assert math.copysign(1.0, p["y"]) == 1.0  # no "-0.0" in the file
    assert p["p90_radius_m"] == 0.988
    assert p["pos_cov_m2"] == [[0.1235, 0.0123], [0.0123, 0.2]]  # symmetrised before rounding


def test_abstained_frame_must_not_carry_positions():
    with pytest.raises(ValueError, match="abstain"):
        sc.frame_to_record(_frame(abstain=True))
    rec = sc.frame_to_record(_frame(abstain=True, players=(_loc(4, xy=None, var=None, r=None),)))
    assert rec["abstain"] is True
    assert rec["player_data"][0] == {
        "x": None,
        "y": None,
        "player_id": 4,
        "is_detected": True,
        "p90_radius_m": None,
        "pos_cov_m2": None,
    }


@pytest.mark.parametrize(
    ("loc", "message"),
    [
        (Localization(track_id=1, xy=np.array([np.nan, 0.0])), "finite"),
        (Localization(track_id=1, xy=np.array([1.0, 2.0, 3.0])), "shape"),
        (Localization(track_id=1, xy=np.zeros(2), cov_xy=np.eye(3)), "shape"),
        (Localization(track_id=1, xy=np.zeros(2), p90_radius_m=-0.1), "radius"),
        (Localization(track_id=1, xy=None, p90_radius_m=1.0), "without a position"),
        (Localization(track_id=-1, xy=np.zeros(2)), "track"),
    ],
)
def test_writer_rejects_contract_violations(loc, message):
    with pytest.raises(ValueError, match=message):
        sc.frame_to_record(_frame(players=(loc,)))


def test_writer_rejects_bad_frame_fields():
    with pytest.raises(ValueError, match="period"):
        sc.frame_to_record(_frame(period=0))
    with pytest.raises(ValueError, match="HH:MM:SS"):
        sc.frame_to_record(_frame(timestamp="0:00:01"))
    tiny_negative = Localization(track_id=1, xy=np.zeros(2), p90_radius_m=-0.0001)
    with pytest.raises(ValueError, match="radius"):
        sc.frame_to_record(_frame(players=(tiny_negative,)))  # not rounded to 0 first


def test_writer_rejects_duplicate_tracks_and_bad_health():
    with pytest.raises(ValueError, match="duplicate"):
        sc.frame_to_record(_frame(players=(_loc(1), _loc(1))))
    with pytest.raises(ValueError, match="calib_health"):
        sc.frame_to_record(_frame(calib_health=1.5))


def test_dumps_is_one_json_line_without_nan():
    line = sc.dumps_record(sc.frame_to_record(_frame()))
    assert "\n" not in line
    assert json.loads(line)["frame"] == 3
    with pytest.raises(ValueError, match="JSON compliant"):
        sc.dumps_record({"x": float("nan")})


def test_round_trip_through_a_file(tmp_path):
    corners = np.array([[-38.73, 39.0], [-13.74, -36.22], [14.04, -35.84], [37.78, 39.0]])
    frames = [
        _frame(frame=0, timestamp="00:00:00.00", corners=corners, calib_health=0.9),
        _frame(frame=1, timestamp="00:00:00.10", abstain=True, players=(_loc(2, None, None, None),)),
    ]
    path = tmp_path / "tracking.jsonl"
    assert sc.write_tracking(path, frames) == 2
    back = list(sc.read_tracking(path))
    assert [f.frame for f in back] == [0, 1]
    np.testing.assert_allclose(back[0].corners, corners)
    assert back[0].calib_health == 0.9
    assert back[1].abstain
    assert back[1].corners is None
    p = {loc.track_id: loc for loc in back[0].players}
    np.testing.assert_allclose(p[2].xy, [-10.0, 5.5])
    np.testing.assert_allclose(p[2].cov_xy, 0.2 * np.eye(2))
    assert p[2].p90_radius_m == 1.0
    assert back[1].players[0].xy is None


def test_written_file_is_byte_stable(tmp_path):
    frames = [_frame(frame=k, timestamp=f"00:00:00.{k}0") for k in range(5)]
    sc.write_tracking(tmp_path / "a.jsonl", frames)
    sc.write_tracking(tmp_path / "b.jsonl", frames)
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


# --- reading real SkillCorner files ---------------------------------------------------------------


def test_reader_parses_real_skillcorner_records():
    frames = list(sc.read_tracking(EXCERPT))
    assert [f.frame for f in frames] == [0, 30, 210]
    pre_match, live, zoomed = frames
    assert pre_match.period is None
    assert pre_match.timestamp is None
    assert pre_match.players == ()
    assert pre_match.corners is None
    assert live.period == 1
    assert live.timestamp == "00:00:02.00"
    assert live.ball.is_detected is True
    assert live.possession.group == "home team"
    assert live.corners is not None
    assert live.corners[0, 1] == 39.0
    assert len(live.players) == 22
    assert all(loc.xy is not None and loc.cov_xy is None for loc in live.players)
    assert not live.abstain  # extension fields default when absent
    assert live.calib_health is None
    assert zoomed.corners is not None
    assert zoomed.corners[0, 1] < 39.0


def test_reader_rejects_partially_null_corners(tmp_path):
    rec = json.loads(EXCERPT.read_text().splitlines()[1])
    rec["image_corners_projection"]["x_top_left"] = None
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="corners"):
        list(sc.read_tracking(path))


# --- match.json -----------------------------------------------------------------------------------


def test_match_json_lists_every_track_once_in_one_of_two_teams():
    match = sc.make_match_json([9, 3, 3, 12], team_of={12: "away"}, pitch_length=100.0, pitch_width=64.0)
    assert sorted(p["id"] for p in match["players"]) == [3, 9, 12]
    teams = {p["id"]: p["team_id"] for p in match["players"]}
    assert teams[12] == match["away_team"]["id"]
    assert teams[3] == teams[9] == match["home_team"]["id"]
    assert match["pitch_length"] == 100.0
    assert match["pitch_width"] == 64.0
    assert match["referees"] == []
    for p in match["players"]:
        assert p["trackable_object"] == p["id"]
        assert set(p) >= {"number", "first_name", "last_name", "start_time", "player_role"}
    assert sc.player_ids_in_match(match) == {3, 9, 12}


def test_match_json_rejects_unknown_team_labels():
    with pytest.raises(ValueError, match="team"):
        sc.make_match_json([1], team_of={1: "left"})
