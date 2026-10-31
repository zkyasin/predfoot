"""Contract tests: predfoot exports load in kloppy and look like SkillCorner open data (EXECUTION.md §5)."""

import json
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from kloppy import skillcorner as kloppy_skillcorner

from predfoot.contracts import Localization
from predfoot.io import skillcorner as sc
from predfoot.io.files import write_json
from predfoot.io.timebase import format_timestamp
from predfoot.pipeline.synthetic import look_at_homography
from tests.fixtures import FIXTURES

EXCERPT = FIXTURES / "skillcorner_opendata_excerpt.jsonl"
N_FRAMES, N_PLAYERS = 30, 22
ABSTAINED = {10, 11, 12, 13}
SIZE = (1920, 1080)


def _synthetic_frames(rng: np.random.Generator) -> list[sc.TrackingFrame]:
    start = rng.uniform([-45.0, -30.0], [45.0, 30.0], size=(N_PLAYERS, 2))
    velocity = rng.uniform(-6.0, 6.0, size=(N_PLAYERS, 2))  # m/s
    H = look_at_homography((0.0, -45.0, 18.0), (0.0, 0.0, 0.0), focal_px=1800.0, image_size=SIZE)
    corners = sc.image_corners_projection(H, SIZE)
    frames = []
    for k in range(N_FRAMES):
        abstain = k in ABSTAINED
        xy = start + velocity * k / sc.EXPORT_FPS
        players = tuple(
            Localization(
                track_id=100 + i,
                xy=None if abstain else xy[i],
                cov_xy=None if abstain else np.diag([0.2, 0.3]),
                p90_radius_m=None if abstain else 1.2,
            )
            for i in range(N_PLAYERS)
        )
        frames.append(
            sc.TrackingFrame(
                frame=k,
                timestamp=format_timestamp(k / sc.EXPORT_FPS),
                players=players,
                corners=None if abstain else corners,
                calib_health=None,
                abstain=abstain,
            )
        )
    return frames


def _positions(frame) -> dict[int, tuple[float, float]]:
    return {int(p.player_id): (d.coordinates.x, d.coordinates.y) for p, d in frame.players_data.items()}


def _load(match_path: Path, tracking_path: Path, **kwargs):
    with match_path.open("rb") as meta, tracking_path.open("rb") as raw:
        return kloppy_skillcorner.load(meta_data=meta, raw_data=raw, **kwargs)


@pytest.fixture
def exported(tmp_path, rng):
    frames = _synthetic_frames(rng)
    tracking = tmp_path / "tracking.jsonl"
    sc.write_tracking(tracking, frames)
    teams = {100 + i: ("home" if i < N_PLAYERS // 2 else "away") for i in range(N_PLAYERS)}
    match = sc.make_match_json([100 + i for i in range(N_PLAYERS)], team_of=teams, n_frames=N_FRAMES)
    write_json(tmp_path / "match.json", match)
    return frames, tmp_path / "match.json", tracking


# --- (a) kloppy loads our exports -----------------------------------------------------------------


def test_kloppy_loads_every_frame_with_our_coordinates(exported):
    frames, match_path, tracking_path = exported
    dataset = _load(match_path, tracking_path, coordinates="skillcorner", include_empty_frames=True)
    assert len(dataset.records) == N_FRAMES
    assert dataset.metadata.frame_rate == sc.EXPORT_FPS
    assert dataset.metadata.pitch_dimensions.pitch_length == 105.0
    for ours, theirs in zip(frames, dataset.records, strict=True):
        assert theirs.frame_id == ours.frame
        assert theirs.timestamp == timedelta(seconds=ours.frame / sc.EXPORT_FPS)
        loaded = _positions(theirs)
        if ours.abstain:
            assert loaded == {}  # null positions are skipped, the frame is kept
            continue
        assert len(loaded) == N_PLAYERS
        for loc in ours.players:
            assert loc.xy is not None
            np.testing.assert_allclose(loaded[loc.track_id], loc.xy, atol=0.005 + 1e-9)  # 1 cm rounding


def test_kloppy_default_coordinates_put_the_far_side_at_the_top(exported):
    """kloppy's normalised frame has y pointing down the screen: our y > 0 (far side) must land above 0.5."""
    frames, match_path, tracking_path = exported
    dataset = _load(match_path, tracking_path)  # kloppy's own coordinate system
    ours = {loc.track_id: loc.xy for loc in frames[0].players}
    for player, pdata in dataset.records[0].players_data.items():
        x, y = ours[int(player.player_id)]
        assert pdata.coordinates.x == pytest.approx((x + 52.5) / 105, abs=1e-4)
        assert pdata.coordinates.y == pytest.approx((34.0 - y) / 68, abs=1e-4)


def test_kloppy_assigns_players_to_the_teams_of_match_json(exported):
    _, match_path, tracking_path = exported
    dataset = _load(match_path, tracking_path, coordinates="skillcorner")
    home, away = dataset.metadata.teams
    assert len(home.players) == len(away.players) == N_PLAYERS // 2


def test_kloppy_skips_frames_without_players_unless_asked(exported, tmp_path):
    _, match_path, tracking_path = exported
    lines = tracking_path.read_text().splitlines()
    empty = json.loads(lines[0])
    empty["player_data"] = []
    path = tmp_path / "with_empty.jsonl"
    path.write_text("\n".join([json.dumps(empty), *lines[1:]]) + "\n")
    assert len(_load(match_path, path, coordinates="skillcorner").records) == N_FRAMES - 1


# --- (b) same keys and value types as SkillCorner open data ---------------------------------------


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _assert_compatible(ours: dict, theirs: dict, where: str) -> None:
    """Our keys include theirs, in the same order, and non-null values have the same JSON type."""
    shared = [k for k in ours if k in theirs]
    assert shared == list(theirs), f"{where}: keys {list(theirs)} expected in order, got {list(ours)}"
    for key in theirs:
        a, b = ours[key], theirs[key]
        if isinstance(b, dict):
            _assert_compatible(a, b, f"{where}.{key}")
        elif a is not None and b is not None:
            assert _json_type(a) == _json_type(b), f"{where}.{key}: {_json_type(a)} vs {_json_type(b)}"


def _real_records() -> list[dict]:
    return [json.loads(line) for line in EXCERPT.read_text().splitlines()]


def test_fixture_is_the_skillcorner_excerpt_we_think_it_is():
    records = _real_records()
    assert [r["frame"] for r in records] == [0, 30, 210]
    assert all(list(r) == list(sc.SKILLCORNER_KEYS) for r in records)


def test_our_records_have_skillcorner_keys_and_types(exported):
    _, _, tracking_path = exported
    ours = [json.loads(line) for line in tracking_path.read_text().splitlines()]
    live = ours[0]
    abstained = ours[min(ABSTAINED)]
    for real in _real_records():
        for record in (live, abstained):
            _assert_compatible(record, real, f"frame {real['frame']}")
            for player in record["player_data"]:
                for real_player in real["player_data"]:
                    _assert_compatible(player, real_player, "player")
    # Values a type check on nulls cannot see: integers stay integers, coordinates are floats.
    assert isinstance(live["frame"], int)
    assert isinstance(live["period"], int)
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}\.\d{2}", live["timestamp"])
    assert all(isinstance(p["player_id"], int) for p in live["player_data"])
    assert all(isinstance(p["x"], float) and isinstance(p["y"], float) for p in live["player_data"])
    assert all(isinstance(v, float) for v in live["image_corners_projection"].values())


def test_kloppy_loads_the_real_excerpt_with_a_generated_match_json(tmp_path):
    real = _real_records()
    ids = {p["player_id"] for r in real for p in r["player_data"]}
    write_json(tmp_path / "match.json", sc.make_match_json(ids))
    dataset = _load(tmp_path / "match.json", EXCERPT, coordinates="skillcorner")
    assert [f.frame_id for f in dataset.records] == [30, 210]  # frame 0 is before kick-off (period null)
    first = _positions(dataset.records[0])
    for p in real[1]["player_data"]:
        assert first[p["player_id"]] == (p["x"], p["y"])


# --- (c) axis rule --------------------------------------------------------------------------------


@pytest.mark.parametrize("pan_x", [-30.0, 0.0, 30.0])
def test_top_image_corners_of_a_main_camera_view_are_on_the_far_side(pan_x):
    H = look_at_homography((0.0, -45.0, 18.0), (pan_x, 0.0, 0.0), focal_px=1800.0, image_size=SIZE)
    record = sc.frame_to_record(sc.TrackingFrame(frame=0, corners=sc.image_corners_projection(H, SIZE)))
    c = record["image_corners_projection"]
    assert c["y_top_left"] > 0
    assert c["y_top_right"] > 0
    assert c["y_bottom_left"] < c["y_top_left"]
    assert c["y_bottom_right"] < c["y_top_right"]
    assert c["x_top_left"] < c["x_top_right"]  # image left is pitch left (x towards the right-hand goal)
    assert c["x_bottom_left"] < c["x_bottom_right"]
