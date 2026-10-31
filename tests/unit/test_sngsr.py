"""SN-GSR parsing on small synthetic labels written by the tests (no SoccerNet data in git)."""

import json

import numpy as np
import polars as pl
import pytest

from predfoot.io import sngsr

W, H = 1920, 1080


def person(ann_id, frame, track, category, box, pitch, raw=None, role=None, team=None, jersey=None):
    x, y, w, h = box
    bottom = {"x_bottom_middle": pitch[0], "y_bottom_middle": pitch[1]} if pitch else None
    raw_bottom = {"x_bottom_middle": raw[0], "y_bottom_middle": raw[1]} if raw else bottom
    return {
        "id": str(ann_id),
        "image_id": f"2001{frame:06d}",
        "track_id": track,
        "supercategory": "object",
        "category_id": category,
        "attributes": {"role": role, "team": team, "jersey": jersey},
        "bbox_image": {"x": x, "y": y, "x_center": x + w / 2, "y_center": y + h / 2, "w": w, "h": h},
        "bbox_pitch": bottom,
        "bbox_pitch_raw": raw_bottom,
    }


def labels(clip="SNGS-001", game_id="4", n_frames=3, action="Goal"):
    images = [
        {
            "is_labeled": True,
            "image_id": f"2001{f:06d}",
            "file_name": f"{f:06d}.jpg",
            "width": W,
            "height": H,
            "has_labeled_person": True,
            "has_labeled_pitch": True,
            "has_labeled_camera": f != 2,
        }
        for f in range(1, n_frames + 1)
    ]
    anns = [
        # A player low in the image, close to the main camera: SoccerNet y > 0.
        person(1, 1, 7, 1, (1150, 760, 63, 170), (34.6, 16.9), (34.5, 17.0), "player", "right", "43"),
        person(2, 1, 5, 2, (1376, 369, 31, 130), (52.0, 0.3), None, "goalkeeper", "right", None),
        person(
            3, 1, 6, 3, (0, 400, 40, 120), (47.8, 1.9), None, "referee", None, None
        ),  # touches the left edge
        person(4, 1, 13, 4, (790, 563, 14, 18), (41.6, -0.3), None, "ball"),  # ball: excluded
        person(
            5, 2, 7, 1, (1150, 950, 63, 129), (34.7, 18.0), (34.7, 18.4), "player", "right", "43"
        ),  # bottom
        person(6, 2, 9, 7, (500, 300, 30, 90), None, None, "other"),  # no pitch position
        {
            "id": "100",
            "image_id": "2001000001",
            "supercategory": "pitch",
            "category_id": 5,
            "lines": {
                "Side line top": [{"x": 0.0, "y": 0.25}, {"x": 0.5, "y": 0.2}],
                "Goal right post left": [{"x": 0.6, "y": 0.28}, {"x": 0.6, "y": 0.35}],
                "Circle right": [{"x": 0.7, "y": 0.5}, {"x": 0.75, "y": 0.55}, {"x": 0.8, "y": 0.5}],
            },
        },
    ]
    return {
        "info": {
            "version": "1.3",
            "game_id": game_id,
            "id": clip[-3:],
            "action_class": action,
            "visibility": "visible",
            "game_time_start": "2 - 36:48",
            "game_time_stop": "2 - 37:18",
            "name": clip,
            "frame_rate": 25,
            "seq_length": n_frames,
        },
        "images": images,
        "annotations": anns,
        "categories": [{"id": 5, "name": "pitch", "lines": list(sngsr.LINE_CLASSES)}],
    }


@pytest.fixture
def parsed():
    return sngsr.parse_labels(labels(), split="train")


def test_clip_row(parsed):
    row = parsed.clip.row(0, named=True)
    assert row["clip_id"] == "SNGS-001"
    assert row["game_id"] == 4
    assert (row["split"], row["action_class"], row["n_frames"]) == ("train", "Goal", 3)
    assert (row["width"], row["height"], row["frame_rate"]) == (1920, 1080, 25.0)
    assert (row["n_frames_labeled_pitch"], row["n_frames_labeled_camera"]) == (3, 2)


def test_persons_exclude_the_ball_and_keep_others(parsed):
    p = parsed.persons
    assert p.height == 5
    assert sorted(p["role"].to_list()) == ["goalkeeper", "other", "player", "player", "referee"]
    assert p.schema == pl.Schema(sngsr.PERSONS_SCHEMA)


def test_foot_is_the_bottom_middle_of_the_box(parsed):
    row = parsed.persons.filter(frame=1, track_id=7).row(0, named=True)
    assert (row["foot_u"], row["foot_v"]) == (1150 + 63 / 2, 760 + 170)
    assert (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]) == (1150, 760, 63, 170)
    assert (row["team"], row["jersey"]) == ("right", "43")


def test_pitch_positions_are_converted_to_the_internal_frame(parsed):
    row = parsed.persons.filter(frame=1, track_id=7).row(0, named=True)
    # Near the main camera: y > 0 in SoccerNet, y < 0 internally (SkillCorner's frame).
    assert (row["x"], row["y"]) == pytest.approx((34.6, -16.9))
    assert (row["x_raw"], row["y_raw"]) == pytest.approx((34.5, -17.0))


def test_missing_pitch_positions_are_null(parsed):
    row = parsed.persons.filter(role="other").row(0, named=True)
    assert row["x"] is None
    assert row["x_raw"] is None


def test_boxes_near_any_border_are_truncated(parsed):
    flags = dict(
        zip(parsed.persons["track_id"].to_list(), parsed.persons["truncated"].to_list(), strict=True)
    )
    by_frame = parsed.persons.select("frame", "track_id", "truncated").rows()
    assert flags[6] is True  # x = 0
    assert (2, 7, True) in by_frame  # y + h = 1079: the feet touch the bottom border
    assert (1, 7, False) in by_frame
    assert flags[5] is False


def test_lines_are_in_pixels_with_a_planar_flag(parsed):
    lines = parsed.lines
    assert lines.schema == pl.Schema(sngsr.LINES_SCHEMA)
    top = lines.filter(line_class="Side line top").sort("point_idx")
    np.testing.assert_allclose(top.select("u", "v").to_numpy(), [[0.0, 270.0], [960.0, 216.0]])
    planar = dict(lines.select("line_class", "is_planar").unique().rows())
    assert planar == {"Side line top": True, "Goal right post left": False, "Circle right": True}
    assert lines.filter(line_class="Circle right")["point_idx"].to_list() == [0, 1, 2]
    assert set(lines["frame"].to_list()) == {1}


def test_frames_table_keeps_the_label_flags(parsed):
    frames = parsed.frames
    assert frames.schema == pl.Schema(sngsr.FRAMES_SCHEMA)
    assert frames["frame"].to_list() == [1, 2, 3]
    assert frames["has_labeled_camera"].to_list() == [True, False, True]


def test_frame_comes_from_the_file_name():
    data = labels()
    data["images"][0]["image_id"] = "anything"
    for ann in data["annotations"]:
        if ann["image_id"] == "2001000001":
            ann["image_id"] = "anything"
    data["images"][0]["file_name"] = "000003.jpg"
    data["images"][2]["file_name"] = "000001.jpg"
    out = sngsr.parse_labels(data, split="train")
    assert out.persons.filter(track_id=7)["frame"].to_list() == [2, 3]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["info"].update(name="SNGS-999"), "does not match"),
        (lambda d: d["images"][0].update(file_name="000751.jpg"), "outside"),
        (lambda d: d["images"][1].update(width=1280), "several image sizes"),
        (lambda d: d["annotations"][0].update(image_id="nope"), "unknown image"),
    ],
)
def test_inconsistent_labels_are_rejected(mutate, message):
    data = labels()
    mutate(data)
    with pytest.raises(sngsr.LabelsError, match=message):
        sngsr.parse_labels(data, split="train", clip_id="SNGS-001")


def test_planar_whitelist():
    assert len(sngsr.LINE_CLASSES) == 26
    assert len(sngsr.NON_PLANAR_LINE_CLASSES) == 6
    assert all("Goal" in c and ("post" in c or "crossbar" in c) for c in sngsr.NON_PLANAR_LINE_CLASSES)
    assert sngsr.is_planar_line("Middle line")
    assert sngsr.is_planar_line("Circle central")
    assert not sngsr.is_planar_line("Goal left post left ")  # trailing space as in SN-Calibration
    assert not sngsr.is_planar_line("Line unknown")  # unknown classes never enter a homography


def test_member_selection():
    names = [
        "sequences_info.json",
        "SNGS-001/",
        "SNGS-001/Labels-GameState.json",
        "SNGS-002/Labels-GameState.json",
        "../SNGS-003/Labels-GameState.json",
        "SNGS-001/../../Labels-GameState.json",
        *(f"SNGS-001/img1/{f:06d}.jpg" for f in range(1, 21)),
        "SNGS-002/img1/000001.jpg",
        "SNGS-001/img1/../../000006.jpg",
    ]
    assert sngsr.labels_members(names) == {
        "SNGS-001": "SNGS-001/Labels-GameState.json",
        "SNGS-002": "SNGS-002/Labels-GameState.json",
    }
    frames = sngsr.frame_members(names, ["SNGS-001"], every=5)
    assert sorted(frames) == [("SNGS-001", f) for f in (1, 6, 11, 16)]
    with pytest.raises(ValueError, match="clip id"):
        sngsr.labels_path("train", "../../etc")


def write_clip(root, split, clip, game_id):
    path = sngsr.labels_path(split, clip, root)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(labels(clip=clip, game_id=game_id)))


def test_build_index_keeps_the_locked_split_apart(tmp_path):
    write_clip(tmp_path, "train", "SNGS-001", "4")
    write_clip(tmp_path, "valid", "SNGS-056", "3")
    write_clip(tmp_path, "test", "SNGS-120", "7")
    written = sngsr.build_index(tmp_path, workers=1)
    index = tmp_path / "index"
    clips = pl.read_parquet(index / "clips.parquet")
    assert clips.select("clip_id", "split", "game_id").rows() == [
        ("SNGS-120", "test", 7),
        ("SNGS-001", "train", 4),
        ("SNGS-056", "valid", 3),
    ]
    assert set(pl.read_parquet(index / "persons.parquet")["clip_id"]) == {"SNGS-001", "SNGS-056"}
    assert set(pl.read_parquet(index / "lines.parquet")["clip_id"]) == {"SNGS-001", "SNGS-056"}
    assert set(pl.read_parquet(index / "test" / "persons.parquet")["clip_id"]) == {"SNGS-120"}
    assert set(pl.read_parquet(index / "frames.parquet")["clip_id"]) == {"SNGS-001", "SNGS-056"}
    assert set(written) == {"clips", "persons", "lines", "frames"} | {
        "test/persons",
        "test/lines",
        "test/frames",
    }


def test_label_noise_floor():
    persons = pl.DataFrame(
        {
            "clip_id": ["a"] * 4 + ["b"],
            "x": [0.0, 0.0, 0.0, 0.0, 1.0],
            "y": [0.0, 0.0, 0.0, None, 1.0],
            "x_raw": [0.3, 0.0, 0.0, 9.0, 1.0],
            "y_raw": [0.4, 1.0, 0.0, 9.0, 1.0],
        }
    )
    floor = sngsr.label_noise_floor(persons).rows(named=True)
    assert floor[0]["clip_id"] == "a"
    assert floor[0]["n"] == 3  # the row with a null position is ignored
    assert floor[0]["p50"] == pytest.approx(0.5)
    assert floor[0]["max"] == pytest.approx(1.0)
    assert floor[1] == {"clip_id": "b", "n": 1, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
