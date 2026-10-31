import importlib.util

import numpy as np
import polars as pl
import pytest

from predfoot.geometry import homography as hg
from predfoot.geometry import ptz
from predfoot.paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location("render_demo", REPO_ROOT / "scripts" / "render_demo.py")
assert _spec is not None
assert _spec.loader is not None
render_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_demo)

CENTRE = [0.0, -45.0, 15.0]


def test_pitch_lines_are_drawn_where_the_camera_sees_the_pitch():
    H = ptz.homography([0.0, 0.35, 0.0, np.log(1500.0)], CENTRE)
    img = np.zeros((1080, 1920, 3), np.uint8)
    render_demo.draw_pitch_lines(img, H, (255, 255, 255), 2)
    assert (img > 0).any(axis=2).mean() > 0.002
    behind = ptz.homography([np.pi, 0.35, 0.0, np.log(1500.0)], CENTRE)  # looking away
    empty = np.zeros_like(img)
    render_demo.draw_pitch_lines(empty, behind, (255, 255, 255), 2)
    assert not empty.any()


def test_gif_stays_under_the_size_cap(tmp_path, rng):
    frames = [rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(6)]
    size = render_demo.write_gif(frames, tmp_path / "demo.gif", width=640, fps=5, max_bytes=200_000)
    assert size <= 200_000
    assert (tmp_path / "demo.gif").stat().st_size == size


def test_gif_too_large_at_the_narrowest_width_is_refused(tmp_path, rng):
    frames = [rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(40)]
    with pytest.raises(ValueError, match="fewer frames"):
        render_demo.write_gif(frames, tmp_path / "demo.gif", width=640, fps=5, max_bytes=100_000)
    assert not (tmp_path / "demo.gif").exists()


def test_keypoint_columns_survive_abstained_frames_first():
    """Seven abstained frames of 16 visible keypoints, then three calibrated ones: more than the 100 rows
    polars reads by default to infer a schema, so our camera's columns must not be dropped."""
    H = ptz.homography([0.0, 0.35, 0.0, np.log(1500.0)], CENTRE)
    xy = np.column_stack([np.linspace(-30.0, 30.0, 16), np.full(16, 34.0)])
    xy[-1] = [0.0, 0.0]
    uv = hg.project(H, xy)
    names = [f"k{i}" for i in range(16)]
    rows = []
    for frame in range(10):
        rows += render_demo.keypoint_rows(frame, names, xy, uv, (1920, 1080), H, None if frame < 7 else H)
    assert len(rows) == 160
    kps = render_demo.table(rows)
    assert kps["ours_px"].drop_nulls().len() == 48
    assert kps["ours_m"].drop_nulls().max() < 1e-6
    assert kps["ref_px"].drop_nulls().len() == 160
    assert set(kps["group"].to_list()) == {"far_touchline", "interior"}
    summary = render_demo.keypoint_summary(kps)
    assert summary["all"]["ours_px"]["n"] == 48
    assert summary["interior"]["n"] == 10


def test_keypoints_outside_the_view_are_skipped():
    H = ptz.homography([0.0, 0.35, 0.0, np.log(1500.0)], CENTRE)
    uv = np.array([[100.0, 200.0], [-5.0, 10.0], [np.nan, 3.0], [1919.5, 1079.5], [1920.0, 5.0]])
    xy = hg.backproject(H, uv[[0, 0, 0, 3, 0]])[0]
    rows = render_demo.keypoint_rows(0, list("abcde"), xy, uv, (1920, 1080), H, H)
    assert [r["keypoint"] for r in rows] == ["a", "d"]


def test_player_summary_splits_the_halves():
    players = pl.DataFrame({
        "y_ref": [-10.0, -5.0, 8.0, 20.0], "err_m": [1.0, 2.0, 3.0, 4.0],
        "p90_radius_m": [5.0, 6.0, 3.0, 4.0], "q": [68.0, 68.0, 132.0, 168.0],
        "hit": [True, True, False, True],
    })  # fmt: skip
    summary = render_demo.player_summary(players)
    assert summary["coverage"] == 0.75
    assert summary["near_half"]["coverage"] == 1.0
    assert summary["far_half"]["n"] == 2
    assert summary["far_half"]["coverage"] == 0.5
    assert render_demo.player_summary(pl.DataFrame()) is None


def test_missing_inputs_are_listed_before_anything_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDFOOT_DATA_DIR", str(tmp_path))
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    missing = render_demo.missing_inputs("117093", tmp_path / "weights", calibration)
    names = {p.name for p in missing}
    assert {"117093_camera_intrinsics.npz", "117093.txt", "117093.mp4", "SV_kp", "SV_lines"} <= names
    assert calibration not in missing
