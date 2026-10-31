import json

import numpy as np
import pytest

from predfoot.keypoints import cache


def test_save_and_reload_detections(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDFOOT_DATA_DIR", str(tmp_path))
    dets = {
        1: {
            "kp_ids": np.array([5, 25]),
            "kp_uv": np.array([[10.0, 20.0], [30.0, 40.0]]),
            "kp_conf": np.array([0.9, 0.8]),
            "line_ids": np.array([2]),
            "line_uv": np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        }
    }
    path = cache.cache_path("SNGS-999", 1)
    cache.save_detections(path, dets, 1.5, "test-version")
    assert path.exists()
    assert not path.with_suffix(".tmp.npz").exists()
    clip_dir = tmp_path / "clip"
    (clip_dir / "img1").mkdir(parents=True)
    (clip_dir / "img1" / "000001.jpg").write_bytes(b"not read: the cache answers first")
    got, seconds, version = cache.cached_detections(clip_dir, "SNGS-999")
    assert seconds == 1.5
    assert version == "test-version"
    np.testing.assert_array_equal(got[1]["kp_ids"], dets[1]["kp_ids"])
    np.testing.assert_allclose(got[1]["line_uv"], dets[1]["line_uv"])


def _detections():
    return {
        3: {
            "kp_ids": np.array([5, 25]),
            "kp_uv": np.array([[10.12345, 20.0], [30.0, 40.98765]]),
            "kp_conf": np.array([0.9, 0.81234]),
            "line_ids": np.array([2]),
            "line_uv": np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        },
        0: {
            "kp_ids": np.zeros(0, dtype=np.int64),
            "kp_uv": np.zeros((0, 2)),
            "kp_conf": np.zeros(0),
            "line_ids": np.zeros(0, dtype=np.int64),
            "line_uv": np.zeros((0, 2, 2)),
        },
    }


def test_detections_round_trip_through_json(tmp_path):
    path = tmp_path / "dets.json"
    path.write_text(cache.dumps_detections(_detections(), "synthetic-1"), encoding="utf-8")
    got, version = cache.load_detections(path)
    assert version == "synthetic-1"
    assert sorted(got) == [0, 3]
    np.testing.assert_array_equal(got[3]["kp_ids"], [5, 25])
    np.testing.assert_allclose(got[3]["kp_uv"], [[10.123, 20.0], [30.0, 40.988]])  # 3 decimals
    np.testing.assert_allclose(got[3]["kp_conf"], [0.9, 0.812])
    np.testing.assert_allclose(got[3]["line_uv"], [[[1.0, 2.0], [3.0, 4.0]]])
    assert got[0]["kp_uv"].shape == (0, 2)
    assert got[0]["line_uv"].shape == (0, 2, 2)
    assert path.read_text(encoding="utf-8").count("\n") == 5  # header, then one line per frame


@pytest.mark.parametrize(
    ("frame_patch", "message"),
    [
        ({"kp_uv": [[1.0, 2.0]]}, "kp_uv has shape"),
        ({"kp_conf": [0.5, None]}, "kp_conf"),
        ({"line_uv": [[1.0, 2.0], [3.0, 4.0]]}, "line_uv has shape"),
        ({"kp_ids": [5, 99]}, "unknown keypoint or line ids"),
        ({"frame": -1}, "frame -1"),
        ({"kp_uv": [[1.0, 2.0], [3.0, float("nan")]]}, "finite"),
    ],
)
def test_malformed_detection_files_are_refused(tmp_path, frame_patch, message):
    frame = json.loads(cache.dumps_detections({3: _detections()[3]}, "v").splitlines()[2])
    raw = {"format": cache.DETECTIONS_FORMAT, "model_version": "v", "frames": [frame | frame_patch]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        cache.load_detections(path)


def test_duplicate_frames_and_other_formats_are_refused(tmp_path):
    frame = json.loads(cache.dumps_detections({3: _detections()[3]}, "v").splitlines()[2])
    path = tmp_path / "dup.json"
    raw = {"format": cache.DETECTIONS_FORMAT, "frames": [frame, frame]}
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="not a new non-negative integer"):
        cache.load_detections(path)
    path.write_text(json.dumps({"format": "predfoot-clip/1", "frames": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a predfoot-detections/1 file"):
        cache.load_detections(path)
    path.write_text(json.dumps({"format": cache.DETECTIONS_FORMAT, "frames": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="'frames' must be a list"):
        cache.load_detections(path)
