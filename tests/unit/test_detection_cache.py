import numpy as np

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
