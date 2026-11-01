"""Detections reach the real stages keyed by the 0-based index of the clip's frames; the detection cache keys
them by image number. A file keyed the cache's way must be refused, not run with the wrong cameras."""

import logging

import pytest

from predfoot.keypoints.cache import by_clip_index
from predfoot.pipeline.clip import ClipInput
from predfoot.pipeline.real import check_detection_frames, real_stages
from predfoot.pipeline.run import ClipMismatch, run_clip
from tests.fixtures import FIXTURES


def _clip(n: int) -> ClipInput:
    return ClipInput.model_validate(
        {
            "format": "predfoot-clip/1",
            "clip_id": "clip-a",
            "fps": 25,
            "image_size": [1920, 1080],
            "frames": [{"frame": i} for i in range(n)],
        }
    )


def test_cache_keys_become_clip_indices_in_image_order():
    cache = {746: "f", 1: "a", 6: "b", 11: "c"}  # SN-GSR images downloaded one in five
    assert by_clip_index(cache) == {0: "a", 1: "b", 2: "c", 3: "f"}
    assert by_clip_index({}) == {}


def test_keys_beyond_the_clip_are_refused():
    with pytest.raises(ClipMismatch, match=r"beyond the 3 frames of clip 'clip-a' \(3\).*0-based index"):
        check_detection_frames({1: {}, 2: {}, 3: {}}, _clip(3))  # keyed from 1, as image numbers are
    cache_keys = {f: {} for f in range(1, 750, 5)}
    with pytest.raises(ClipMismatch, match=r"\(151, 156, 161, 166, 171, …\)"):
        check_detection_frames(cache_keys, _clip(150))


def test_a_missing_frame_0_is_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="predfoot.pipeline.real"):
        check_detection_frames({0: {}, 2: {}}, _clip(5))  # frames without detections may be left out
        assert caplog.records == []
        check_detection_frames({1: {}, 2: {}}, _clip(5))
    assert "no entry for frame 0 of clip 'clip-a'" in caplog.text
    check_detection_frames({}, _clip(5))  # nothing to check: every frame abstains


def test_run_clip_checks_the_stages_before_writing_anything(tmp_path):
    stages = real_stages({25: {}}, conformal=None, version="test")
    out = tmp_path / "out"
    with pytest.raises(ClipMismatch, match="beyond the 20 frames"):
        run_clip(FIXTURES / "golden" / "clip.json", out, stages=stages)
    assert not out.exists()
