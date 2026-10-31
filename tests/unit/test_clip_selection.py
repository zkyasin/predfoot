import polars as pl
import pytest
from pydantic import ValidationError

from predfoot import paths
from predfoot.io.sngsr import ClipSelection, load_clip_selection

SPIKE = paths.REPO_ROOT / "configs" / "spike_clips.yaml"


def test_spike_clips_are_varied_dev_clips():
    selection = load_clip_selection(SPIKE)
    assert len(selection.clips) == 5
    assert len({c.game_id for c in selection.clips}) >= 4
    assert len({c.action_class for c in selection.clips}) == 5
    assert {c.split for c in selection.clips} == {"train", "valid"}
    assert selection.every == 5


@pytest.mark.parametrize(
    "clips",
    [
        [
            {
                "clip_id": "SNGS-120",
                "split": "test",
                "game_id": 7,
                "action_class": "Goal",
                "reason": "locked split",
            }
        ],
        [
            {
                "clip_id": "../etc",
                "split": "train",
                "game_id": 4,
                "action_class": "Goal",
                "reason": "bad clip id",
            }
        ],
        [
            {
                "clip_id": "SNGS-001",
                "split": "train",
                "game_id": 4,
                "action_class": "Goal",
                "reason": "duplicated",
            }
        ]
        * 2,
    ],
)
def test_invalid_selections_are_rejected(clips):
    with pytest.raises(ValidationError):
        ClipSelection.model_validate({"purpose": "test", "every": 5, "clips": clips})


@pytest.mark.data
def test_spike_clips_match_the_index():
    index = paths.index_dir() / "clips.parquet"
    if not index.exists():
        pytest.skip("SN-GSR index not built (scripts/build_index.py)")
    clips = pl.read_parquet(index)
    for clip in load_clip_selection(SPIKE).clips:
        row = clips.filter(clip_id=clip.clip_id).row(0, named=True)
        assert (row["split"], row["game_id"], row["action_class"]) == (
            clip.split,
            clip.game_id,
            clip.action_class,
        )
        assert row["n_frames_labeled_camera"] >= 0.97 * row["n_frames"]
