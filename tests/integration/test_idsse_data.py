"""Checks on the parsed IDSSE tables (run ``scripts/prepare_idsse.py`` first)."""

import numpy as np
import polars as pl
import pytest

from predfoot.io import idsse

pytestmark = pytest.mark.data


def _require(match_id: str) -> idsse.IdssePaths:
    paths = idsse.match_paths(match_id)
    if not paths.players.exists():
        pytest.skip(f"IDSSE match {match_id} is not prepared; run scripts/prepare_idsse.py")
    return paths


@pytest.mark.parametrize("match_id", sorted(idsse.IDSSE_MATCHES))
def test_parsed_match_is_25_hz_with_two_periods_and_plausible_positions(match_id):
    paths = _require(match_id)
    players = pl.read_parquet(paths.players)
    roster = pl.read_parquet(paths.roster)

    assert players.columns == ["frame", "period", "t_s", "player_id", "team", "x", "y"]
    assert sorted(players["period"].unique().to_list()) == [1, 2]
    assert not players.select(pl.struct("player_id", "frame").is_duplicated().any()).item()
    assert set(players["player_id"]) <= set(roster["player_id"])
    assert players.select(pl.col("x", "y").is_finite().all()).row(0) == (True, True)
    assert players["x"].abs().max() < 60.0
    assert players["y"].abs().max() < 40.0

    starts = players.group_by("period").agg(pl.col("frame").min()).sort("period")["frame"].to_list()
    assert starts == [10_000, 100_000]
    first = players.filter(pl.col("period") == 1)
    np.testing.assert_allclose(
        first["t_s"].to_numpy(), (first["frame"].to_numpy() - 10_000) / idsse.IDSSE_FPS
    )

    # 11 players per team at kick-off, and each player's frames are consecutive (25 Hz, no holes).
    kickoff = players.filter(pl.col("frame") == 10_000).group_by("team").len()
    assert sorted(kickoff["len"].to_list()) == [11, 11]
    spans = players.group_by("player_id", "period").agg(
        (pl.col("frame").max() - pl.col("frame").min() + 1 - pl.len()).alias("holes")
    )
    assert spans["holes"].sum() == 0


@pytest.mark.slow
def test_parser_matches_kloppy_on_the_first_frames():
    sportec = pytest.importorskip("kloppy.sportec")
    paths = _require("J03WPY")
    dataset = sportec.load_tracking(
        meta_data=str(paths.match_information),
        raw_data=str(paths.positions),
        coordinates="sportec",
        limit=500,
    )
    ours = pl.read_parquet(paths.players).filter(pl.col("frame") < 10_000 + 500)
    theirs = pl.DataFrame(
        [
            {
                "frame": frame.frame_id,
                "player_id": player.player_id,
                "kx": d.coordinates.x,
                "ky": d.coordinates.y,
            }
            for frame in dataset.frames
            for player, d in frame.players_data.items()
        ]
    )
    joined = ours.join(theirs, on=["frame", "player_id"], how="full", coalesce=True)
    assert joined.height == ours.height == theirs.height > 10_000
    np.testing.assert_allclose(joined["x"].to_numpy(), joined["kx"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(joined["y"].to_numpy(), joined["ky"].to_numpy(), atol=1e-9)
