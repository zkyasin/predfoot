"""The IDSSE (DFL) parser on a tiny synthetic feed with the same structure as the real files."""

import pytest

from predfoot.io import idsse

MATCH_INFORMATION = """<?xml version="1.0" encoding="utf-8"?>
<PutDataRequest>
  <MatchInformation>
    <Teams>
      <Team TeamId="CLU-A" Role="guest">
        <Players>
          <Player PersonId="OBJ-A1" ShirtNumber="1" Starting="true" PlayingPosition="TW" />
          <Player PersonId="OBJ-A9" ShirtNumber="9" Starting="false" PlayingPosition="STZ" />
        </Players>
      </Team>
      <Team TeamId="CLU-H" Role="home">
        <Players>
          <Player PersonId="OBJ-H7" ShirtNumber="7" Starting="true" PlayingPosition="RA" />
        </Players>
      </Team>
    </Teams>
  </MatchInformation>
</PutDataRequest>
"""

POSITIONS = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<PutDataRequest>
<Positions>
<MetaData MatchId="M" Type="pitch-size"><PitchSize X="105.00" Y="68.00" /></MetaData>
<FrameSet GameSection="firstHalf" TeamId="CLU-H" PersonId="OBJ-H7">
<Frame N="10000" T="t" X="1.00" Y="2.00" D="0" S="0" A="0" M="1"/>
<Frame N="10001" T="t" X="1.50" Y="2.25" D="0" S="0" A="0" M="1"/>
</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="CLU-A" PersonId="OBJ-A1">
<Frame N="10000" T="t" X="-50.00" Y="0.00" D="0" S="0" A="0" M="1"/>
<Frame N="10001" T="t" X="-50.10" Y="0.10" D="0" S="0" A="0" M="1"/>
</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="referee" PersonId="OBJ-REF">
<Frame N="10000" T="t" X="0.00" Y="0.00" D="0" S="0" A="0" M="1"/>
</FrameSet>
<FrameSet GameSection="secondHalf" TeamId="CLU-A" PersonId="OBJ-A9">
<Frame N="100025" T="t" X="10.00" Y="-3.00" D="0" S="0" A="0" M="46"/>
</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="OBJ-BALL">
<Frame N="10000" T="t" X="0.00" Y="0.00" Z="0.10" D="0" S="0" A="0" M="1" BallPossession="1" BallStatus="1"/>
<Frame N="10001" T="t" X="0.50" Y="0.00" Z="0.20" D="0" S="0" A="0" M="1" BallPossession="2" BallStatus="0"/>
</FrameSet>
<FrameSet GameSection="secondHalf" TeamId="BALL" PersonId="OBJ-BALL">
<Frame N="100000" T="t" X="0.00" Y="0.00" Z="0.00" S="0" M="46" BallPossession="2" BallStatus="1"/>
</FrameSet>
</Positions>
</PutDataRequest>
"""


@pytest.fixture
def feed(tmp_path):
    info = tmp_path / "info.xml"
    pos = tmp_path / "positions.xml"
    info.write_text(MATCH_INFORMATION, encoding="utf-8")
    pos.write_text(POSITIONS, encoding="utf-8")
    return info, pos


def test_roster_maps_guest_to_away_and_keeps_no_names(feed):
    roster = idsse.parse_match_information(feed[0])
    assert roster.columns == ["player_id", "team", "team_id", "shirt", "position", "starting"]
    by_id = {r["player_id"]: r for r in roster.iter_rows(named=True)}
    assert by_id["OBJ-A1"]["team"] == "away"
    assert by_id["OBJ-A1"]["position"] == "TW"
    assert by_id["OBJ-H7"]["team"] == "home"
    assert by_id["OBJ-A9"]["starting"] is False


def test_positions_are_parsed_per_player_with_period_time(feed):
    roster = idsse.parse_match_information(feed[0])
    players, ball = idsse.parse_positions(feed[1], roster)
    assert players.columns == ["frame", "period", "t_s", "player_id", "team", "x", "y"]
    assert set(players["player_id"]) == {"OBJ-H7", "OBJ-A1", "OBJ-A9"}  # the referee is skipped
    h7 = players.filter(players["player_id"] == "OBJ-H7")
    assert h7["frame"].to_list() == [10000, 10001]
    assert h7["t_s"].to_list() == pytest.approx([0.0, 0.04])
    assert h7["x"].to_list() == [1.0, 1.5]
    assert h7["y"].to_list() == [2.0, 2.25]
    assert h7["team"].to_list() == ["home", "home"]
    sub = players.filter(players["player_id"] == "OBJ-A9").row(0, named=True)
    assert (sub["period"], sub["t_s"], sub["team"]) == (2, pytest.approx(1.0), "away")

    assert ball.columns == ["frame", "period", "t_s", "x", "y", "z", "alive", "possession"]
    assert ball["frame"].to_list() == [10000, 10001, 100000]
    assert ball["alive"].to_list() == [True, False, True]
    assert ball["possession"].to_list() == [1, 2, 2]
    assert ball["z"].to_list() == pytest.approx([0.1, 0.2, 0.0])


def test_match_paths_follow_the_dfl_file_names(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDFOOT_DATA_DIR", str(tmp_path))
    paths = idsse.match_paths("J03WPY")
    assert paths.positions.name == "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml"
    assert paths.match_information.parent == tmp_path.resolve() / "idsse" / "raw"
    assert paths.players == tmp_path.resolve() / "idsse" / "J03WPY.parquet"
    with pytest.raises(ValueError, match="unknown IDSSE match"):
        idsse.match_paths("NOPE")
