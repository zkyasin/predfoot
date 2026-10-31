"""IDSSE tracking data (Bassek, Rein, Weber & Memmert 2025, CC BY 4.0, doi:10.6084/m9.figshare.28196177).

Seven Bundesliga matches of DFL/Sportec tracking at 25 Hz. The raw XML files come from the public Hugging
Face mirror ``pysport/idsse-data`` (the files kloppy's ``sportec.load_open_tracking_data`` reads) and are
parsed once into Parquet under ``PREDFOOT_DATA_DIR/idsse/``:

- ``<id>.parquet``: one row per player and frame: ``frame``, ``period``, ``t_s``, ``player_id``, ``team``,
  ``x``, ``y``;
- ``<id>_ball.parquet``: ``frame``, ``period``, ``t_s``, ``x``, ``y``, ``z``, ``alive``, ``possession``;
- ``<id>_roster.parquet``: ``player_id``, ``team``, ``team_id``, ``shirt``, ``position``, ``starting``.

Frame and time: ``frame`` is the native DFL frame number (first half from 10 000, second half from 100 000,
25 Hz); ``t_s`` is the time in seconds since the first frame of the period.

Pitch frame: positions are kept in DFL's native frame: origin at the centre spot, metres, x along the long
axis and y along the short axis, right-handed like the internal frame (ADR-0001). There is no broadcast
camera in this data set, so which touchline is the "far" one is undefined; physical KPIs only use distances
between positions of the same player, which do not depend on that choice (or on any rotation or reflection
of the pitch).

The positions file is read with a streaming parser (``xml.etree.ElementTree.iterparse``) into compact
arrays, in about 5 s and 1.2 GB per match. kloppy's Sportec reader keeps every frame as Python objects, which
is much heavier for full matches; a data test checks our parser against it on the first frames of a match.
"""

from __future__ import annotations

import array
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from predfoot.paths import data_dir

#: Native frame rate of the DFL positions feed.
IDSSE_FPS = 25

#: Hugging Face dataset that mirrors the IDSSE XML files.
IDSSE_REPO_ID = "pysport/idsse-data"

#: Match id -> DFL competition id (first and second Bundesliga, season 2022/23).
IDSSE_MATCHES: dict[str, str] = {
    "J03WMX": "DFL-COM-000001",
    "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}

_GAME_SECTIONS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}
_TEAM_ROLES = {"home": "home", "guest": "away"}


@dataclass(frozen=True)
class IdssePaths:
    """Raw inputs and parsed outputs of one match."""

    match_information: Path
    positions: Path
    players: Path
    ball: Path
    roster: Path


def idsse_dir() -> Path:
    """Parsed IDSSE tables."""
    return data_dir() / "idsse"


def match_paths(match_id: str) -> IdssePaths:
    """Where the raw XML files and the parsed tables of ``match_id`` live."""
    if match_id not in IDSSE_MATCHES:
        raise ValueError(f"unknown IDSSE match {match_id!r}; expected one of {sorted(IDSSE_MATCHES)}")
    competition = IDSSE_MATCHES[match_id]
    raw = idsse_dir() / "raw"
    out = idsse_dir()
    return IdssePaths(
        match_information=raw / f"DFL_02_01_matchinformation_{competition}_DFL-MAT-{match_id}.xml",
        positions=raw / f"DFL_04_03_positions_raw_observed_{competition}_DFL-MAT-{match_id}.xml",
        players=out / f"{match_id}.parquet",
        ball=out / f"{match_id}_ball.parquet",
        roster=out / f"{match_id}_roster.parquet",
    )


def download_match(match_id: str) -> IdssePaths:
    """Download the match-information and positions XML files of ``match_id`` (skipped when up to date)."""
    from huggingface_hub import hf_hub_download

    paths = match_paths(match_id)
    paths.positions.parent.mkdir(parents=True, exist_ok=True)
    for target in (paths.match_information, paths.positions):
        hf_hub_download(
            repo_id=IDSSE_REPO_ID,
            repo_type="dataset",
            filename=target.name,
            local_dir=target.parent,
        )
    return paths


def parse_match_information(path: Path) -> pl.DataFrame:
    """Read the roster of a DFL match-information file (no names: ids, shirt numbers and positions only)."""
    rows: list[dict[str, object]] = []
    root = ET.parse(path).getroot()
    for team in root.iter("Team"):
        role = team.get("Role", "")
        if role not in _TEAM_ROLES:
            raise ValueError(f"{path.name}: unknown team role {role!r}")
        for player in team.iter("Player"):
            rows.append(
                {
                    "player_id": player.get("PersonId"),
                    "team": _TEAM_ROLES[role],
                    "team_id": team.get("TeamId"),
                    "shirt": int(player.get("ShirtNumber", "-1")),
                    "position": player.get("PlayingPosition", ""),
                    "starting": player.get("Starting") == "true",
                }
            )
    if not rows:
        raise ValueError(f"{path.name}: no players found")
    return pl.DataFrame(rows).sort("team", "shirt")


def parse_positions(path: Path, roster: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stream a DFL positions file into a player table and a ball table (see the module docstring).

    Players absent from ``roster`` (officials, if a feed contains them) are skipped.
    """
    team_of = dict(zip(roster["player_id"].to_list(), roster["team"].to_list(), strict=True))

    p_frame, p_period, p_player = array.array("q"), array.array("b"), array.array("l")
    p_x, p_y = array.array("d"), array.array("d")
    b_frame, b_period = array.array("q"), array.array("b")
    b_x, b_y, b_z = array.array("d"), array.array("d"), array.array("d")
    b_alive, b_possession = array.array("b"), array.array("b")
    player_codes: dict[str, int] = {}

    period: int | None = None
    target: str | None = None  # "ball", a player id, or None to skip the current FrameSet
    code = -1
    frameset: ET.Element | None = None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if elem.tag == "Frame":
            if event != "end":
                continue
            if target is not None and period is not None:
                frame = int(elem.attrib["N"])
                if target == "ball":
                    b_frame.append(frame)
                    b_period.append(period)
                    b_x.append(float(elem.attrib["X"]))
                    b_y.append(float(elem.attrib["Y"]))
                    b_z.append(float(elem.attrib.get("Z", "nan")))
                    b_alive.append(int(elem.attrib.get("BallStatus", "0")) == 1)
                    b_possession.append(int(elem.attrib.get("BallPossession", "0")))
                else:
                    p_frame.append(frame)
                    p_period.append(period)
                    p_player.append(code)
                    p_x.append(float(elem.attrib["X"]))
                    p_y.append(float(elem.attrib["Y"]))
            if frameset is not None:
                frameset.remove(elem)  # keep memory flat: the FrameSet never holds more than one frame
        elif elem.tag == "FrameSet":
            if event == "start":
                frameset = elem
                period = _GAME_SECTIONS.get(elem.get("GameSection", ""))
                if elem.get("TeamId") == "BALL":
                    target = "ball"
                else:
                    person = elem.get("PersonId", "")
                    target = person if person in team_of else None
                    if target is not None:
                        code = player_codes.setdefault(person, len(player_codes))
            else:
                frameset, target, period = None, None, None
                elem.clear()

    if not p_frame:
        raise ValueError(f"{path.name}: no player frames found")
    if not b_frame:
        raise ValueError(f"{path.name}: no ball frames found")

    ids = np.array(list(player_codes), dtype=object)
    players = pl.DataFrame(
        {
            "frame": np.frombuffer(p_frame, dtype=np.int64),
            "period": np.frombuffer(p_period, dtype=np.int8),
            "player_id": ids[np.frombuffer(p_player, dtype=np.int64)].astype(str),
            "x": np.frombuffer(p_x, dtype=np.float64),
            "y": np.frombuffer(p_y, dtype=np.float64),
        }
    ).with_columns(pl.col("player_id").replace_strict(team_of).alias("team"))
    ball = pl.DataFrame(
        {
            "frame": np.frombuffer(b_frame, dtype=np.int64),
            "period": np.frombuffer(b_period, dtype=np.int8),
            "x": np.frombuffer(b_x, dtype=np.float64),
            "y": np.frombuffer(b_y, dtype=np.float64),
            "z": np.frombuffer(b_z, dtype=np.float64),
            "alive": np.frombuffer(b_alive, dtype=np.int8).astype(bool),
            "possession": np.frombuffer(b_possession, dtype=np.int8),
        }
    )

    # Time since the first frame of each period, over every object of the feed.
    starts = (
        pl.concat([players.select("period", "frame"), ball.select("period", "frame")])
        .group_by("period")
        .agg(pl.col("frame").min().alias("period_start"))
    )

    def with_time(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.join(starts, on="period")
            .with_columns(((pl.col("frame") - pl.col("period_start")) / IDSSE_FPS).alias("t_s"))
            .drop("period_start")
        )

    players = with_time(players).select("frame", "period", "t_s", "player_id", "team", "x", "y")
    ball = with_time(ball).select("frame", "period", "t_s", "x", "y", "z", "alive", "possession")
    duplicated = players.select(pl.struct("player_id", "frame").is_duplicated().any()).item()
    if duplicated:
        raise ValueError(f"{path.name}: a player has two positions in the same frame")
    return players.sort("player_id", "frame"), ball.sort("frame")


def prepare_match(match_id: str, *, force: bool = False) -> IdssePaths:
    """Download (if needed) and parse one match into Parquet; a no-op when the tables already exist."""
    paths = match_paths(match_id)
    if not force and paths.players.exists() and paths.ball.exists() and paths.roster.exists():
        return paths
    if not (paths.positions.exists() and paths.match_information.exists()):
        download_match(match_id)
    roster = parse_match_information(paths.match_information)
    players, ball = parse_positions(paths.positions, roster)
    paths.players.parent.mkdir(parents=True, exist_ok=True)
    players.write_parquet(paths.players, compression="zstd")
    ball.write_parquet(paths.ball, compression="zstd")
    roster.write_parquet(paths.roster, compression="zstd")
    return paths


def load_players(match_id: str) -> pl.DataFrame:
    """Player positions of a parsed match (run :func:`prepare_match` first)."""
    return pl.read_parquet(match_paths(match_id).players)


def load_ball(match_id: str) -> pl.DataFrame:
    """Ball positions of a parsed match."""
    return pl.read_parquet(match_paths(match_id).ball)


def load_roster(match_id: str) -> pl.DataFrame:
    """Roster of a parsed match."""
    return pl.read_parquet(match_paths(match_id).roster)
