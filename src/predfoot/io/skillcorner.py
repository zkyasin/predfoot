"""SkillCorner open-data format: tracking writer and reader, image corners, minimal ``match.json`` (ADR-0001).

Format checked on 2026-10-30 against SkillCorner open data (match 1874553, MIT): one JSON object per line
with ``frame``, ``timestamp`` (``HH:MM:SS.ss``), ``period``, ``ball_data``, ``possession``,
``image_corners_projection`` and ``player_data`` (``x``, ``y``, ``player_id``, ``is_detected``), at 10 fps, in
the internal pitch frame.

predfoot adds two frame fields, ``calib_health`` and ``abstain``, and two player fields, ``p90_radius_m`` and
``pos_cov_m2``. Every record carries ``ball_data`` and ``possession`` (with null values when unknown): kloppy
reads ``frame["ball_data"]`` and ``frame["possession"].get(...)`` without checking that the keys exist.

kloppy's reader for this format (``player_data``, its "V3" path) looks every ``player_id`` up in the players
of ``match.json`` and raises ``KeyError`` on an unknown id, so :func:`make_match_json` lists every track.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from predfoot.contracts import FloatArray, Localization
from predfoot.io.files import atomic_write_chunks
from predfoot.io.timebase import parse_timestamp

#: Frame rate of SkillCorner open data and of predfoot exports.
EXPORT_FPS = 10
DEFAULT_PITCH_LENGTH_M = 105.0
DEFAULT_PITCH_WIDTH_M = 68.0
#: Top image corners are clipped this far beyond the far touchline (y = 34 + 5 = 39 m on a 68 m wide pitch).
CORNER_MARGIN_M = 5.0
#: χ²₂ quantile at 0.90, -2 ln(0.1) ≈ 4.605. ``pos_cov_m2`` is calibrated so that its ellipse
#: {p : (p - x̂)ᵀ Σ⁻¹ (p - x̂) ≤ CHI2_2_090} is the 90 % prediction region.
CHI2_2_090 = -2.0 * math.log(0.10)

#: Corner order of ``image_corners_projection`` and of :attr:`TrackingFrame.corners`.
CORNER_NAMES = ("top_left", "bottom_left", "bottom_right", "top_right")
CORNER_KEYS = tuple(f"{axis}_{name}" for name in CORNER_NAMES for axis in ("x", "y"))
SKILLCORNER_KEYS = (
    "frame",
    "timestamp",
    "period",
    "ball_data",
    "possession",
    "image_corners_projection",
    "player_data",
)
EXTENSION_KEYS = ("calib_health", "abstain")
PLAYER_KEYS = ("x", "y", "player_id", "is_detected")
PLAYER_EXTENSION_KEYS = ("p90_radius_m", "pos_cov_m2")
POSSESSION_GROUPS = ("home team", "away team")

# Decimals written to the file. Positions and corners: 1 cm, like SkillCorner. Covariances: 1e-4 m², which
# moves eigenvalues by at most 1e-4 m² (Weyl); the validator uses that as its tolerance.
XY_DECIMALS = 2
RADIUS_DECIMALS = 3
COV_DECIMALS = 4
HEALTH_DECIMALS = 3

Team = Literal["home", "away"]
HOME_TEAM_ID = 1
AWAY_TEAM_ID = 2


@dataclass(frozen=True)
class Ball:
    """``ball_data`` of a record; predfoot does not track the ball, so its exports keep the defaults."""

    x: float | None = None
    y: float | None = None
    z: float | None = None
    is_detected: bool | None = None


@dataclass(frozen=True)
class Possession:
    """``possession`` of a record (``group`` is ``"home team"``, ``"away team"`` or None)."""

    player_id: int | None = None
    group: str | None = None


@dataclass(frozen=True)
class TrackingFrame:
    """One record of a SkillCorner-style tracking file, in the internal pitch frame.

    ``frame`` is the export frame index (10 fps) and ``timestamp`` its time; ``players`` carry ``player_id``
    as ``track_id``. ``corners`` has shape (4, 2) in :data:`CORNER_NAMES` order, or is None when there is no
    camera.
    """

    frame: int
    players: tuple[Localization, ...] = ()
    timestamp: str | None = None
    period: int | None = 1
    corners: FloatArray | None = None
    ball: Ball = Ball()
    possession: Possession = Possession()
    calib_health: float | None = None
    abstain: bool = False


# --- image corners --------------------------------------------------------------------------------


def image_corners_projection(
    H: ArrayLike,
    image_size: tuple[int, int],
    *,
    pitch_width: float = DEFAULT_PITCH_WIDTH_M,
    margin_m: float = CORNER_MARGIN_M,
) -> FloatArray | None:
    """Ground footprint of the image, as SkillCorner's ``image_corners_projection``; shape (4, 2) or None.

    ``H`` maps pitch points to pixels (``CalibResult.H``). The image corners are the outer pixel edges, at
    (-0.5, -0.5) and (W - 0.5, H - 0.5) in OpenCV's convention. A top corner that projects beyond
    ``y_max = pitch_width / 2 + margin_m`` (39 m by default) or above the horizon is replaced by the point
    where the image's side border crosses y = y_max on the ground. That is our reading of SkillCorner's
    clipping: their top corners sit at y = 39.0 exactly, with x left free (we saw x = -137 m). Bottom corners
    are not clipped (we saw y = -48 m). Returns None when a bottom corner is above the horizon or beyond the
    clipping line, or when a top corner above the horizon sits on a border that never crosses the clipping
    line (a border heading along the touchline); we do not know how SkillCorner handles that last case.
    """
    h = np.asarray(H, dtype=np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        raise ValueError("H must be a finite 3×3 matrix")
    if h[2, 2] == 0.0:
        raise ValueError("H[2, 2] = 0: the centre spot is not in front of the camera")
    h = h / h[2, 2]  # sign convention of CalibResult: w > 0 in front of the camera
    width, height = image_size
    y_max = pitch_width / 2 + margin_m
    top, bottom = -0.5, height - 0.5
    sides = []
    for u in (-0.5, width - 0.5):
        side = _clipped_side(h, u, top, bottom, y_max)
        if side is None:
            return None
        sides.append(side)
    (top_left, bottom_left), (top_right, bottom_right) = sides
    corners = np.array([top_left, bottom_left, bottom_right, top_right])
    return corners if np.all(np.isfinite(corners)) else None


def _clipped_side(
    h: FloatArray, u: float, v_top: float, v_bottom: float, y_max: float
) -> tuple[FloatArray, FloatArray] | None:
    """(top, bottom) ground points of the image border at column ``u``, the top one clipped at y = y_max."""
    qb = np.linalg.solve(h, [u, v_bottom, 1.0])
    qt = np.linalg.solve(h, [u, v_top, 1.0])
    if qb[2] <= 0.0 or qb[1] / qb[2] >= y_max:
        return None
    bottom = qb[:2] / qb[2]
    if qt[2] > 0.0 and qt[1] / qt[2] <= y_max:
        return qt[:2] / qt[2], bottom
    # Pixels along the border map to q(t) = qb + t (qt - qb) in homogeneous pitch coordinates, so the crossing
    # of y = y_max solves a linear equation: q_y(t) - y_max q_w(t) = 0. It is valid only in front of the
    # camera.
    a = qb[1] - y_max * qb[2]
    b = (qt[1] - qb[1]) - y_max * (qt[2] - qb[2])
    if b <= 0.0:
        return None
    q = qb + (-a / b) * (qt - qb)
    if q[2] <= 0.0:
        return None
    return np.array([q[0] / q[2], y_max]), bottom


# --- writing --------------------------------------------------------------------------------------


def _number(value: float, decimals: int, what: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{what} must be finite, got {v}")
    return round(v, decimals) + 0.0  # + 0.0 turns -0.0 into 0.0


def _optional_number(value: float | None, decimals: int, what: str) -> float | None:
    return None if value is None else _number(value, decimals, what)


def _player_record(loc: Localization, abstain: bool) -> dict[str, Any]:
    tid = int(loc.track_id)
    if tid < 0:
        raise ValueError(f"track ids must be non-negative, got {tid}")
    record: dict[str, Any] = {"x": None, "y": None, "player_id": tid, "is_detected": bool(loc.is_detected)}
    if loc.xy is None:
        if loc.cov_xy is not None or loc.p90_radius_m is not None:
            raise ValueError(f"track {tid}: uncertainty without a position")
        return record | {"p90_radius_m": None, "pos_cov_m2": None}
    if abstain:
        raise ValueError(f"track {tid} has a position in a frame that abstains")
    xy = np.asarray(loc.xy, dtype=np.float64)
    if xy.shape != (2,):
        raise ValueError(f"track {tid}: xy has shape {xy.shape}, expected (2,)")
    record["x"] = _number(xy[0], XY_DECIMALS, f"track {tid} x")
    record["y"] = _number(xy[1], XY_DECIMALS, f"track {tid} y")
    radius = None
    if loc.p90_radius_m is not None:
        if not loc.p90_radius_m >= 0:  # also refuses NaN
            raise ValueError(f"track {tid}: p90 radius must be ≥ 0, got {loc.p90_radius_m}")
        radius = _number(loc.p90_radius_m, RADIUS_DECIMALS, f"track {tid} radius")
    record["p90_radius_m"] = radius
    record["pos_cov_m2"] = None if loc.cov_xy is None else _cov_record(loc.cov_xy, tid)
    return record


def _cov_record(cov_xy: ArrayLike, tid: int) -> list[list[float]]:
    cov = np.asarray(cov_xy, dtype=np.float64)
    if cov.shape != (2, 2):
        raise ValueError(f"track {tid}: covariance has shape {cov.shape}, expected (2, 2)")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"track {tid}: covariance must be finite")
    cov = 0.5 * (cov + cov.T)
    if np.linalg.eigvalsh(cov)[0] < -1e-9 * max(1.0, float(np.trace(cov))):
        raise ValueError(f"track {tid}: covariance is not positive semi-definite")
    c00, c01, c11 = (_number(v, COV_DECIMALS, "covariance") for v in (cov[0, 0], cov[0, 1], cov[1, 1]))
    return [[c00, c01], [c01, c11]]


def _corners_record(corners: FloatArray | None) -> dict[str, float | None]:
    if corners is None:
        return dict.fromkeys(CORNER_KEYS)
    c = np.asarray(corners, dtype=np.float64)
    if c.shape != (4, 2):
        raise ValueError(f"corners have shape {c.shape}, expected (4, 2)")
    values = [_number(v, XY_DECIMALS, "corner") for v in c.reshape(-1)]
    return dict(zip(CORNER_KEYS, values, strict=True))


def frame_to_record(frame: TrackingFrame) -> dict[str, Any]:
    """JSON-ready record of one frame. Raises ``ValueError`` when the frame breaks the export contract."""
    if frame.frame < 0:
        raise ValueError(f"frame index must be non-negative, got {frame.frame}")
    if frame.period is not None and frame.period < 1:
        raise ValueError(f"period must be ≥ 1 or None, got {frame.period}")
    if frame.timestamp is not None:
        parse_timestamp(frame.timestamp)  # raises on anything but HH:MM:SS.ss
    if frame.calib_health is not None and not 0.0 <= frame.calib_health <= 1.0:
        raise ValueError(f"calib_health must be in [0, 1], got {frame.calib_health}")
    if frame.possession.group not in (None, *POSSESSION_GROUPS):
        raise ValueError(f"possession group must be one of {POSSESSION_GROUPS} or None")
    players = []
    seen: set[int] = set()
    for loc in sorted(frame.players, key=lambda p: p.track_id):
        if loc.track_id in seen:
            raise ValueError(f"frame {frame.frame}: duplicate track {loc.track_id}")
        seen.add(loc.track_id)
        players.append(_player_record(loc, frame.abstain))
    ball = frame.ball
    return {
        "frame": int(frame.frame),
        "timestamp": frame.timestamp,
        "period": frame.period,
        "ball_data": {
            "x": _optional_number(ball.x, XY_DECIMALS, "ball x"),
            "y": _optional_number(ball.y, XY_DECIMALS, "ball y"),
            "z": _optional_number(ball.z, XY_DECIMALS, "ball z"),
            "is_detected": ball.is_detected,
        },
        "possession": {"player_id": frame.possession.player_id, "group": frame.possession.group},
        "image_corners_projection": _corners_record(frame.corners),
        "calib_health": _optional_number(frame.calib_health, HEALTH_DECIMALS, "calib_health"),
        "abstain": bool(frame.abstain),
        "player_data": players,
    }


def dumps_record(record: Mapping[str, Any]) -> str:
    """One JSONL line (without the newline), with the separators SkillCorner uses. NaN is refused."""
    return json.dumps(record, allow_nan=False)


def write_tracking(path: Path, frames: Iterable[TrackingFrame]) -> int:
    """Write frames as JSONL, atomically; returns the number of records."""
    count = 0

    def lines() -> Iterator[bytes]:
        nonlocal count
        for frame in frames:
            count += 1
            yield (dumps_record(frame_to_record(frame)) + "\n").encode("utf-8")

    atomic_write_chunks(Path(path), lines())
    return count


# --- reading --------------------------------------------------------------------------------------


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Raw records of a JSONL file, line by line (blank lines are skipped)."""
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({err.msg})") from err
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: a record must be a JSON object")
            yield record


def _corners_from_record(raw: Mapping[str, Any] | None) -> FloatArray | None:
    if raw is None:
        return None
    values = [raw.get(k) for k in CORNER_KEYS]
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError("image corners are partially null")
    return np.array(values, dtype=np.float64).reshape(4, 2)


def _player_from_record(raw: Mapping[str, Any]) -> Localization:
    x, y = raw.get("x"), raw.get("y")
    if (x is None) != (y is None):
        raise ValueError(f"player {raw.get('player_id')}: x and y must both be null or both be numbers")
    cov = raw.get("pos_cov_m2")
    return Localization(
        track_id=int(raw["player_id"]),
        xy=None if x is None else np.array([x, y], dtype=np.float64),
        cov_xy=None if cov is None else np.array(cov, dtype=np.float64),
        p90_radius_m=raw.get("p90_radius_m"),
        is_detected=bool(raw.get("is_detected")),
    )


def record_to_frame(record: Mapping[str, Any]) -> TrackingFrame:
    """Parse one record, from predfoot or from SkillCorner (extension fields then keep their defaults)."""
    ball = record.get("ball_data") or {}
    possession = record.get("possession") or {}
    return TrackingFrame(
        frame=int(record["frame"]),
        players=tuple(_player_from_record(p) for p in record.get("player_data") or ()),
        timestamp=record.get("timestamp"),
        period=record.get("period"),
        corners=_corners_from_record(record.get("image_corners_projection")),
        ball=Ball(
            x=ball.get("x"),
            y=ball.get("y"),
            z=ball.get("z"),
            is_detected=ball.get("is_detected"),
        ),
        possession=Possession(player_id=possession.get("player_id"), group=possession.get("group")),
        calib_health=record.get("calib_health"),
        abstain=bool(record.get("abstain", False)),
    )


def read_tracking(path: Path) -> Iterator[TrackingFrame]:
    """Frames of a SkillCorner-style JSONL file (predfoot output or SkillCorner open data)."""
    for lineno, record in enumerate(iter_records(path), start=1):
        try:
            yield record_to_frame(record)
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"{path}: record {lineno}: {err}") from err


# --- match.json -----------------------------------------------------------------------------------


def make_match_json(
    track_ids: Iterable[int],
    *,
    team_of: Mapping[int, Team] | None = None,
    pitch_length: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width: float = DEFAULT_PITCH_WIDTH_M,
    match_id: int | None = None,
    n_frames: int | None = None,
    fps: float = EXPORT_FPS,
) -> dict[str, Any]:
    """Minimal SkillCorner ``match.json`` that kloppy accepts: two anonymous teams, one player per track id.

    Tracks absent from ``team_of`` go to the home team: kloppy needs a team for every player, and an arbitrary
    split would look like information we do not have.
    """
    team_of = dict(team_of or {})
    for tid, team in team_of.items():
        if team not in ("home", "away"):
            raise ValueError(f"track {tid}: team must be 'home' or 'away', got {team!r}")
    players = []
    for tid in sorted({int(t) for t in track_ids}):
        team_id = AWAY_TEAM_ID if team_of.get(tid) == "away" else HOME_TEAM_ID
        players.append(
            {
                "id": tid,
                "trackable_object": tid,
                "team_id": team_id,
                "number": None,
                "first_name": "",
                "last_name": f"Track {tid}",
                "short_name": f"Track {tid}",
                "start_time": "00:00:00",
                "end_time": None,
                "player_role": {"id": None, "name": "Unknown", "acronym": None},
            }
        )
    periods = []
    if n_frames:
        periods.append(
            {
                "period": 1,
                "name": "period_1",
                "start_frame": 0,
                "end_frame": n_frames - 1,
                "duration_frames": n_frames,
                "duration_minutes": round(n_frames / fps / 60, 2),
            }
        )
    return {
        "id": match_id,
        "home_team_score": None,
        "away_team_score": None,
        "date_time": None,
        "home_team": {"id": HOME_TEAM_ID, "name": "Home (anonymous)", "short_name": "Home", "acronym": "HOM"},
        "away_team": {"id": AWAY_TEAM_ID, "name": "Away (anonymous)", "short_name": "Away", "acronym": "AWY"},
        "home_team_coach": None,
        "away_team_coach": None,
        "match_periods": periods,
        "referees": [],
        "players": players,
        "ball": {"trackable_object": None},
        "pitch_length": float(pitch_length),
        "pitch_width": float(pitch_width),
    }


def player_ids_in_match(match: Mapping[str, Any]) -> set[int]:
    """Ids that may appear as ``player_id`` in the tracking file of this ``match.json``."""
    return {int(p["id"]) for p in match.get("players", [])}
