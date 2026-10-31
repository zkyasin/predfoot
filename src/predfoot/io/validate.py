"""Validation of tracking files: JSON Schema ``frame-1.0.json`` plus the rules a schema cannot express.

Checked in code (EXECUTION.md §3.2):

- every ``pos_cov_m2`` is symmetric and positive semi-definite (up to the writer's 1e-4 m² rounding);
- top image corners are clipped at y = pitch_width / 2 + 5 m, and the footprint top-left → bottom-left →
  bottom-right → top-right turns counter-clockwise (a y-mirrored pitch turns it clockwise);
- ``player_id`` is unique within a frame, and listed in ``match.json`` when one is given (kloppy raises
  ``KeyError`` otherwise);
- frames increase, timestamps advance at the export rate, and a frame inside a period has a timestamp;
- no NaN or Infinity tokens and no duplicate keys (Python's ``json`` accepts both silently).

The schema itself carries the structural rules: required keys (``ball_data`` and ``possession`` always
present), types, ranges, all-or-nothing corners, null positions when a frame abstains, no uncertainty
without a position.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from predfoot.io.skillcorner import (
    CORNER_MARGIN_M,
    CORNER_NAMES,
    DEFAULT_PITCH_WIDTH_M,
    EXPORT_FPS,
    EXTENSION_KEYS,
    PLAYER_EXTENSION_KEYS,
)
from predfoot.io.timebase import parse_timestamp

SCHEMA_RESOURCE = "schema/frame-1.0.json"
#: Eigenvalue tolerance, m²: rounding each covariance entry to 1e-4 moves eigenvalues by at most 1e-4 (Weyl).
PSD_TOL_M2 = 1e-4
SYMMETRY_TOL_M2 = 1e-6
#: Corners are written with 2 decimals.
CORNER_TOL_M = 0.005
#: Timestamps are written in centiseconds, so a difference of two can be off by 0.01 s.
TIMESTAMP_TOL_S = 0.011


@dataclass(frozen=True)
class Issue:
    """One problem found in a file; ``line`` is 1-based, 0 for problems about the whole file."""

    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}" if self.line else self.message


@dataclass
class Report:
    """Outcome of :func:`validate_tracking`."""

    path: Path
    n_records: int = 0
    issues: list[Issue] = field(default_factory=list)
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues


def load_schema(*, extensions: bool = True) -> dict[str, Any]:
    """The frame schema. With ``extensions=False`` predfoot's extra fields become optional, so that genuine
    SkillCorner open-data records validate; unknown keys are still refused."""
    schema: dict[str, Any] = json.loads(
        resources.files("predfoot.io").joinpath(SCHEMA_RESOURCE).read_text(encoding="utf-8")
    )
    if not extensions:
        schema = copy.deepcopy(schema)
        schema["required"] = [k for k in schema["required"] if k not in EXTENSION_KEYS]
        player = schema["$defs"]["player"]
        player["required"] = [k for k in player["required"] if k not in PLAYER_EXTENSION_KEYS]
    return schema


@cache
def _validator(extensions: bool) -> Draft202012Validator:
    return Draft202012Validator(load_schema(extensions=extensions))


def _schema_message(err: ValidationError) -> str:
    where = "/".join(str(p) for p in err.absolute_path) or "record"
    path = list(err.schema_path)
    hint = ""
    if path[:1] == ["then"]:
        hint = " (the frame abstains: x, y, p90_radius_m and pos_cov_m2 must be null)"
    elif "then" in path or "else" in path:
        hint = " (x and y are null together, and a null position has no uncertainty)"
    return f"{where}: {err.message}{hint}"


def validate_record(
    record: Mapping[str, Any], *, pitch_width: float = DEFAULT_PITCH_WIDTH_M, extensions: bool = True
) -> list[str]:
    """Problems in one record (empty when valid). Semantic rules run only on schema-valid records."""
    errors = sorted(_validator(extensions).iter_errors(record), key=lambda e: list(map(str, e.absolute_path)))
    if errors:
        return [_schema_message(e) for e in errors]
    issues = []
    seen: set[int] = set()
    for player in record["player_data"]:
        pid = player["player_id"]
        if pid in seen:
            issues.append(f"duplicate player_id {pid}")
        seen.add(pid)
        cov = player.get("pos_cov_m2")
        if cov is not None:
            issues.extend(f"player {pid}: {msg}" for msg in _covariance_issues(np.array(cov, dtype=float)))
    issues.extend(_corner_issues(record["image_corners_projection"], pitch_width))
    return issues


def _covariance_issues(cov: np.ndarray) -> list[str]:
    if abs(cov[0, 1] - cov[1, 0]) > SYMMETRY_TOL_M2:
        return ["pos_cov_m2 is not symmetric"]
    smallest = float(np.linalg.eigvalsh(cov)[0])
    if smallest < -PSD_TOL_M2:
        return [f"pos_cov_m2 is not positive semi-definite (smallest eigenvalue {smallest:.4g} m²)"]
    return []


def _corner_issues(corners: Mapping[str, float | None], pitch_width: float) -> list[str]:
    if corners["y_top_left"] is None:
        return []
    y_max = pitch_width / 2 + CORNER_MARGIN_M
    issues = []
    for side in ("left", "right"):
        top = corners[f"y_top_{side}"]
        assert top is not None  # all-or-nothing is enforced by the schema
        if top > y_max + CORNER_TOL_M:
            issues.append(
                f"image corner y_top_{side} = {top} m is not clipped at y = {y_max:g} m "
                "(pitch_width / 2 + 5 m)"
            )
    if _footprint_area(corners) <= 0.0:
        issues.append(
            "the image footprint (top-left, bottom-left, bottom-right, top-right) must turn "
            "counter-clockwise; a clockwise one means a mirrored pitch (y flipped)"
        )
    return issues


def _footprint_area(corners: Mapping[str, float | None]) -> float:
    """Signed (shoelace) area of the footprint in corner order, m².

    The image rectangle, traversed top-left → bottom-left → bottom-right → top-right, maps to the ground with
    image right ≈ +x and image down ≈ -y: a reflection, so the footprint turns counter-clockwise (area > 0).
    The homography is a diffeomorphism on the ground in front of the camera, so this holds for any pan, tilt
    or zoom of an upright camera, including views along the touchline where far corners have smaller y than
    near ones. Mirroring y reverses it.
    """
    xs = [float(corners[f"x_{name}"] or 0.0) for name in CORNER_NAMES]  # all numbers (checked by the schema)
    ys = [float(corners[f"y_{name}"] or 0.0) for name in CORNER_NAMES]
    return 0.5 * sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i] for i in range(4))


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _parse(line: str) -> dict[str, Any]:
    try:
        obj = json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as err:
        raise ValueError(f"invalid JSON ({err.msg})") from err
    if not isinstance(obj, dict):
        raise ValueError("a record must be a JSON object")
    return obj


def validate_tracking(
    path: Path,
    *,
    match: Mapping[str, Any] | None = None,
    pitch_width: float | None = None,
    fps: float = EXPORT_FPS,
    extensions: bool = True,
    max_issues: int = 100,
) -> Report:
    """Validate a JSONL tracking file record by record, then across records.

    ``pitch_width`` defaults to the one in ``match`` (``match.json``), else 68 m.
    """
    if pitch_width is None:
        pitch_width = float(match["pitch_width"]) if match else DEFAULT_PITCH_WIDTH_M
    known_ids = None if match is None else {int(p["id"]) for p in match.get("players", [])}
    report = Report(path=Path(path))
    unknown_reported: set[int] = set()
    previous: dict[str, Any] | None = None

    def add(line: int, message: str) -> None:
        if len(report.issues) < max_issues:
            report.issues.append(Issue(line, message))
        else:
            report.truncated = True

    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            report.n_records += 1
            try:
                record = _parse(line)
            except ValueError as err:
                add(lineno, str(err))
                continue
            messages = validate_record(record, pitch_width=pitch_width, extensions=extensions)
            for message in messages:
                add(lineno, message)
            if messages:
                continue
            for message in _sequence_issues(previous, record, fps):
                add(lineno, message)
            if known_ids is not None:
                for player in record["player_data"]:
                    pid = player["player_id"]
                    if pid not in known_ids and pid not in unknown_reported:
                        unknown_reported.add(pid)
                        add(lineno, f"player_id {pid} is not in match.json (kloppy raises KeyError)")
            previous = record
    if report.n_records == 0:
        add(0, "the file has no records")
    return report


def _sequence_issues(previous: Mapping[str, Any] | None, record: Mapping[str, Any], fps: float) -> list[str]:
    issues = []
    if record["period"] is not None and record["timestamp"] is None:
        issues.append(f"frame {record['frame']} is in period {record['period']} but has no timestamp")
    if previous is None:
        return issues
    if record["frame"] <= previous["frame"]:
        issues.append(
            f"frame {record['frame']} does not increase (previous record: frame {previous['frame']})"
        )
        return issues
    same_period = record["period"] is not None and record["period"] == previous["period"]
    if same_period and record["timestamp"] is not None and previous["timestamp"] is not None:
        dt = parse_timestamp(record["timestamp"]) - parse_timestamp(previous["timestamp"])
        expected = (record["frame"] - previous["frame"]) / fps
        if abs(dt - expected) > TIMESTAMP_TOL_S:
            issues.append(
                f"timestamp {record['timestamp']} is {dt:.2f} s after the previous record, but frames "
                f"{previous['frame']} → {record['frame']} at {fps:g} fps are {expected:.2f} s apart"
            )
    return issues
