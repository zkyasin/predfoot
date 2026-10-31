"""Compare the outputs of two runs: a golden check that tolerates the rounding of other platforms.

Structure, ids, flags and texts must match exactly. Numbers may differ by one unit of the last decimal the
writer keeps (:mod:`predfoot.io.skillcorner`: 1 cm for positions and corners, 1 mm for radii, 1e-4 m² for
covariances, 1e-3 for health): the solver's floating point is not bit-identical between macOS (Accelerate)
and Linux (OpenBLAS), so a value close to a rounding boundary can round the other way, by one unit.

    python -m predfoot.io.compare OUT_DIR EXPECTED_DIR

compares ``tracking.jsonl``, ``match.json``, ``meta.json`` (without the commit and the package version) and
the deterministic part of ``metrics.json`` (no timings, no block counts); exit code 1 lists the differences.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from predfoot.io.skillcorner import (
    CORNER_KEYS,
    COV_DECIMALS,
    HEALTH_DECIMALS,
    RADIUS_DECIMALS,
    XY_DECIMALS,
)

#: One unit of the last written decimal, per field.
UNIT = {
    "x": 10.0**-XY_DECIMALS,
    "y": 10.0**-XY_DECIMALS,
    "corner": 10.0**-XY_DECIMALS,
    "p90_radius_m": 10.0**-RADIUS_DECIMALS,
    "pos_cov_m2": 10.0**-COV_DECIMALS,
    "calib_health": 10.0**-HEALTH_DECIMALS,
}
#: meta.json fields that change with every commit or release.
VOLATILE_META = ("git_sha", "predfoot_version")
METRICS_KEYS = ("clip_id", "frames", "players")
#: Differences listed at most.
MAX_DIFFERENCES = 50


def normalised_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if k not in VOLATILE_META}


def normalised_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """The deterministic part of metrics.json."""
    return {k: metrics.get(k) for k in METRICS_KEYS}


def _number_differs(got: Any, want: Any, unit: float) -> bool:
    if got is None or want is None:
        return got is not want
    if isinstance(got, bool) or isinstance(want, bool):
        return True
    if not isinstance(got, int | float) or not isinstance(want, int | float):
        return True
    return not (math.isfinite(got) and abs(got - want) <= unit * (1 + 1e-6))


def _player_differences(got: Mapping[str, Any], want: Mapping[str, Any], where: str) -> list[str]:
    if list(got) != list(want):
        return [f"{where}: keys {list(got)} != {list(want)}"]
    out = [
        f"{where} {key}: {got[key]!r} != {want[key]!r}"
        for key in ("player_id", "is_detected")
        if got[key] != want[key]
    ]
    for key in ("x", "y", "p90_radius_m"):
        if _number_differs(got[key], want[key], UNIT[key]):
            out.append(f"{where} {key}: {got[key]!r} != {want[key]!r} ± {UNIT[key]:g}")
    g, w = got["pos_cov_m2"], want["pos_cov_m2"]
    if g is None or w is None:
        if g is not w:
            out.append(f"{where} pos_cov_m2: {g!r} != {w!r}")
    else:
        flat_g = [v for row in g for v in row] if isinstance(g, list) else []
        flat_w = [v for row in w for v in row]
        if len(flat_g) != len(flat_w) or any(
            _number_differs(a, b, UNIT["pos_cov_m2"]) for a, b in zip(flat_g, flat_w, strict=False)
        ):
            out.append(f"{where} pos_cov_m2: {g!r} != {w!r} ± {UNIT['pos_cov_m2']:g}")
    return out


def record_differences(got: Mapping[str, Any], want: Mapping[str, Any]) -> list[str]:
    """Differences between two tracking records (empty when they match)."""
    where = f"frame {want.get('frame')}"
    if list(got) != list(want):
        return [f"{where}: keys {list(got)} != {list(want)}"]
    out: list[str] = []
    for key, value in want.items():
        if key == "player_data":
            ids = [p.get("player_id") for p in got[key]]
            if ids != [p.get("player_id") for p in value]:
                out.append(f"{where}: player ids {ids} != {[p.get('player_id') for p in value]}")
                continue
            for g, w in zip(got[key], value, strict=True):
                out += _player_differences(g, w, f"{where} player {w.get('player_id')}")
        elif key == "image_corners_projection":
            if list(got[key]) != list(value) or list(value) != list(CORNER_KEYS):
                out.append(f"{where}: corner keys {list(got[key])}")
                continue
            out += [
                f"{where} {corner}: {got[key][corner]!r} != {value[corner]!r} ± {UNIT['corner']:g}"
                for corner in CORNER_KEYS
                if _number_differs(got[key][corner], value[corner], UNIT["corner"])
            ]
        elif key == "calib_health":
            if _number_differs(got[key], value, UNIT["calib_health"]):
                out.append(f"{where} calib_health: {got[key]!r} != {value!r}")
        elif got[key] != value:  # frame, timestamp, period, ball_data, possession, abstain: exact
            out.append(f"{where} {key}: {got[key]!r} != {value!r}")
    return out


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tracking_differences(got_path: Path, want_path: Path) -> list[str]:
    got, want = _records(got_path), _records(want_path)
    frames_got, frames_want = [r.get("frame") for r in got], [r.get("frame") for r in want]
    if frames_got != frames_want:
        return [f"tracking.jsonl: frames {frames_got[:10]}… != {frames_want[:10]}…"]
    out: list[str] = []
    for g, w in zip(got, want, strict=True):
        out += record_differences(g, w)
    return out


def _dict_differences(name: str, got: Mapping[str, Any], want: Mapping[str, Any]) -> list[str]:
    keys = sorted(set(got) | set(want))
    return [f"{name} {k}: {got.get(k)!r} != {want.get(k)!r}" for k in keys if got.get(k) != want.get(k)]


def compare_runs(out_dir: Path, expected_dir: Path) -> list[str]:
    """Differences between the outputs in ``out_dir`` and the golden files in ``expected_dir`` (whose
    meta.json and metrics.json may already be normalised). Empty when the run reproduces them."""
    out_dir, expected_dir = Path(out_dir), Path(expected_dir)

    def load(path: Path) -> dict[str, Any]:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"{path} is not a JSON object")
        return obj

    diffs = tracking_differences(out_dir / "tracking.jsonl", expected_dir / "tracking.jsonl")
    diffs += _dict_differences("match.json", load(out_dir / "match.json"), load(expected_dir / "match.json"))
    diffs += _dict_differences(
        "meta.json",
        normalised_meta(load(out_dir / "meta.json")),
        normalised_meta(load(expected_dir / "meta.json")),
    )
    diffs += _dict_differences(
        "metrics.json",
        normalised_metrics(load(out_dir / "metrics.json")),
        normalised_metrics(load(expected_dir / "metrics.json")),
    )
    return diffs


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m predfoot.io.compare OUT_DIR EXPECTED_DIR", file=sys.stderr)
        return 2
    diffs = compare_runs(Path(args[0]), Path(args[1]))
    if not diffs:
        print(f"{args[0]} reproduces {args[1]}")
        return 0
    for line in diffs[:MAX_DIFFERENCES]:
        print(line, file=sys.stderr)
    more = f" (first {MAX_DIFFERENCES} shown)" if len(diffs) > MAX_DIFFERENCES else ""
    print(f"{len(diffs)} difference(s){more}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
