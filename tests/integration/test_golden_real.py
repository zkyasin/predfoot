"""Golden test of the real stages (T5.3): calibration from synthetic network detections, Mondrian conformal
regions, localization and export, compared with the outputs committed in ``tests/fixtures/golden/expected``.

Structure, ids, flags and texts must match exactly. Numbers may differ by one unit of their last written
decimal: the solver's floating point is not bit-identical between macOS (Accelerate) and Linux CI
(OpenBLAS), and a value near a rounding boundary can then round the other way. After an intended change of
the outputs: ``uv run python scripts/make_golden.py --update``, then review the diff of ``expected/``.
"""

import copy
import importlib.util
import json
import math

import numpy as np
import pytest

from predfoot.io.skillcorner import (
    CORNER_KEYS,
    COV_DECIMALS,
    HEALTH_DECIMALS,
    RADIUS_DECIMALS,
    XY_DECIMALS,
)
from predfoot.io.validate import validate_tracking
from predfoot.paths import REPO_ROOT
from predfoot.pipeline.synthetic import SPECTATOR_TRACK_ID, make_broadcast_clip
from tests.fixtures import FIXTURES

_spec = importlib.util.spec_from_file_location("make_golden", REPO_ROOT / "scripts" / "make_golden.py")
assert _spec is not None
assert _spec.loader is not None
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)

GOLDEN = FIXTURES / "golden"
EXPECTED = GOLDEN / "expected"
#: One unit of the last decimal the writer keeps, per field.
UNIT = {
    "x": 10.0**-XY_DECIMALS,
    "y": 10.0**-XY_DECIMALS,
    "corner": 10.0**-XY_DECIMALS,
    "p90_radius_m": 10.0**-RADIUS_DECIMALS,
    "pos_cov_m2": 10.0**-COV_DECIMALS,
    "calib_health": 10.0**-HEALTH_DECIMALS,
}
HINT = "if the change is intended: uv run python scripts/make_golden.py --update, and review the diff"


def _close(got, want, unit, where):
    if want is None or got is None:
        assert got is want, f"{where}: {got!r} != {want!r} ({HINT})"
        return
    assert math.isfinite(got), f"{where}: {got!r}"
    assert abs(got - want) <= unit * (1 + 1e-6), f"{where}: {got} != {want} ± {unit} ({HINT})"


def assert_same_player(got, want, where):
    assert got.keys() == want.keys(), where
    assert got["player_id"] == want["player_id"], where
    assert got["is_detected"] == want["is_detected"], where
    for key in ("x", "y", "p90_radius_m"):
        _close(got[key], want[key], UNIT[key], f"{where} {key}")
    if want["pos_cov_m2"] is None or got["pos_cov_m2"] is None:
        assert got["pos_cov_m2"] is want["pos_cov_m2"], f"{where} pos_cov_m2 ({HINT})"
    else:
        for i in range(2):
            for j in range(2):
                _close(got["pos_cov_m2"][i][j], want["pos_cov_m2"][i][j], UNIT["pos_cov_m2"], f"{where} cov")


def assert_same_record(got, want):
    where = f"frame {want['frame']}"
    assert list(got) == list(want), f"{where}: keys {list(got)} != {list(want)}"
    for key, value in want.items():
        if key == "player_data":
            ids = [p["player_id"] for p in got[key]]
            assert ids == [p["player_id"] for p in value], f"{where}: player ids {ids} ({HINT})"
            for g, w in zip(got[key], value, strict=True):
                assert_same_player(g, w, f"{where} player {w['player_id']}")
        elif key == "image_corners_projection":
            assert list(got[key]) == list(CORNER_KEYS) == list(value), where
            for corner in CORNER_KEYS:
                _close(got[key][corner], value[corner], UNIT["corner"], f"{where} {corner}")
        elif key == "calib_health":
            _close(got[key], value, UNIT["calib_health"], f"{where} calib_health")
        else:  # frame, timestamp, period, ball_data, possession, abstain: exact
            assert got[key] == value, f"{where} {key}: {got[key]!r} != {value!r} ({HINT})"


def assert_same_tracking(got_path, want_path):
    got = [json.loads(line) for line in got_path.read_text(encoding="utf-8").splitlines()]
    want = [json.loads(line) for line in want_path.read_text(encoding="utf-8").splitlines()]
    assert [r["frame"] for r in got] == [r["frame"] for r in want], HINT
    for g, w in zip(got, want, strict=True):
        assert_same_record(g, w)


@pytest.fixture(scope="module")
def golden_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("golden")
    return out, golden.run_golden(GOLDEN, out)


def test_tracking_matches_the_golden_file(golden_run):
    out, _ = golden_run
    assert_same_tracking(out / "tracking.jsonl", EXPECTED / "tracking.jsonl")


def test_match_meta_and_metrics_match_the_golden_files(golden_run):
    out, _ = golden_run

    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    assert load(out / "match.json") == load(EXPECTED / "match.json"), HINT
    assert golden.normalised_meta(load(out / "meta.json")) == load(EXPECTED / "meta.json"), HINT
    assert golden.normalised_metrics(load(out / "metrics.json")) == load(EXPECTED / "metrics.json"), HINT


def test_the_fixture_exercises_every_path(golden_run):
    out, result = golden_run
    assert (result.n_native, result.n_export, result.n_abstained) == (20, 8, 1)
    assert result.blocks_computed == 3  # blocks of 3 export frames
    records = [json.loads(line) for line in (out / "tracking.jsonl").read_text().splitlines()]
    # native frames 7 (three keypoints, no line) and 8 (no detection) cannot be calibrated: export frame 3,
    # halfway between them, abstains; frames 2, 10 and 15 calibrate despite their outlying keypoints
    assert [r["frame"] for r in records if r["abstain"]] == [3]
    spectator = [p for r in records for p in r["player_data"] if p["player_id"] == SPECTATOR_TRACK_ID]
    assert len(spectator) == 4  # export frames 4-7, above the horizon: no position
    assert all(p["x"] is None and p["p90_radius_m"] is None for p in spectator)
    bins = {key for _, _, key, _ in golden.player_bins(GOLDEN)}
    assert bins == {"0-0", "0-1", "1-0", "1-1", "2-0", "2-1"}  # "2-0" falls back to the pooled quantile
    assert min(margin for *_, margin in golden.player_bins(GOLDEN)) >= golden.MIN_EDGE_MARGIN
    report = validate_tracking(out / "tracking.jsonl", match=json.loads((out / "match.json").read_text()))
    assert report.ok, report.issues


def test_positions_are_close_to_the_truth(golden_run):
    """Not a golden comparison: the recovered camera puts the players where they are, within centimetres."""
    out, _ = golden_run
    _, truth = make_broadcast_clip()
    errors = []
    for line in (out / "tracking.jsonl").read_text().splitlines():
        record = json.loads(line)
        native = record["frame"] * 2.5
        i0, w = int(native), native - int(native)
        for p in record["player_data"]:
            if p["x"] is None:
                continue
            tid = p["player_id"]
            want = truth[(i0, tid)] if w == 0 else (1 - w) * truth[(i0, tid)] + w * truth[(i0 + 1, tid)]
            errors.append(np.hypot(p["x"] - want[0], p["y"] - want[1]))
    errors = np.array(errors)
    assert len(errors) == 85
    assert np.median(errors) < 0.05
    assert errors.max() < 0.25


def test_the_comparison_catches_changes():
    want = json.loads((EXPECTED / "tracking.jsonl").read_text().splitlines()[0])
    assert_same_record(copy.deepcopy(want), want)
    nudged = copy.deepcopy(want)
    nudged["player_data"][0]["x"] += 0.01  # one unit of the last decimal: tolerated
    assert_same_record(nudged, want)
    for change in (
        lambda r: r["player_data"][0].__setitem__("x", r["player_data"][0]["x"] + 0.02),
        lambda r: r["player_data"][0].__setitem__("p90_radius_m", r["player_data"][0]["p90_radius_m"] * 1.1),
        lambda r: r["player_data"][0].__setitem__("player_id", 999),
        lambda r: r["player_data"].pop(),
        lambda r: r.__setitem__("abstain", True),
        lambda r: r.pop("calib_health"),
        lambda r: r["player_data"][0].__setitem__("pos_cov_m2", None),
    ):
        changed = copy.deepcopy(want)
        change(changed)
        with pytest.raises(AssertionError):
            assert_same_record(changed, want)
