"""Golden comparison of run outputs: exact on structure and ids, one written decimal of slack on numbers."""

import copy
import json
import shutil

import pytest

from predfoot.io import compare, skillcorner
from tests.fixtures import FIXTURES

EXPECTED = FIXTURES / "golden" / "expected"


def _first_record():
    return json.loads((EXPECTED / "tracking.jsonl").read_text().splitlines()[0])


def test_units_follow_the_writer():
    assert compare.UNIT["x"] == 10.0**-skillcorner.XY_DECIMALS == 0.01
    assert compare.UNIT["p90_radius_m"] == 10.0**-skillcorner.RADIUS_DECIMALS
    assert compare.UNIT["pos_cov_m2"] == 10.0**-skillcorner.COV_DECIMALS
    assert compare.UNIT["calib_health"] == 10.0**-skillcorner.HEALTH_DECIMALS


def test_one_unit_of_the_last_decimal_is_tolerated():
    want = _first_record()
    assert compare.record_differences(copy.deepcopy(want), want) == []
    nudged = copy.deepcopy(want)
    nudged["player_data"][0]["x"] += 0.01
    nudged["player_data"][0]["p90_radius_m"] -= 0.001
    nudged["player_data"][0]["pos_cov_m2"][0][1] += 1e-4
    nudged["image_corners_projection"]["x_top_left"] += 0.01
    assert compare.record_differences(nudged, want) == []


def _shift(record, key, delta):
    record["player_data"][0][key] += delta


@pytest.mark.parametrize(
    "change",
    [
        lambda r: _shift(r, "x", 0.02),
        lambda r: _shift(r, "p90_radius_m", 0.003),
        lambda r: r["player_data"][0].__setitem__("pos_cov_m2", None),
        lambda r: r["player_data"][0].__setitem__("x", None),
        lambda r: r["player_data"][0].__setitem__("x", True),
        lambda r: r["player_data"][0].__setitem__("player_id", 999),
        lambda r: r["player_data"].pop(),
        lambda r: r.__setitem__("abstain", True),
        lambda r: r.__setitem__("timestamp", "00:00:09.00"),
        lambda r: r.pop("calib_health"),
        lambda r: r["image_corners_projection"].__setitem__("y_top_left", 0.0),
    ],
)
def test_real_changes_are_reported(change):
    want = _first_record()
    changed = copy.deepcopy(want)
    change(changed)
    assert compare.record_differences(changed, want) != []


def test_compare_runs_and_its_command_line(tmp_path, capsys):
    out = tmp_path / "out"
    shutil.copytree(EXPECTED, out)
    meta = json.loads((out / "meta.json").read_text())
    meta |= {"git_sha": "0" * 40, "predfoot_version": "9.9.9"}  # volatile: ignored
    (out / "meta.json").write_text(json.dumps(meta))
    assert compare.compare_runs(out, EXPECTED) == []
    assert compare.main([str(out), str(EXPECTED)]) == 0
    lines = (out / "tracking.jsonl").read_text().splitlines()
    record = json.loads(lines[0])
    record["player_data"][0]["y"] += 0.5
    (out / "tracking.jsonl").write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n")
    metrics = json.loads((out / "metrics.json").read_text())
    metrics["frames"]["abstained"] = 0
    (out / "metrics.json").write_text(json.dumps(metrics))
    diffs = compare.compare_runs(out, EXPECTED)
    assert len(diffs) == 2
    assert diffs[0].startswith("frame 0 player")
    assert diffs[1].startswith("metrics.json frames")
    assert compare.main([str(out), str(EXPECTED)]) == 1
    assert "2 difference(s)" in capsys.readouterr().err
    assert compare.main([str(out)]) == 2
