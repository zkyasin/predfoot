import copy
import json

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from predfoot.cli import app
from predfoot.io import skillcorner as sc
from predfoot.io.validate import load_schema, validate_record, validate_tracking
from tests.fixtures import FIXTURES

EXCERPT = FIXTURES / "skillcorner_opendata_excerpt.jsonl"
CORNERS = {
    "x_top_left": -38.73,
    "y_top_left": 39.0,
    "x_bottom_left": -13.74,
    "y_bottom_left": -36.22,
    "x_bottom_right": 14.04,
    "y_bottom_right": -35.84,
    "x_top_right": 37.78,
    "y_top_right": 39.0,
}


def _player(pid=7, **kw):
    p = {
        "x": 12.31,
        "y": -4.87,
        "player_id": pid,
        "is_detected": True,
        "p90_radius_m": 0.84,
        "pos_cov_m2": [[0.061, 0.018], [0.018, 0.094]],
    }
    p.update(kw)
    return p


def _record(frame=500, **kw):
    r = {
        "frame": frame,
        "timestamp": f"00:00:{frame // 10:02d}.{frame % 10}0" if frame < 600 else "00:01:00.00",
        "period": 1,
        "ball_data": {"x": None, "y": None, "z": None, "is_detected": False},
        "possession": {"player_id": None, "group": None},
        "image_corners_projection": dict(CORNERS),
        "calib_health": 0.93,
        "abstain": False,
        "player_data": [_player(7), _player(9, x=-3.0, y=20.5)],
    }
    r.update(kw)
    return r


def _abstained(r):
    r["abstain"] = True
    for p in r["player_data"]:
        p.update(x=None, y=None, p90_radius_m=None, pos_cov_m2=None)
    return r


def test_schema_is_a_valid_2020_12_schema():
    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_valid_record_has_no_issue():
    assert validate_record(_record()) == []
    assert validate_record(_abstained(_record())) == []


def _set(path, value):
    def mutate(r):
        target = r
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return r

    return mutate


def _drop(key):
    def mutate(r):
        del r[key]
        return r

    return mutate


def _rename_player_key(old, new):
    def mutate(r):
        r["player_data"][0][new] = r["player_data"][0].pop(old)
        return r

    return mutate


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_drop("ball_data"), "'ball_data' is a required property"),
        (_drop("possession"), "'possession' is a required property"),
        (_rename_player_key("player_id", "track_id"), "'player_id' is a required property"),
        (_set(("player_data", 0, "x"), None), "null together"),
        (_set(("player_data", 0, "p90_radius_m"), -0.1), "less than the minimum"),
        (_set(("player_data", 0, "pos_cov_m2"), [[0.1, 0.0]]), "pos_cov_m2"),
        (_set(("calib_health",), 1.2), "greater than the maximum"),
        (_set(("timestamp",), "0:00:50"), "does not match"),
        (_set(("possession", "group"), "home"), "group"),
        (_set(("image_corners_projection", "x_top_left"), None), "image_corners_projection"),
        (_set(("abstain",), True), "abstain"),
        (_set(("player_data", 0, "pos_cov_m2"), [[0.1, 0.05], [0.0, 0.1]]), "symmetric"),
        (_set(("player_data", 0, "pos_cov_m2"), [[0.1, 0.5], [0.5, 0.1]]), "positive semi-definite"),
        (_set(("image_corners_projection", "y_top_left"), 39.5), "clipped"),
        (_set(("player_data", 1, "player_id"), 7), "duplicate player_id 7"),
    ],
)
def test_each_rule_is_enforced(mutate, expected):
    issues = validate_record(mutate(_record()))
    assert issues, "the mutated record should fail"
    assert any(expected in msg for msg in issues), issues


def test_covariance_rounding_is_tolerated():
    # Rank-one covariance rounded to 1e-4 m²: λ_min ≈ -5e-5, within the writer's rounding.
    r = _record()
    r["player_data"][0]["pos_cov_m2"] = [[1.0, 1.00005], [1.00005, 1.0]]
    assert validate_record(r) == []


def test_mirrored_footprint_is_caught():
    """A y-flipped homography puts the top image corners on the near side."""
    r = _record()
    c = r["image_corners_projection"]
    for side in ("left", "right"):
        c[f"y_top_{side}"], c[f"y_bottom_{side}"] = -c[f"y_top_{side}"], -c[f"y_bottom_{side}"]
    issues = validate_record(r)
    assert any("far side" in msg for msg in issues), issues


def test_clipping_limit_follows_the_pitch_width():
    r = _record()
    assert validate_record(r, pitch_width=64.0)  # 39 > 32 + 5
    r["image_corners_projection"]["y_top_left"] = r["image_corners_projection"]["y_top_right"] = 37.0
    assert validate_record(r, pitch_width=64.0) == []


def _write(tmp_path, lines, name="tracking.jsonl"):
    path = tmp_path / name
    path.write_text("".join(line + "\n" for line in lines))
    return path


def _dump(records):
    return [json.dumps(r) for r in records]


def test_valid_file(tmp_path):
    path = _write(tmp_path, _dump([_record(500), _record(501), _abstained(_record(502))]))
    report = validate_tracking(path)
    assert report.ok, report.issues
    assert report.n_records == 3


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (['{"frame": NaN}'], "NaN"),
        (['{"frame": 1, "frame": 2}'], "duplicate key"),
        (["{not json"], "invalid JSON"),
        (["[1, 2]"], "JSON object"),
    ],
)
def test_unparseable_lines_are_reported_with_their_number(tmp_path, lines, expected):
    path = _write(tmp_path, [json.dumps(_record(500)), *lines])
    report = validate_tracking(path)
    assert not report.ok
    assert report.issues[0].line == 2
    assert expected in report.issues[0].message


def test_frames_must_increase(tmp_path):
    report = validate_tracking(_write(tmp_path, _dump([_record(501), _record(500)])))
    assert any("increase" in i.message for i in report.issues)


def test_timestamps_must_follow_the_export_rate(tmp_path):
    late = _record(501, timestamp="00:00:50.04")  # 25 fps spacing in a 10 fps file
    report = validate_tracking(_write(tmp_path, _dump([_record(500), late])))
    assert any("10 fps" in i.message for i in report.issues)
    assert validate_tracking(_write(tmp_path, _dump([_record(500), late]), "b.jsonl"), fps=25).ok


def test_a_frame_in_a_period_needs_a_timestamp(tmp_path):
    report = validate_tracking(_write(tmp_path, _dump([_record(500, timestamp=None)])))
    assert any("timestamp" in i.message for i in report.issues)


def test_player_ids_must_be_in_match_json(tmp_path):
    path = _write(tmp_path, _dump([_record(500)]))
    match = sc.make_match_json([7])
    report = validate_tracking(path, match=match)
    assert [i.message for i in report.issues] == ["player_id 9 is not in match.json (kloppy raises KeyError)"]
    assert validate_tracking(path, match=sc.make_match_json([7, 9])).ok


def test_issue_count_is_capped(tmp_path):
    bad = [json.dumps(_set(("calib_health",), 2.0)(_record(500 + k))) for k in range(50)]
    report = validate_tracking(_write(tmp_path, bad), max_issues=10)
    assert len(report.issues) == 10
    assert report.truncated


def test_real_skillcorner_file_passes_in_skillcorner_mode():
    report = validate_tracking(EXCERPT, extensions=False)
    assert report.ok, report.issues
    strict = validate_tracking(EXCERPT)
    assert any("'calib_health' is a required property" in i.message for i in strict.issues)


def test_skillcorner_mode_still_rejects_unknown_keys():
    records = [json.loads(line) for line in EXCERPT.read_text().splitlines()]
    r = copy.deepcopy(records[1])
    r["player_data"][0]["track_id"] = 1
    assert validate_record(r, extensions=False)


# --- CLI ---------------------------------------------------------------------------------------------------


def test_cli_validate_exit_codes(tmp_path):
    good = _write(tmp_path, _dump([_record(500)]), "good.jsonl")
    bad = _write(tmp_path, _dump([_set(("calib_health",), 2.0)(_record(500))]), "bad.jsonl")
    match = tmp_path / "match.json"
    match.write_text(json.dumps(sc.make_match_json([7, 9])))
    ok = CliRunner().invoke(app, ["validate", str(good), "--match", str(match)])
    assert ok.exit_code == 0, ok.output
    assert "OK" in ok.output
    ko = CliRunner().invoke(app, ["validate", str(bad)])
    assert ko.exit_code == 1
    assert "line 1" in ko.output
    real = CliRunner().invoke(app, ["validate", str(EXCERPT), "--skillcorner"])
    assert real.exit_code == 0, real.output
