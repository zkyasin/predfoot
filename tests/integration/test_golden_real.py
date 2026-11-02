"""Golden test of the real stages: calibration from synthetic network detections, Mondrian conformal
regions, localization and export, compared with the outputs committed in ``tests/fixtures/golden/expected``.

The comparison (:func:`predfoot.io.compare.compare_runs`) is exact on structure, ids, flags and texts, and
allows one unit of the last written decimal on numbers: the solver's floating point is not bit-identical
between macOS and Linux CI. After an intended change of the outputs:
``uv run python scripts/make_golden.py --update``, then review the diff of ``expected/``.
"""

import importlib.util
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from predfoot.cli import app
from predfoot.io.compare import compare_runs
from predfoot.io.validate import validate_tracking
from predfoot.paths import REPO_ROOT
from predfoot.pipeline.synthetic import SPECTATOR_TRACK_ID, make_broadcast_clip
from tests.cli_output import PLAIN_ENV, plain
from tests.fixtures import FIXTURES

_spec = importlib.util.spec_from_file_location("make_golden", REPO_ROOT / "scripts" / "make_golden.py")
assert _spec is not None
assert _spec.loader is not None
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)

GOLDEN = FIXTURES / "golden"
EXPECTED = GOLDEN / "expected"
HINT = "if the change is intended: uv run python scripts/make_golden.py --update, and review the diff"


@pytest.fixture(scope="module")
def golden_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("golden")
    return out, golden.run_golden(GOLDEN, out)


def test_the_real_stages_reproduce_the_golden_outputs(golden_run):
    out, _ = golden_run
    diffs = compare_runs(out, EXPECTED)
    assert diffs == [], f"{len(diffs)} difference(s), {HINT}:\n" + "\n".join(diffs[:20])


def test_predfoot_run_with_detections_reproduces_them_too(tmp_path):
    """The command the Docker workflow runs in the cli image, with the default block size."""
    args = ["run", str(GOLDEN / "clip.json"), "--out", str(tmp_path)]
    args += ["--detections", str(GOLDEN / "detections.json"), "--conformal", str(GOLDEN / "conformal.json")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "8 frames at 10 fps from 20 native frames, 1 abstained" in result.output
    assert compare_runs(tmp_path, EXPECTED) == []


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
    bins = golden.player_bins(GOLDEN)
    assert {key for _, _, key, _ in bins} == {"0-0", "0-1", "1-0", "1-1", "2-0", "2-1"}  # "2-0": pooled
    assert min(margin for *_, margin in bins) >= golden.MIN_EDGE_MARGIN
    meta = json.loads((out / "meta.json").read_text())
    assert meta["config"]["stages"].startswith("real+synthetic-detections-1+det@")
    assert meta["conformal"]["kind"] == "mondrian"
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


def test_conformal_needs_detections(tmp_path):
    args = ["run", str(GOLDEN / "clip.json"), "--out", str(tmp_path)]
    args += ["--conformal", str(GOLDEN / "conformal.json")]
    result = CliRunner(env=PLAIN_ENV).invoke(app, args)
    assert result.exit_code == 2
    assert "needs --detections" in plain(result)


def test_a_bad_detections_file_is_a_usage_error(tmp_path):
    bad = tmp_path / "dets.json"
    bad.write_text(json.dumps({"format": "predfoot-clip/1"}), encoding="utf-8")
    args = ["run", str(GOLDEN / "clip.json"), "--out", str(tmp_path / "out"), "--detections", str(bad)]
    result = CliRunner(env=PLAIN_ENV).invoke(app, args)
    assert result.exit_code == 2
    assert "not a predfoot-detections/1 file" in plain(result)


def test_detections_keyed_by_image_number_are_a_usage_error(tmp_path):
    """Keys shifted by one, as image numbers are: refused before anything runs, not run with the cameras of
    the neighbouring frames."""
    dets = json.loads((GOLDEN / "detections.json").read_text(encoding="utf-8"))
    for entry in dets["frames"]:
        entry["frame"] += 1
    shifted = tmp_path / "dets.json"
    shifted.write_text(json.dumps(dets), encoding="utf-8")
    out = tmp_path / "out"
    args = ["run", str(GOLDEN / "clip.json"), "--out", str(out), "--detections", str(shifted)]
    result = CliRunner(env=PLAIN_ENV).invoke(app, args)
    assert result.exit_code == 2, result.output
    text = plain(result)
    assert "--detections" in text
    assert "beyond the 20 frames of clip 'synthetic-golden-001' (20)" in text
    assert "keyed by the 0-based index of the clip's frames" in text
    assert not out.exists()


def test_a_conformal_model_with_an_infinite_edge_is_refused_up_front(tmp_path):
    model = json.loads((GOLDEN / "conformal.json").read_text(encoding="utf-8"))
    model["edges"]["mpp"] = [0.035, float("inf")]
    path = tmp_path / "conformal.json"
    path.write_text(json.dumps(model), encoding="utf-8")  # Infinity, as Python writes it
    out = tmp_path / "out"
    args = [
        "run",
        str(GOLDEN / "clip.json"),
        "--out",
        str(out),
        "--detections",
        str(GOLDEN / "detections.json"),
    ]
    result = CliRunner(env=PLAIN_ENV).invoke(app, [*args, "--conformal", str(path)])
    assert result.exit_code == 2, result.output
    assert "bin edges that are not sorted finite numbers" in plain(result)
    assert not out.exists()
