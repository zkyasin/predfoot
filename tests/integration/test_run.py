"""End-to-end tracer bullet: ``predfoot run`` on the synthetic clip, resume after a hard kill (T0.7, T5.1)."""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from kloppy import skillcorner as kloppy_skillcorner
from typer.testing import CliRunner

from predfoot.cli import app
from predfoot.io.skillcorner import read_tracking
from predfoot.io.validate import validate_tracking
from predfoot.pipeline.run import PARTS_DIR, RunConfig, run_clip
from predfoot.pipeline.synthetic import SPECTATOR_TRACK_ID, make_broadcast_clip
from tests.fixtures import FIXTURES

CLIP = FIXTURES / "synthetic_clip.json"
OUTPUTS = ("tracking.jsonl", "match.json", "meta.json", "metrics.json", "minimap.png")
#: Everything but metrics.json, which records timings and reused blocks.
STABLE = ("tracking.jsonl", "match.json", "meta.json", "minimap.png")


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("clean")
    result = run_clip(CLIP, out, RunConfig(block_size=7))
    return out, result


def _bytes(out, names=STABLE):
    return {name: (out / name).read_bytes() for name in names}


def test_run_writes_every_output_and_cleans_up(clean_run):
    out, result = clean_run
    for name in OUTPUTS:
        assert (out / name).stat().st_size > 0, name
    assert not (out / PARTS_DIR).exists()
    assert (result.n_native, result.n_export, result.n_abstained) == (75, 30, 4)
    assert (result.blocks_computed, result.blocks_reused) == (5, 0)


def test_output_is_valid_and_loads_in_kloppy(clean_run):
    out, _ = clean_run
    match = json.loads((out / "match.json").read_text())
    report = validate_tracking(out / "tracking.jsonl", match=match)
    assert report.ok, report.issues
    with (out / "match.json").open("rb") as meta, (out / "tracking.jsonl").open("rb") as raw:
        dataset = kloppy_skillcorner.load(
            meta_data=meta, raw_data=raw, coordinates="skillcorner", include_empty_frames=True
        )
    assert len(dataset.records) == 30


def test_positions_match_the_ground_truth(clean_run):
    """Inverse homography + 25 → 10 fps interpolation reproduce the true positions to the export's 1 cm."""
    out, _ = clean_run
    _, truth = make_broadcast_clip()
    checked = 0
    for frame in read_tracking(out / "tracking.jsonl"):
        native = frame.frame * 2.5
        i0, w = int(native), native - int(native)
        for loc in frame.players:
            if loc.xy is None:
                continue
            expected = (
                truth[(i0, loc.track_id)]
                if w == 0
                else ((1 - w) * truth[(i0, loc.track_id)] + w * truth[(i0 + 1, loc.track_id)])
            )
            # 5 mm export rounding, plus foot points stored to 0.001 px: ≤ 0.1 mm where 1 px spans 0.2 m.
            np.testing.assert_allclose(loc.xy, expected, atol=0.005 + 2e-4)
            checked += 1
    assert checked > 300


def test_abstentions_and_rejections_are_explicit(clean_run):
    out, _ = clean_run
    frames = list(read_tracking(out / "tracking.jsonl"))
    assert [f.frame for f in frames if f.abstain] == [16, 17, 18, 19]  # native frames 40-49 failed
    assert all(f.corners is None for f in frames if f.abstain)
    spectator = [p for f in frames for p in f.players if p.track_id == SPECTATOR_TRACK_ID]
    assert spectator
    assert all(p.xy is None for p in spectator)  # detected above the horizon: no ground position
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["frames"]["abstained"] == 4
    assert metrics["players"]["unlocated_in_live_frames"] == len(spectator)


def test_meta_records_provenance(clean_run):
    out, _ = clean_run
    meta = json.loads((out / "meta.json").read_text())
    assert meta["schema_version"] == "1.0"
    assert len(meta["config_hash"]) == 64
    assert meta["input"]["file"] == CLIP.name
    assert meta["export_fps"] == 10.0
    assert meta["git_sha"] is None or len(meta["git_sha"]) == 40
    assert "placeholder" in meta["stages"]["localization"]


def test_output_does_not_depend_on_the_block_size(clean_run, tmp_path):
    out, _ = clean_run
    run_clip(CLIP, tmp_path, RunConfig(block_size=1000))
    assert _bytes(tmp_path) == _bytes(out)


_CRASHING_RUN = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    import predfoot.pipeline.run as run

    real, calls = run._compute_block, 0

    def crash_after_two_blocks(*args, **kwargs):
        global calls
        calls += 1
        if calls > 2:
            os._exit(17)  # like SIGKILL: no exception, no cleanup, no flush
        return real(*args, **kwargs)

    run._compute_block = crash_after_two_blocks
    run.run_clip(Path(sys.argv[1]), Path(sys.argv[2]), run.RunConfig(block_size=int(sys.argv[3])))
    """
)


def _crash(out, block_size=7):
    proc = subprocess.run(
        [sys.executable, "-c", _CRASHING_RUN, str(CLIP), str(out), str(block_size)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 17, proc.stderr


def test_killed_run_resumes_to_byte_identical_outputs(clean_run, tmp_path):
    out, _ = clean_run
    _crash(tmp_path)
    parts = tmp_path / PARTS_DIR
    assert sorted(p.name for p in parts.glob("block-*.jsonl")) == ["block-000000.jsonl", "block-000001.jsonl"]
    assert not (tmp_path / "tracking.jsonl").exists()
    (parts / ".block-000002.jsonl.tmp").write_text('{"frame": 14, "trunc')  # a write cut short by the kill
    result = run_clip(CLIP, tmp_path, RunConfig(block_size=7))
    assert (result.blocks_computed, result.blocks_reused) == (3, 2)
    assert _bytes(tmp_path) == _bytes(out)
    assert not parts.exists()


def test_stale_blocks_from_another_configuration_are_discarded(clean_run, tmp_path):
    out, _ = clean_run
    _crash(tmp_path, block_size=3)  # blocks of 3 frames: the wrong size for the next run
    result = run_clip(CLIP, tmp_path, RunConfig(block_size=7))
    assert result.blocks_reused == 0
    assert _bytes(tmp_path) == _bytes(out)


def test_rerun_of_a_finished_run_is_identical(clean_run, tmp_path):
    out, _ = clean_run
    run_clip(CLIP, tmp_path, RunConfig(block_size=7))
    run_clip(CLIP, tmp_path, RunConfig(block_size=7))
    assert _bytes(tmp_path) == _bytes(out)


def test_cli_run_then_validate(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(CLIP), "--out", str(tmp_path), "--fps", "10"])
    assert result.exit_code == 0, result.output
    assert "30 frames at 10 fps" in result.output
    check = runner.invoke(
        app, ["validate", str(tmp_path / "tracking.jsonl"), "--match", str(tmp_path / "match.json")]
    )
    assert check.exit_code == 0, check.output


def test_cli_run_reports_invalid_clips(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "predfoot-clip/1", "clip_id": "x"}))
    result = CliRunner().invoke(app, ["run", str(bad), "--out", str(tmp_path / "out")])
    assert result.exit_code == 2
    assert "invalid clip file" in result.output


def test_cli_run_rejects_a_non_positive_rate(tmp_path):
    result = CliRunner().invoke(app, ["run", str(CLIP), "--out", str(tmp_path), "--fps", "0"])
    assert result.exit_code != 0
    assert "--fps" in result.output


def test_blocks_from_other_stages_are_never_reused(tmp_path, monkeypatch):
    import dataclasses

    import predfoot.pipeline.run as run_module

    real, calls = run_module._compute_block, []

    def fail_on_third_block(*args, **kwargs):
        calls.append(1)
        if len(calls) > 2:
            raise RuntimeError("interrupted")
        return real(*args, **kwargs)

    monkeypatch.setattr(run_module, "_compute_block", fail_on_third_block)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_clip(CLIP, tmp_path, RunConfig(block_size=7))
    monkeypatch.setattr(run_module, "_compute_block", real)
    config = RunConfig(block_size=7)
    other = dataclasses.replace(run_module.stub_stages(config), version="calib-v2")
    result = run_clip(CLIP, tmp_path, config, stages=other)
    assert result.blocks_reused == 0
    assert json.loads((tmp_path / "meta.json").read_text())["config"]["stages"] == "calib-v2"


def test_run_config_rejects_bad_settings():
    with pytest.raises(ValueError, match="export_fps"):
        RunConfig(export_fps=0)
    with pytest.raises(ValueError, match="block_size"):
        RunConfig(block_size=0)
