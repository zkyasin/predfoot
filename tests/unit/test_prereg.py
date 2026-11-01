"""Guards and frozen artifacts of the preregistered test run (scripts/freeze_prereg.py, run_test_once.py).

The guards run on throw-away git repositories; nothing here touches data, the real lock or test files.
"""

import importlib.util
import json
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import typer

from predfoot.calib import temporal as tm
from predfoot.contracts import CalibResult, CalibStatus
from predfoot.eval import frozen as fz
from predfoot.eval import positions as ps
from predfoot.eval import prereg as pr
from predfoot.health import detector as hd
from predfoot.health.features import FEATURES
from predfoot.keypoints.cache import cache_path, load_detections, save_detections
from predfoot.paths import REPO_ROOT

FINAL_TEXT = "# Preregistration of the test run\n\n- Version: 1\n- Status: approved by the owner.\n"
CLIP = ("SNGS-999", "test", 7)  # the one fake test clip pinned by the throw-away locks
MODEL = "pnlcalib-sv@fake"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden" / "detections.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_test_once = _load("run_test_once")
freeze_prereg = _load("freeze_prereg")


def git(root, *args):
    cmd = ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args]
    subprocess.run(cmd, check=True, capture_output=True)


def commit_all(root, message):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)


def write_lock(root, frozen_at="2026-11-01T00:00:00+00:00", **changes):
    """A lock of the current preregistration and artifacts at HEAD, pinning the fake test clip."""
    fields = {
        "code_commit": pr.head_commit(root),
        "frozen_at_utc": frozen_at,
        "config_digests": {"detection_model": MODEL},
        "test_split": {"clips": 1, "clip_ids_sha256": fz.clip_list_digest([(CLIP[0], CLIP[2])])},
        "environment": pr.environment(),
    } | changes
    lock = pr.make_lock(root, Path("artifacts/prereg/v1"), **fields)
    (root / pr.LOCK_PATH).write_text(lock.to_json())
    return lock


@pytest.fixture
def repo(tmp_path):
    """A repository whose preregistration is frozen: code, artifacts and a matching lock, all committed."""
    root = tmp_path / "repo"
    for d in ("docs", "src", "scripts", "configs", "artifacts/prereg/v1"):
        (root / d).mkdir(parents=True)
    (root / pr.PREREG_PATH).write_text(FINAL_TEXT)
    (root / "src" / "m.py").write_text("x = 1\n")
    (root / "scripts" / "s.py").write_text("y = 2\n")
    (root / "configs" / "c.yaml").write_text("a: 1\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (root / "uv.lock").write_text("version = 1\n")
    (root / ".gitignore").write_text("__pycache__/\ndata/\n")
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    commit_all(root, "code")
    (root / "artifacts/prereg/v1/regions.json").write_text('{"q": 1.0}\n')
    write_lock(root)
    commit_all(root, "freeze")
    return root


def test_a_matching_lock_passes(repo):
    lock = pr.verify_lock(repo)
    assert lock.preregistration["version"] == 1
    assert list(lock.artifacts) == ["artifacts/prereg/v1/regions.json"]
    assert "frozen on 2026-11-01" in lock.readme_line()


def test_a_tampered_preregistration_is_refused_even_when_committed(repo):
    (repo / pr.PREREG_PATH).write_text(FINAL_TEXT + "- P1 now uses the oracle.\n")
    commit_all(repo, "edit the preregistration")
    with pytest.raises(pr.PreregError, match="preregistration changed"):
        pr.verify_lock(repo)


def test_a_modified_artifact_is_refused(repo):
    (repo / "artifacts/prereg/v1/regions.json").write_text('{"q": 1.5}\n')
    commit_all(repo, "retune")
    with pytest.raises(pr.PreregError, match="modified since the freeze"):
        pr.verify_lock(repo)


def test_an_added_or_removed_artifact_is_refused(repo):
    (repo / "artifacts/prereg/v1/extra.json").write_text("{}\n")
    commit_all(repo, "add")
    with pytest.raises(pr.PreregError, match="unexpected"):
        pr.verify_lock(repo)
    (repo / "artifacts/prereg/v1/extra.json").unlink()
    (repo / "artifacts/prereg/v1/regions.json").unlink()
    commit_all(repo, "remove")
    with pytest.raises(pr.PreregError, match="missing"):
        pr.verify_lock(repo)


@pytest.mark.parametrize(
    "path", ["src/m.py", "scripts/s.py", "configs/c.yaml", "pyproject.toml", "uv.lock", ".python-version"]
)
def test_a_code_change_since_the_lock_commit_is_refused(repo, path):
    (repo / path).write_text("changed = True\n")
    commit_all(repo, "change after the freeze")
    with pytest.raises(pr.PreregError, match="code changed since the lock commit"):
        pr.verify_lock(repo)


def test_a_change_outside_the_protected_paths_is_accepted(repo):
    (repo / "README.md").write_text("Preregistration frozen.\n")
    commit_all(repo, "readme")
    pr.verify_lock(repo)


def test_the_protected_set_is_the_code_s_not_the_lock_s(repo):
    lock = json.loads((repo / pr.LOCK_PATH).read_text())
    (repo / pr.LOCK_PATH).write_text(json.dumps(lock | {"protected_paths": []}))
    commit_all(repo, "a lock that protects nothing")  # refused anyway: the lock changed after its freeze
    with pytest.raises(pr.PreregError):
        pr.verify_lock(repo)
    lock = pr.read_lock(repo)
    (repo / "src" / "m.py").write_text("x = 4\n")
    commit_all(repo, "code change")
    with pytest.raises(pr.PreregError, match="code changed"):
        pr.verify_code(repo, lock)


def test_editing_the_preregistration_and_the_lock_together_is_refused(repo):
    (repo / pr.PREREG_PATH).write_text(FINAL_TEXT + "- P1 now uses the oracle.\n")
    lock = json.loads((repo / pr.LOCK_PATH).read_text())
    lock["preregistration"]["sha256"] = pr.sha256_file(repo / pr.PREREG_PATH)
    (repo / pr.LOCK_PATH).write_text(json.dumps(lock))
    commit_all(repo, "edit both")
    pr.verify_hashes(repo, pr.read_lock(repo))  # the hashes alone would pass
    with pytest.raises(pr.PreregError, match="changed since the freeze commit"):
        pr.verify_lock(repo)


def test_pointing_the_lock_at_a_newer_code_commit_is_refused(repo):
    (repo / "src" / "m.py").write_text("x = 5\n")
    commit_all(repo, "code change after the freeze")
    lock = json.loads((repo / pr.LOCK_PATH).read_text())
    (repo / pr.LOCK_PATH).write_text(json.dumps(lock | {"code_commit": pr.head_commit(repo)}))
    commit_all(repo, "move the lock")
    with pytest.raises(pr.PreregError, match="changed since the freeze commit"):
        pr.verify_lock(repo)


def test_an_uncommitted_lock_hidden_from_git_is_refused(tmp_path):
    root = tmp_path / "r"
    (root / "docs").mkdir(parents=True)
    (root / "src").mkdir()
    (root / pr.PREREG_PATH).write_text(FINAL_TEXT)
    (root / "src" / "m.py").write_text("x = 1\n")
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    commit_all(root, "code")
    (root / "artifacts/prereg/v1").mkdir(parents=True)
    (root / "artifacts/prereg/v1/regions.json").write_text("{}\n")
    write_lock(root)
    (root / ".git/info/exclude").write_text("artifacts/\ndocs/preregistration.lock.json\n")
    assert pr.dirty_paths(root) == []  # invisible to git status
    with pytest.raises(pr.PreregError, match="not tracked"):
        pr.verify_lock(root)


def test_a_freeze_commit_with_other_changes_is_refused(repo):
    (repo / pr.LOCK_PATH).unlink()
    commit_all(repo, "remove the lock")
    base = pr.head_commit(repo)
    write_lock(repo, code_commit=base)
    (repo / "README.md").write_text("also in the freeze commit\n")
    commit_all(repo, "freeze with extras")
    with pytest.raises(pr.PreregError, match="adds more than the lock"):
        pr.verify_lock(repo)


def test_a_re_freeze_is_accepted_and_ties_nothing_to_the_old_lock(repo):
    old = pr.lock_sha256(repo)
    (repo / pr.LOCK_PATH).unlink()
    (repo / "src" / "m.py").write_text("x = 6\n")
    commit_all(repo, "fix the code, drop the lock")
    write_lock(repo, frozen_at="2026-11-02T00:00:00+00:00")
    commit_all(repo, "re-freeze")
    lock = pr.verify_lock(repo)
    assert pr.freeze_commit(repo) == pr.head_commit(repo)
    assert lock.code_commit != pr.head_commit(repo)
    assert pr.lock_sha256(repo) != old


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_a_tracked_file_hiding_its_changes_is_refused(repo, flag):
    git(repo, "update-index", flag, "src/m.py")
    (repo / "src" / "m.py").write_text("x = 7\n")
    assert pr.dirty_paths(repo) == []  # git status shows nothing
    with pytest.raises(pr.PreregError, match="hidden from git status"):
        pr.verify_lock(repo)


def test_an_ignored_file_under_a_protected_path_is_refused_but_bytecode_is_not(repo):
    (repo / "src" / "__pycache__").mkdir()
    (repo / "src" / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"\0")
    pr.verify_lock(repo)
    (repo / "src" / "data").mkdir()
    (repo / "src" / "data" / "__init__.py").write_text("shadow = True\n")  # ignored by 'data/'
    with pytest.raises(pr.PreregError, match="ignored"):
        pr.verify_lock(repo)


def test_artifact_hashes_skip_hidden_files(repo):
    (repo / "artifacts/prereg/v1/.DS_Store").write_bytes(b"finder")
    pr.verify_hashes(repo, pr.read_lock(repo))


def test_the_environment_must_be_the_frozen_one(repo):
    lock = pr.read_lock(repo)
    pr.verify_environment(lock)
    bumped = pr.Lock(**{**lock.__dict__, "environment": lock.environment | {"scipy": "0.0"}})
    with pytest.raises(pr.PreregError, match=r"scipy 0\.0 at the freeze"):
        pr.verify_environment(bumped)
    with pytest.raises(pr.PreregError, match="no environment"):
        pr.verify_environment(pr.Lock(**{**lock.__dict__, "environment": {}}))


def test_the_readme_line_binds_the_artifacts_and_the_lock(repo):
    lock = pr.read_lock(repo)
    other = pr.Lock(**{**lock.__dict__, "artifacts": {"artifacts/prereg/v1/regions.json": "0" * 64}})
    assert lock.readme_line() != other.readme_line()
    assert pr.lock_sha256(repo)[:16] in lock.readme_line(pr.lock_sha256(repo))


def test_a_dirty_tree_is_refused(repo):
    (repo / "notes.txt").write_text("untracked\n")
    with pytest.raises(pr.PreregError, match="not clean"):
        pr.verify_lock(repo)
    (repo / "notes.txt").unlink()
    (repo / "src" / "m.py").write_text("x = 2\n")  # uncommitted code change
    with pytest.raises(pr.PreregError, match="not clean"):
        pr.verify_lock(repo)


def test_the_freeze_refuses_a_dirty_tree_and_a_draft(repo, monkeypatch):
    monkeypatch.setattr(freeze_prereg, "REPO_ROOT", repo)
    (repo / "notes.txt").write_text("untracked\n")
    with pytest.raises(typer.Exit) as exc:
        freeze_prereg.main(calib_from=None)
    assert exc.value.exit_code == 2
    (repo / "notes.txt").unlink()
    (repo / pr.PREREG_PATH).write_text(FINAL_TEXT.replace("approved by the owner", "DRAFT"))
    commit_all(repo, "back to draft")
    with pytest.raises(typer.Exit) as exc:
        freeze_prereg.main(calib_from=None)
    assert exc.value.exit_code == 2


def test_the_freeze_refuses_to_overwrite_a_lock(repo, monkeypatch):
    monkeypatch.setattr(freeze_prereg, "REPO_ROOT", repo)
    monkeypatch.setattr(freeze_prereg, "load_dev_clips", lambda **_: pytest.fail("must refuse first"))
    with pytest.raises(typer.Exit) as exc:
        freeze_prereg.main(calib_from=None)
    assert exc.value.exit_code == 2


def test_the_draft_markers_are_refused():
    for text in ("# Preregistration (DRAFT)\n", "choice (proposed, pending owner approval)\n"):
        with pytest.raises(pr.PreregError, match="not final"):
            pr.require_final_text(text)
    pr.require_final_text(FINAL_TEXT)


def test_the_current_preregistration_is_version_1_and_final():
    text = (REPO_ROOT / pr.PREREG_PATH).read_text()
    assert pr.prereg_version(text) == 1
    pr.require_final_text(text)  # no draft marker: the freeze accepts it


def test_a_second_evaluation_is_refused(tmp_path):
    markers = [tmp_path / "data" / "STAGE_C_STARTED.json", tmp_path / "run" / "STAGE_C_STARTED.json"]
    pr.claim_once(markers, {"started": "now"})
    assert all(json.loads(m.read_text()) == {"started": "now"} for m in markers)
    with pytest.raises(pr.PreregError, match="second evaluation"):
        pr.claim_once(markers, {"started": "again"})
    markers[1].chmod(0o644)
    markers[1].unlink()  # deleting the local run folder does not reopen the evaluation
    with pytest.raises(pr.PreregError, match="second evaluation"):
        pr.claim_once(markers, {"started": "again"})


def _context(repo, tmp_path):
    return run_test_once.Context(
        dry=False,
        root=repo,
        run_dir=tmp_path / "run",
        markers=[tmp_path / "markers" / "STAGE_C_STARTED.json", tmp_path / "run" / "STAGE_C_STARTED.json"],
        cache_root=tmp_path / "cache",
        report=tmp_path / "test_run.md",
        persons_path=tmp_path / "persons.parquet",
        lines_path=tmp_path / "lines.parquet",
        clips=[CLIP],
    )


@pytest.mark.parametrize("tamper", ["no lock", "preregistration", "code"])
def test_stage_c_is_not_reachable_without_a_matching_lock(repo, tmp_path, monkeypatch, tamper):
    if tamper == "no lock":
        (repo / pr.LOCK_PATH).unlink()
    elif tamper == "preregistration":
        (repo / pr.PREREG_PATH).write_text(FINAL_TEXT + "edited\n")
    else:
        (repo / "src" / "m.py").write_text("x = 3\n")
    commit_all(repo, tamper)

    def no_labels(*args, **kwargs):
        pytest.fail("stage C read a table before the lock was checked")

    monkeypatch.setattr(run_test_once.pl, "read_parquet", no_labels)
    ctx = _context(repo, tmp_path)
    with pytest.raises(pr.PreregError):
        run_test_once.stage_c(ctx)
    assert not any(m.exists() for m in ctx.markers)


def test_stage_c_refuses_when_a_test_report_exists(repo, tmp_path, monkeypatch):
    ctx = _context(repo, tmp_path)
    ctx.report.parent.mkdir(parents=True, exist_ok=True)
    ctx.report.write_text("# Preregistered test run\n")
    with pytest.raises(pr.PreregError, match="already ran"):
        run_test_once.stage_c(ctx)
    assert not any(m.exists() for m in ctx.markers)


def test_stages_a_and_b_refuse_without_the_lock(repo, tmp_path):
    (repo / pr.LOCK_PATH).unlink()
    commit_all(repo, "no lock")
    ctx = _context(repo, tmp_path)
    for stage in (run_test_once.stage_a, run_test_once.stage_b):
        with pytest.raises(pr.PreregError, match="no lock"):
            stage(ctx)


# --- stages on a fake data folder -------------------------------------------------------------------------


def _golden_by_image():
    dets, _ = load_detections(GOLDEN)
    return {1 + 5 * i: d for i, d in dets.items()}  # image numbers of frames downloaded one in five


def _put_detections(ctx, repo, version=MODEL):
    """The fake clip's detections in the test cache of the current lock (what stage A's networks write)."""
    dets = _golden_by_image()
    root = ctx.cache_root / f"lock-{pr.lock_sha256(repo)[:16]}"
    save_detections(cache_path(CLIP[0], len(dets), root=root), dets, 0.0, version)


LABEL_SCHEMAS = {
    "persons": {"clip_id": pl.String, "frame": pl.Int32, "track_id": pl.Int64, "foot_u": pl.Float64,
                "foot_v": pl.Float64, "x": pl.Float64, "y": pl.Float64, "x_raw": pl.Float64,
                "y_raw": pl.Float64, "truncated": pl.Boolean},
    "lines": {"clip_id": pl.String, "frame": pl.Int32, "line_class": pl.String, "point_idx": pl.Int32,
              "u": pl.Float64, "v": pl.Float64, "is_planar": pl.Boolean},
}  # fmt: skip


def _put_label_tables(ctx):
    """Label tables without a single row, with the columns of the index (no label exists in these tests)."""
    pl.DataFrame(schema=LABEL_SCHEMAS["persons"]).write_parquet(ctx.persons_path)
    pl.DataFrame(schema=LABEL_SCHEMAS["lines"]).write_parquet(ctx.lines_path)


@pytest.fixture
def world(repo, tmp_path, monkeypatch):
    """Stages A, B and C on the throw-away repository with a valid lock: one fake test clip (empty frame
    files, the golden synthetic detections already in the lock's test cache), default noise parameters as
    the frozen artifacts, no download and no weights. Nothing reads real data."""
    data = tmp_path / "data"
    monkeypatch.setenv("PREDFOOT_DATA_DIR", str(data))
    img = data / "sn-gsr" / "test" / CLIP[0] / "img1"
    img.mkdir(parents=True)
    for f in _golden_by_image():
        (img / f"{f:06d}.jpg").write_bytes(b"")
    ctx = _context(repo, tmp_path)
    _put_detections(ctx, repo)
    _put_label_tables(ctx)
    art = fz.Artifacts(tm.NoiseParams(), tm.SmootherConfig().digest(), {}, None, {})  # type: ignore[arg-type]
    monkeypatch.setattr(run_test_once, "artifacts", lambda ctx: art)
    monkeypatch.setattr(run_test_once, "verify_weights", lambda lock: None)
    fake = SimpleNamespace(run=lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(run_test_once, "subprocess", fake)
    return ctx


def _fresh(ctx):
    """A new invocation: same folders, nothing in memory."""
    return run_test_once.Context(**{**ctx.__dict__, "lock": None, "lock_sha": "", "log_lines": []})


def test_stages_a_and_b_record_the_lock_and_resume(world):
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)
    a = json.loads((world.work_dir / "stage_a.json").read_text())
    b = json.loads((world.work_dir / "stage_b.json").read_text())
    assert a["lock_sha256"] == b["lock_sha256"] == world.lock_sha
    assert a["cache"][CLIP[0]]["model_version"] == MODEL
    _, meta = ps.load_calib(world.work_dir / "stage_b" / f"{CLIP[0]}.npz")
    assert meta["lock_sha256"] == world.lock_sha
    _, meta = fz.load_outputs(world.work_dir / "calib_ptz.npz")
    assert meta["lock_sha256"] == world.lock_sha
    assert meta["params_digest"] == tm.NoiseParams().digest()
    again = _fresh(world)
    run_test_once.stage_b(again)
    assert any("already done for this lock" in line for line in again.log_lines)


def test_a_re_freeze_reuses_nothing_from_the_old_lock(world, repo):
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)
    old = world.work_dir
    (repo / pr.LOCK_PATH).unlink()
    commit_all(repo, "drop the lock")
    write_lock(repo, frozen_at="2026-11-03T00:00:00+00:00")
    commit_all(repo, "re-freeze")
    ctx = _fresh(world)
    with pytest.raises(RuntimeError, match=r"stage_a\.json is missing"):
        run_test_once.stage_b(ctx)  # the new lock's run folder is empty: stage B cannot reuse the old one
    assert ctx.work_dir != old
    with pytest.raises(RuntimeError, match=r"stage_a\.json is missing"):
        run_test_once.stage_c(_fresh(world))
    assert not any(m.exists() for m in world.markers)
    # outputs of the old lock copied into the new lock's folder are refused, not evaluated
    for name in ("stage_a.json", "stage_b.json", "calib_perframe.npz", "calib_ptz.npz"):
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        (ctx.work_dir / name).write_bytes((old / name).read_bytes())
    _put_detections(ctx, repo)
    with pytest.raises(pr.PreregError, match="another lock"):
        run_test_once.stage_c(_fresh(world))
    assert not any(m.exists() for m in world.markers)


def test_a_truncated_part_file_is_recomputed(world):
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)
    part = world.work_dir / "stage_b" / f"{CLIP[0]}.npz"
    expected, _ = ps.load_calib(part)
    part.write_bytes(part.read_bytes()[:100])
    (world.work_dir / "stage_b.json").unlink()
    ctx = _fresh(world)
    run_test_once.stage_b(ctx)
    assert any("unreadable part file" in line for line in ctx.log_lines)
    got, _ = ps.load_calib(part)
    assert freeze_prereg.same_results(got[CLIP[0]], expected[CLIP[0]])


@pytest.mark.parametrize("change", ["cache", "extra frame", "params", "clip list", "camera file"])
def test_stage_c_checks_every_label_free_input_before_the_claim(world, monkeypatch, change):
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)
    ctx = _fresh(world)
    if change == "cache":
        _put_detections(ctx, world.root, version="pnlcalib-sv@other")
    elif change == "extra frame":
        img = Path(run_test_once.data_dir()) / "sn-gsr" / "test" / CLIP[0] / "img1"
        (img / f"{max(_golden_by_image()) + 5:06d}.jpg").write_bytes(b"")
    elif change == "params":
        b = json.loads((world.work_dir / "stage_b.json").read_text())
        (world.work_dir / "stage_b.json").write_text(json.dumps(b | {"params_digest": "OLD"}))
    elif change == "clip list":
        ctx.clips = [CLIP, ("SNGS-998", "test", 8)]
    else:
        (world.work_dir / "calib_ptz.npz").write_bytes(b"not the stage B file")

    def no_labels(*args, **kwargs):
        pytest.fail("a table was read before every label-free input was checked")

    monkeypatch.setattr(run_test_once.pl, "read_parquet", no_labels)
    with pytest.raises(pr.PreregError):
        run_test_once.stage_c(ctx)
    assert not any(m.exists() for m in world.markers)


@pytest.mark.data
def test_the_dev_label_tables_have_the_columns_stage_c_reads():
    """build_index writes the test split's tables with the dev tables' columns: schemas only, dev only."""
    for name, columns in run_test_once.LABEL_COLUMNS.items():
        path = run_test_once.index_dir() / f"{name}.parquet"
        if not path.exists():
            pytest.skip(f"no dev index at {path}")
        assert set(columns) <= set(pl.read_parquet_schema(path))
        assert set(columns) <= set(LABEL_SCHEMAS[name])


@pytest.mark.parametrize("damage", ["persons missing", "lines empty", "lines truncated", "column renamed"])
def test_stage_c_refuses_missing_or_damaged_label_tables_before_the_claim(world, monkeypatch, damage):
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)
    if damage == "persons missing":
        world.persons_path.unlink()
    elif damage == "lines empty":
        world.lines_path.write_bytes(b"")
    elif damage == "lines truncated":
        world.lines_path.write_bytes(world.lines_path.read_bytes()[:-20])
    else:
        pl.read_parquet(world.persons_path).rename({"x_raw": "x_raw_m"}).write_parquet(world.persons_path)
    monkeypatch.setattr(run_test_once.pl, "read_parquet", lambda *a, **k: pytest.fail("a label was read"))
    with pytest.raises(pr.PreregError, match="label table"):
        run_test_once.stage_c(_fresh(world))
    assert not any(m.exists() for m in world.markers)


def test_stage_a_refuses_detections_of_another_model(world, repo):
    _put_detections(world, repo, version="pnlcalib-sv@other")
    with pytest.raises(pr.PreregError, match="fitted on"):
        run_test_once.stage_a(world)


def test_stage_a_refuses_another_clip_list(world):
    world.clips = [("SNGS-998", "test", 7)]
    with pytest.raises(pr.PreregError, match="not the frozen one"):
        run_test_once.stage_a(world)


def test_stage_c_claims_before_reading_labels_and_a_crash_still_leaves_a_report(world, monkeypatch):
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)

    def read_after_claim(*args, **kwargs):
        assert all(m.exists() for m in world.markers), "labels read before the markers were written"
        raise OSError("disk unplugged")

    monkeypatch.setattr(run_test_once.pl, "read_parquet", read_after_claim)
    ctx = _fresh(world)
    with pytest.raises(OSError, match="disk unplugged"):
        run_test_once.stage_c(ctx)
    marker = json.loads(world.markers[0].read_text())
    assert marker["lock_sha256"] == ctx.lock_sha
    assert marker["freeze_commit"] == pr.freeze_commit(world.root)
    report = world.report.read_text()
    assert "Stage C failed" in report
    assert "OSError: disk unplugged" in report
    assert "No result was computed" in report
    assert "disk unplugged" in (world.work_dir / "stage_c.log").read_text()  # with its traceback
    with pytest.raises(pr.PreregError, match="already ran"):
        run_test_once.stage_c(_fresh(world))
    world.report.unlink()
    with pytest.raises(pr.PreregError, match="second evaluation"):
        run_test_once.stage_c(_fresh(world))
    assert not world.report.exists()  # a refused second stage C writes no failure report


def test_stage_c_asks_after_every_check_and_before_the_claim(world, monkeypatch):
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)
    asked = []

    def answer(yes):
        def confirm():
            asked.append(not any(m.exists() for m in world.markers))
            return yes

        return confirm

    monkeypatch.setattr(run_test_once.pl, "read_parquet", lambda *a, **k: pytest.fail("a label was read"))
    with pytest.raises(pr.PreregError, match="not confirmed"):
        run_test_once.stage_c(_fresh(world), answer(False))
    assert asked == [True]
    assert not any(m.exists() for m in world.markers)
    assert not world.report.exists()
    ctx = _fresh(world)
    ctx.clips = [CLIP, ("SNGS-998", "test", 8)]
    with pytest.raises(pr.PreregError, match="not the frozen one"):
        run_test_once.stage_c(ctx, answer(True))
    assert asked == [True]  # a failed check never reaches the question


def test_the_real_run_asks_before_stage_c_unless_yes(world, monkeypatch):
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)
    monkeypatch.setattr(run_test_once, "real_context", lambda: _fresh(world))
    for reply in (lambda *a, **k: False, lambda *a, **k: (_ for _ in ()).throw(typer.Abort())):
        monkeypatch.setattr(run_test_once.typer, "confirm", reply)
        with pytest.raises(typer.Exit) as exc:
            run_test_once.main(stage="c")
        assert exc.value.exit_code == 2  # refused, nothing claimed
        assert not any(m.exists() for m in world.markers)
    monkeypatch.setattr(run_test_once.typer, "confirm", lambda *a, **k: pytest.fail("--yes asks nothing"))
    monkeypatch.setattr(run_test_once, "_evaluate", lambda *a: {"criteria": {"P1": {"pass": True}}})
    run_test_once.main(stage="c", yes=True)
    assert all(m.exists() for m in world.markers)


@pytest.mark.parametrize("how", ["KeyboardInterrupt", "SIGTERM", "SIGHUP"])
def test_an_interruption_after_the_claim_still_leaves_a_report(world, monkeypatch, how):
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers belong to the main thread")
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}

    def interrupt(*args, **kwargs):
        if how == "KeyboardInterrupt":
            raise KeyboardInterrupt
        sig = getattr(signal, how)
        if signal.getsignal(sig) in (signal.SIG_DFL, signal.SIG_IGN, None):  # would end pytest
            pytest.fail(f"stage C holds its claim without a {how} handler")
        signal.raise_signal(sig)  # the handler runs before raise_signal returns
        pytest.fail(f"{how} did not interrupt stage C")

    monkeypatch.setattr(run_test_once.pl, "read_parquet", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_test_once.stage_c(_fresh(world))
    assert all(m.exists() for m in world.markers)
    report = world.report.read_text()
    assert "Stage C failed" in report
    assert how in report
    assert {s: signal.getsignal(s) for s in before} == before  # the handlers are restored


def test_a_second_signal_is_ignored_and_an_ignored_signal_stays_ignored():
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers belong to the main thread")
    guard = run_test_once.SignalGuard()
    with pytest.raises(run_test_once.Interrupted, match="SIGTERM"):
        guard.handler(signal.SIGTERM, None)
    guard.handler(signal.SIGTERM, None)  # while the failure report is written
    old = signal.signal(signal.SIGHUP, signal.SIG_IGN)  # as under nohup
    try:
        with run_test_once.signals_raise():
            assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
            assert signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, signal.SIG_IGN, None)
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, old)


def test_a_claim_that_fails_between_its_markers_still_leaves_a_report(world, monkeypatch):
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)
    claim = pr.claim_once

    def no_marker(markers, payload):
        raise OSError("data folder not writable")

    monkeypatch.setattr(run_test_once.pr, "claim_once", no_marker)
    with pytest.raises(OSError, match="data folder"):
        run_test_once.stage_c(_fresh(world))
    assert not world.report.exists()  # nothing written: the evaluation is still unclaimed
    assert not any(m.exists() for m in world.markers)

    def half_claim(markers, payload):
        claim(markers[:1], payload)
        raise OSError("run folder not writable")

    monkeypatch.setattr(run_test_once.pr, "claim_once", half_claim)
    with pytest.raises(OSError, match="run folder"):
        run_test_once.stage_c(_fresh(world))
    assert world.markers[0].exists()
    assert not world.markers[1].exists()
    assert "OSError: run folder not writable" in world.report.read_text()


def test_a_failure_after_the_results_puts_every_result_in_the_report(world, monkeypatch):
    run_test_once.stage_a(world)
    run_test_once.stage_b(world)

    def results_then_crash(ctx, prepared, lock, payload):
        results = {"criteria": {"P1": {"coverage": 0.91, "pass": True}}, "regions": {}}
        run_test_once.write_json(ctx.work_dir / "results.json", results)
        raise TypeError("unsupported format string passed to NoneType.__format__")

    monkeypatch.setattr(run_test_once, "_evaluate", results_then_crash)
    with pytest.raises(TypeError):
        run_test_once.stage_c(_fresh(world))
    report = world.report.read_text()
    assert "TypeError: unsupported format string" in report
    assert '"coverage": 0.91' in report
    assert "No result was computed" not in report


def test_the_dry_run_never_falls_back_to_the_test_clip_list(tmp_path, monkeypatch):
    monkeypatch.setattr(run_test_once, "test_clips", lambda: pytest.fail("the test clip list was read"))
    ctx = _context(tmp_path, tmp_path)
    ctx.dry, ctx.clips = True, []
    with pytest.raises(pr.PreregError, match="no stand-in clips"):
        run_test_once.ensure_clips(ctx)
    with pytest.raises(typer.BadParameter):
        run_test_once.dry_context(tmp_path / "scratch", [], 3, None)


def test_the_smoke_check_fails_on_any_difference():
    ok = {"frames": 10, "frames_with_same_ids": 10, "max_kp_px_diff": 0.0, "max_line_px_diff": 0.0}
    run_test_once.check_smoke(ok | {"model_version": MODEL}, MODEL)
    run_test_once.check_smoke({"skipped": "torch unavailable"}, MODEL)
    for bad in ({"frames_with_same_ids": 9}, {"max_kp_px_diff": 0.5}, {"model_version": "other"}):
        with pytest.raises(RuntimeError, match="do not reproduce"):
            run_test_once.check_smoke(ok | {"model_version": MODEL} | bad, MODEL)
    for frames in (-1, 11):
        with pytest.raises(typer.BadParameter):
            run_test_once.main(stage="a", smoke_frames=frames)


# --- frozen artifacts -------------------------------------------------------------------------------------


def synthetic_players(rng, games=(2, 3, 4, 5, 6, 9, 7), clips_per_game=4, n=120, filled=False):
    """Player rows of two covariance sources with errors drawn from their covariances (scaled per source)."""
    rows = []
    for g in games:
        for c in range(clips_per_game):
            source = np.where(rng.random(n) < 0.5, "smoother", "frame")
            if filled:
                source = np.full(n, "gap")
            sd = np.where(source == "frame", 1.3, 0.8)
            rows.append(
                pl.DataFrame(
                    {
                        "clip": f"g{g}-c{c}",
                        "game": g,
                        "frame": np.arange(n) * 5,
                        "track_id": np.zeros(n, dtype=np.int64),
                        "ex_lab": sd * rng.standard_normal(n),
                        "ey_lab": sd * rng.standard_normal(n),
                        "c00": np.ones(n),
                        "c01": np.zeros(n),
                        "c11": np.ones(n),
                        "mpp": rng.uniform(0.02, 0.2, n),
                        "n_inliers": rng.integers(6, 30, n),
                        "dof": np.full(n, np.nan),
                        "cov_source": source,
                        "filled": np.full(n, filled),
                    }
                )
            )
    return pl.concat(rows)


def test_frozen_regions_equal_the_eval_dev_models_fitted_on_all_six_dev_games(rng):
    """In a leave-one-game-out over the six dev games plus a new game, the fold that holds out the new game
    calibrates on all six dev games: the frozen models must be exactly that fold's models."""
    eval_dev = _load("eval_dev")
    df = synthetic_players(rng)
    extra = synthetic_players(rng, clips_per_game=3, n=30, filled=True)
    fold_models = {}
    _, _, held, extra_held = eval_dev.logo(df, "lab", rng, n_boot=5, extra=extra, models_out=fold_models)
    dev = pl.col("game") != 7
    frozen = ps.fit_frozen_regions(df.filter(dev), "lab", extra.filter(dev))
    assert frozen.to_dict() == fold_models[7].to_dict()
    assert set(frozen.models) == {None, "frame", "smoother", "gap"}
    # through JSON, the frozen models give the held-out game the same regions as the dev evaluation
    loaded = ps.RegionModels.from_dict(json.loads(json.dumps(frozen.to_dict())))
    test = held.filter(pl.col("game") == 7)
    s = ps.score_table(test, "lab")
    q = loaded.quantile_for({k: s[k] for k in ps.FEATURE_NAMES}, s["source"])
    np.testing.assert_array_equal(s["score"] <= q, test["hit_mond"].to_numpy())
    np.testing.assert_array_equal(s["norm"] <= loaded.r_const, test["hit_const"].to_numpy())
    # and the camera-level helper of the freeze builds the same models from one table with a filled column
    both = pl.concat([df, extra]).filter(dev)
    assert fz.fit_camera_regions(both, "lab").to_dict() == frozen.to_dict()
    assert extra_held is not None


def test_region_endpoint_uses_the_frozen_models_and_reports_filled_rows_apart(rng):
    df = synthetic_players(rng, games=(2, 3, 4, 5, 6, 9))
    extra = synthetic_players(rng, games=(2, 3, 4, 5, 6, 9), clips_per_game=3, n=30, filled=True)
    rm = ps.fit_frozen_regions(df, "lab", extra)
    test = pl.concat(
        [synthetic_players(rng, games=(7, 8, 11)), synthetic_players(rng, games=(7,), filled=True)]
    )
    out = fz.region_endpoint(test, rm, "lab", np.random.default_rng(0), n_boot=50)
    assert out["n"] == 3 * 4 * 120
    assert out["filled"]["n"] == 4 * 120
    assert [g["game"] for g in out["per_game"]] == [7, 8, 11]
    assert out["coverage_ci"][0] <= out["coverage"] <= out["coverage_ci"][1]
    assert out["coverage_clips_ci"][0] <= out["coverage_clips"] <= out["coverage_clips_ci"][1]
    assert out["area_ratio"] == pytest.approx(out["median_area"] / (np.pi * rm.r_const**2))


def test_clip_weighted_coverage_weighs_every_clip_equally():
    # one clip of 90 rows all covered, one of 10 rows none covered: pooled 90 %, clips weighted equally 50 %
    hit = np.r_[np.ones(90, bool), np.zeros(10, bool)]
    clips = np.array(["a"] * 90 + ["b"] * 10)
    cov, (lo, hi) = ps.clip_weighted_coverage(hit, clips, np.random.default_rng(0), n_boot=200)
    assert hit.mean() == pytest.approx(0.9)
    assert cov == pytest.approx(0.5)
    assert lo == 0.0
    assert hi == 1.0  # a resample of two clips is (0, 0), (0, 1), (1, 0) or (1, 1)
    # the order of the rows does not matter
    perm = np.random.default_rng(1).permutation(100)
    assert ps.clip_weighted_coverage(hit[perm], clips[perm], np.random.default_rng(0), 200)[0] == cov


def test_p1_follows_the_preregistered_weighting(monkeypatch, rng):
    df = synthetic_players(rng, games=(2, 3, 4, 5, 6, 9))
    rm = ps.fit_frozen_regions(df, "lab")
    test = synthetic_players(rng, games=(7, 8, 11))
    out = fz.region_endpoint(test, rm, "lab", np.random.default_rng(0), n_boot=50)
    assert set(fz.P1_COVERAGE_KEYS) == set(fz.P1_WEIGHTING_TITLES)
    pooled = fz.p1_criterion(out)
    assert (pooled["weighting"], pooled["coverage"], pooled["ci"]) == (
        "player-frames",
        out["coverage"],
        out["coverage_ci"],
    )
    monkeypatch.setattr(fz, "P1_WEIGHTING", "clips")
    clips = fz.p1_criterion(out)
    assert (clips["weighting"], clips["coverage"], clips["ci"]) == (
        "clips",
        out["coverage_clips"],
        out["coverage_clips_ci"],
    )
    assert clips["pass"] == (out["coverage_clips_ci"][0] <= 0.9 <= out["coverage_clips_ci"][1])
    assert fz.analysis_config()["p1_weighting"] == "clips"  # the weighting is part of the digested plan
    assert fz.p1_criterion({"n": 0}) == {"pass": None}


def failure_frames(rng, games=(2, 3, 4), n=300):
    x = rng.standard_normal((len(games) * n, len(FEATURES)))
    x[rng.random(x.shape) < 0.05] = np.nan  # missing features are imputed with the training medians
    logit = 1.5 * np.nan_to_num(x[:, 0]) - np.nan_to_num(x[:, 3]) + 0.5 * rng.standard_normal(len(x))
    df = pl.DataFrame({name: x[:, j] for j, name in enumerate(FEATURES)})
    return df.with_columns(
        pl.Series("fail", logit > 1.2),
        pl.Series("game", np.repeat(games, n)),
        pl.Series("clip", [f"c{i // 50}" for i in range(len(x))]),
    )


def test_the_frozen_detector_reproduces_scikit_learn_through_json(rng):
    df = failure_frames(rng)
    det = hd.FrozenDetector.fit(df)
    x = df.select(FEATURES).to_numpy()
    xi = np.where(np.isfinite(x), x, hd.training_medians(x))
    y = df["fail"].to_numpy().astype(int)
    gb, lr = hd.make_boosting().fit(xi, y), hd.make_logistic().fit(xi, y)
    loaded = hd.FrozenDetector.from_dict(json.loads(json.dumps(det.to_dict())))
    scores = loaded.scores(df)
    np.testing.assert_allclose(scores["boosting"], gb.predict_proba(xi)[:, 1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(scores["logistic"], lr.predict_proba(xi)[:, 1], rtol=0, atol=1e-12)
    assert det.best_single in FEATURES
    assert hd.BOOSTING_PARAMS["early_stopping"] is False


def test_saved_output_cameras_round_trip_exactly(tmp_path, rng):
    frames = np.arange(4) * 5
    h = np.eye(3) + 0.01 * rng.standard_normal((3, 3))
    h[2, 2] = 1.0
    results = [
        CalibResult(
            CalibStatus.OK, H=h, cov_theta=np.eye(8), s2=1.5, rms_px=0.7, n_inliers=12, cond=3.0, dof=20.0
        ),
        CalibResult(CalibStatus.FEW_POINTS),
        CalibResult(CalibStatus.OK, H=h, cov_theta=2 * np.eye(8), n_inliers=9),
        CalibResult(CalibStatus.OK, H=h, cov_theta=np.eye(8), s2=None, rms_px=1.0, n_inliers=7, dof=None),
    ]
    cc = ps.ClipCalib("SNGS-001", 2, "train", frames, results, np.array([False, False, True, False]), [])
    flags = tm.FrameFlags.empty(4)
    flags.smoothed[[0, 2]] = True
    flags.filled[1] = True
    diag = {"nis": np.array([0.5, np.nan, 1.0, 2.0])}
    fz.save_outputs(tmp_path / "out.npz", [ps.Output(cc, None, results, flags, diag)], {"stage": "B"})
    loaded, meta = fz.load_outputs(tmp_path / "out.npz")
    res, fl, dg = loaded["SNGS-001"]
    assert meta["stage"] == "B"
    assert meta["clips"] == ["SNGS-001"]
    assert freeze_prereg.same_results(res, results)
    for name, values in flags.as_dict().items():
        np.testing.assert_array_equal(getattr(fl, name), values)
    np.testing.assert_array_equal(dg["nis"], diag["nis"])


# --- end to end: frozen artifacts, endpoints, criteria, report --------------------------------------------


def _camera_players(rng, games, scale, filled_rows=False):
    """Synthetic player rows of one camera: both references (the oracle on about half of the rows) and the
    smoothed labels, errors scaled by ``scale``."""
    df = synthetic_players(rng, games=games, clips_per_game=4, n=60)
    if filled_rows:
        df = pl.concat([df, synthetic_players(rng, games=games, clips_per_game=4, n=10, filled=True)])
    n = df.height
    has_or = rng.random(n) < 0.5
    return df.with_columns(
        pl.col("ex_lab") * scale,
        pl.col("ey_lab") * scale,
        pl.Series("ex_or", np.where(has_or, scale * rng.standard_normal(n), np.nan)),
        pl.Series("ey_or", np.where(has_or, scale * rng.standard_normal(n), np.nan)),
        pl.Series("ex_sm", 0.5 * scale * rng.standard_normal(n)),
        pl.Series("ey_sm", 0.5 * scale * rng.standard_normal(n)),
    )


def _camera_frames(rng, games):
    """Frame rows of one camera: health features, line distances, output flags."""
    f = failure_frames(rng, games=games, n=200)
    n = f.height
    line_px = np.where(f["fail"].to_numpy(), 15.0, 2.0)
    flags = {name: np.zeros(n, dtype=bool) for name in fz.OUTPUT_TYPES} | {"smoothed": np.ones(n, dtype=bool)}
    return f.drop("fail").with_columns(
        pl.Series("frame", np.arange(n) * 5),
        pl.Series("ok", np.ones(n, dtype=bool)),
        pl.Series("ok_out", rng.random(n) < 0.97),
        pl.Series("trusted", np.ones(n, dtype=bool)),
        pl.Series("line_px", line_px),
        pl.Series("line_px_out", line_px * 0.5),
        *[pl.Series(k, v) for k, v in flags.items()],
    )


def test_artifacts_endpoints_criteria_and_report_end_to_end(tmp_path, rng):
    dev = fz.DEV_GAMES
    tables = fz.DevTables(
        perframe=fz.CameraTables(_camera_players(rng, dev, 1.0), _camera_frames(rng, dev)),
        ptz=fz.CameraTables(_camera_players(rng, dev, 1.0, filled_rows=True), _camera_frames(rng, dev)),
        smoothing=ps.SmoothingInfo(
            fits={"all": {"params": tm.NoiseParams().to_dict(), "training_clips": 24}}
        ),
        params=tm.NoiseParams(),
        cfg=tm.SmootherConfig(),
    )
    written = fz.write_artifacts(tables, tmp_path / "v1", {"dev_clips": 24})
    assert set(written) == {"analysis", "smoother", "detector"} | {
        f"regions_{c}_{r}" for c in fz.CAMERAS for r in fz.REFERENCES
    }
    art = fz.load_artifacts(tmp_path / "v1")
    for c in fz.CAMERAS:
        for r in fz.REFERENCES:
            fitted = fz.fit_camera_regions(fz.camera_table(tables, c).players, r)
            assert art.regions[c, r].to_dict() == fitted.to_dict()
    assert art.params == tm.NoiseParams()

    # test games: the smoothed camera keeps the dev error scale, the per-frame camera triples it, the smoothed
    # camera abstains on every tenth frame, and the per-frame camera cannot place the feet on frame 35
    test_games = fz.TEST_GAMES
    perframe = _camera_players(np.random.default_rng(1), test_games, 3.0).filter(pl.col("frame") != 35)
    ptz = _camera_players(np.random.default_rng(1), test_games, 1.0).filter(pl.col("frame") % 50 != 0)
    frames = _camera_frames(rng, test_games)
    res = fz.evaluate(
        {"perframe": fz.CameraTables(perframe, frames), "ptz": fz.CameraTables(ptz, frames)}, art, n_boot=200
    )

    prim, crit = res["regions"]["ptz"]["lab"], res["criteria"]
    assert fz.PRIMARY == ("ptz", "lab")
    s = ps.score_table(ptz.filter(~pl.col("filled")), "lab")
    q = art.regions["ptz", "lab"].quantile_for({k: s[k] for k in ps.FEATURE_NAMES}, s["source"])
    assert prim["coverage"] == pytest.approx(np.mean(s["score"] <= q))
    lo, hi = prim["coverage_ci"]
    assert fz.P1_WEIGHTING == "player-frames"
    assert crit["P1"] == {
        "weighting": "player-frames",
        "coverage": prim["coverage"],
        "ci": prim["coverage_ci"],
        "pass": lo <= 0.9 <= hi,
    }
    per_clip = ptz.filter(~pl.col("filled")).with_columns(pl.Series("hit", s["score"] <= q))
    clip_means = per_clip.group_by("clip").agg(pl.col("hit").mean())["hit"].to_numpy()
    assert prim["coverage_clips"] == pytest.approx(clip_means.mean())
    area = np.pi * q * np.sqrt(np.linalg.det(s["cov"]))
    ratio = np.median(area) / (np.pi * art.regions["ptz", "lab"].r_const ** 2)
    assert crit["P2"]["ratio"] == pytest.approx(ratio)
    assert crit["P2"]["pass"] == (ratio < 1)
    assert res["regions"]["perframe"]["lab"]["coverage"] < prim["coverage"] - 0.2  # a swapped arm shows

    df = hd.failure_table(frames)
    y = df["fail"].to_numpy()
    scores = art.detector.scores(df)
    from sklearn.metrics import roc_auc_score

    diff = roc_auc_score(y, scores["boosting"]) - roc_auc_score(y, scores[art.detector.best_single])
    assert res["detector"]["best_single_dev"] == art.detector.best_single
    assert crit["P3"] == {"diff": pytest.approx(diff), "pass": diff > 0}

    abstained = perframe.filter(pl.col("frame") % 50 == 0)
    un = prim["unpublished"]
    assert (un["n"], un["frames"]) == (abstained.height, abstained.select("clip", "frame").unique().height)
    n_all = prim["n"] + abstained.height
    assert un["share"] == pytest.approx(abstained.height / n_all)
    assert un["coverage_as_misses"] == pytest.approx(prim["coverage"] * prim["n"] / n_all)
    misses = abstained.group_by("clip").len().rename({"len": "missing"})
    clip_cov = (
        per_clip.group_by("clip")
        .agg(pl.col("hit").sum(), pl.len())
        .join(misses, on="clip", how="left")
        .select(pl.col("hit") / (pl.col("len") + pl.col("missing").fill_null(0)))
    )
    assert un["coverage_clips_as_misses"] == pytest.approx(clip_cov.to_series().mean())
    pf_un = res["regions"]["perframe"]["lab"]["unpublished"]  # feet only the smoothed camera places
    assert pf_un["n"] == ptz.filter(pl.col("frame") == 35).height
    assert pf_un["frames"] == ptz.filter(pl.col("frame") == 35).select("clip", "frame").unique().height
    assert (
        res["regions"]["ptz"]["or"]["unpublished"]["n"]
        == abstained.filter(pl.col("ex_or").is_not_nan()).height
    )

    rs = res["residual"]
    both = frames.filter(pl.col("ok") & pl.col("ok_out"))
    assert rs["frames"] == both.height
    assert rs["fail_share_perframe"] == pytest.approx(np.mean(both["line_px"].to_numpy() > 10))

    md = fz.render_report(json.loads(fz.dump(res)), ["# Synthetic"])
    assert "| P1 |" in md
    assert "the other camera publishes and this camera does not" in md
    assert (
        f"| PTZ-smoothed camera | trusted line oracle | {res['regions']['ptz']['or']['unpublished']['n']} |"
        in md
    )
    assert md.count("**met**") + md.count("**not met**") == 3
