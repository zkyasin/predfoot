"""Guards and frozen artifacts of the preregistered test run (scripts/freeze_prereg.py, run_test_once.py).

The guards run on throw-away git repositories; nothing here touches data, the real lock or test files.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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
from predfoot.paths import REPO_ROOT

FINAL_TEXT = "# Preregistration of the test run\n\n- Version: 1\n- Status: approved by the owner.\n"


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
    (root / "artifacts/prereg/v1/regions.json").write_text('{"q": 1.0}\n')
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    commit_all(root, "code")
    lock = pr.make_lock(
        root,
        Path("artifacts/prereg/v1"),
        code_commit=pr.head_commit(root),
        frozen_at_utc="2026-11-01T00:00:00+00:00",
        config_digests={},
    )
    (root / pr.LOCK_PATH).write_text(lock.to_json())
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


@pytest.mark.parametrize("path", ["src/m.py", "scripts/s.py", "configs/c.yaml"])
def test_a_code_change_since_the_lock_commit_is_refused(repo, path):
    (repo / path).write_text("changed = True\n")
    commit_all(repo, "change after the freeze")
    with pytest.raises(pr.PreregError, match="code changed since the lock commit"):
        pr.verify_lock(repo)


def test_a_change_outside_the_protected_paths_is_accepted(repo):
    (repo / "README.md").write_text("Preregistration frozen.\n")
    commit_all(repo, "readme")
    pr.verify_lock(repo)


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


def test_the_current_preregistration_is_version_1_and_still_a_draft():
    text = (REPO_ROOT / pr.PREREG_PATH).read_text()
    assert pr.prereg_version(text) == 1
    with pytest.raises(pr.PreregError):
        pr.require_final_text(text)


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
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stage_b.json").write_text("{}")
    return run_test_once.Context(
        dry=False,
        root=repo,
        run_dir=run_dir,
        markers=[tmp_path / "data" / "STAGE_C_STARTED.json", run_dir / "STAGE_C_STARTED.json"],
        cache_root=tmp_path / "cache",
        report=tmp_path / "test_run.md",
        persons_path=tmp_path / "persons.parquet",
        lines_path=tmp_path / "lines.parquet",
        clips=[("SNGS-999", "test", 7)],
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
    assert out["area_ratio"] == pytest.approx(out["median_area"] / (np.pi * rm.r_const**2))


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
