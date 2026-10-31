"""Logic of scripts/eval_dev.py and scripts/error_process.py that the headline numbers depend on."""

import importlib.util
import sys

import numpy as np
import polars as pl
import pytest

from predfoot.calib import temporal as tm
from predfoot.contracts import CalibResult, CalibStatus
from predfoot.paths import REPO_ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


eval_dev = _load("eval_dev")
error_process = _load("error_process")


def players(rng, n_per_clip=400, games=(1, 2, 3, 4), clips_per_game=4, folds=False, scale=None):
    """Synthetic player rows: errors drawn from each row's covariance (times ``scale`` per source)."""
    rows = []
    for g in games:
        for c in range(clips_per_game):
            for fold in games if folds else (None,):
                m = n_per_clip
                source = np.where(rng.random(m) < 0.5, "smoother", "frame") if scale else np.full(m, "frame")
                sd = np.where(source == "smoother", scale["smoother"], scale["frame"]) if scale else 1.0
                rows.append(
                    pl.DataFrame(
                        {
                            "clip": f"g{g}-c{c}",
                            "game": g,
                            "frame": np.arange(m) * 5,
                            "track_id": np.zeros(m, dtype=np.int64),
                            "ex_lab": sd * rng.standard_normal(m),
                            "ey_lab": sd * rng.standard_normal(m),
                            "c00": np.ones(m),
                            "c01": np.zeros(m),
                            "c11": np.ones(m),
                            "mpp": rng.uniform(0.02, 0.2, m),
                            "n_inliers": rng.integers(6, 30, m),
                            "dof": np.full(m, np.nan),
                            "fold": fold if folds else None,
                            "cov_source": source,
                        }
                    )
                )
    return pl.concat(rows)


def test_logo_folds_never_calibrate_on_the_held_out_game(rng, monkeypatch):
    df = players(rng, n_per_clip=60, folds=True)
    calls = []
    real_fit = eval_dev.MondrianConformal.fit

    def spy(scores, groups, *args, **kwargs):
        calls.append(np.unique(groups))
        return real_fit(scores, groups, *args, **kwargs)

    monkeypatch.setattr(eval_dev.MondrianConformal, "fit", staticmethod(spy))
    rows, summary, held, _ = eval_dev.logo(df, "lab", rng, n_boot=10, fold_col="fold")
    assert [r["game"] for r in rows] == [1, 2, 3, 4]
    assert len(calls) == 4  # one source, one model per fold
    for g, groups in zip((1, 2, 3, 4), calls, strict=True):
        assert not any(str(c).startswith(f"g{g}-") for c in groups)
    expected = df.filter(pl.col("fold") == pl.col("game"))
    assert held.height == expected.height == summary["n"]
    assert held.select("clip", "frame", "fold").unique().height == held.height


def test_mondrian_bins_are_formed_within_each_covariance_source(rng):
    """Smoother rows whose scores are 100 times the per-frame ones (on the smoke clips smoothed and
    passed-through frames differ by about 10x). One quantile per bin fits the large scores and gives the
    per-frame rows ellipses ten times too wide in radius; per-source models cover both alike."""
    df = players(rng, scale={"smoother": 10.0, "frame": 1.0})

    def by_source(d):
        _, summary, held, _ = eval_dev.logo(d, "lab", rng, n_boot=10)
        held = held.drop("cov_source", strict=False).join(
            df.select("clip", "frame", "cov_source"), on=["clip", "frame"]
        )
        return summary["cov_mond"], {
            s: (g["hit_mond"].mean(), g["area_mond"].median())
            for (s,), g in held.partition_by("cov_source", as_dict=True).items()
        }

    overall, split = by_source(df)
    assert overall >= 0.9
    assert abs(split["smoother"][0] - split["frame"][0]) < 0.04
    _, pooled = by_source(df.drop("cov_source"))
    assert pooled["frame"][0] > 0.99
    assert pooled["frame"][1] > 50 * split["frame"][1]


def test_cov_sources_follow_the_flags_and_duplicates():
    flags = tm.FrameFlags.empty(8)
    flags.smoothed[0] = flags.duplicate[1] = flags.projected[2] = flags.duplicate[3] = True
    flags.passthrough[4] = flags.repaired[5] = flags.filled[6] = flags.duplicate[7] = True
    assert eval_dev.cov_sources(flags, 8).tolist() == [
        "smoother", "smoother", "frame", "frame", "frame", "smoother", "gap", "gap",
    ]  # fmt: skip
    assert eval_dev.cov_sources(None, 2).tolist() == ["frame", "frame"]


def test_a_source_with_too_few_clips_uses_the_model_of_all_sources(rng):
    """A source seen in three clips (one per game) has at most two calibration clips per fold: at the clip
    level its conformal quantile would be its largest score (here 10⁸, one wild row), so it takes the model
    fitted on all sources instead."""
    alpha, g_min = eval_dev.ALPHA, eval_dev.MIN_SOURCE_CLIPS
    assert (1 - alpha) * (1 + 1 / g_min) < 1 <= (1 - alpha) * (1 + 1 / (g_min - 1))
    df = players(rng, n_per_clip=100)
    odd = pl.col("clip").is_in(["g1-c0", "g2-c0", "g3-c0"])
    df = df.with_columns(
        pl.when(odd).then(pl.lit("odd")).otherwise(pl.col("cov_source")).alias("cov_source"),
        pl.when(odd & (pl.col("frame") == 0)).then(1e4).otherwise(pl.col("ex_lab")).alias("ex_lab"),
    )
    _, summary, held, _ = eval_dev.logo(df, "lab", rng, n_boot=10)
    odd_rows = held.filter(pl.col("cov_source") == "odd")
    assert odd_rows.height == 300
    assert odd_rows["area_mond"].max() < 100  # about π · χ²₂(0.9); its own quantile would give 3 · 10⁸
    assert summary["cov_mond"] >= 0.88


def test_filled_rows_calibrate_their_own_pooled_gap_quantile(rng, monkeypatch):
    """Filled frames (source "gap") have covariances about a hundred times wider than smoothed ones relative
    to their errors. Calibrated on the filled rows of the other games with one pooled quantile, they cover at
    the nominal level; scored with the smoother's quantile they covered 100 % with regions a hundred times too
    large. The held-out game's filled rows are never used to calibrate."""
    head = players(rng, scale={"smoother": 10.0, "frame": 10.0}, folds=True)
    filled = players(rng, n_per_clip=20, clips_per_game=10, folds=True).with_columns(
        pl.lit("gap").alias("cov_source")
    )  # 30 calibration clips per fold: level (1 − α)(1 + 1/30) = 0.93
    calls = []
    real_fit = eval_dev.MondrianConformal.fit

    def spy(scores, groups, features, n_bins, *args, **kwargs):
        calls.append((np.unique(groups), dict(n_bins)))
        return real_fit(scores, groups, features, n_bins, *args, **kwargs)

    monkeypatch.setattr(eval_dev.MondrianConformal, "fit", staticmethod(spy))
    _, summary, held, extra = eval_dev.logo(head, "lab", rng, n_boot=10, fold_col="fold", extra=filled)
    assert extra is not None
    assert extra.height == filled.filter(pl.col("fold") == pl.col("game")).height
    assert 0.87 <= extra["hit_mond"].mean() <= 0.97
    assert extra["area_mond"].median() < 0.05 * held["area_mond"].median()
    assert summary["cov_mond"] >= 0.88  # 12 calibration clips per fold: conservative level 0.975
    assert len(calls) == 4 * 4  # per fold: all sources, then frame, gap and smoother
    bins = eval_dev.MONDRIAN_BINS
    for g, fold_calls in zip((1, 2, 3, 4), (calls[i : i + 4] for i in range(0, 16, 4)), strict=True):
        assert [b for _, b in fold_calls] == [bins, bins, {}, bins]  # the gap source gets one pooled quantile
        for groups, _ in fold_calls:
            assert not any(str(c).startswith(f"g{g}-") for c in groups)


def test_saved_calibrations_round_trip_exactly(tmp_path, rng):
    h = rng.standard_normal((3, 3))
    cov = np.eye(8) * 1e-6
    results = [
        CalibResult(
            CalibStatus.OK, H=h, cov_theta=cov, s2=2.5, rms_px=1.5, n_inliers=12, cond=30.0, dof=17.0
        ),
        CalibResult(CalibStatus.OK, H=h, cov_theta=cov, n_inliers=8),
        CalibResult(CalibStatus.FEW_POINTS, n_inliers=3),
    ]
    dup = np.array([False, True, False])
    cc = eval_dev.ClipCalib("SNGS-001", 2, "train", np.array([1, 6, 11]), results, dup, [])
    path = tmp_path / "calib.npz"
    eval_dev.save_calib(path, [cc], {"git_sha": "abc"})
    loaded, meta = eval_dev.load_calib(path)
    assert meta["clips"] == ["SNGS-001"]
    assert meta["git_sha"] == "abc"
    for a, b in zip(loaded["SNGS-001"], results, strict=True):
        assert (a.status, a.n_inliers) == (b.status, b.n_inliers)
        assert (a.s2, a.rms_px, a.cond, a.dof) == (b.s2, b.rms_px, b.cond, b.dof)  # None stays None
        for got, want in ((a.H, b.H), (a.cov_theta, b.cov_theta)):
            assert (got is None) == (want is None)
            if want is not None:
                np.testing.assert_array_equal(got, want)


def test_kpi_proxy_on_a_straight_constant_speed_run():
    """A player at 6 m/s (above the 5.56 m/s HSR threshold) seen exactly: no bias, no speed error."""
    n = 40
    x = 6.0 * 0.2 * np.arange(n)
    d = pl.DataFrame(
        {
            "clip": "c",
            "frame": 5 * np.arange(n),
            "track_id": 1,
            "x_pred": x,
            "y_pred": np.zeros(n),
            "ex_sm": np.zeros(n),
            "ey_sm": np.zeros(n),
            "ex_lab": np.zeros(n),
            "ey_lab": np.zeros(n),
        }
    )
    p = error_process.kpi_proxy(d)
    assert p is not None
    assert p["runs"] == 1
    assert p["distance_bias"] == pytest.approx(0.0, abs=1e-12)
    assert p["hsr_bias"] == pytest.approx(0.0, abs=1e-12)
    assert p["hsr_sm_m"] == pytest.approx(6.0 * 0.2 * (n - 1), rel=1e-9)
    assert p["speed_rms"] == pytest.approx(0.0, abs=1e-9)
    shifted = d.with_columns(pl.col("x_pred") + 0.5, pl.col("ex_sm") + 0.5, pl.col("ex_lab") + 0.5)
    q = error_process.kpi_proxy(shifted)
    assert q is not None
    assert q["distance_bias"] == pytest.approx(0.0, abs=1e-12)  # a constant offset leaves distances alone


def test_restrict_keeps_the_shared_finite_player_frames():
    keys = {"clip": ["a", "a", "a", "b"], "frame": [1, 6, 11, 1], "track_id": [1, 1, 1, 2]}
    a = pl.DataFrame({**keys, "ex_lab": [0.1, np.nan, 0.3, 0.4], "ex_or": [np.nan, 0.0, 0.0, 0.0]})
    b = pl.DataFrame({**keys, "ex_lab": [0.1, 0.2, np.nan, 0.4], "ex_or": [0.0] * 4}).filter(
        pl.col("clip") == "a"
    )
    ra, rb = error_process._restrict(a, b)
    assert ra.select("clip", "frame").rows() == rb.select("clip", "frame").rows() == [("a", 1)]


def test_compare_labels_are_distinct(tmp_path):
    same_folder = [tmp_path / "smoke" / "ptz.parquet", tmp_path / "smoke" / "none.parquet"]
    assert error_process.camera_labels(same_folder) == ["smoke/ptz", "smoke/none"]
    runs = [tmp_path / "run_a" / "players.parquet", tmp_path / "run_b" / "players.parquet"]
    assert error_process.camera_labels(runs) == ["run_a", "run_b"]
    assert error_process.camera_labels([tmp_path / "players.parquet"])[0]
