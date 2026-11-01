"""scripts/make_figures.py on synthetic run folders: LOGO fits at several levels, figures and report."""

import importlib.util
import json
import sys

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from predfoot.paths import REPO_ROOT


def _load(name):
    key = f"scripts_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


figures = _load("make_figures")


def write_run(path, rng, smoothed, games=(1, 2, 3, 4), clips_per_game=8, n=100):
    """players.parquet of a fake eval_dev.py run: errors drawn from each row's covariance; the smoothed run
    has its rows marked with their held-out fold (fold == game) and a few filled rows of the gap source."""
    parts = []
    for g in games:
        for c in range(clips_per_game):
            sd = rng.uniform(0.5, 2.0, n)
            e = sd[:, None] * rng.standard_normal((n, 2))
            filled = (rng.random(n) < 0.05) if smoothed else np.zeros(n, bool)
            parts.append(
                pl.DataFrame(
                    {
                        "clip": f"g{g}-c{c}",
                        "game": g,
                        "frame": 1 + 5 * np.arange(n),
                        "track_id": np.zeros(n, dtype=np.int64),
                        "ex_lab": e[:, 0],
                        "ey_lab": e[:, 1],
                        "ex_or": np.where(rng.random(n) < 0.4, np.nan, 1.3 * e[:, 0]),
                        "ey_or": 1.3 * e[:, 1],
                        "c00": sd**2,
                        "c01": np.zeros(n),
                        "c11": sd**2,
                        "mpp": rng.uniform(0.02, 0.08, n),
                        "n_inliers": rng.integers(6, 30, n),
                        "dof": np.full(n, np.nan),
                        "fold": np.full(n, g) if smoothed else np.full(n, None),
                        "cov_source": np.where(filled, "gap", "smoother" if smoothed else "frame"),
                        "filled": filled,
                    },
                    schema_overrides={"fold": pl.Int64},
                )
            )
    path.mkdir(parents=True)
    pl.concat(parts).write_parquet(path / "players.parquet")
    return path


def test_levels_cover_about_the_nominal_level_and_reproduce_the_default_fit(tmp_path, rng):
    head, filled = figures.load_run(write_run(tmp_path / "none", rng, smoothed=False))
    assert filled.height == 0
    rows = figures.levels(head, filled, "none", n_boot=20)
    assert [(r.ref, r.alpha) for r in rows] == [(ref, a) for ref in figures.REFS for a in figures.ALPHAS]
    n_cal = 3 * 8  # calibration clips per fold
    for ref in figures.REFS:
        series = [r for r in rows if r.ref == ref]
        assert [r.nominal for r in series] == sorted(r.nominal for r in series)
        cover = [r.cov_mond for r in series]
        assert cover == sorted(cover)
        assert [r.area_mond for r in series] == sorted(r.area_mond for r in series)
        if ref == "lab":  # errors drawn from the covariances: calibrated at every level
            for r in series:
                assert 1 - r.alpha - 0.03 <= r.cov_mond <= (1 - r.alpha) * (1 + 1 / n_cal) + 0.03
        at90 = next(r for r in series if r.alpha == figures.TERCILE_ALPHA)
        assert [t["bin"] for t in at90.terciles] == [0, 1, 2]
        assert sum(t["n"] for t in at90.terciles) == at90.n
        assert all(not r.terciles for r in series if r.alpha != figures.TERCILE_ALPHA)
        assert [t["bin"] for t in at90.size_terciles] == [0, 1, 2]
        assert sum(t["clips"] for t in at90.size_terciles) == at90.clips
        for r in series:
            assert r.cov_clip_ci[0] <= r.cov_clip <= r.cov_clip_ci[1]
    # eval_dev.py's own sequence at 90 %: one generator, the labels' fit first, then the oracle's.
    eval_dev = figures.eval_dev
    rng = np.random.default_rng(0)
    for ref in ("lab", "or"):
        _, summary, held, _ = eval_dev.logo(head, ref, rng, n_boot=20, extra=filled)
        at90 = next(r for r in rows if r.ref == ref and r.alpha == 0.1)
        assert (at90.cov_mond, at90.cov_mond_ci, at90.area_mond) == (
            summary["cov_mond"],
            summary["cov_mond_ci"],
            summary["a_mond"],
        )
        per_clip = held.group_by("clip").agg(pl.col("hit_mond").mean())["hit_mond"]
        assert at90.cov_clip == pytest.approx(per_clip.mean())


def test_clip_weighted_coverage_weighs_every_clip_once(rng):
    # A large clip fully covered and three small ones half covered: pooling the player-frames favours the
    # large clip; weighting clips equally does not.
    sizes, covered = [400, 20, 20, 20], [1.0, 0.5, 0.5, 0.5]
    held = pl.DataFrame(
        {
            "clip": np.repeat([f"c{k}" for k in range(4)], sizes),
            "hit_mond": np.concatenate([np.arange(n) < c * n for n, c in zip(sizes, covered, strict=True)]),
        }
    )
    assert held["hit_mond"].mean() == pytest.approx(430 / 460)
    cov, (lo, hi) = figures.clip_weighted(held, rng, n_boot=200)
    assert cov == pytest.approx(0.625)
    assert 0.5 <= lo <= cov <= hi <= 1.0
    terciles = figures.size_tercile_rows(held)
    assert [t["clips"] for t in terciles] == [3, 0, 1]  # three clips of 20 at or below both edges
    assert terciles[0]["cov"] == pytest.approx(0.5)
    assert np.isnan(terciles[1]["cov"])
    assert terciles[2]["cov"] == pytest.approx(1.0)


def test_load_run_refuses_rows_of_another_fold(tmp_path, rng):
    run = write_run(tmp_path / "ptz", rng, smoothed=True)
    head, filled = figures.load_run(run)
    assert filled.height > 0
    assert not head["filled"].any()
    players = pl.read_parquet(run / "players.parquet").with_columns(pl.lit(1, dtype=pl.Int64).alias("fold"))
    players.write_parquet(run / "players.parquet")
    with pytest.raises(ValueError, match="fold == game"):
        figures.load_run(run)


def test_main_writes_four_small_figures_and_the_report(tmp_path, rng):
    none = write_run(tmp_path / "runs" / "a_none", rng, smoothed=False)
    ptz = write_run(tmp_path / "runs" / "b_ptz", rng, smoothed=True)
    proxy = {"distance_bias": 0.27, "hsr_bias": 0.67, "raw_distance_bias": 0.18, "raw_hsr_bias": 0.43}
    kpi = tmp_path / "error_process.json"
    kpi.write_text(
        json.dumps(
            {
                "cameras": {
                    "none": {"kpi_proxy": proxy},
                    "ptz": {"kpi_proxy": {**proxy, "distance_bias": 0.03, "hsr_bias": 0.06}},
                }
            }
        )
    )
    out, report = tmp_path / "figures", tmp_path / "dev_figures.md"
    args = ["--none-run", str(none), "--ptz-run", str(ptz), "--kpi-json", str(kpi)]
    args += ["--out-dir", str(out), "--report", str(report), "--n-boot", "10"]
    result = CliRunner().invoke(figures.app, args)
    assert result.exit_code == 0, result.output
    for name in ("reliability", "area", "coverage_by_distance", "kpi_proxy"):
        png = out / f"{name}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert png.stat().st_size < 200_000
    text = report.read_text()
    assert "runs/a_none" in text
    assert "runs/b_ptz" in text
    assert str(tmp_path) not in text  # run folders are named, local paths are not
    # four levels, three distance terciles, three clip-size terciles
    assert text.count("| SN-GSR raw labels | per-frame camera | ") == 4 + 3 + 3
    assert "Clips weighted equally [95 % CI]" in text
    assert "## Coverage by clip size at 90%" in text
    assert "regions fitted on oracle errors" in text
    assert "| raw labels (noise floor) | +18.0% | +43.0% |" in text
    assert "| PTZ-smoothed camera | +3.0% | +6.0% |" in text
