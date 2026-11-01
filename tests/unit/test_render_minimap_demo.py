"""scripts/render_minimap_demo.py on synthetic run folders: clip choice, window and the animation."""

import importlib.util
import sys

import numpy as np
import polars as pl
from PIL import Image
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


demo = _load("render_minimap_demo")
FAST = ("g1-c0", "g2-c1", "g3-c2", "g1-c3")  # the top third of the 12 clips by camera speed
STEADY = "g2-c1"  # the fast clip whose smoothed camera jitters least
UNLABELLED_K = 3  # cached frame index without usable labels in every clip


def write_run(path, rng, smoothed, games=(1, 2, 3), clips_per_game=4, n_frames=20, n_players=6):
    """Tables of a fake eval_dev.py run: a common-mode error per frame (the camera) plus player noise.

    The smoothed camera's common-mode steps are five times smaller (fifty for STEADY); the per-frame run
    abstains on the 8th frame of every clip; the 4th frame of every clip has no usable label (both cameras
    publish positions there, but the run has no player row for it)."""
    players, frames = [], []
    for g in games:
        for c in range(clips_per_game):
            clip = f"g{g}-c{c}"
            jitter = (0.01 if clip == STEADY else 0.1) if smoothed else 0.5
            common = np.cumsum(jitter * rng.standard_normal((n_frames, 2)), axis=0)
            for k in range(n_frames):
                frame = 1 + 5 * k
                abstains = not smoothed and k == 7
                frames.append(
                    {
                        "clip": clip,
                        "game": g,
                        "frame": frame,
                        "ok_out": not abstains,
                        "line_px_out": None if abstains else float(rng.uniform(2, 15)),
                        "cam_speed_px_s": 300.0 if clip in FAST else 50.0,
                        "filled": False,
                    }
                )
                if abstains or k == UNLABELLED_K:
                    continue
                truth = rng.uniform([-40, -30], [40, 30], (n_players, 2))
                e = common[k] + 0.05 * rng.standard_normal((n_players, 2))
                players.append(
                    pl.DataFrame(
                        {
                            "clip": clip,
                            "game": g,
                            "frame": frame,
                            "track_id": np.arange(n_players),
                            "x_pred": truth[:, 0] + e[:, 0],
                            "y_pred": truth[:, 1] + e[:, 1],
                            "ex_lab": e[:, 0],
                            "ey_lab": e[:, 1],
                            "ex_sm": e[:, 0],
                            "ey_sm": e[:, 1],
                            "c00": np.full(n_players, 0.3),
                            "c01": np.full(n_players, 0.05),
                            "c11": np.full(n_players, 0.5),
                            "mpp": rng.uniform(0.02, 0.08, n_players),
                            "n_inliers": rng.integers(6, 30, n_players),
                            "dof": np.full(n_players, np.nan),
                            "cov_source": "smoother" if smoothed else "frame",
                            "filled": False,
                        }
                    )
                )
    path.mkdir(parents=True)
    pl.concat(players).write_parquet(path / "players.parquet")
    pl.DataFrame(frames).write_parquet(path / "frames.parquet")
    return path


def test_the_clip_is_the_fast_one_whose_smoothing_removes_most_jitter(tmp_path, rng):
    runs = {
        "none": demo.load_run(write_run(tmp_path / "none", rng, smoothed=False)),
        "ptz": demo.load_run(write_run(tmp_path / "ptz", rng, smoothed=True)),
    }
    assert runs["none"].players["q_mond"].is_finite().all()
    table = demo.clip_table(runs)
    assert table.height == 12
    assert table.filter(pl.col("clip").is_in(FAST))["speed_ptz"].min() == 300.0
    assert demo.choose_clip(table) == STEADY
    ratio = table.filter(pl.col("clip").is_in(FAST)).sort("step_ratio")["step_ratio"].to_list()
    assert ratio[0] < 0.5 * ratio[1]  # the player noise keeps the steady clip's ratio above zero
    slow_best = table.filter(~pl.col("clip").is_in(FAST))["step_ratio"].min()
    assert slow_best > ratio[0]


def test_the_window_starts_where_the_per_frame_camera_jitters_most():
    frames = 1 + 5 * np.arange(12)
    common = np.zeros(12)
    common[6:10] = [0.0, 2.0, -2.0, 2.0]  # large steps between frames 6 and 9
    rows = pl.DataFrame(
        {
            "clip": "c",
            "frame": np.repeat(frames, 3),
            "track_id": np.tile([0, 1, 2], 12),
            "ex_sm": np.repeat(common, 3),
            "ey_sm": np.zeros(36),
            "filled": False,
        }
    )
    assert demo.choose_start(rows, frames, 4) == 6
    assert demo.choose_start(rows, frames, 20) == 0


def test_a_panel_abstains_only_without_a_camera_and_is_not_evaluated_without_labels(tmp_path, rng):
    run = demo.load_run(write_run(tmp_path / "none", rng, smoothed=False))
    frame = 1 + 5 * np.arange(20)
    shown = demo.frame_view(run, "g1-c0", int(frame[0]))
    assert (shown.state, shown.rows.height, shown.filled) == (demo.SHOWN, 6, False)
    abstains = demo.frame_view(run, "g1-c0", int(frame[7]))
    assert (abstains.state, abstains.rows.height) == (demo.ABSTAINS, 0)
    assert np.isnan(abstains.line_px)
    # A camera that published positions on a frame without usable labels is not an abstention.
    unlabelled = demo.frame_view(run, "g1-c0", int(frame[UNLABELLED_K]))
    assert (unlabelled.state, unlabelled.rows.height) == (demo.UNLABELLED, 0)
    expected = run.frames.filter((pl.col("clip") == "g1-c0") & (pl.col("frame") == int(frame[UNLABELLED_K])))
    assert unlabelled.line_px == expected["line_px_out"][0]


def test_main_writes_a_small_animation_and_a_still(tmp_path, rng):
    none = write_run(tmp_path / "none", rng, smoothed=False)
    ptz = write_run(tmp_path / "ptz", rng, smoothed=True)
    gif, png = tmp_path / "demo.gif", tmp_path / "demo.png"
    args = ["--none-run", str(none), "--ptz-run", str(ptz), "--seconds", "2", "--start", "0"]
    args += ["--gif", str(gif), "--png", str(png), "--report", str(tmp_path / "demo.md"), "--width", "6"]
    result = CliRunner().invoke(demo.app, args)
    assert result.exit_code == 0, result.output
    text = (tmp_path / "demo.md").read_text()
    assert f"**Clip {STEADY} (game 2).**" in text
    # The window holds frames 0-9: the per-frame camera abstains on the 8th, both are unlabelled on the 4th.
    assert "| per-frame camera | " in text
    assert " | 1 of 10 | 1 of 10 |" in text
    assert "| PTZ-smoothed camera | " in text
    assert " | 0 of 10 | 1 of 10 |" in text
    assert "Not evaluated: the camera published positions" in text
    assert f"t = {5 * UNLABELLED_K / 25:.1f} s, " in text
    assert str(tmp_path) not in text
    with Image.open(gif) as im:
        assert im.n_frames == 10  # 2 s at 5 fps
        assert im.info["duration"] == 200
    assert gif.stat().st_size < 1_000_000
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plain_style_writes_the_showcase_files_and_no_report(tmp_path, rng, monkeypatch):
    none = write_run(tmp_path / "none", rng, smoothed=False)
    ptz = write_run(tmp_path / "ptz", rng, smoothed=True)
    encoded = {}

    def fake_mp4(frames, path, fps, **kwargs):
        encoded.update(n=len(frames), fps=fps, shape=frames[0].shape, kwargs=kwargs)
        path.write_bytes(b"mp4")
        return 3

    monkeypatch.setattr(demo.render_demo, "write_mp4", fake_mp4)
    out = {k: tmp_path / f"smoothing.{k}" for k in ("gif", "mp4", "jpg")}
    report = tmp_path / "report.md"
    args = [
        "--none-run",
        str(none),
        "--ptz-run",
        str(ptz),
        "--seconds",
        "2",
        "--start",
        "0",
        "--style",
        "plain",
    ]
    args += ["--gif", str(out["gif"]), "--mp4", str(out["mp4"]), "--poster", str(out["jpg"])]
    args += ["--report", str(report), "--width", "6"]
    result = CliRunner().invoke(demo.app, args)
    assert result.exit_code == 0, result.output
    assert not report.exists()
    assert f"clip {STEADY} (game 2)" in result.output
    with Image.open(out["gif"]) as im:
        assert im.n_frames == 10
        assert im.info["duration"] == 200
    assert encoded["n"] == 10
    assert encoded["fps"] == 5.0
    h, w, _ = encoded["shape"]
    assert h % 2 == 0
    assert w % 2 == 0
    assert out["jpg"].read_bytes()[:2] == b"\xff\xd8"


def test_plain_panels_have_plain_titles_and_words_and_no_numbers(tmp_path, rng):
    from matplotlib.figure import Figure

    run = demo.load_run(write_run(tmp_path / "none", rng, smoothed=False))
    frame = 1 + 5 * np.arange(20)
    window = (-50.0, 50.0, -35.0, 35.0)
    cases = [(0, "none", "Frame by frame", None), (7, "none", "Frame by frame", "no position"),
             (UNLABELLED_K, "ptz", "Smoothed over time", "not checked")]  # fmt: skip
    for k, cam, title, word in cases:
        ax = Figure().subplots()
        demo.draw_panel(ax, cam, demo.frame_view(run, "g1-c0", int(frame[k])), window, plain=True)
        assert ax.get_title(loc="left") == title
        texts = [t.get_text() for t in ax.texts]
        assert texts == ([] if word is None else [word])
        assert not any(c.isdigit() for t in [title, *texts] for c in t)
