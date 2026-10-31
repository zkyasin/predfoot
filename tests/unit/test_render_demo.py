import importlib.util
import json

import numpy as np
import polars as pl

from predfoot.geometry import ptz
from predfoot.paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location("render_demo", REPO_ROOT / "scripts" / "render_demo.py")
assert _spec is not None
assert _spec.loader is not None
render_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_demo)


def test_pitch_lines_are_drawn_where_the_camera_sees_the_pitch():
    H = ptz.homography([0.0, 0.35, 0.0, np.log(1500.0)], [0.0, -45.0, 15.0])
    img = np.zeros((1080, 1920, 3), np.uint8)
    render_demo.draw_pitch_lines(img, H, (255, 255, 255), 2)
    assert (img > 0).any(axis=2).mean() > 0.002
    behind = ptz.homography([np.pi, 0.35, 0.0, np.log(1500.0)], [0.0, -45.0, 15.0])  # looking away
    empty = np.zeros_like(img)
    render_demo.draw_pitch_lines(empty, behind, (255, 255, 255), 2)
    assert not empty.any()


def test_gif_stays_under_the_size_cap(tmp_path, rng):
    frames = [rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(6)]
    size = render_demo.write_gif(frames, tmp_path / "demo.gif", width=640, fps=5, max_bytes=200_000)
    assert size <= 200_000
    assert (tmp_path / "demo.gif").stat().st_size == size


def test_conformal_models_per_covariance_source(tmp_path, rng):
    n = 3000
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    err = rng.multivariate_normal([0, 0], cov * 2.0, size=n)
    table = pl.DataFrame({
        "ex_lab": err[:, 0], "ey_lab": err[:, 1], "c00": np.full(n, cov[0, 0]), "c01": np.zeros(n),
        "c11": np.full(n, cov[1, 1]), "clip": rng.integers(0, 30, n).astype(str),
        "cov_source": np.where(np.arange(n) % 3 == 0, "frame", "smoother"),
        "mpp": rng.uniform(0.02, 0.08, n), "n_inliers": rng.integers(5, 20, n),
    })  # fmt: skip
    table.write_parquet(tmp_path / "players.parquet")
    models = render_demo.fit_conformal(tmp_path / "players.parquet")
    assert set(models) == {"frame", "smoother"}
    # scores are 2 × χ²₂, so the pooled 90 % quantile is near 2 × 4.6
    assert 7.0 < models["smoother"].pooled < 11.5


def test_dev_noise_params_take_the_median_over_folds(tmp_path):
    folds = {str(g): {"params": {"kappa2": k, "q_pan": 1e-3, "q_tilt": 1e-4, "q_logf": 5e-3}} for g, k in
             ((2, 1.8), (3, 2.0), (4, 1.9))}  # fmt: skip
    (tmp_path / "smoothing.json").write_text(json.dumps({"folds": folds}))
    params = render_demo.dev_noise_params(tmp_path)
    assert params.kappa2 == 1.9
    assert params.q_pan == 1e-3
