import json

import numpy as np
import polars as pl
import pytest

from predfoot.calib.temporal import FrameFlags, NoiseParams
from predfoot.demo import calibration as dc


def _players(rng, n=3000, sources=("frame", "smoother"), passthrough=True):
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    err = rng.multivariate_normal([0, 0], cov * 2.0, size=n)
    source = np.array(sources)[np.arange(n) % len(sources)]
    return pl.DataFrame({
        "ex_lab": err[:, 0], "ey_lab": err[:, 1], "c00": np.full(n, cov[0, 0]), "c01": np.zeros(n),
        "c11": np.full(n, cov[1, 1]), "clip": rng.integers(0, 30, n).astype(str),
        "game": rng.integers(0, 6, n),
        "cov_source": source, "passthrough": (source == "frame") & passthrough,
        "mpp": rng.uniform(0.02, 0.08, n), "n_inliers": rng.integers(5, 20, n),
    })  # fmt: skip


def _smoothing(kappas=(1.8, 2.0, 1.9)):
    folds = {str(g): {"params": {"kappa2": k, "q_pan": 1e-3, "q_tilt": 1e-4, "q_logf": 5e-3}}
             for g, k in zip((2, 3, 4), kappas, strict=True)}  # fmt: skip
    return {"smooth": "ptz", "fit_scope": "fold", "config_digest": "abc", "folds": folds}


def test_models_per_covariance_source(rng):
    models = dc.fit_models(_players(rng))
    assert set(models) == {"frame", "smoother"}
    # scores are 2 × χ²₂, so the pooled 90 % quantile is near 2 × 4.6
    assert 7.0 < models["smoother"].pooled < 11.5
    assert set(models["frame"].edges) == {"mpp", "n_inliers"}


def test_noise_params_are_the_median_over_folds():
    params = dc.median_noise_params(_smoothing())
    assert params.kappa2 == 1.9
    assert params.q_pan == 1e-3


def test_dev_runs_must_match_the_camera(tmp_path, rng):
    ptz, none = tmp_path / "ptz", tmp_path / "none"
    for run in (ptz, none):
        run.mkdir()
    _players(rng).write_parquet(ptz / "players.parquet")
    (ptz / "smoothing.json").write_text(json.dumps(_smoothing()))
    _players(rng, sources=("frame",), passthrough=False).write_parquet(none / "players.parquet")
    cal = dc.from_dev_run(ptz, "ptz")
    assert cal.noise is not None
    assert cal.noise.kappa2 == 1.9
    assert cal.provenance["smoother_config_digest"] == "abc"
    assert set(dc.from_dev_run(none, "none").models) == {"frame"}
    with pytest.raises(ValueError, match=r"smoothing\.json"):
        dc.from_dev_run(none, "ptz")
    with pytest.raises(ValueError, match="per-frame"):
        dc.from_dev_run(ptz, "none")


def test_json_round_trip_and_checks(rng):
    models = dc.fit_models(_players(rng))
    cams = {
        "ptz": dc.CameraCalibration(models, NoiseParams(kappa2=1.9), {"dev_run": "x"}),
        "none": dc.CameraCalibration({"frame": models["frame"]}),
    }
    back = dc.from_json(dc.to_json(cams))
    assert back["ptz"].noise == NoiseParams(kappa2=1.9)
    assert back["none"].noise is None
    for name, cal in cams.items():
        for src, m in cal.models.items():
            assert back[name].models[src].to_json() == m.to_json()
    # an unknown source falls back to the per-frame model
    assert back["ptz"].model_for("projected") is back["ptz"].models["frame"]

    d = json.loads(dc.to_json(cams))
    with pytest.raises(ValueError, match="format"):
        dc.from_json(json.dumps(d | {"format": "other"}))
    no_noise = json.loads(dc.to_json(cams))
    no_noise["cameras"]["ptz"]["noise"] = None
    with pytest.raises(ValueError, match="noise"):
        dc.from_json(json.dumps(no_noise))
    bad_q = json.loads(dc.to_json(cams))
    bad_q["cameras"]["none"]["models"]["frame"]["pooled"] = -1.0
    with pytest.raises(ValueError, match="not a valid conformal model"):
        dc.from_json(json.dumps(bad_q))
    no_frame = json.loads(dc.to_json(cams))
    del no_frame["cameras"]["none"]["models"]["frame"]
    with pytest.raises(ValueError, match="'frame'"):
        dc.from_json(json.dumps(no_frame))


def test_committed_calibration_loads():
    cams = dc.load()
    assert set(cams) == set(dc.CAMERAS)
    ptz, none = cams["ptz"], cams["none"]
    assert set(ptz.models) == {"frame", "projected", "smoother"}
    assert set(none.models) == {"frame"}
    assert ptz.noise is not None
    assert 1.0 < ptz.noise.kappa2 < 4.0
    assert ptz.provenance["games"] == none.provenance["games"] == 6
    # the per-frame Laplace covariance needs inflating on the dev clips: q̂ well above χ²₂(0.90) = 4.6
    assert none.models["frame"].pooled > 4.6


def test_duplicates_take_the_source_of_the_frame_they_copy():
    flags = FrameFlags.empty(5)
    flags.smoothed[0] = True
    flags.duplicate[1] = True
    flags.projected[2] = True
    flags.duplicate[3] = True
    flags.passthrough[4] = True
    assert dc.cov_sources(flags) == ["smoother", "smoother", "projected", "projected", "frame"]
