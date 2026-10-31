"""Uncertainty calibration that the SoccerTrack v2 demo borrows from the SN-GSR dev evaluation.

SoccerTrack v2 has no ground truth to calibrate 90 % regions on, so the demo reuses what
``scripts/eval_dev.py`` fitted on the SN-GSR dev clips (labels as reference, all six games), for each camera
the demo can run:

- ``ptz``, the camera of the PTZ smoother: one Mondrian conformal model per covariance source (``smoother``,
  ``projected``, ``frame``; ``frame`` rows are raw cameras passed through by the smoother, with covariance
  κ² Σθ) and the smoother's noise parameters, the median of the six label-free fold fits;
- ``none``, the per-frame camera with its raw Laplace covariance Σθ: one model (``frame``).

Scores, features and bins are those of the dev evaluation: squared Mahalanobis distances of the errors
against the labels, clips as groups, bins of metres per pixel at the feet (3) × RANSAC inliers (2), α = 0.1.

``dev_calibration.json``, next to this module, holds both cameras with aggregate numbers only (bin edges,
quantiles, bin counts and four noise parameters), so the demo runs without SN-GSR data.
``scripts/fit_demo_calibration.py`` rebuilds it from two dev runs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from predfoot.calib.temporal import FrameFlags, NoiseParams
from predfoot.uncertainty.conformal import MondrianConformal
from predfoot.uncertainty.delta import mahalanobis2

FORMAT = "predfoot-demo-calibration/1"
DEFAULT_PATH = Path(__file__).with_name("dev_calibration.json")
CAMERAS = ("ptz", "none")
ALPHA = 0.1
BINS = {"mpp": 3, "n_inliers": 2}
NOISE_KEYS = ("kappa2", "q_pan", "q_tilt", "q_logf")


@dataclass(frozen=True)
class CameraCalibration:
    """Conformal models per covariance source of one camera, and the smoother noise for ``ptz``."""

    models: dict[str, MondrianConformal]
    noise: NoiseParams | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def model_for(self, source: str) -> MondrianConformal:
        """Model of a covariance source; the per-frame (``frame``) model when that source has none."""
        return self.models.get(source, self.models["frame"])


def cov_sources(flags: FrameFlags) -> list[str]:
    """Covariance source of each output camera of the PTZ smoother, the rule of ``scripts/eval_dev.py``:
    ``smoother`` (smoothed, repaired, filled), ``projected`` or ``frame`` (raw, passed through); a duplicate
    takes the source of the frame it copies."""
    out: list[str] = []
    for k in range(len(flags.smoothed)):
        if k and flags.duplicate[k]:
            out.append(out[k - 1])
        elif flags.smoothed[k] or flags.repaired[k] or flags.filled[k]:
            out.append("smoother")
        elif flags.projected[k]:
            out.append("projected")
        else:
            out.append("frame")
    return out


def fit_models(players: pl.DataFrame) -> dict[str, MondrianConformal]:
    """Mondrian models per covariance source (column ``cov_source``), fitted on every labelled player-frame
    of a dev players table (columns of ``scripts/eval_dev.py``)."""
    d = players.filter(pl.col("ex_lab").is_not_nan())
    if d.is_empty():
        raise ValueError("no labelled player-frames to fit on")
    e = d.select("ex_lab", "ey_lab").to_numpy()
    cov = np.stack([d["c00"], d["c01"], d["c01"], d["c11"]], axis=1).reshape(-1, 2, 2)
    score, groups, source = mahalanobis2(e, cov), d["clip"].to_numpy(), d["cov_source"].to_numpy()
    feats = {"mpp": d["mpp"].to_numpy(), "n_inliers": d["n_inliers"].to_numpy().astype(float)}
    models = {}
    for src in np.unique(source):
        sel = source == src
        models[str(src)] = MondrianConformal.fit(
            score[sel], groups[sel], {k: v[sel] for k, v in feats.items()}, BINS, ALPHA
        )
    return models


def median_noise_params(smoothing: Mapping[str, Any]) -> NoiseParams:
    """Median over the folds of the noise parameters in a dev run's ``smoothing.json``."""
    folds = list(smoothing["folds"].values())
    if not folds:
        raise ValueError("smoothing.json has no fold fits")
    return NoiseParams(**{k: float(np.median([f["params"][k] for f in folds])) for k in NOISE_KEYS})


def from_dev_run(run: Path, camera: str) -> CameraCalibration:
    """Calibration of ``camera`` from a dev run of ``scripts/eval_dev.py`` made with that camera.

    ``ptz`` needs a run with ``--smooth ptz`` (``smoothing.json`` present); ``none`` needs a per-frame run,
    whose rows all have the raw per-frame covariance. ValueError when the run does not match.
    """
    players = pl.read_parquet(run / "players.parquet")
    sources = set(players["cov_source"].unique().to_list())
    smoothing_path = run / "smoothing.json"
    provenance: dict[str, Any] = {
        "dev_run": run.name,
        "player_frames": int(players.filter(pl.col("ex_lab").is_not_nan()).height),
        "clips": int(players["clip"].n_unique()),
        "games": int(players["game"].n_unique()),
    }
    noise = None
    if camera == "ptz":
        if not smoothing_path.exists():
            raise ValueError(f"{run} has no smoothing.json: not a PTZ-smoother dev run")
        smoothing = json.loads(smoothing_path.read_text())
        if smoothing.get("smooth") != "ptz":
            raise ValueError(f"{run} was made with --smooth {smoothing.get('smooth')}, not ptz")
        noise = median_noise_params(smoothing)
        n_folds, scope = len(smoothing["folds"]), smoothing["fit_scope"]
        provenance |= {
            "noise": f"median of the {n_folds} fold fits (fit scope {scope})",
            "smoother_config_digest": smoothing["config_digest"],
        }
    elif camera == "none":
        if smoothing_path.exists() or sources != {"frame"} or bool(players["passthrough"].any()):
            raise ValueError(f"{run} is not a per-frame dev run (--smooth none)")
    else:
        raise ValueError(f"unknown camera {camera!r}, expected one of {CAMERAS}")
    return CameraCalibration(models=fit_models(players), noise=noise, provenance=provenance)


def to_json(cameras: Mapping[str, CameraCalibration]) -> str:
    out: dict[str, Any] = {"format": FORMAT, "alpha": ALPHA, "bins": BINS, "cameras": {}}
    for name, cal in cameras.items():
        out["cameras"][name] = {
            "provenance": cal.provenance,
            "noise": cal.noise.to_dict() if cal.noise is not None else None,
            "models": {src: json.loads(m.to_json()) for src, m in sorted(cal.models.items())},
        }
    return json.dumps(out, indent=2) + "\n"


def _check(name: str, cal: CameraCalibration) -> None:
    if "frame" not in cal.models:
        raise ValueError(f"camera {name!r} has no per-frame ('frame') conformal model")
    if name == "ptz" and cal.noise is None:
        raise ValueError("camera 'ptz' has no smoother noise parameters")
    for src, m in cal.models.items():
        q = np.asarray([m.pooled, *m.quantiles.values()], dtype=np.float64)
        sorted_edges = all(np.all(np.diff(np.asarray(e, dtype=np.float64)) >= 0) for e in m.edges.values())
        if not (np.all(np.isfinite(q) & (q > 0)) and sorted_edges and set(m.edges) <= set(BINS)):
            raise ValueError(f"camera {name!r}, source {src!r}: not a valid conformal model")


def from_json(text: str) -> dict[str, CameraCalibration]:
    """Cameras of a file written by :func:`to_json`, checked; ValueError when it is not one."""
    try:
        d = json.loads(text)
        if d.get("format") != FORMAT:
            raise ValueError(f"format {d.get('format')!r}, expected {FORMAT!r}")
        out = {
            name: CameraCalibration(
                models={src: MondrianConformal.from_json(json.dumps(m)) for src, m in c["models"].items()},
                noise=NoiseParams.from_dict(c["noise"]) if c["noise"] is not None else None,
                provenance=dict(c["provenance"]),
            )
            for name, c in d["cameras"].items()
        }
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as err:
        raise ValueError(f"not a demo calibration file: {err}") from err
    for name, cal in out.items():
        _check(name, cal)
    return out


def load(path: Path = DEFAULT_PATH) -> dict[str, CameraCalibration]:
    """The calibration file (default: the committed ``dev_calibration.json``)."""
    return from_json(path.read_text(encoding="utf-8"))
