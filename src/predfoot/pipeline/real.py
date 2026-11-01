"""Real stages for ``predfoot run``: cached network detections → robust calibration (optionally smoothed over
time) → delta-method positions scaled by conformal quantiles. They keep the signatures of the stubs in
:mod:`predfoot.pipeline.stub`.

The calibrated covariance of a player is (q̂ / χ²₂(0.90)) · Σp, so its 90 % ellipse is exactly the conformal
region {e : eᵀ Σp⁻¹ e ≤ q̂}, and ``p90_radius_m`` = √(q̂ · λmax(Σp)) is the circle that contains it.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from predfoot.calib.observations import observations_from_detections
from predfoot.calib.solver import SolverConfig
from predfoot.calib.solver import calibrate as solve
from predfoot.calib.temporal import (
    CentreEstimate,
    NoiseParams,
    SmootherConfig,
    duplicate_mask,
    smooth_results,
)
from predfoot.contracts import CalibResult, CalibStatus, FootObs, Localization
from predfoot.geometry import homography as hg
from predfoot.keypoints.cache import load_detections
from predfoot.pipeline import stub
from predfoot.pipeline.clip import ClipFrame, ClipInput
from predfoot.pipeline.run import ClipMismatch, Stages
from predfoot.uncertainty.conformal import MondrianConformal
from predfoot.uncertainty.delta import CHI2_2_90, position_covariance

log = logging.getLogger(__name__)


def real_stages(
    detections: Mapping[int, dict[str, Any]],
    conformal: MondrianConformal | None,
    version: str,
    solver: SolverConfig | None = None,
    seed: int = 0,
    temporal: tuple[SmootherConfig, NoiseParams] | None = None,
    game_centre: CentreEstimate | None = None,
) -> Stages:
    """Stages over ``detections`` keyed by the clip's frame index (0 … n−1).

    Without ``conformal``, regions are the uncalibrated Laplace ellipses (q̂ = χ²₂(0.90)); with it, q̂ comes
    from the Mondrian bin of each player (metres per pixel at the feet, number of RANSAC inliers).

    With ``temporal = (config, params)``, the first call calibrates every frame once, flags duplicate
    detections and smooths the camera with :func:`predfoot.calib.temporal.smooth_results` (the config's
    ``fps`` is the frame rate of the clip's frame index); later calls return the cached outputs. When the
    clip's own projection centre is unreliable, ``game_centre`` (the median centre of the match's clips,
    :func:`predfoot.calib.temporal.game_centre`) is used if it explains the clip, as in the dev evaluation;
    otherwise the clip passes through. The version then carries the smoother's config digest, a hash of its
    parameters and of the game centre, so cached blocks of another camera are never reused, and the conformal
    quantiles must come from the same smoother. The dev evaluation calibrates smoothed and passed-through
    frames separately (their covariances have different scales); a single ``conformal`` model here does not.
    """
    cfg = solver or SolverConfig()
    if temporal is not None:
        version = f"{version}+ptz-{temporal[0].digest()}-{temporal[1].digest()}"
        if game_centre is not None:
            version += f"-{game_centre.digest()}"
    smoothed: dict[int, CalibResult] = {}
    done = False

    def solve_frame(frame: int) -> CalibResult:
        det = detections.get(frame)
        if det is None:
            return CalibResult(CalibStatus.FEW_POINTS)
        xy, uv, w, lines = observations_from_detections(det)
        rng = np.random.default_rng(seed * 100_003 + frame)
        result, _ = solve(xy, uv, rng, weights=w, config=cfg, lines=lines)
        return result

    def smooth_clip_once() -> None:
        nonlocal done
        if done or temporal is None:
            return
        frames = sorted(detections)
        raw = [solve_frame(f) for f in frames]
        dup = duplicate_mask([detections[f] for f in frames])
        out, _, _ = smooth_results(frames, raw, dup, temporal[0], temporal[1], game=game_centre)
        smoothed.update(zip(frames, out, strict=True))
        done = True

    def calibrate(frame: ClipFrame) -> CalibResult:
        if temporal is None:
            return solve_frame(frame.frame)
        smooth_clip_once()
        return smoothed.get(frame.frame, CalibResult(CalibStatus.FEW_POINTS))

    def localize(calib: CalibResult, feet: Sequence[FootObs]) -> list[Localization]:
        if not calib.ok or calib.H is None or calib.cov_theta is None:
            return [Localization(f.track_id, xy=None) for f in feet]
        if not feet:
            return []
        uv = np.array([f.uv for f in feet], dtype=np.float64)
        xy, cov, valid = position_covariance(calib.H, calib.cov_theta, uv)
        if conformal is None:
            q = np.full(len(feet), CHI2_2_90)
        else:
            mpp = np.sqrt(np.abs(np.linalg.det(hg.jacobian_backproject_uv(calib.H, uv))))
            q = conformal.quantile_for({"mpp": mpp, "n_inliers": np.full(len(feet), float(calib.n_inliers))})
        out = []
        for i, f in enumerate(feet):
            if not valid[i] or not np.isfinite(q[i]):
                out.append(Localization(f.track_id, xy=None))
                continue
            radius = float(np.sqrt(q[i] * np.linalg.eigvalsh(cov[i])[-1]))
            out.append(
                Localization(f.track_id, xy=xy[i], cov_xy=cov[i] * (q[i] / CHI2_2_90), p90_radius_m=radius)
            )
        return out

    return Stages(
        version=version,
        calibrate=calibrate,
        detect=stub.detect,
        localize=localize,
        health=stub.health,
        description=_description(conformal, smoothed=temporal is not None),
        conformal=None if conformal is None else conformal_provenance(conformal),
        check_clip=lambda clip: check_detection_frames(detections, clip),
    )


def check_detection_frames(detections: Mapping[int, Any], clip: ClipInput) -> None:
    """Detections are keyed by the 0-based index of the clip's frames (``frame`` in ``predfoot-clip/1``).

    Raises :class:`ClipMismatch` when a key is beyond the clip's last frame, as with a file keyed by image
    number (the detection cache numbers SN-GSR images from 1, one in five). Logs a warning when frame 0 has no
    entry: a file keyed from 1 would give every frame the camera of the next one.
    """
    n = len(clip.frames)
    beyond = sorted(f for f in detections if f >= n)
    if beyond:
        shown = ", ".join(map(str, beyond[:5])) + (", …" if len(beyond) > 5 else "")
        raise ClipMismatch(
            f"the detections name frames beyond the {n} frames of clip {clip.clip_id!r} ({shown}): they must "
            "be keyed by the 0-based index of the clip's frames, not by image number"
        )
    if detections and 0 not in detections:
        log.warning(
            "the detections have no entry for frame 0 of clip %r: check that they are keyed by the 0-based "
            "index of the clip's frames (a file keyed from 1 gives every frame the camera of the next one)",
            clip.clip_id,
        )


def conformal_provenance(model: MondrianConformal) -> dict[str, Any]:
    """What meta.json records about a conformal model: enough to recognise it and to recompute every q̂.

    meta.json is strict JSON, so a non-finite quantile (an empty calibration set gives +inf) or bin edge is
    written as a string, "inf".
    """

    def number(value: float) -> float | str:
        return float(value) if math.isfinite(value) else str(float(value))

    return {
        "kind": "mondrian",
        "sha256": hashlib.sha256(model.to_json().encode("utf-8")).hexdigest(),
        "alpha": model.alpha,
        "pooled": number(model.pooled),
        "edges": {
            name: [e if math.isfinite(e) else number(e) for e in es] for name, es in model.edges.items()
        },
        "quantiles": {key: number(q) for key, q in model.quantiles.items()},
    }


def _description(conformal: MondrianConformal | None, smoothed: bool = False) -> dict[str, str]:
    regions = (
        "scaled by the Mondrian conformal quantile of each player's bin (metres per pixel at the feet, "
        "RANSAC inliers)"
        if conformal is not None
        else "uncalibrated Laplace regions (q = χ²₂(0.90))"
    )
    return {
        "calibration": "RANSAC + soft-L1 Levenberg-Marquardt on the cached keypoint and line detections; "
        "Laplace covariance of the 8 homography parameters"
        + (
            "; then fixed-centre pan-tilt-zoom Kalman/RTS smoothing over the clip (predfoot.calib.temporal)"
            if smoothed
            else ""
        ),
        "detection": "foot points read from the input",
        "localization": f"inverse homography; delta-method covariance from the calibration, {regions}",
        "health": "none: calib_health is null; a frame abstains when its calibration failed",
    }


def stages_from_files(
    detections: Path, conformal: MondrianConformal | None = None, solver: SolverConfig | None = None
) -> Stages:
    """Real stages over a detections file (``predfoot-detections/1``, keyed by the clip's frame index).

    The version names the detection model, a hash of the file and the conformal model, so that a resumed run
    never reuses blocks computed from other detections or other quantiles.
    """
    path = Path(detections)
    dets, model_version = load_detections(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    tag = (
        "laplace"
        if conformal is None
        else f"mondrian@{hashlib.sha256(conformal.to_json().encode('utf-8')).hexdigest()[:12]}"
    )
    return real_stages(dets, conformal, f"real+{model_version or 'unknown'}+det@{digest}+{tag}", solver)
