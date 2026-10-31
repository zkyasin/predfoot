"""Real stages for ``predfoot run``: cached network detections → robust calibration → delta-method positions
scaled by conformal quantiles. They keep the signatures of the stubs in :mod:`predfoot.pipeline.stub`.

The calibrated covariance of a player is (q̂ / χ²₂(0.90)) · Σp, so its 90 % ellipse is exactly the conformal
region {e : eᵀ Σp⁻¹ e ≤ q̂}, and ``p90_radius_m`` = √(q̂ · λmax(Σp)) is the circle that contains it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from predfoot.calib.observations import observations_from_detections
from predfoot.calib.solver import SolverConfig
from predfoot.calib.solver import calibrate as solve
from predfoot.contracts import CalibResult, CalibStatus, FootObs, Localization
from predfoot.geometry import homography as hg
from predfoot.keypoints.cache import load_detections
from predfoot.pipeline import stub
from predfoot.pipeline.clip import ClipFrame
from predfoot.pipeline.run import Stages
from predfoot.uncertainty.conformal import MondrianConformal
from predfoot.uncertainty.delta import CHI2_2_90, position_covariance


def real_stages(
    detections: Mapping[int, dict[str, Any]],
    conformal: MondrianConformal | None,
    version: str,
    solver: SolverConfig | None = None,
    seed: int = 0,
) -> Stages:
    """Stages over ``detections`` keyed by the clip's frame index (0 … n−1).

    Without ``conformal``, regions are the uncalibrated Laplace ellipses (q̂ = χ²₂(0.90)); with it, q̂ comes
    from the Mondrian bin of each player (metres per pixel at the feet, number of RANSAC inliers).
    """
    cfg = solver or SolverConfig()

    def calibrate(frame: ClipFrame) -> CalibResult:
        det = detections.get(frame.frame)
        if det is None:
            return CalibResult(CalibStatus.FEW_POINTS)
        xy, uv, w, lines = observations_from_detections(det)
        rng = np.random.default_rng(seed * 100_003 + frame.frame)
        result, _ = solve(xy, uv, rng, weights=w, config=cfg, lines=lines)
        return result

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
        description=_description(conformal),
        conformal=None if conformal is None else conformal_provenance(conformal),
    )


def conformal_provenance(model: MondrianConformal) -> dict[str, Any]:
    """What meta.json records about a conformal model: enough to recognise it and to recompute every q̂."""
    return {
        "kind": "mondrian",
        "sha256": hashlib.sha256(model.to_json().encode("utf-8")).hexdigest(),
        "alpha": model.alpha,
        "pooled": model.pooled,
        "edges": model.edges,
        "quantiles": model.quantiles,
    }


def _description(conformal: MondrianConformal | None) -> dict[str, str]:
    regions = (
        "scaled by the Mondrian conformal quantile of each player's bin (metres per pixel at the feet, "
        "RANSAC inliers)"
        if conformal is not None
        else "uncalibrated Laplace regions (q = χ²₂(0.90))"
    )
    return {
        "calibration": "RANSAC + soft-L1 Levenberg-Marquardt on the cached keypoint and line detections; "
        "Laplace covariance of the 8 homography parameters",
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
