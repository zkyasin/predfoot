"""Stub stages of the tracer bullet, with the signatures that :mod:`predfoot.pipeline.real` keeps.

- calibrate: ClipFrame → CalibResult. Reads the homography from the input.
- detect: ClipFrame → list[FootObs]. Reads the foot points from the input, as the real stages do.
- localize: CalibResult, FootObs → Localization. Inverse homography, exact; the uncertainty is a
  placeholder, a constant 90 % radius with an isotropic covariance.
- health: CalibResult → (calib_health, abstain). No score; abstain when calibration failed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from predfoot.contracts import CalibResult, CalibStatus, FootObs, Localization
from predfoot.io.skillcorner import CHI2_2_090
from predfoot.pipeline.clip import ClipFrame

#: Placeholder 90 % radius (m) of the stub stages. Not a measured value.
PLACEHOLDER_RADIUS_M = 1.0
#: Condition number above which an input homography is treated as degenerate.
MAX_CONDITION = 1e12


def calibrate(frame: ClipFrame) -> CalibResult:
    """Calibration read from the input, normalised as the contract requires (H[2, 2] = 1)."""
    if frame.homography is None:
        return CalibResult(status=CalibStatus.FEW_POINTS)
    h = np.array(frame.homography, dtype=np.float64)
    if abs(h[2, 2]) < 1e-12:
        return CalibResult(status=CalibStatus.DEGENERATE)
    h = h / h[2, 2]
    if not np.isfinite(cond := float(np.linalg.cond(h))) or cond > MAX_CONDITION:
        return CalibResult(status=CalibStatus.DEGENERATE, cond=cond)
    return CalibResult(status=CalibStatus.OK, H=h)


def detect(frame: ClipFrame) -> list[FootObs]:
    """Foot points read from the input; exact, so no pixel covariance."""
    return [FootObs(track_id=p.track_id, uv=np.array([p.u, p.v], dtype=np.float64)) for p in frame.feet]


def localize(
    calib: CalibResult, feet: Sequence[FootObs], *, radius_m: float = PLACEHOLDER_RADIUS_M
) -> list[Localization]:
    """Pitch positions by inverse homography, with a placeholder uncertainty.

    A foot point whose back-projected ray meets the ground behind the camera (homogeneous w ≤ 0, i.e. above
    the horizon) gets no position. The placeholder covariance is (r² / χ²₂(0.90)) I, whose 90 % ellipse is
    the circle of radius ``radius_m``, so ``p90_radius_m`` and ``pos_cov_m2`` agree.
    """
    if not calib.ok or calib.H is None:
        return [Localization(track_id=f.track_id, xy=None) for f in feet]
    if not feet:
        return []
    uv1 = np.column_stack([np.array([f.uv for f in feet], dtype=np.float64), np.ones(len(feet))])
    q = np.linalg.solve(calib.H, uv1.T).T  # CalibResult.H has H[2, 2] = 1: w > 0 in front of the camera
    cov = (radius_m**2 / CHI2_2_090) * np.eye(2)
    out = []
    for foot, qi in zip(feet, q, strict=True):
        if qi[2] <= 0.0:
            out.append(Localization(track_id=foot.track_id, xy=None))
        else:
            out.append(
                Localization(
                    track_id=foot.track_id, xy=qi[:2] / qi[2], cov_xy=cov.copy(), p90_radius_m=radius_m
                )
            )
    return out


def health(calib: CalibResult) -> tuple[float | None, bool]:
    """(calib_health, abstain): no score; abstain exactly when calibration failed."""
    return None, not calib.ok
