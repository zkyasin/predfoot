"""Per-frame calibration-health features, computed from the calibration and its inputs only (no labels).

They feed the failure detector: a frame whose camera is wrong should look different on at least one of them
(few or clustered correspondences, large residuals, an ill-conditioned fit, a jump from the previous frame,
large predicted uncertainty).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from predfoot.contracts import CalibResult
from predfoot.geometry import homography as hg
from predfoot.uncertainty.delta import position_covariance

FloatArray = NDArray[np.float64]

FEATURES = [
    "n_kp",
    "n_lines",
    "n_corr",
    "inlier_ratio",
    "rms_px",
    "log_s2",
    "log_cond",
    "kp_conf_mean",
    "hull_area",
    "log_jump_px",
    "log_sigma_centre",
]

#: Image points (fractions of width, height) used to compare consecutive calibrations.
_GRID = np.array([[x, y] for x in (0.1, 0.5, 0.9) for y in (0.45, 0.7, 0.95)])


def _hull_area(points: FloatArray) -> float:
    """Area of the convex hull of 2D points (monotone chain), 0 for fewer than 3 points."""
    if len(points) < 3:
        return 0.0
    pts = sorted(map(tuple, np.asarray(points, dtype=np.float64)))

    def half(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in seq:
            while (
                len(out) >= 2
                and (
                    (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                    - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])
                )
                <= 0
            ):
                out.pop()
            out.append(p)
        return out

    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    x, y = np.array(hull).T
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def calibration_jump_px(H: ArrayLike, H_prev: ArrayLike, size: tuple[int, int]) -> float:
    """Mean image displacement (px) of a grid of ground points between two consecutive calibrations."""
    grid = _GRID * np.array(size)
    ground, valid = hg.backproject(H, grid)
    if not valid.any():
        return float("nan")
    moved = hg.project(H_prev, ground[valid])
    return float(np.mean(np.linalg.norm(moved - grid[valid], axis=1)))


def calibration_features(
    result: CalibResult,
    uv_corr: ArrayLike,
    n_kp: int,
    n_lines: int,
    kp_conf: ArrayLike,
    H_prev: ArrayLike | None = None,
    size: tuple[int, int] = (1920, 1080),
) -> dict[str, float]:
    """Features of one frame. Values that need a calibration are NaN when it failed."""
    uv = np.asarray(uv_corr, dtype=np.float64).reshape(-1, 2)
    conf = np.asarray(kp_conf, dtype=np.float64)
    feats = {
        "n_kp": float(n_kp),
        "n_lines": float(n_lines),
        "n_corr": float(len(uv)),
        "inlier_ratio": result.n_inliers / len(uv) if len(uv) else 0.0,
        "kp_conf_mean": float(conf.mean()) if conf.size else 0.0,
        "hull_area": _hull_area(uv) / (size[0] * size[1]),
        "rms_px": float("nan"),
        "log_s2": float("nan"),
        "log_cond": float("nan"),
        "log_jump_px": float("nan"),
        "log_sigma_centre": float("nan"),
    }
    if not result.ok or result.H is None or result.cov_theta is None:
        return feats
    feats["rms_px"] = float(result.rms_px or 0.0)
    feats["log_s2"] = float(np.log(max(result.s2 or 0.0, 1e-9)))
    feats["log_cond"] = float(np.log10(max(result.cond or 1.0, 1.0)))
    if H_prev is not None:
        feats["log_jump_px"] = float(np.log1p(calibration_jump_px(result.H, H_prev, size)))
    centre = np.array([[size[0] / 2, size[1] * 0.75]])  # a typical player position in the lower half
    _, cov, valid = position_covariance(result.H, result.cov_theta, centre)
    if valid[0]:
        feats["log_sigma_centre"] = float(np.log(np.sqrt(max(np.linalg.eigvalsh(cov[0])[-1], 1e-12))))
    return feats
