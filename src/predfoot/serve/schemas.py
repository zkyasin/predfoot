"""JSON bodies of the HTTP API. Pixels refer to the original frame (OpenCV convention); pitch quantities are
in metres, in the internal pitch frame (ADR-0001)."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

#: At most this many feet per /v1/localize request (a broadcast frame shows about 25 people).
MAX_FEET = 1000
#: Tolerances of the covariance checks, on the correlation scale (entries of Σθ span 10 orders of magnitude).
SYMMETRY_TOL = 1e-6
PSD_TOL = 1e-6
#: H[2, 2] must be 1 within this tolerance: θ, the parameters of cov_theta, are the first 8 entries of H.
H22_TOL = 1e-6
MAX_CONDITION = 1e12

PositiveFloat = Annotated[FiniteFloat, Field(gt=0)]
Row2 = Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]
Row3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Row8 = Annotated[list[FiniteFloat], Field(min_length=8, max_length=8)]
Matrix2 = Annotated[list[Row2], Field(min_length=2, max_length=2)]
Matrix3 = Annotated[list[Row3], Field(min_length=3, max_length=3)]
Matrix8 = Annotated[list[Row8], Field(min_length=8, max_length=8)]
CalibStatusName = Literal["ok", "few_points", "degenerate", "unhealthy"]


def checked_covariance(rows: list[list[float]], name: str) -> list[list[float]]:
    """A symmetric positive semi-definite matrix, symmetrised; raises ValueError otherwise.

    The checks run on the correlation scale D^-1/2 C D^-1/2, where entries of very different magnitudes (the
    perspective terms of θ are ~1e-3, the others ~1e2) weigh the same. A covariance has correlations within
    [-1, 1]; a matrix whose rescaled entries overflow is refused, and the comparisons are written so that NaN
    fails them.
    """
    c = np.asarray(rows, dtype=np.float64)
    d = np.diag(c)
    if np.any(d < 0):
        raise ValueError(f"{name} has a negative variance")
    s = 1.0 / np.sqrt(np.where(d > 0, d, 1.0))
    with np.errstate(over="ignore", invalid="ignore"):
        r = c * s[:, None] * s[None, :]
    if not np.isfinite(r).all():
        raise ValueError(f"{name} is not a covariance: its correlations overflow")
    if not np.max(np.abs(r - r.T)) <= SYMMETRY_TOL:
        raise ValueError(f"{name} is not symmetric")
    if not np.linalg.eigvalsh(0.5 * (r + r.T))[0] >= -PSD_TOL:
        raise ValueError(f"{name} is not positive semi-definite")
    return list((0.5 * c + 0.5 * c.T).tolist())  # halves first: no overflow near the float limit


class Foot(BaseModel):
    """Ground contact point of one person: the bottom centre of the box."""

    model_config = ConfigDict(extra="forbid")

    track_id: int = Field(ge=0, description="Anonymous track id (player_id of the SkillCorner format).")
    u: FiniteFloat = Field(description="Pixels, original frame.")
    v: FiniteFloat = Field(description="Pixels, original frame.")
    cov_uv: Matrix2 | None = Field(
        default=None, description="2 × 2 covariance of (u, v) in px²; omitted means exact pixels."
    )

    @field_validator("cov_uv")
    @classmethod
    def _cov_uv(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        return None if value is None else checked_covariance(value, "cov_uv")


class LocalizeRequest(BaseModel):
    """A calibrated camera and the feet to place on the pitch."""

    model_config = ConfigDict(extra="forbid")

    homography: Matrix3 = Field(
        description="H mapping pitch points (x, y, 1) to image points (u, v, 1), scaled so that H[2][2] = 1."
    )
    cov_theta: Matrix8 = Field(
        description="8 × 8 covariance of θ, the first 8 entries of H in row-major order."
    )
    dof: PositiveFloat | None = Field(
        default=None,
        description="Degrees of freedom of the noise scale in cov_theta (the dof of /v1/calibrate). "
        "Without a conformal quantile, the regions then use 2·F⁻¹(0.90; 2, dof) instead of χ²₂(0.90).",
    )
    feet: list[Foot] = Field(max_length=MAX_FEET)
    conformal_quantile: PositiveFloat | None = Field(
        default=None,
        description="Threshold q̂ of the 90 % regions {e : eᵀ Σp⁻¹ e ≤ q̂}, fitted by conformal calibration "
        "on held-out games. Default: the uncalibrated χ²₂(0.90) = 4.605, which covers far less than 90 % on "
        "real broadcast frames.",
    )

    @field_validator("homography")
    @classmethod
    def _homography(cls, value: list[list[float]]) -> list[list[float]]:
        h = np.asarray(value, dtype=np.float64)
        if abs(h[2, 2] - 1.0) > H22_TOL:
            raise ValueError("H[2][2] must be 1: cov_theta is about the first 8 entries of H scaled that way")
        if not np.linalg.cond(h) < MAX_CONDITION:  # also refuses NaN (singular H)
            raise ValueError("the homography is singular")
        return value

    @field_validator("cov_theta")
    @classmethod
    def _cov_theta(cls, value: list[list[float]]) -> list[list[float]]:
        return checked_covariance(value, "cov_theta")


class Position(BaseModel):
    """Pitch position of one foot point with its calibrated uncertainty; all null above the horizon."""

    track_id: int
    x: float | None = Field(description="Metres, internal pitch frame.")
    y: float | None = Field(description="Metres, internal pitch frame.")
    pos_cov_m2: list[list[float]] | None = Field(
        description="Calibrated 2 × 2 covariance (m²): its χ²₂(0.90) = 4.605 ellipse is the 90 % region."
    )
    p90_radius_m: float | None = Field(description="Radius of the circle that contains the 90 % ellipse.")


class LocalizeResponse(BaseModel):
    quantile: float = Field(description="Threshold q of the regions {e : eᵀ Σp⁻¹ e ≤ q}.")
    quantile_source: Literal["conformal", "laplace_f", "laplace_chi2"]
    positions: list[Position]


class Keypoint(BaseModel):
    id: int = Field(description="PnLCalib keypoint id (1-57).")
    u: float
    v: float
    conf: float


class LineDetection(BaseModel):
    id: int = Field(description="PnLCalib line id (1-23).")
    ends: list[list[float]] = Field(description="The two visible ends [[u, v], [u, v]], pixels.")
    conf: list[float]


class CalibrateResponse(BaseModel):
    """Calibration of one frame. ``homography``, ``cov_theta``, ``dof`` and ``rms_px`` are null unless
    ``status`` is ok; ``homography``, ``cov_theta`` and ``dof`` can be passed as they are to /v1/localize."""

    status: CalibStatusName
    homography: list[list[float]] | None
    cov_theta: list[list[float]] | None
    dof: float | None
    rms_px: float | None = Field(description="RMS reprojection error of the inlier keypoints, pixels.")
    n_inliers: int
    image_size: tuple[int, int] = Field(description="(width, height) in pixels.")
    image_corners_projection: dict[str, float | None] = Field(
        description="Ground footprint of the image, as in SkillCorner open data (nulls when unknown)."
    )
    keypoints: list[Keypoint]
    lines: list[LineDetection]
    model_version: str
    timings_ms: dict[str, float] = Field(
        description="decode, load (the networks, on the first request only), networks, calibration."
    )


class Health(BaseModel):
    status: Literal["ok"]
    version: str = Field(description="predfoot package version.")
    models: dict[str, str] = Field(description="Loaded models and their versions (networks load lazily).")
