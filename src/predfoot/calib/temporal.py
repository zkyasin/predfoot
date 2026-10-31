"""Temporal camera smoothing v1: a fixed-centre pan-tilt-zoom Kalman/RTS smoother (PTZ-RTS).

Per-frame calibrations jitter from frame to frame, which inflates distances and speeds. Inside a clip the main
broadcast camera is a pan-tilt-zoom pinhole with one projection centre (:mod:`predfoot.geometry.ptz`): its
homography has 4 degrees of freedom per frame, φ = (pan, tilt, roll, log f), and they move smoothly in time.
Post-match data may use past and future frames, hence a smoother rather than a filter.

Per clip, without labels:

1. :func:`estimate_centre`: median of the closed-form centres, PTZ misfit gate at that median, then a robust
   joint refit of C and every frame's φ on the inliers (in this order: refitting before gating put the centre
   0.8-2.5 m off in the prototype). An unreliable estimate gives way to the game centre, which is refused when
   it does not explain the clip (:func:`measure_at_centre`).
2. :func:`measure_clip`: each OK homography is projected onto the PTZ manifold at C with the shape of its
   Laplace covariance (:func:`project_to_ptz`). Frames the model cannot explain (another camera, a wrong
   RANSAC basin) are flagged ``misfit`` and pass through unchanged.
3. :func:`smooth_clip`: constant-velocity Kalman filter and RTS smoother on φ inside shot segments
   (:func:`segments`), with deletion of outlying measurements to a fixed point.
4. :func:`to_calib_results`: H and cov_theta rebuilt as :class:`~predfoot.contracts.CalibResult`, so position
   covariances, conformal calibration, health and export are unchanged.

The noise parameters (κ², q) are fitted by maximum likelihood on the innovations of other games
(:func:`fit_noise`); every other setting is fixed a priori in :class:`SmootherConfig`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares, minimize
from scipy.sparse import lil_matrix
from scipy.stats import chi2

from predfoot.contracts import CalibResult, CalibStatus, FloatArray, IntArray
from predfoot.geometry import homography as hg
from predfoot.geometry import ptz

BoolArray = NDArray[np.bool_]

#: Misfit gate, deletion test and NLL truncation: χ²₄ quantile at 0.999.
GATE4 = float(chi2.ppf(0.999, 4))
#: χ²₄ quantile at 0.99, used by consistency (NEES) checks.
NEES4_99 = float(chi2.ppf(0.99, 4))
#: Median of χ²₄: robust scale of the PTZ misfits, κ²_clip = median(d²) / MED_CHI2_4.
MED_CHI2_4 = float(chi2.ppf(0.5, 4))

N_PHI = 4
N_STATE = 2 * N_PHI
_TINY = 1e-300

CentreSource = Literal["clip", "median", "game"]


@dataclass(frozen=True)
class SmootherConfig:
    """Settings fixed a priori (never tuned on labels). Angles in radians, times in seconds."""

    image_size: tuple[int, int] = (1920, 1080)
    fps: float = 25.0  # native frame rate: t = native frame / fps
    misfit_level: float = 0.999  # PTZ misfit gate, χ²₄ quantile on d² / κ²_clip
    gate_level: float = 0.999  # deletion test and NLL truncation, χ²₄ quantile
    max_gap_s: float = 1.0  # a longer gap between used measurements closes the segment
    max_rate: tuple[float, float, float, float] = (0.70, 0.26, 0.035, 1.0)  # rad/s (pan, tilt, roll), 1/s
    # λ: process noise × λ on steps next to a frame without an accepted measurement (failed, duplicate,
    # misfit, not projected). A measurement deleted as an outlier keeps λ = 1, so its deletion and prediction
    # statistics are the same leave-one-out test under one model.
    gap_noise_factor: float = 20.0
    q_roll: float = 1e-8  # rad²/s³, fixed
    floor_px: float = 0.0  # fixed pixel floor added to the measurement covariance; off in v1
    v0: tuple[float, float, float, float] = (1.0, 0.3, 0.05, 0.5)  # prior sd of the rates at a segment start
    min_segment: int = 3  # fewer used measurements: the segment is "short" (projection only)
    split_run: int = 3  # this many consecutive flagged measurements are tried as their own segment
    # A single-step manoeuvre must lower the innovation NLL of the other measurements by more than this. At 0
    # it is a likelihood comparison of two fixed models on the same data, the log det of the freed step being
    # the Occam penalty (on synthetic clips, 0 and χ²₁(0.999)/2 reject 7σ spikes alike; 0 also follows kinks).
    step_gain: float = 0.0
    centre_min_frames: int = 10
    centre_max_sd_m: float = 1.0
    centre_max_refit: int = 300  # the joint centre refit uses at most this many evenly spaced inliers
    max_misfit_ratio: float = 4.0  # a game centre is refused when κ²_clip at it exceeds this × the reference
    beta: float = 0.0  # ablation: add β·R_k to the smoothed covariance of a used frame

    @property
    def principal_point(self) -> tuple[float, float]:
        return ((self.image_size[0] - 1) / 2, (self.image_size[1] - 1) / 2)

    @property
    def misfit_gate(self) -> float:
        return float(chi2.ppf(self.misfit_level, N_PHI))

    @property
    def gate(self) -> float:
        return float(chi2.ppf(self.gate_level, N_PHI))

    def digest(self) -> str:
        """First 12 hex characters of the SHA-256 of the settings."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class NoiseParams:
    """Fitted noise: measurement inflation κ² and process spectral densities q (per s³).

    The defaults are geometric means of the prototype's per-clip fits; :func:`fit_noise` replaces them.
    """

    kappa2: float = 2.0
    q_pan: float = 1e-3  # rad²/s³
    q_tilt: float = 1e-4  # rad²/s³
    q_logf: float = 6e-3  # 1/s³

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> NoiseParams:
        return cls(**{k: float(d[k]) for k in ("kappa2", "q_pan", "q_tilt", "q_logf")})

    def log_vector(self) -> FloatArray:
        return np.log(np.array([self.kappa2, self.q_pan, self.q_tilt, self.q_logf], dtype=np.float64))

    @classmethod
    def from_log_vector(cls, h: ArrayLike) -> NoiseParams:
        e = np.exp(np.asarray(h, dtype=np.float64))
        return cls(float(e[0]), float(e[1]), float(e[2]), float(e[3]))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class CentreEstimate:
    """Projection centre of a clip's main camera (metres, internal pitch frame)."""

    centre: FloatArray  # (3,)
    cov: FloatArray  # (3, 3) m²
    n_frames: int  # PTZ inliers behind the estimate (for a game centre: summed over its clips)
    kappa2_clip: float  # misfit scale at the median centre: median d² / median of χ²₄
    median_d2: float  # median misfit of the refitted inliers at the returned centre
    reliable: bool
    source: CentreSource  # "clip": joint refit; "median": median of closed-form centres; "game"

    @property
    def sd_m(self) -> float:
        """Largest standard deviation of the centre (m); inf when its covariance is unknown."""
        if not np.all(np.isfinite(self.cov)):
            return float("inf")
        return float(np.sqrt(max(np.linalg.eigvalsh(self.cov)[-1], 0.0)))

    def digest(self) -> str:
        """First 12 hex characters of the SHA-256 of what the smoother uses: centre and misfit scale."""
        d = {"centre": np.round(self.centre, 6).tolist(), "kappa2_clip": round(float(self.kappa2_clip), 9)}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ClipMeasurements:
    """PTZ measurements of one clip; plain arrays, so tests can build instances directly."""

    clip_id: str
    frames: IntArray  # (T,) native frame indices, increasing
    times: FloatArray  # (T,) seconds
    centre: CentreEstimate | None
    z: FloatArray  # (T, 4) φ̂, pan unwrapped over the accepted frames; NaN where not projected
    sigma_phi: FloatArray  # (T, 4, 4) unscaled Σφ; identity where not projected
    d2: FloatArray  # (T,) PTZ misfit; NaN where not projected
    raw_ok: BoolArray
    duplicate: BoolArray
    projected: BoolArray  # the PTZ projection converged
    misfit: BoolArray  # d² / κ²_clip above the misfit gate
    accepted: BoolArray  # raw_ok & ~duplicate & projected & ~misfit
    kappa2_clip: float = float("nan")  # median d² of OK, non-duplicate, projected frames / MED_CHI2_4

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class SmoothedTrack:
    """Smoothed state of one clip: φ and its rates (T, 8), NaN outside segments."""

    mean: FloatArray  # (T, 8)
    cov: FloatArray  # (T, 8, 8)
    R: FloatArray  # (T, 4, 4) measurement covariance
    segment: IntArray  # (T,) -1 outside segments
    used: BoolArray  # measurement used by the final pass
    rejected: BoolArray  # accepted but deleted by the robust passes
    short: BoolArray  # inside a segment with fewer than min_segment used measurements
    manoeuvre: BoolArray  # inside a manoeuvre segment: process noise × λ on all its steps
    # process noise × λ on the step into this frame beyond the gap rule (manoeuvre segments and single-step
    # manoeuvres); what fit_noise must reuse to evaluate the same model
    loose: BoolArray
    nis: FloatArray  # (T,) innovation χ² of the final pass; NaN for the first two used of a segment
    # (T,) deletion statistic (used) or prediction statistic (rejected; outside segments, from the
    # extrapolated edge state of the segment it would join)
    deletion: FloatArray
    nll: float
    n_iter: int  # smoothing passes run by the robust loop


@dataclass(frozen=True)
class FrameFlags:
    """What the temporal stage did to each frame; a raw-OK frame carries exactly one flag."""

    smoothed: BoolArray
    repaired: BoolArray  # rejected measurement replaced by the smoothed state
    filled: BoolArray  # failed raw calibration inside a segment
    projected: BoolArray  # PTZ projection only (short segment, or the projection ablation)
    passthrough: BoolArray  # raw homography, covariance scaled by κ²
    duplicate: BoolArray  # same detections as the previous frame: output copied
    rejected: BoolArray  # deleted measurement with no smoothed state to replace it: abstains
    segment: IntArray

    @classmethod
    def empty(cls, n: int) -> FrameFlags:
        def z() -> BoolArray:
            return np.zeros(n, dtype=bool)

        return cls(
            smoothed=z(),
            repaired=z(),
            filled=z(),
            projected=z(),
            passthrough=z(),
            duplicate=z(),
            rejected=z(),
            segment=np.full(n, -1, dtype=np.int64),
        )

    def as_dict(self) -> dict[str, NDArray[Any]]:
        return asdict(self)


# --- measurement step ---------------------------------------------------------------------------------


#: Detection fields that must all repeat for a frame to count as a duplicate (slow-motion replays).
DUPLICATE_KEYS = ("kp_ids", "kp_uv", "line_ids", "line_uv")


def duplicate_mask(detections: Sequence[Mapping[str, Any]]) -> BoolArray:
    """True where a frame's detections equal those of the previous frame (keypoints and line ends).

    Two frames without any detection are not duplicates: that is a detection dropout, not a repeated frame.
    """
    out = np.zeros(len(detections), dtype=bool)
    for k in range(1, len(detections)):
        a, b = detections[k], detections[k - 1]
        if len(np.asarray(a["kp_ids"])) + len(np.asarray(a["line_ids"])) == 0:
            continue
        out[k] = all(np.array_equal(np.asarray(a[key]), np.asarray(b[key])) for key in DUPLICATE_KEYS)
    return out


def _usable(r: CalibResult) -> bool:
    return r.ok and r.H is not None and r.cov_theta is not None


def whiten(cov_theta: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """(d, W) with d = √diag Σθ and W = Λ^(-1/2) Vᵀ from the correlation matrix V Λ Vᵀ (Λ floored at
    1e-12 λmax), so that |W (δθ / d)|² = δθᵀ Σθ⁺ δθ."""
    cov = np.asarray(cov_theta, dtype=np.float64)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    lam, vec = np.linalg.eigh(0.5 * (corr + corr.T))
    lam = np.maximum(lam, 1e-12 * lam.max())
    return d, np.asarray((vec / np.sqrt(lam)).T, dtype=np.float64)


def project_to_ptz(
    theta_hat: ArrayLike,
    cov_theta: ArrayLike,
    centre: ArrayLike,
    phi0: ArrayLike,
    pp: tuple[float, float] = ptz.PP_1080P,
) -> tuple[FloatArray, FloatArray, float] | None:
    """Mahalanobis projection of a homography onto the PTZ manifold at a fixed centre.

    φ̂ = argmin |W (θ(φ, C) − θ̂) / d|², Σφ = (JᵀJ)⁺ with J = W diag(1/d) ∂θ/∂φ, and d² the minimum. φ̂ depends
    only on the shape of Σθ; Σφ carries its scale. Returns None when the fit fails or is not finite.
    """
    th = np.asarray(theta_hat, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    with np.errstate(all="ignore"):
        d, w = whiten(cov_theta)
    if not (np.all(np.isfinite(d)) and np.all(d > 0) and np.all(np.isfinite(w))):
        return None
    wd = w / d[None, :]

    def fun(p: FloatArray) -> FloatArray:
        return np.asarray(wd @ (ptz.theta(p, c, pp) - th), dtype=np.float64)

    def jac(p: FloatArray) -> FloatArray:
        return np.asarray(wd @ ptz.jacobian_theta_phi(p, c, pp), dtype=np.float64)

    p0 = np.asarray(phi0, dtype=np.float64)
    if not np.all(np.isfinite(p0)):
        return None
    try:
        with np.errstate(all="ignore"):
            sol = least_squares(
                fun, p0, jac=jac, method="lm", x_scale=1e-2, xtol=1e-12, ftol=1e-12, gtol=1e-12
            )
            phi = np.asarray(sol.x, dtype=np.float64)
            j = jac(phi)
            sigma = np.linalg.pinv(j.T @ j)
            r = fun(phi)
    except (ValueError, np.linalg.LinAlgError):
        return None
    d2 = float(r @ r)
    if sol.status <= 0 or not (np.all(np.isfinite(phi)) and np.all(np.isfinite(sigma)) and np.isfinite(d2)):
        return None
    return phi, 0.5 * (sigma + sigma.T), d2


def _calibrations(
    results: Sequence[CalibResult], keep: BoolArray
) -> dict[int, tuple[FloatArray, FloatArray]]:
    """(H, Σθ) of the OK frames selected by ``keep``."""
    out: dict[int, tuple[FloatArray, FloatArray]] = {}
    for k, r in enumerate(results):
        if keep[k] and r.ok and r.H is not None and r.cov_theta is not None:
            out[k] = (r.H, r.cov_theta)
    return out


def _project_frames(
    calibs: Mapping[int, tuple[FloatArray, FloatArray]], centre: FloatArray, pp: tuple[float, float]
) -> dict[int, tuple[FloatArray, FloatArray, float]]:
    """PTZ projections at ``centre`` of the frames that converge (started from ptz.phi_given_centre)."""
    out: dict[int, tuple[FloatArray, FloatArray, float]] = {}
    for k, (h, cov) in calibs.items():
        try:
            phi0 = ptz.phi_given_centre(h, centre, pp)
        except np.linalg.LinAlgError:
            continue
        proj = project_to_ptz(hg.theta_from_h(h), cov, centre, phi0, pp)
        if proj is not None:
            out[k] = proj
    return out


def _refit_centre(
    calibs: Sequence[tuple[FloatArray, FloatArray]],
    c0: FloatArray,
    phis0: FloatArray,
    kappa2: float,
    pp: tuple[float, float],
) -> tuple[FloatArray, FloatArray, FloatArray] | None:
    """Robust joint fit of (C, φ_1 … φ_n) on whitened θ residuals; returns (C, Σ_C, φ) or None."""
    n = len(calibs)
    th = np.stack([hg.theta_from_h(h) for h, _ in calibs])
    wd = np.stack([w / d[None, :] for d, w in (whiten(cov) for _, cov in calibs)])

    def resid(x: FloatArray) -> FloatArray:
        t = ptz.thetas(x[3:].reshape(n, N_PHI), x[:3], pp)
        return np.asarray(np.einsum("nij,nj->ni", wd, t - th).ravel(), dtype=np.float64)

    sparsity = lil_matrix((8 * n, 3 + N_PHI * n), dtype=int)
    for t in range(n):
        sparsity[8 * t : 8 * t + 8, :3] = 1
        sparsity[8 * t : 8 * t + 8, 3 + N_PHI * t : 3 + N_PHI * (t + 1)] = 1
    x0 = np.concatenate([c0, np.asarray(phis0, dtype=np.float64).ravel()])
    try:
        with np.errstate(all="ignore"):
            sol = least_squares(
                resid,
                x0,
                jac_sparsity=sparsity,
                method="trf",
                loss="soft_l1",
                f_scale=2.0 * np.sqrt(kappa2),
                x_scale="jac",
                max_nfev=100,
            )
            jac = sol.jac.toarray() if hasattr(sol.jac, "toarray") else np.asarray(sol.jac)
            cov = kappa2 * np.linalg.pinv(jac.T @ jac)[:3, :3]
    except (ValueError, np.linalg.LinAlgError):
        return None
    x = np.asarray(sol.x, dtype=np.float64)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(cov))):
        return None
    return x[:3], 0.5 * (cov + cov.T), x[3:].reshape(n, N_PHI)


def estimate_centre(
    results: Sequence[CalibResult], duplicate: ArrayLike, cfg: SmootherConfig
) -> CentreEstimate | None:
    """Projection centre of a clip from its own per-frame calibrations (no labels).

    1. closed-form centres of the OK, non-duplicate frames (positive height); None with fewer than
       ``centre_min_frames``; 2. C0 = their coordinate-wise median; 3. PTZ projection of every frame at C0,
       κ²_clip = median(d²) / MED_CHI2_4 and inliers d² / κ²_clip ≤ the misfit gate; 4. soft-L1 joint refit of
       C and every inlier's φ (at most ``centre_max_refit`` evenly spaced inliers, so long clips stay linear
       in memory); 5. Σ_C from the Gauss-Newton curvature; the refit is kept only if it lowers the median
       misfit of the refitted inliers (else C0, source "median", with an unknown covariance: the curvature of
       a rejected refit says nothing about C0); 6. reliable when there are enough inliers and the centre's
       largest standard deviation is at most ``centre_max_sd_m``.
    """
    pp = cfg.principal_point
    usable = _calibrations(results, ~np.asarray(duplicate, dtype=bool))
    centres = []
    for h, _ in usable.values():
        out = ptz.decompose(h, pp)
        if out is not None and np.all(np.isfinite(out[1])) and out[1][2] > 0:
            centres.append(out[1])
    if len(centres) < cfg.centre_min_frames:
        return None
    c0 = np.median(np.asarray(centres), axis=0)
    first = _project_frames(usable, c0, pp)
    if not first:
        return None
    kappa2 = max(float(np.median([v[2] for v in first.values()])) / MED_CHI2_4, _TINY)
    inliers = [k for k, v in first.items() if v[2] / kappa2 <= cfg.misfit_gate]
    if len(inliers) > cfg.centre_max_refit:
        pick = np.unique(np.linspace(0, len(inliers) - 1, cfg.centre_max_refit).round().astype(int))
        sub = [inliers[i] for i in pick]
    else:
        sub = inliers
    med_c0 = float(np.median([first[k][2] for k in sub]))
    centre, med = c0, med_c0
    source: CentreSource = "median"
    cov = np.full((3, 3), np.inf)
    refit = _refit_centre([usable[k] for k in sub], c0, np.stack([first[k][0] for k in sub]), kappa2, pp)
    if refit is not None:
        c_hat, cov_hat, phis = refit
        if c_hat[2] > 0:
            second = [
                project_to_ptz(hg.theta_from_h(usable[k][0]), usable[k][1], c_hat, phis[j], pp)
                for j, k in enumerate(sub)
            ]
            med_hat = float(np.median([v[2] if v is not None else np.inf for v in second]))
            if med_hat < med_c0:
                centre, source, med, cov = c_hat, "clip", med_hat, cov_hat
    est = CentreEstimate(
        centre=np.asarray(centre, dtype=np.float64),
        cov=np.asarray(cov, dtype=np.float64),
        n_frames=len(inliers),
        kappa2_clip=kappa2,
        median_d2=med,
        reliable=False,
        source=source,
    )
    return replace(est, reliable=len(inliers) >= cfg.centre_min_frames and est.sd_m <= cfg.centre_max_sd_m)


def game_centre(estimates: Sequence[CentreEstimate | None]) -> CentreEstimate | None:
    """Coordinate-wise median of the reliable centres of one game's clips (covariance: sample cov / n)."""
    rel = [e for e in estimates if e is not None and e.reliable]
    if not rel:
        return None
    cs = np.stack([e.centre for e in rel])
    cov = np.cov(cs.T, ddof=1) / len(rel) if len(rel) >= 2 else rel[0].cov
    return CentreEstimate(
        centre=np.median(cs, axis=0),
        cov=np.asarray(cov, dtype=np.float64),
        n_frames=sum(e.n_frames for e in rel),
        kappa2_clip=float(np.median([e.kappa2_clip for e in rel])),
        median_d2=float(np.median([e.median_d2 for e in rel])),
        reliable=True,
        source="game",
    )


def choose_centre(own: CentreEstimate | None, game: CentreEstimate | None) -> CentreEstimate | None:
    """Fallback order: own reliable estimate → game centre → none.

    An unreliable own estimate is never used: nothing confirms it, and a centre a few metres off makes the
    outputs worse than the raw calibrations, so such a clip passes through instead.
    """
    if own is not None and own.reliable:
        return own
    return game


def measure_at_centre(
    clip_id: str,
    frames: ArrayLike,
    results: Sequence[CalibResult],
    duplicate: ArrayLike,
    own: CentreEstimate | None,
    game: CentreEstimate | None,
    cfg: SmootherConfig,
) -> tuple[ClipMeasurements, bool]:
    """Measurements at :func:`choose_centre`'s centre, and True when a game centre was refused.

    The per-frame misfit gate is relative to the clip's own scale κ²_clip, so a centre that is wrong for the
    whole clip (a clip filmed from elsewhere) inflates every d² alike and passes it. A game centre is kept
    only if κ²_clip at it is at most ``max_misfit_ratio`` times the scale at the clip's own median centre
    (without an own estimate: the game's median scale); otherwise the clip passes through. The clip's own
    reliable estimate needs no such check: it minimises that misfit.
    """
    centre = choose_centre(own, game)
    meas = measure_clip(clip_id, frames, results, duplicate, centre, cfg)
    if centre is None or centre is own:
        return meas, False
    ref = own.kappa2_clip if own is not None else centre.kappa2_clip
    if np.isfinite(meas.kappa2_clip) and meas.kappa2_clip <= cfg.max_misfit_ratio * ref:
        return meas, False
    return measure_clip(clip_id, frames, results, duplicate, None, cfg), True


def measure_clip(
    clip_id: str,
    frames: ArrayLike,
    results: Sequence[CalibResult],
    duplicate: ArrayLike,
    centre: CentreEstimate | None,
    cfg: SmootherConfig,
) -> ClipMeasurements:
    """Project every OK frame at the centre, gate the misfits and unwrap the pan over the accepted frames.

    Without a centre, no frame is accepted (the whole clip passes through).
    """
    fr = np.asarray(frames, dtype=np.int64)
    n = len(fr)
    if len(results) != n:
        raise ValueError("one calibration per frame is needed")
    if n > 1 and np.any(np.diff(fr) <= 0):
        raise ValueError("frames must be strictly increasing")
    raw_ok = np.array([_usable(r) for r in results], dtype=bool).reshape(n)
    dup = np.asarray(duplicate, dtype=bool).reshape(n)
    z = np.full((n, N_PHI), np.nan)
    sigma = np.tile(np.eye(N_PHI), (n, 1, 1))
    d2 = np.full(n, np.nan)
    projected = np.zeros(n, dtype=bool)
    if centre is not None:
        for k, (phi, sig, dd) in _project_frames(
            _calibrations(results, raw_ok), centre.centre, cfg.principal_point
        ).items():
            z[k], sigma[k], d2[k], projected[k] = phi, sig, dd, True
    base = raw_ok & ~dup & projected
    kappa2 = max(float(np.median(d2[base])) / MED_CHI2_4, _TINY) if base.any() else float("nan")
    misfit = np.zeros(n, dtype=bool)
    if np.isfinite(kappa2):
        misfit[projected] = d2[projected] / kappa2 > cfg.misfit_gate
    accepted = base & ~misfit
    acc = np.flatnonzero(accepted)
    if len(acc):
        z[acc, 0] = np.unwrap(z[acc, 0])
    return ClipMeasurements(
        clip_id=clip_id,
        frames=fr,
        times=fr / cfg.fps,
        centre=centre,
        z=z,
        sigma_phi=sigma,
        d2=d2,
        raw_ok=raw_ok,
        duplicate=dup,
        projected=projected,
        misfit=misfit,
        accepted=accepted,
        kappa2_clip=kappa2,
    )


def _floor_cov(z: FloatArray, cfg: SmootherConfig) -> FloatArray:
    """floor_px² diag(1/f², 1/f², 1/r², 1/r²), r = half the image diagonal; zero where φ is unknown."""
    out = np.zeros((len(z), N_PHI, N_PHI))
    if cfg.floor_px <= 0:
        return out
    ok = np.isfinite(z[:, 3])
    r2 = (np.hypot(*cfg.image_size) / 2) ** 2
    inv_f2 = np.exp(-2.0 * z[ok, 3])
    for i, v in enumerate((inv_f2, inv_f2, np.full(ok.sum(), 1 / r2), np.full(ok.sum(), 1 / r2))):
        out[ok, i, i] = cfg.floor_px**2 * v
    return out


def measurement_cov(meas: ClipMeasurements, params: NoiseParams, cfg: SmootherConfig) -> FloatArray:
    """R_k = κ² Σφ_k (+ the pixel floor when ``floor_px`` > 0), shape (T, 4, 4)."""
    return np.asarray(params.kappa2 * meas.sigma_phi + _floor_cov(meas.z, cfg), dtype=np.float64)


# --- state-space model --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Batch:
    """Segments stacked as padded arrays: row b is one segment, step k one cached frame inside it."""

    clip: IntArray  # (B,) position of the segment's clip in the input sequence
    segment: IntArray  # (B,) segment id inside its clip
    index: IntArray  # (B, L) frame index in the clip, -1 on padding
    length: IntArray  # (B,)
    dt: FloatArray  # (B, L) seconds since the previous step; 0 at the first step and on padding
    lam: FloatArray  # (B, L) process-noise factor of the step into k (from the accepted mask); 0 on padding
    has: BoolArray  # (B, L) a used measurement at step k
    z: FloatArray  # (B, L, 4) measurement, 0 where none
    sigma: FloatArray  # (B, L, 4, 4) unscaled Σφ, identity where none
    floor: FloatArray  # (B, L, 4, 4) fixed floor covariance, 0 where none
    term: BoolArray  # (B, L) measurement entering the NLL (third and later of its segment)

    @property
    def size(self) -> int:
        return len(self.length)


@dataclass(frozen=True)
class Forward:
    """Kalman filter output over a batch; the moments are stored only when asked for (for the RTS pass)."""

    nll: float
    n_terms: int
    nis: FloatArray  # (B, L) innovation χ² of the NLL terms (third and later used), NaN elsewhere
    m_pred: FloatArray | None = None  # (B, L, 8)
    P_pred: FloatArray | None = None  # (B, L, 8, 8)
    m_filt: FloatArray | None = None
    P_filt: FloatArray | None = None

    @property
    def mean_nis(self) -> float:
        v = self.nis[np.isfinite(self.nis)]
        return float(v.mean()) if v.size else float("nan")


def stack_clips(
    clips: Sequence[ClipMeasurements],
    masks: Sequence[BoolArray],
    segs: Sequence[IntArray],
    cfg: SmootherConfig,
    loose: Sequence[BoolArray] | None = None,
) -> Batch:
    """Stack every segment of every clip; ``masks`` mark the used measurements, ``segs`` the segment ids.

    λ = ``gap_noise_factor`` on the steps next to a frame that has no accepted measurement and on the steps
    marked in ``loose`` (index i: the step from frame i − 1 to frame i); a measurement that is accepted but
    left out of ``masks`` (deleted as an outlier) does not change the process noise.
    """
    rows: list[tuple[int, int, IntArray]] = []
    for ci, seg in enumerate(segs):
        for s in np.unique(seg[seg >= 0]):
            rows.append((ci, int(s), np.flatnonzero(seg == s)))
    b_n = len(rows)
    length = max((len(r[2]) for r in rows), default=1)
    index = np.full((b_n, length), -1, dtype=np.int64)
    lengths = np.zeros(b_n, dtype=np.int64)
    dt = np.zeros((b_n, length))
    lam = np.zeros((b_n, length))
    has = np.zeros((b_n, length), dtype=bool)
    z = np.zeros((b_n, length, N_PHI))
    sigma = np.tile(np.eye(N_PHI), (b_n, length, 1, 1))
    floor = np.zeros((b_n, length, N_PHI, N_PHI))
    term = np.zeros((b_n, length), dtype=bool)
    floors = [_floor_cov(c.z, cfg) for c in clips]
    for b, (ci, _, idx) in enumerate(rows):
        meas, n = clips[ci], len(idx)
        h = np.asarray(masks[ci], dtype=bool)[idx]
        if not (h[0] and h[-1]):
            raise ValueError("a segment must start and end with a used measurement")
        index[b, :n], lengths[b] = idx, n
        dt[b, 1:n] = np.diff(meas.times[idx])
        acc = np.asarray(meas.accepted, dtype=bool)[idx]
        wide = ~(acc[1:] & acc[:-1])
        if loose is not None:
            wide |= np.asarray(loose[ci], dtype=bool)[idx][1:]
        lam[b, 1:n] = np.where(wide, cfg.gap_noise_factor, 1.0)
        has[b, :n] = h
        z[b, :n][h] = meas.z[idx][h]
        sigma[b, :n][h] = meas.sigma_phi[idx][h]
        floor[b, :n][h] = floors[ci][idx][h]
        term[b, :n] = h & (np.cumsum(h) > 2)
    return Batch(
        clip=np.array([r[0] for r in rows], dtype=np.int64),
        segment=np.array([r[1] for r in rows], dtype=np.int64),
        index=index,
        length=lengths,
        dt=dt,
        lam=lam,
        has=has,
        z=z,
        sigma=sigma,
        floor=floor,
        term=term,
    )


def _transition(dt: FloatArray) -> FloatArray:
    """Constant-velocity F = [[I, dt I], [0, I]], shape (..., 8, 8)."""
    f = np.broadcast_to(np.eye(N_STATE), (*dt.shape, N_STATE, N_STATE)).copy()
    i = np.arange(N_PHI)
    f[..., i, i + N_PHI] = dt[..., None]
    return f


def _process_noise(dt: FloatArray, lam: FloatArray, qc: FloatArray) -> FloatArray:
    """Q = [[dt³/3 Qc, dt²/2 Qc], [dt²/2 Qc, dt Qc]] with Qc = λ diag(q), shape (..., 8, 8)."""
    q = lam[..., None] * qc
    out = np.zeros((*dt.shape, N_STATE, N_STATE))
    i = np.arange(N_PHI)
    out[..., i, i] = (dt**3 / 3)[..., None] * q
    out[..., i, i + N_PHI] = (dt**2 / 2)[..., None] * q
    out[..., i + N_PHI, i] = (dt**2 / 2)[..., None] * q
    out[..., i + N_PHI, i + N_PHI] = dt[..., None] * q
    return out


def _cholesky(s: FloatArray) -> FloatArray:
    try:
        return np.asarray(np.linalg.cholesky(s), dtype=np.float64)
    except np.linalg.LinAlgError:
        tr = np.trace(s, axis1=-2, axis2=-1)
        return np.asarray(np.linalg.cholesky(s + (1e-12 * tr)[..., None, None] * np.eye(s.shape[-1])))


def _sym(a: FloatArray) -> FloatArray:
    return np.asarray(0.5 * (a + np.swapaxes(a, -1, -2)), dtype=np.float64)


def forward(
    batch: Batch,
    params: NoiseParams,
    cfg: SmootherConfig,
    store: bool = False,
    nll_gate: float | None = None,
) -> Forward:
    """Vectorised constant-velocity Kalman filter over every segment of a batch.

    A segment starts at its first used measurement s with m0 = (z_s, 0) and P0 = blkdiag(R_s, diag(v0²)).
    Updates use the Joseph form; there is no online gating (robustness comes from :func:`smooth_clip`). The
    NLL sums ½ [log det S_k + NIS_k] over used measurements after the first two of each segment (constants
    dropped). With ``nll_gate``, NIS_k is truncated at that value: that robust variant is unbounded below
    (as S shrinks every term saturates while log det S → −∞), so :func:`fit_noise` does not use it.
    """
    b_n, length = batch.has.shape
    nis = np.full((b_n, length), np.nan)
    if b_n == 0:
        empty = np.zeros((0, length, N_STATE))
        e2 = np.zeros((0, length, N_STATE, N_STATE))
        return Forward(0.0, 0, nis, empty, e2, empty, e2) if store else Forward(0.0, 0, nis)
    qc = np.array([params.q_pan, params.q_tilt, cfg.q_roll, params.q_logf])
    r_all = params.kappa2 * batch.sigma + batch.floor
    f_all = _transition(batch.dt)
    q_all = _process_noise(batch.dt, batch.lam, qc)
    m = np.zeros((b_n, N_STATE))
    m[:, :N_PHI] = batch.z[:, 0]
    p = np.zeros((b_n, N_STATE, N_STATE))
    p[:, :N_PHI, :N_PHI] = r_all[:, 0]
    p[:, N_PHI:, N_PHI:] = np.diag(np.square(np.asarray(cfg.v0, dtype=np.float64)))
    if store:
        m_pred, p_pred = np.empty((b_n, length, N_STATE)), np.empty((b_n, length, N_STATE, N_STATE))
        m_filt, p_filt = np.empty_like(m_pred), np.empty_like(p_pred)
        m_pred[:, 0], m_filt[:, 0], p_pred[:, 0], p_filt[:, 0] = m, m, p, p
    nll = 0.0
    eye = np.eye(N_STATE)
    for k in range(1, length):
        f = f_all[:, k]
        mp = np.einsum("bij,bj->bi", f, m)
        pp = f @ p @ np.swapaxes(f, -1, -2) + q_all[:, k]
        h = batch.has[:, k]
        if h.any():
            r = r_all[:, k]
            s = _sym(pp[:, :N_PHI, :N_PHI] + r)
            chol = _cholesky(s)
            nu = batch.z[:, k] - mp[:, :N_PHI]
            sol = np.linalg.solve(s, np.concatenate([nu[..., None], pp[:, :N_PHI, :]], axis=2))
            nis_k = np.einsum("bi,bi->b", nu, sol[..., 0])
            gain = np.swapaxes(sol[..., 1:], -1, -2)  # K = P⁻ Hᵀ S⁻¹, (B, 8, 4)
            ikh = np.broadcast_to(eye, pp.shape).copy()
            ikh[:, :, :N_PHI] -= gain
            p_up = _sym(ikh @ pp @ np.swapaxes(ikh, -1, -2) + gain @ r @ np.swapaxes(gain, -1, -2))
            m = np.where(h[:, None], mp + np.einsum("bij,bj->bi", gain, nu), mp)
            p = np.where(h[:, None, None], p_up, pp)
            nis[h, k] = nis_k[h]
            t = batch.term[:, k]
            if t.any():
                logdet = 2.0 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=1)
                nis_t = nis_k[t] if nll_gate is None else np.minimum(nis_k[t], nll_gate)
                nll += 0.5 * float(np.sum(logdet[t] + nis_t))
        else:
            m, p = mp, pp
        if store:
            m_pred[:, k], p_pred[:, k], m_filt[:, k], p_filt[:, k] = mp, pp, m, p
    n_terms = int(batch.term.sum())
    term_nis = np.where(batch.term, nis, np.nan)
    if store:
        return Forward(nll, n_terms, term_nis, m_pred, p_pred, m_filt, p_filt)
    return Forward(nll, n_terms, term_nis)


def rts(fwd: Forward, batch: Batch) -> tuple[FloatArray, FloatArray]:
    """Rauch-Tung-Striebel smoother inside each segment, backwards from its last used measurement.

    G_k = P_k F_{k+1}ᵀ (P⁻_{k+1})⁻¹ (batched solve with the SPD P⁻), m_k|T = m_k + G_k (m_{k+1}|T − m⁻_{k+1})
    and P_k|T = P_k + G_k (P_{k+1}|T − P⁻_{k+1}) G_kᵀ.
    """
    if fwd.m_filt is None or fwd.P_filt is None or fwd.m_pred is None or fwd.P_pred is None:
        raise ValueError("the forward pass must store its moments (store=True)")
    ms, ps = fwd.m_filt.copy(), fwd.P_filt.copy()
    f_all = _transition(batch.dt)
    for k in range(batch.has.shape[1] - 2, -1, -1):
        a = np.flatnonzero(batch.length > k + 1)
        if not len(a):
            continue
        f, pf, pp = f_all[a, k + 1], fwd.P_filt[a, k], fwd.P_pred[a, k + 1]
        gt = np.linalg.solve(pp, f @ pf)
        g = np.swapaxes(gt, -1, -2)
        ms[a, k] = fwd.m_filt[a, k] + np.einsum("bij,bj->bi", g, ms[a, k + 1] - fwd.m_pred[a, k + 1])
        ps[a, k] = _sym(pf + g @ (ps[a, k + 1] - pp) @ gt)
    return ms, ps


# --- segments and robust smoothing ---------------------------------------------------------------------


def segments(
    meas: ClipMeasurements,
    R: FloatArray,
    mask: BoolArray,
    cfg: SmootherConfig,
    cuts: Collection[int] = (),
    forced: Collection[int] = (),
) -> IntArray:
    """Segment id per frame (-1 outside), from the used measurements ``mask``.

    A new segment starts at a used measurement b (previous used a) when t_b − t_a > max_gap_s (with a 1e-9 s
    tolerance, so a nominal 1 s gap is bridged wherever it falls), when an external cut (native frame c,
    frame_a < c ≤ frame_b) or a forced boundary (frame index c, a < c ≤ b) falls in between, or on a
    persistent jump: b breaks the rate limit relative to the last non-spike measurement in some coordinate
    (|Δz_i| > max_rate_i Δt + 3 √(R_a,ii + R_b,ii)), the next used measurement c breaks it too and c is
    consistent with b. A lone spike never cuts (it goes to the deletion test). A segment spans the frames
    from its first to its last used measurement, so nothing is extrapolated.
    """
    n = len(meas)
    seg = np.full(n, -1, dtype=np.int64)
    used = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(used):
        return seg
    rate = np.asarray(cfg.max_rate, dtype=np.float64)
    var = np.clip(np.diagonal(R, axis1=1, axis2=2), 0.0, None)
    cut_frames = np.sort(np.asarray(list(cuts), dtype=np.int64))
    forced_idx = np.sort(np.asarray(list(forced), dtype=np.int64))

    def breaks(a: int, b: int) -> bool:
        tol = rate * (meas.times[b] - meas.times[a]) + 3.0 * np.sqrt(var[a] + var[b])
        return bool(np.any(np.abs(meas.z[b] - meas.z[a]) > tol))

    starts = [0]
    ref = int(used[0])
    for j in range(1, len(used)):
        a, b = int(used[j - 1]), int(used[j])
        new = (
            meas.times[b] - meas.times[a] > cfg.max_gap_s + 1e-9
            or bool(np.any((forced_idx > a) & (forced_idx <= b)))
            or bool(np.any((cut_frames > meas.frames[a]) & (cut_frames <= meas.frames[b])))
        )
        if not new and breaks(ref, b):
            if j + 1 < len(used) and breaks(ref, int(used[j + 1])) and not breaks(b, int(used[j + 1])):
                new = True
            else:
                continue  # lone spike: keep it in the segment, but never as the reference
        if new:
            starts.append(j)
        ref = b
    for s, (i0, i1) in enumerate(pairwise([*starts, len(used)])):
        seg[used[i0] : used[i1 - 1] + 1] = s
    return seg


@dataclass(frozen=True)
class _Pass:
    mask: BoolArray
    seg: IntArray
    mean: FloatArray
    cov: FloatArray
    nis: FloatArray
    nll: float
    deletion: FloatArray


def _smooth_pass(
    meas: ClipMeasurements,
    R: FloatArray,
    mask: BoolArray,
    seg: IntArray,
    params: NoiseParams,
    cfg: SmootherConfig,
    loose: BoolArray,
    forced: Collection[int] = (),
    cuts: Collection[int] = (),
) -> _Pass:
    n = len(meas)
    batch = stack_clips([meas], [mask], [seg], cfg, [loose])
    fwd = forward(batch, params, cfg, store=True)
    ms, ps = rts(fwd, batch)
    mean, cov, nis = np.full((n, N_STATE), np.nan), np.full((n, N_STATE, N_STATE), np.nan), np.full(n, np.nan)
    for b in range(batch.size):
        idx = batch.index[b, : batch.length[b]]
        mean[idx], cov[idx] = ms[b, : len(idx)], ps[b, : len(idx)]
        nis[idx] = np.where(batch.has[b, : len(idx)], fwd.nis[b, : len(idx)], np.nan)
    deletion = _deletion(meas, R, mean, cov, mask, seg)
    edge = _edge_statistics(meas, R, mean, cov, mask, seg, loose, params, cfg, forced, cuts)
    return _Pass(mask, seg, mean, cov, nis, fwd.nll, np.where(np.isnan(deletion), edge, deletion))


def _edge_statistics(
    meas: ClipMeasurements,
    R: FloatArray,
    mean: FloatArray,
    cov: FloatArray,
    mask: BoolArray,
    seg: IntArray,
    loose: BoolArray,
    params: NoiseParams,
    cfg: SmootherConfig,
    forced: Collection[int],
    cuts: Collection[int],
) -> FloatArray:
    """Prediction statistic rᵀ (R + P)⁻¹ r of deleted measurements outside every segment; NaN elsewhere.

    The state comes from the edge of each segment the measurement would join if it were used (time gap at
    most ``max_gap_s``, no forced boundary or cut in between), extrapolated with the motion model: forwards
    from the last frame of the segment before it, backwards (x_{j−1} = F⁻¹ (x_j − w_j)) from the first frame
    of the segment after it, the two fused when it would bridge them; NaN when no segment can take it.
    Without it, a measurement deleted at a segment edge, or stranded when deletions open a gap longer than
    ``max_gap_s``, could never come back.
    """
    out = np.full(len(meas), np.nan)
    acc = np.asarray(meas.accepted, dtype=bool)
    todo = np.flatnonzero(acc & ~mask & (seg < 0))
    used = np.flatnonzero(mask)
    if not len(todo) or not len(used):
        return out
    qc = np.array([params.q_pan, params.q_tilt, cfg.q_roll, params.q_logf])
    forced_idx = np.asarray(sorted(forced), dtype=np.int64)
    cut_frames = np.asarray(sorted(cuts), dtype=np.int64)
    t = meas.times

    def step(i: int, backwards: bool) -> tuple[FloatArray, FloatArray]:
        """F (or F⁻¹) and Q of the step from frame i − 1 to frame i."""
        dt = np.asarray(t[i] - t[i - 1])
        wide = not (acc[i] and acc[i - 1]) or bool(loose[i])
        lam = np.asarray(cfg.gap_noise_factor if wide else 1.0)
        return _transition(-dt if backwards else dt), _process_noise(dt, lam, qc)

    def blocked(a: int, b: int) -> bool:
        return bool(
            t[b] - t[a] > cfg.max_gap_s + 1e-9
            or np.any((forced_idx > a) & (forced_idx <= b))
            or np.any((cut_frames > meas.frames[a]) & (cut_frames <= meas.frames[b]))
        )

    for k in todo:
        preds: list[tuple[FloatArray, FloatArray]] = []
        j = int(np.searchsorted(used, k))
        if j > 0 and not blocked(int(used[j - 1]), int(k)):
            e = int(used[j - 1])
            m, p = mean[e], cov[e]
            for i in range(e + 1, int(k) + 1):
                f, q = step(i, backwards=False)
                m, p = f @ m, f @ p @ f.T + q
            preds.append((m, p))
        if j < len(used) and not blocked(int(k), int(used[j])):
            s = int(used[j])
            m, p = mean[s], cov[s]
            for i in range(s, int(k), -1):
                finv, q = step(i, backwards=True)
                m, p = finv @ m, finv @ (p + q) @ finv.T
            preds.append((m, p))
        if not preds:
            continue
        m, p = min(preds, key=lambda mp: float(np.trace(mp[1][:N_PHI, :N_PHI])))
        try:
            if len(preds) == 2:
                w1, w2 = np.linalg.inv(preds[0][1]), np.linalg.inv(preds[1][1])
                p = np.linalg.inv(w1 + w2)
                m = p @ (w1 @ preds[0][0] + w2 @ preds[1][0])
            r = meas.z[k] - m[:N_PHI]
            out[k] = float(r @ np.linalg.solve(_sym(R[k] + p[:N_PHI, :N_PHI]), r))
        except np.linalg.LinAlgError:
            continue  # no statistic: the measurement stays deleted
    return out


def _deletion(
    meas: ClipMeasurements, R: FloatArray, mean: FloatArray, cov: FloatArray, mask: BoolArray, seg: IntArray
) -> FloatArray:
    """Deletion statistic rᵀ (R − P|T)⁻¹ r of used measurements (exact leave-one-out in a linear-Gaussian
    model) and prediction statistic rᵀ (R + P|T)⁻¹ r of the other accepted ones, r = z − m|T[:4]."""
    out = np.full(len(meas), np.nan)
    idx = np.flatnonzero(meas.accepted & (seg >= 0) & np.isfinite(mean[:, 0]))
    if not len(idx):
        return out
    r = meas.z[idx] - mean[idx, :N_PHI]
    p = cov[idx, :N_PHI, :N_PHI]
    m = _sym(np.where(mask[idx][:, None, None], R[idx] - p, R[idx] + p))
    w, v = np.linalg.eigh(m)
    scale = np.linalg.eigvalsh(R[idx])[:, -1:]
    w = np.maximum(w, np.maximum(1e-9 * w[:, -1:], 1e-12 * scale))
    proj = np.einsum("bji,bj->bi", v, r)
    out[idx] = np.sum(proj**2 / w, axis=1)
    return out


def _runs(order: IntArray, member: BoolArray, seg: IntArray | None = None) -> list[IntArray]:
    """Maximal runs of consecutive entries of ``order`` where ``member`` holds (within one segment)."""
    runs: list[list[int]] = []
    cur: list[int] = []
    for pos, k in enumerate(order):
        if member[pos] and (not cur or seg is None or seg[k] == seg[cur[-1]]):
            cur.append(int(k))
            continue
        if cur:
            runs.append(cur)
        cur = [int(k)] if member[pos] else []
    if cur:
        runs.append(cur)
    return [np.array(r, dtype=np.int64) for r in runs]


def _short(seg: IntArray, mask: BoolArray, cfg: SmootherConfig) -> BoolArray:
    """Frames inside a segment with fewer than ``min_segment`` used measurements."""
    counts = np.bincount(seg[mask], minlength=int(seg.max()) + 1) if mask.any() else np.zeros(1, dtype=int)
    return np.asarray((seg >= 0) & (counts[np.maximum(seg, 0)] < cfg.min_segment), dtype=bool)


def smooth_clip(
    meas: ClipMeasurements, params: NoiseParams, cfg: SmootherConfig, cuts: Collection[int] = ()
) -> SmoothedTrack:
    """Segment, filter and smooth one clip, deleting outlying measurements.

    Deletion works in two phases. Shrink: every used measurement whose deletion statistic is above the gate
    is flagged, and again on the remaining ones, until none is left. Grow: flagged measurements come back one
    at a time, the most consistent first (smallest prediction statistic, at most the gate), and a release is
    kept only if no used measurement then goes above the gate. Releasing from a clean core avoids masking (two
    outliers that support each other pass a one-at-a-time deletion test) and gives back the neighbours an
    outlier pulled above the gate. A deleted measurement keeps λ = 1 on its steps, so its prediction
    statistic is the leave-one-out test its deletion statistic was, under the same model. Both phases are
    monotone, so the loop ends, and at the end no used measurement has a deletion statistic above the gate.
    A flagged measurement outside every segment (at a clip or segment edge) is tested against the
    extrapolated edge state of the segment it would join (:func:`_edge_statistics`); with no such segment it
    stays deleted.

    A measurement can also fail because the camera moved faster than the fitted process noise allows, and
    deleting it then makes the smoother cut the corner. Two manoeuvre hypotheses are tried, each kept only if
    it brings the deleted measurements back with no used measurement above the gate:

    - a run of ``split_run`` or more consecutive flagged measurements becomes its own segment with process
      noise × λ on all its steps (deletion passes again), kept if every member is then used in a segment that
      is not short: a real camera move fits the looser motion model, a burst of mutually inconsistent
      calibrations does not and stays deleted;
    - a single flagged measurement comes back with process noise × λ on one step, into it or out of it, if
      that looser step first lowers the innovation NLL of the other measurements (the flagged one still
      deleted) by more than ``step_gain``, and the step on its other side stays tight to a used neighbour.
      A kink in the camera path strains its neighbours until the step is freed; once a spike is deleted its
      neighbours fit, so the looser step brings nothing and the spike stays deleted.

    Segments with fewer than ``min_segment`` used measurements are marked short.
    """
    n = len(meas)
    gate = cfg.gate
    R = measurement_cov(meas, params, cfg)
    acc = np.asarray(meas.accepted, dtype=bool)
    order = np.flatnonzero(acc)
    passes = 0

    def run(fl: BoolArray, forced: frozenset[int], man: BoolArray) -> _Pass:
        nonlocal passes
        passes += 1
        mask = acc & ~fl
        seg = segments(meas, R, mask, cfg, cuts, forced)
        return _smooth_pass(meas, R, mask, seg, params, cfg, man, forced, cuts)

    def above(state: _Pass) -> BoolArray:
        d = state.deletion
        return np.asarray(state.mask & np.isfinite(d) & (d > gate), dtype=bool)

    def settle(fl: BoolArray, forced: frozenset[int], man: BoolArray) -> tuple[BoolArray, _Pass]:
        fl = fl.copy()
        state = run(fl, forced, man)
        while (new := above(state)).any():  # shrink: flags only grow
            fl |= new
            state = run(fl, forced, man)
        frozen = np.zeros(n, dtype=bool)
        while True:  # grow: flags only shrink, each candidate is tried once
            d = state.deletion
            cand = np.flatnonzero(fl & ~frozen & np.isfinite(d) & (d <= gate))
            if not len(cand):
                return fl, state
            k = cand[np.argmin(d[cand])]
            trial = fl.copy()
            trial[k] = False
            s2 = run(trial, forced, man)
            if above(s2).any():
                frozen[k] = True
            else:
                fl, state = trial, s2

    def anchored(state: _Pass, k: int, i: int, lse: BoolArray) -> bool:
        """Frame k keeps a tight step to a used neighbour on the side opposite the loosened step i."""
        j = k + 1 if i == k else k - 1
        t = max(j, k)
        return bool(
            0 <= j < n
            and state.mask[j]
            and state.seg[j] == state.seg[k] >= 0
            and acc[t]
            and acc[t - 1]
            and not lse[t]
        )

    forced: frozenset[int] = frozenset()
    manoeuvre = np.zeros(n, dtype=bool)
    loose = np.zeros(n, dtype=bool)
    flags, state = settle(np.zeros(n, dtype=bool), forced, loose)
    tried: set[tuple[int, int, int]] = set()

    def next_trial() -> tuple[int, int, int] | None:
        for r in _runs(order, flags[order]):
            if len(r) >= cfg.split_run and (key := (0, int(r[0]), int(r[-1]))) not in tried:
                return key
        for k in np.flatnonzero(flags):
            for i in (int(k), int(k) + 1):
                key = (1, int(k), i)
                if 0 < i < n and acc[i] and acc[i - 1] and not loose[i] and key not in tried:
                    return key
        return None

    for _ in range(4 * n):
        if (key := next_trial()) is None:
            break
        tried.add(key)
        kind, a, b = key
        if kind == 0:  # frames a..b as a manoeuvre segment
            r = order[(order >= a) & (order <= b)]
            t_forced = forced | {a, b + 1}
            t_loose = loose.copy()
            t_loose[a + 1 : b + 1] = True
            trial = flags.copy()
            trial[r] = False
            f2, s2 = settle(trial, t_forced, t_loose)
            if s2.mask[r].all() and not _short(s2.seg, s2.mask, cfg)[r].any():
                flags, state, forced, loose = f2, s2, t_forced, t_loose
                manoeuvre[a : b + 1] = True
        else:  # frame a back with a single-step manoeuvre on step b (into a, or out of it)
            t_loose = loose.copy()
            t_loose[b] = True
            # Same data and segments, looser step: the forward pass alone gives the NLL.
            looser = forward(stack_clips([meas], [state.mask], [state.seg], cfg, [t_loose]), params, cfg)
            if state.nll - looser.nll <= cfg.step_gain:
                continue  # with frame a still deleted, the other measurements do not ask for that step
            trial = flags.copy()
            trial[a] = False
            s2 = run(trial, forced, t_loose)
            if not above(s2).any() and anchored(s2, a, b, t_loose):
                loose = t_loose
                flags, state = settle(trial, forced, loose)
    seg, mask = state.seg, state.mask
    return SmoothedTrack(
        mean=state.mean,
        cov=state.cov,
        R=R,
        segment=seg,
        used=mask.copy(),
        rejected=acc & flags,
        short=_short(seg, mask, cfg),
        manoeuvre=manoeuvre & (seg >= 0),
        loose=loose,
        nis=state.nis,
        deletion=state.deletion,
        nll=state.nll,
        n_iter=passes,
    )


# --- noise fit ----------------------------------------------------------------------------------------


def fit_noise(
    clips: Sequence[ClipMeasurements], cfg: SmootherConfig, init: NoiseParams | None = None, rounds: int = 2
) -> tuple[NoiseParams, dict[str, Any]]:
    """Label-free maximum-likelihood fit of (κ², q_pan, q_tilt, q_logf) on the innovations of ``clips``.

    Each round cleans every clip with the smoother itself (:func:`smooth_clip` with the current parameters:
    deletions, manoeuvres), then minimises the Gaussian innovation NLL of the remaining measurements, segments
    and freed steps held fixed, by Nelder-Mead in log units from the simplex h + I. The next round re-cleans
    with the fitted parameters (skipped when the cleaned measurements do not change).

    Why not the gate-truncated NLL on every accepted measurement: it bounds the pull of an outlier, but it is
    unbounded below, and on two dev clips Nelder-Mead ran off to κ² ≈ 1e-13, q ≈ 1e-31. Deleting outliers
    first (exact leave-one-out χ²₄ test at ``gate_level``) keeps the robustness with a proper likelihood.
    """
    params = init or NoiseParams()
    diag: dict[str, Any] = {"rounds": 0, "nfev": 0}
    prev: list[BoolArray] | None = None
    for _ in range(max(rounds, 1)):
        tracks = [smooth_clip(c, params, cfg) for c in clips]
        masks = [t.used for t in tracks]
        if prev is not None and all(np.array_equal(a, b) for a, b in zip(prev, masks, strict=True)):
            break
        prev = masks
        batch = stack_clips(clips, masks, [t.segment for t in tracks], cfg, [t.loose for t in tracks])
        if not batch.term.any():
            diag |= {"converged": False, "n_terms": 0}
            break
        h0 = params.log_vector()

        def objective(h: FloatArray, batch: Batch = batch) -> float:
            try:
                with np.errstate(all="ignore"):
                    nll = forward(batch, NoiseParams.from_log_vector(h), cfg).nll
            except np.linalg.LinAlgError:
                return float("inf")
            return nll if np.isfinite(nll) else float("inf")

        sol = minimize(
            objective,
            h0,
            method="Nelder-Mead",
            options={
                "initial_simplex": np.vstack([h0, h0 + np.eye(len(h0))]),
                "xatol": 0.02,
                "fatol": 0.5,
                "maxfev": 400,
            },
        )
        params = NoiseParams.from_log_vector(sol.x)
        diag["rounds"] += 1
        diag["nfev"] += int(sol.nfev)
        diag["converged"] = bool(sol.success)
        diag["deleted"] = int(sum(int(t.rejected.sum()) for t in tracks))
        diag["n_segments"] = batch.size
        fwd = forward(batch, params, cfg)
        nis = fwd.nis[np.isfinite(fwd.nis)]
        diag |= {
            "nll": float(fwd.nll),
            "n_terms": fwd.n_terms,
            "mean_nis": fwd.mean_nis,
            "mean_nis_truncated": float(np.minimum(nis, cfg.gate).mean()),
            "median_nis": float(np.median(nis)),
        }
    diag.setdefault("converged", False)
    diag.setdefault("n_terms", 0)
    diag.setdefault("nll", float("nan"))
    return params, diag


# --- outputs ------------------------------------------------------------------------------------------


def _cov_theta(a: FloatArray, p: FloatArray) -> FloatArray:
    cov = _sym(a @ p @ a.T)
    return np.asarray(cov + 1e-12 * np.diag(np.diag(cov)), dtype=np.float64)


def _ptz_result(
    phi: FloatArray, p: FloatArray, centre: FloatArray, raw: CalibResult, raw_ok: bool, cfg: SmootherConfig
) -> CalibResult:
    pp = cfg.principal_point
    a = ptz.jacobian_theta_phi(phi, centre, pp)
    return CalibResult(
        CalibStatus.OK,
        H=ptz.homography(phi, centre, pp),
        cov_theta=_cov_theta(a, p),
        s2=raw.s2,
        rms_px=raw.rms_px,
        n_inliers=raw.n_inliers if raw_ok else 0,
        cond=raw.cond,
        dof=None,
    )


def to_calib_results(
    raw: Sequence[CalibResult],
    meas: ClipMeasurements,
    track: SmoothedTrack,
    params: NoiseParams,
    cfg: SmootherConfig,
) -> tuple[list[CalibResult], FrameFlags]:
    """Per-frame outputs, rules applied in order:

    1. duplicate detections (k > 0): copy of output k − 1;
    2. inside a segment that is not short (and not a misfit or failed projection): smoothed state,
       cov_theta = A (P|T + β R_k [used]) Aᵀ with A = ∂θ/∂φ, dof None; flag smoothed / repaired / filled;
    3. used frame of a short segment: PTZ projection, cov_theta = A R_k Aᵀ; flag projected;
    4. measurement deleted by the robust passes with no smoothed state to replace it (outside every segment,
       or in a short one): abstains (status UNHEALTHY), since its raw camera was judged an outlier; flag
       rejected;
    5. any other raw-OK frame: raw H with cov_theta = κ² Σθ, dof None; flag passthrough. This covariance is
       per-frame, while a smoothed one pools the whole segment, and their scales differ: the evaluation
       calibrates the conformal quantiles of the two kinds separately;
    6. failed raw calibration outside segments: the raw result (abstains).
    """
    n = len(raw)
    flags = FrameFlags.empty(n)
    out: list[CalibResult] = []
    centre = meas.centre.centre if meas.centre is not None else None
    for k in range(n):
        r = raw[k]
        if k > 0 and meas.duplicate[k]:
            out.append(out[k - 1])
            flags.duplicate[k] = True
            continue
        raw_ok = bool(meas.raw_ok[k])
        off_model = raw_ok and (bool(meas.misfit[k]) or not bool(meas.projected[k]))
        in_seg = track.segment[k] >= 0 and bool(np.isfinite(track.mean[k, 0]))
        smooth = in_seg and not track.short[k] and not off_model
        flag = (
            flags.smoothed
            if smooth and track.used[k]
            else flags.repaired
            if smooth and track.rejected[k]
            else flags.filled
            if smooth and not raw_ok
            else None
        )
        if centre is not None and flag is not None:
            p = track.cov[k, :N_PHI, :N_PHI]
            if track.used[k] and cfg.beta:
                p = p + cfg.beta * track.R[k]
            out.append(_ptz_result(track.mean[k, :N_PHI], p, centre, r, raw_ok, cfg))
            flag[k] = True
        elif centre is not None and in_seg and track.short[k] and track.used[k]:
            out.append(_ptz_result(meas.z[k], track.R[k], centre, r, raw_ok, cfg))
            flags.projected[k] = True
        elif track.rejected[k]:
            out.append(CalibResult(CalibStatus.UNHEALTHY, n_inliers=r.n_inliers))
            flags.rejected[k] = True
        elif raw_ok:
            out.append(replace(r, cov_theta=params.kappa2 * np.asarray(r.cov_theta), dof=None))
            flags.passthrough[k] = True
        else:
            out.append(r)
    flags.segment[:] = track.segment
    return out, flags


def projection_results(
    raw: Sequence[CalibResult], meas: ClipMeasurements, cfg: SmootherConfig
) -> tuple[list[CalibResult], FrameFlags]:
    """Ablation "projection only": accepted (and projected duplicate) frames get H(φ̂, C) with
    cov_theta = A Σφ Aᵀ; the other OK frames keep their raw result; dof None everywhere."""
    n = len(raw)
    flags = FrameFlags.empty(n)
    out: list[CalibResult] = []
    centre = meas.centre.centre if meas.centre is not None else None
    for k in range(n):
        r = raw[k]
        dup = k > 0 and bool(meas.duplicate[k])
        on_model = bool(meas.accepted[k]) or (dup and bool(meas.projected[k]) and not bool(meas.misfit[k]))
        if centre is not None and on_model:
            out.append(_ptz_result(meas.z[k], meas.sigma_phi[k], centre, r, True, cfg))
        elif meas.raw_ok[k]:
            out.append(replace(r, dof=None))
        else:
            out.append(r)
        if dup:
            flags.duplicate[k] = True
        elif centre is not None and on_model:
            flags.projected[k] = True
        elif meas.raw_ok[k]:
            flags.passthrough[k] = True
    return out, flags


def empty_track(meas: ClipMeasurements, params: NoiseParams, cfg: SmootherConfig) -> SmoothedTrack:
    """Track of a clip with nothing to smooth (every frame passes through or abstains)."""
    n = len(meas)
    return SmoothedTrack(
        mean=np.full((n, N_STATE), np.nan),
        cov=np.full((n, N_STATE, N_STATE), np.nan),
        R=measurement_cov(meas, params, cfg),
        segment=np.full(n, -1, dtype=np.int64),
        used=np.zeros(n, dtype=bool),
        rejected=np.zeros(n, dtype=bool),
        short=np.zeros(n, dtype=bool),
        manoeuvre=np.zeros(n, dtype=bool),
        loose=np.zeros(n, dtype=bool),
        nis=np.full(n, np.nan),
        deletion=np.full(n, np.nan),
        nll=0.0,
        n_iter=0,
    )


def smooth_results(
    frames: ArrayLike,
    results: Sequence[CalibResult],
    duplicate: ArrayLike,
    cfg: SmootherConfig,
    params: NoiseParams,
    cuts: Collection[int] = (),
    game: CentreEstimate | None = None,
) -> tuple[list[CalibResult], FrameFlags, SmoothedTrack]:
    """Production wrapper: centre (the clip's own reliable estimate, else ``game``, the median centre of the
    match's clips from :func:`game_centre`, if it explains the clip) → measurements → smoothing → one
    CalibResult per frame. The same choice as the dev evaluation, so both score the same camera."""
    own = estimate_centre(results, duplicate, cfg)
    meas, _ = measure_at_centre("clip", frames, results, duplicate, own, game, cfg)
    track = smooth_clip(meas, params, cfg, cuts) if meas.accepted.any() else empty_track(meas, params, cfg)
    out, flags = to_calib_results(results, meas, track, params, cfg)
    return out, flags, track
