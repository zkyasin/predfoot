"""KPI uncertainty card: "HI distance = X ± Y m (nominal 90 %)" for one player-match (T4.4).

Inputs: the player's observed positions (native rate, NaN where missing, one array per period) and the
position-error model of the camera that produced them (:mod:`predfoot.kpi.noise`). Nothing else: no clean
track and no labels.

Method: simulation-extrapolation (SIMEX, Cook & Stefanski 1994) on the observed track. Fresh error tracks
e*_b are drawn from the model and added to the observed track y at doses λ = 1 and λ = 2 (y + √λ·e*_b, so
that the total error variance is (1 + λ) times the original), and the 10 fps pipeline and the KPIs run on
each copy. With K(λ) the mean KPI at dose λ (K(0) = K(y)):

- ``linear``: K̂ = K(0) − (K(1) − K(0)), the noise-on-noise bias correction, right when the KPI error grows
  linearly with the error variance;
- ``quadratic``: the parabola through λ = 0, 1, 2 extrapolated to λ = −1 (no error), K̂ = 3K(0) − 3K(1) +
  K(2), which also follows a curved response (error-induced distance of a slow player grows like σ, not σ²;
  speed-threshold KPIs respond convexly).

An extrapolation can leave the KPI's support (every card KPI is ≥ 0): the estimate and the interval are then
projected onto it, and the line says so (``truncated``).

Uncertainty of K̂ = observed + Σ_λ c_λ·D̄_λ, with D_λ,b = K(y + √λ·e*_b) − K(y) and c = (−1) for the linear,
(−3, 1) for the quadratic extrapolation:

- Monte Carlo variance: var_b(Σ_λ c_λ·D_λ,b) / B. The doses share each e*_b, so their effects are correlated
  and must be combined draw by draw.
- Sampling variance over the observed error (the track is one draw). It is *not* the spread of one dose's
  effect sd(D₁)²: for KPIs that jump with the track (effort counts, thresholds, PSV-99) the averages K(1) and
  K(2) are much smoother in y than K(y), so the extrapolation amplifies the observed noise, up to 9 times in
  variance for the quadratic one. It is estimated by a nested parametric bootstrap (``n_outer`` = R > 1):
  pseudo-observations y'_r = y + e'_r, the same extrapolation on each with ``n_inner`` inner draws, and
  var_r(K̂_r) minus the mean inner Monte Carlo variance, floored at sd(D₁)² (the value for a KPI linear in the
  error). The pseudo-observations carry twice the error variance of y, which makes this conservative. With
  R ≤ 1 the sampling variance is sd(D₁)², right only for KPIs linear in the error.

The interval is K̂ ± t·s, with s² the sum of the two variances and t the Student quantile with R − 1 degrees
of freedom (B − 1 without the nested bootstrap). It is model-based: it assumes the error model is right and
the extrapolation unbiased, and is not calibrated. :meth:`CardLine.text` therefore says "nominal" unless a
calibration factor was applied (``scripts/kpi_card.py`` fits one by leave-one-match-out on IDSSE). The naive
interval is K(y) ± t·sd(D₁): no bias correction, and the spread of one dose as the sampling error of K(y).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t as student_t

from predfoot.kpi.config import KpiConfig
from predfoot.kpi.metrics import compute_kpis
from predfoot.kpi.noise import SpectralErrorModel, sample_player_errors
from predfoot.kpi.speed import FloatArray, pipeline

METHODS = ("linear", "quadratic")
#: Doses of each extrapolation (the coefficients of :func:`extrapolate` hold for these only).
DOSES = {"linear": (1.0,), "quadratic": (1.0, 2.0)}
#: Outer estimates and their inner Monte Carlo variances, per method and KPI.
NestedEstimates = dict[str, dict[str, tuple[list[float], list[float]]]]


@dataclass(frozen=True)
class CardLine:
    """One KPI of a card: the observed value, the bias-corrected estimate and its interval."""

    kpi: str
    observed: float
    raw_estimate: float  # the extrapolated value, before projection onto the support
    half_width: float  # t·s, times ``calibration``
    sd: float  # s: standard error of the raw estimate (sampling and Monte Carlo)
    n_draws: int
    level: float
    lower: float | None = 0.0  # lower end of the KPI's support (None: unbounded)
    calibration: float = 1.0  # factor applied to the half-width (1: model-based, nominal level)
    naive_half_width: float = math.nan  # t·sd(D₁), the half-width of the naive interval

    @property
    def estimate(self) -> float:
        """The bias-corrected estimate, projected onto the KPI's support."""
        if self.lower is None or not math.isfinite(self.raw_estimate):
            return self.raw_estimate
        return max(self.raw_estimate, self.lower)

    @property
    def truncated(self) -> bool:
        """Whether the extrapolation left the support (the estimate was moved onto it)."""
        return self.lower is not None and self.raw_estimate < self.lower

    @property
    def bias(self) -> float:
        """Estimated error-induced bias of the observed value."""
        return self.observed - self.estimate

    @property
    def interval(self) -> tuple[float, float]:
        lo, hi = self.raw_estimate - self.half_width, self.raw_estimate + self.half_width
        if self.lower is not None:
            lo, hi = max(lo, self.lower), max(hi, self.lower)
        return lo, hi

    @property
    def naive_interval(self) -> tuple[float, float]:
        lo, hi = self.observed - self.naive_half_width, self.observed + self.naive_half_width
        if self.lower is not None:
            lo, hi = max(lo, self.lower), max(hi, self.lower)
        return lo, hi

    def covers(self, truth: float, *, naive: bool = False) -> bool:
        lo, hi = self.naive_interval if naive else self.interval
        return bool(lo <= truth <= hi)

    def calibrated(self, factor: float) -> CardLine:
        """The same line with the half-width multiplied by ``factor`` (e.g. a leave-one-match-out factor)."""
        return replace(self, half_width=self.half_width * factor, calibration=self.calibration * factor)

    def text(self, label: str, unit: str = "", digits: int = 0) -> str:
        """Card wording, e.g. ``HI distance = 812 ± 95 m (nominal 90 %)``; an interval cut by the support
        is written out, e.g. ``Sprint efforts = 0.0 [0.0, 2.1] (nominal 90 %)``."""
        u = f" {unit}" if unit else ""
        level = f"{100 * self.level:g} %"
        level = level if self.calibration != 1.0 else f"nominal {level}"
        lo, hi = self.interval
        if lo > self.raw_estimate - self.half_width:
            return f"{label} = {self.estimate:.{digits}f}{u} [{lo:.{digits}f}, {hi:.{digits}f}] ({level})"
        return f"{label} = {self.estimate:.{digits}f} ± {self.half_width:.{digits}f}{u} ({level})"


def extrapolate(
    observed: float, draws: ArrayLike, draws2: ArrayLike | None = None
) -> tuple[float, float, float, int]:
    """The SIMEX estimate from KPI values at dose 1 (and dose 2 for the quadratic extrapolation), with its
    Monte Carlo variance, the spread sd(D₁) of one dose's effect and the number of usable draws. Draws are
    kept in pairs: a draw is dropped when either of its doses is not finite."""
    d1 = np.asarray(draws, dtype=np.float64) - observed
    d2 = None if draws2 is None else np.asarray(draws2, dtype=np.float64) - observed
    ok = np.isfinite(d1) if d2 is None else np.isfinite(d1) & np.isfinite(d2)
    d1 = d1[ok]
    b = int(d1.size)
    if b < 2 or not math.isfinite(observed):
        return math.nan, math.nan, math.nan, b
    combined = -d1 if d2 is None else -3.0 * d1 + d2[ok]
    estimate = observed + float(np.mean(combined))
    return estimate, float(np.var(combined, ddof=1)) / b, float(np.std(d1, ddof=1)), b


def card_line(
    kpi: str,
    observed: float,
    draws: ArrayLike,
    level: float = 0.9,
    draws2: ArrayLike | None = None,
    *,
    outer: tuple[Sequence[float], Sequence[float]] | None = None,
    lower: float | None = 0.0,
) -> CardLine:
    """Summarise bootstrap values of one KPI: ``draws`` at dose 1 (K(y + e*_b)) and, for the quadratic
    extrapolation, ``draws2`` at dose 2 (K(y + √2·e*_b)), paired by draw. ``outer`` holds the nested
    bootstrap's estimates on the pseudo-observations and their inner Monte Carlo variances (same
    extrapolation); without it the sampling variance is the spread of one dose. See the module docstring."""
    raw, mc_var, sd1, b = extrapolate(observed, draws, draws2)
    if not math.isfinite(raw):
        return CardLine(kpi, observed, math.nan, math.nan, math.nan, b, level, lower)
    sampling_var, df = sd1**2, b - 1
    if outer is not None:
        est = np.asarray(outer[0], dtype=np.float64)
        mc = np.asarray(outer[1], dtype=np.float64)
        ok = np.isfinite(est) & np.isfinite(mc)
        if ok.sum() >= 2:
            sampling_var = max(sd1**2, float(np.var(est[ok], ddof=1)) - float(np.mean(mc[ok])))
            df = int(ok.sum()) - 1
    s = math.sqrt(sampling_var + mc_var)
    t = float(student_t.ppf(0.5 + level / 2.0, df))
    naive = float(student_t.ppf(0.5 + level / 2.0, b - 1)) * sd1
    return CardLine(kpi, observed, raw, t * s, s, b, level, lower, naive_half_width=naive)


def kpi_draws(
    parts: Sequence[ArrayLike],
    model: SpectralErrorModel,
    cfg: KpiConfig,
    rng: np.random.Generator,
    *,
    n_draws: int = 50,
    doses: Sequence[float] = (1.0, 2.0),
    chunk: int = 25,
) -> tuple[dict[str, float], dict[float, dict[str, list[float]]]]:
    """Observed KPIs of a player-match and their bootstrap values at each error dose (one error draw is
    shared by all doses, scaled by √λ). Minutes played are left out."""
    observed_parts: list[FloatArray] = [np.asarray(p, dtype=np.float64) for p in parts]
    observed = compute_kpis([pipeline(p, cfg, resample_to=cfg.output_hz) for p in observed_parts], cfg)
    names = [k for k in observed if k != "minutes"]
    draws: dict[float, dict[str, list[float]]] = {lam: {k: [] for k in names} for lam in doses}
    done = 0
    while done < n_draws:
        size = min(chunk, n_draws - done)
        errors = [
            sample_player_errors(model, p.shape[0], size, rng, hz=cfg.native_hz) for p in observed_parts
        ]
        for j in range(size):
            for lam in doses:
                scale = math.sqrt(lam)
                kins = [
                    pipeline(p + scale * e[:, j], cfg, resample_to=cfg.output_hz)
                    for p, e in zip(observed_parts, errors, strict=True)
                ]
                values = compute_kpis(kins, cfg)
                for k in names:
                    draws[lam][k].append(values[k])
        done += size
    return {k: observed[k] for k in names}, draws


def nested_estimates(
    parts: Sequence[ArrayLike],
    model: SpectralErrorModel,
    cfg: KpiConfig,
    rng: np.random.Generator,
    *,
    n_outer: int = 10,
    n_inner: int = 10,
    doses: Sequence[float] = (1.0, 2.0),
) -> NestedEstimates:
    """Nested parametric bootstrap: for ``n_outer`` pseudo-observations y'_r = y + e'_r (one model draw
    each), the linear (and, with dose 2, quadratic) estimates on y'_r from ``n_inner`` inner draws, with
    their Monte Carlo variances. Raw estimates: no projection onto the support."""
    if tuple(doses) not in DOSES.values():
        raise ValueError(f"doses must be one of {list(DOSES.values())}")
    methods = [m for m in METHODS if len(DOSES[m]) <= len(doses)]
    observed_parts = [np.asarray(p, dtype=np.float64) for p in parts]
    out: NestedEstimates = {}
    for _ in range(n_outer):
        pseudo = [
            p + sample_player_errors(model, p.shape[0], 1, rng, hz=cfg.native_hz)[:, 0]
            for p in observed_parts
        ]
        obs, draws = kpi_draws(pseudo, model, cfg, rng, n_draws=n_inner, doses=doses)
        for method in methods:
            per_kpi = out.setdefault(method, {})
            for k, v in obs.items():
                second = draws[2.0][k] if method == "quadratic" else None
                estimate, mc_var, _, _ = extrapolate(v, draws[1.0][k], second)
                ests, mcs = per_kpi.setdefault(k, ([], []))
                ests.append(estimate)
                mcs.append(mc_var)
    return out


def kpi_card(
    parts: Sequence[ArrayLike],
    model: SpectralErrorModel,
    cfg: KpiConfig,
    rng: np.random.Generator,
    *,
    n_draws: int = 50,
    level: float = 0.9,
    method: str = "quadratic",
    n_outer: int = 10,
    n_inner: int = 10,
    lower: float | None = 0.0,
) -> dict[str, CardLine]:
    """The card of one player-match from its observed positions (one (n, 2) array per period at the native
    rate) and the error model of its camera, with the ``linear`` or ``quadratic`` extrapolation. The nested
    bootstrap (``n_outer`` > 1) costs ``n_outer``·(1 + 2·``n_inner``) more pipeline runs."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    doses = DOSES[method]
    observed, draws = kpi_draws(parts, model, cfg, rng, n_draws=n_draws, doses=doses)
    nested = (
        nested_estimates(parts, model, cfg, rng, n_outer=n_outer, n_inner=n_inner, doses=doses)
        if n_outer > 1
        else None
    )
    return {
        k: card_line(
            k,
            v,
            draws[1.0][k],
            level,
            draws[2.0][k] if method == "quadratic" else None,
            outer=nested[method][k] if nested is not None else None,
            lower=lower,
        )
        for k, v in observed.items()
    }
