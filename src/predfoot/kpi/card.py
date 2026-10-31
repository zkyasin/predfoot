"""KPI uncertainty card: "HI distance = X ± Y m (90 %)" for one player-match (T4.4).

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

The 90 % interval is K̂ ± t·s, t being the Student quantile with B − 1 degrees of freedom and s² the variance
of one dose's effect, sd(D₁)² with D₁ = K(y + e*) − K(y), plus the Monte Carlo variance of K̂. The naive
interval has the same half-width around K(y), without bias correction. ``scripts/kpi_card.py`` measures the
coverage on IDSSE.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t as student_t

from predfoot.kpi.config import KpiConfig
from predfoot.kpi.metrics import compute_kpis
from predfoot.kpi.noise import SpectralErrorModel, sample_player_errors
from predfoot.kpi.speed import FloatArray, pipeline


@dataclass(frozen=True)
class CardLine:
    """One KPI of a card: the observed value, the bias-corrected estimate and its interval half-width."""

    kpi: str
    observed: float
    estimate: float
    half_width: float
    sd: float  # standard deviation of the bootstrap differences D
    n_draws: int
    level: float

    @property
    def bias(self) -> float:
        """Estimated error-induced bias of the observed value."""
        return self.observed - self.estimate

    @property
    def interval(self) -> tuple[float, float]:
        return self.estimate - self.half_width, self.estimate + self.half_width

    @property
    def naive_interval(self) -> tuple[float, float]:
        return self.observed - self.half_width, self.observed + self.half_width

    def covers(self, truth: float, *, naive: bool = False) -> bool:
        lo, hi = self.naive_interval if naive else self.interval
        return bool(lo <= truth <= hi)

    def text(self, label: str, unit: str = "", digits: int = 0) -> str:
        """Card wording, e.g. ``HI distance = 812 ± 95 m (90 %)``."""
        u = f" {unit}" if unit else ""
        return (
            f"{label} = {self.estimate:.{digits}f} ± {self.half_width:.{digits}f}{u} ({100 * self.level:g} %)"
        )


def card_line(
    kpi: str,
    observed: float,
    draws: ArrayLike,
    level: float = 0.9,
    draws2: ArrayLike | None = None,
) -> CardLine:
    """Summarise bootstrap values of one KPI: ``draws`` at dose 1 (K(y + e*_b)) and, for the quadratic
    extrapolation, ``draws2`` at dose 2 (K(y + √2·e*_b)); see the module docstring."""
    d1 = np.asarray(draws, dtype=np.float64) - observed
    d1 = d1[np.isfinite(d1)]
    b = d1.size
    if b < 2 or not math.isfinite(observed):
        return CardLine(kpi, observed, math.nan, math.nan, math.nan, b, level)
    sd1 = float(np.std(d1, ddof=1))
    t = float(student_t.ppf(0.5 + level / 2.0, b - 1))
    if draws2 is None:
        return CardLine(
            kpi, observed, observed - float(np.mean(d1)), t * sd1 * math.sqrt(1.0 + 1.0 / b), sd1, b, level
        )
    d2 = np.asarray(draws2, dtype=np.float64) - observed
    d2 = d2[np.isfinite(d2)]
    if d2.size < 2:
        return CardLine(kpi, observed, math.nan, math.nan, math.nan, b, level)
    estimate = observed - 3.0 * float(np.mean(d1)) + float(np.mean(d2))
    mc_var = 9.0 * sd1**2 / b + float(np.var(d2, ddof=1)) / d2.size
    return CardLine(kpi, observed, estimate, t * math.sqrt(sd1**2 + mc_var), sd1, b, level)


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


def kpi_card(
    parts: Sequence[ArrayLike],
    model: SpectralErrorModel,
    cfg: KpiConfig,
    rng: np.random.Generator,
    *,
    n_draws: int = 50,
    level: float = 0.9,
    method: str = "quadratic",
) -> dict[str, CardLine]:
    """The card of one player-match from its observed positions (one (n, 2) array per period at the native
    rate) and the error model of its camera, with the ``linear`` or ``quadratic`` extrapolation."""
    if method not in ("linear", "quadratic"):
        raise ValueError(f"unknown method {method!r}")
    doses = (1.0,) if method == "linear" else (1.0, 2.0)
    observed, draws = kpi_draws(parts, model, cfg, rng, n_draws=n_draws, doses=doses)
    return {
        k: card_line(k, v, draws[1.0][k], level, draws[2.0][k] if method == "quadratic" else None)
        for k, v in observed.items()
    }
