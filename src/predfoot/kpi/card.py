"""KPI uncertainty card: "HI distance = X ± Y m (90 %)" for one player-match (T4.4).

Inputs: the player's observed positions (native rate, NaN where missing, one array per period) and the
position-error model of the camera that produced them (:mod:`predfoot.kpi.noise`). Nothing else: no clean
track and no labels.

Method: a parametric bootstrap on the observed track ("noise on noise"). B fresh error tracks e*_b are drawn
from the model and added to the observed track y, and the 10 fps pipeline and the KPIs run on each
y + e*_b. If a KPI responds to a second dose of error as it did to the first (the first, linear step of
SIMEX, Cook & Stefanski 1994), D_b = K(y + e*_b) − K(y) is distributed like K(y) − K(x), x being the
error-free track, so:

- the bias estimate is b̂ = mean(D) and the bias-corrected estimate K̂ = K(y) − b̂;
- the 90 % interval is K̂ ± t·sd(D)·sqrt(1 + 1/B), t being the Student quantile with B − 1 degrees of
  freedom (the second factor covers the Monte Carlo error of b̂);
- the naive interval has the same half-width around K(y), without bias correction.

The linear step is right when the KPI error grows like the error variance (a moving player's distance); it
under-corrects where the error grows more slowly, such as the "ghost" distance of a nearly motionless
player, which grows like σ. ``scripts/kpi_card.py`` measures the resulting coverage on IDSSE.
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


def card_line(kpi: str, observed: float, draws: ArrayLike, level: float = 0.9) -> CardLine:
    """Summarise B bootstrap values K(y + e*_b) of one KPI around its observed value K(y)."""
    d = np.asarray(draws, dtype=np.float64) - observed
    d = d[np.isfinite(d)]
    b = d.size
    if b < 2 or not math.isfinite(observed):
        return CardLine(kpi, observed, math.nan, math.nan, math.nan, b, level)
    sd = float(np.std(d, ddof=1))
    t = float(student_t.ppf(0.5 + level / 2.0, b - 1))
    return CardLine(
        kpi, observed, observed - float(np.mean(d)), t * sd * math.sqrt(1.0 + 1.0 / b), sd, b, level
    )


def kpi_card(
    parts: Sequence[ArrayLike],
    model: SpectralErrorModel,
    cfg: KpiConfig,
    rng: np.random.Generator,
    *,
    n_draws: int = 100,
    level: float = 0.9,
    chunk: int = 25,
) -> dict[str, CardLine]:
    """The card of one player-match from its observed positions (one (n, 2) array per period at the native
    rate) and the error model of its camera. KPIs are those of :func:`~predfoot.kpi.metrics.compute_kpis`,
    without minutes played."""
    observed_parts: list[FloatArray] = [np.asarray(p, dtype=np.float64) for p in parts]
    kins = [pipeline(p, cfg, resample_to=cfg.output_hz) for p in observed_parts]
    observed = compute_kpis(kins, cfg)
    names = [k for k in observed if k != "minutes"]
    draws: dict[str, list[float]] = {k: [] for k in names}
    done = 0
    while done < n_draws:
        size = min(chunk, n_draws - done)
        errors = [
            sample_player_errors(model, p.shape[0], size, rng, hz=cfg.native_hz) for p in observed_parts
        ]
        for j in range(size):
            kin_b = [
                pipeline(p + e[:, j], cfg, resample_to=cfg.output_hz)
                for p, e in zip(observed_parts, errors, strict=True)
            ]
            values = compute_kpis(kin_b, cfg)
            for k in names:
                draws[k].append(values[k])
        done += size
    return {k: card_line(k, observed[k], draws[k], level) for k in names}
