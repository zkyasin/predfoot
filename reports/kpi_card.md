# KPI uncertainty card: a consistency check on IDSSE (T4.4)

> **Consistency check, not validation.** The "observed" tracks are clean IDSSE tracking plus errors
> simulated by the same model the card assumes (fitted on SN-GSR dev, `reports/kpi_budget_spectral.md`).
> Coverage here checks the card's arithmetic and its noise-on-noise approximation; it says nothing about
> whether the error model is right for real broadcast tracking.

- **Command**: `uv run python scripts/kpi_card.py --n-draws 20`
- **Git commit**: `cae18db`; date 2026-10-31T17:34:30; runtime 294.5 s
- **Error model** per camera: `matern32:copula:white` from `reports/error_process_temporal_dev.json` (`predfoot.kpi.noise.fit_error_model`).
- **Data**: IDSSE (Bassek, Rein, Weber & Memmert 2025, CC BY 4.0), 7 matches, 142 outfield player-matches with ≥ 45 min; 20 error draws per card, each at two doses; coverage intervals: 1000 bootstrap resamples over matches; seed 20261031.

## The card

From the observed track and the camera's error model only (`predfoot.kpi.card`): fresh errors
are drawn from the model and added to the observed track at one and two extra doses of error
variance, and the 10 fps KPI pipeline runs on each copy (SIMEX, Cook & Stefanski 1994). The
**linear** estimate removes the change caused by one extra dose (noise on noise); the **quadratic**
one extrapolates the parabola through zero, one and two extra doses back to no error. The 90 %
interval adds the spread of one dose's effect and the Monte Carlo error; the **naive** interval has
the same half-width around the observed value. The truth is the clean track through the same 10 fps
pipeline.

## Coverage of the clean truth

Coverage of 90 % intervals across player-matches [95 % CI over matches]; LOMO: leave-one-match-out
calibrated half-width (median scale factor in brackets); bias: relative error of the KPI summed over
player-matches, before (observed) and after correction (estimate).

| Camera | KPI | Quadratic | Linear | Naive | Quadratic, LOMO-calibrated | Median half-width (rel.) | Bias, observed | Bias after linear | Bias after quadratic |
|---|---|---|---|---|---|---|---|---|---|
| none | Distance | 0 % [0, 0] | 0 % [0, 0] | 0 % [0, 0] | 88 % (×7.98) | 109 m (1 %) | +46.2 % | +13.0 % | +6.3 % |
| none | HI distance (≥ 20 km/h) | 0 % [0, 0] | 0 % [0, 0] | 0 % [0, 0] | 88 % (×4.73) | 128 m (17 %) | +228.1 % | -106.8 % | -56.2 % |
| none | Sprint distance (> 25 km/h) | 74 % [68, 81] | 0 % [0, 0] | 0 % [0, 0] | 89 % (×1.35) | 98 m (63 %) | +423.5 % | -338.3 % | +12.1 % |
| none | HSR efforts | 54 % [44, 63] | 42 % [32, 54] | 67 % [59, 75] | 89 % (×2.17) | 8.0 (22 %) | -15.7 % | -22.7 % | -15.6 % |
| none | Sprint efforts | 54 % [50, 58] | 49 % [46, 53] | 78 % [73, 84] | 89 % (×2.03) | 3.8 (48 %) | -28.2 % | -33.2 % | -25.7 % |
| none | PSV-99 | 40 % [34, 46] | 34 % [26, 41] | 0 % [0, 0] | 91 % (×2.70) | 2.0 km/h (7 %) | +24.2 % | +8.9 % | +9.1 % |
| ptz | Distance | 21 % [3, 41] | 0 % [0, 0] | 0 % [0, 0] | 87 % (×3.54) | 82 m (1 %) | +22.6 % | +4.9 % | +1.7 % |
| ptz | HI distance (≥ 20 km/h) | 56 % [40, 72] | 4 % [1, 7] | 0 % [0, 0] | 89 % (×1.88) | 79 m (11 %) | +76.0 % | -23.5 % | +8.3 % |
| ptz | Sprint distance (> 25 km/h) | 18 % [12, 24] | 7 % [3, 12] | 0 % [0, 0] | 89 % (×2.97) | 55 m (36 %) | +124.4 % | -57.1 % | +54.5 % |
| ptz | HSR efforts | 65 % [57, 75] | 62 % [54, 70] | 94 % [90, 99] | 89 % (×1.62) | 7.5 (20 %) | -3.3 % | -8.9 % | -5.6 % |
| ptz | Sprint efforts | 55 % [46, 63] | 60 % [50, 69] | 87 % [83, 92] | 90 % (×1.89) | 3.6 (46 %) | -13.8 % | -19.5 % | -16.8 % |
| ptz | PSV-99 | 63 % [54, 72] | 63 % [50, 75] | 4 % [1, 8] | 89 % (×1.77) | 1.5 km/h (5 %) | +12.2 % | +3.0 % | +3.5 % |

![Cards for HI distance](figures/kpi_card.png)

## Example cards

**J03WN1, home #10**

| KPI | Card, none camera (90 %) | Card, ptz camera (90 %) | Clean truth |
|---|---|---|---|
| Distance | 7134 ± 67 m (observed 9653) | 6727 ± 68 m (observed 8047) | 6670 m |
| HI distance (≥ 20 km/h) | 355 ± 87 m (observed 1790) | 633 ± 71 m (observed 970) | 575 m |
| Sprint distance (> 25 km/h) | 248 ± 76 m (observed 763) | 315 ± 45 m (observed 369) | 174 m |
| HSR efforts | 18.4 ± 6.9 (observed 23.0) | 19.6 ± 6.9 (observed 20.0) | 25.0 |
| Sprint efforts | 1.8 ± 3.3 (observed 5.0) | 1.5 ± 2.8 (observed 5.0) | 6.0 |
| PSV-99 | 29.3 ± 2.3 km/h (observed 35.1) | 30.9 ± 1.7 km/h (observed 33.8) | 30.2 km/h |

**J03WR9, away #13**

| KPI | Card, none camera (90 %) | Card, ptz camera (90 %) | Clean truth |
|---|---|---|---|
| Distance | 9393 ± 107 m (observed 13093) | 8894 ± 96 m (observed 10845) | 8834 m |
| HI distance (≥ 20 km/h) | 383 ± 127 m (observed 2202) | 807 ± 72 m (observed 1197) | 698 m |
| Sprint distance (> 25 km/h) | 184 ± 96 m (observed 851) | 328 ± 59 m (observed 421) | 189 m |
| HSR efforts | 35.1 ± 7.9 (observed 32.0) | 36.0 ± 6.5 (observed 36.0) | 36.0 |
| Sprint efforts | 8.3 ± 3.2 (observed 6.0) | 11.1 ± 3.9 (observed 10.0) | 8.0 |
| PSV-99 | 34.9 ± 2.3 km/h (observed 37.6) | 30.2 ± 1.8 km/h (observed 32.6) | 30.1 km/h |

**J03WPY, home #8**

| KPI | Card, none camera (90 %) | Card, ptz camera (90 %) | Clean truth |
|---|---|---|---|
| Distance | 11659 ± 113 m (observed 16116) | 11126 ± 100 m (observed 13503) | 10991 m |
| HI distance (≥ 20 km/h) | 556 ± 115 m (observed 2915) | 1133 ± 89 m (observed 1682) | 1053 m |
| Sprint distance (> 25 km/h) | 396 ± 111 m (observed 1213) | 472 ± 57 m (observed 606) | 347 m |
| HSR efforts | 50.2 ± 9.7 (observed 45.0) | 40.1 ± 7.4 (observed 44.0) | 46.0 |
| Sprint efforts | 11.1 ± 4.2 (observed 12.0) | 9.4 ± 3.9 (observed 12.0) | 16.0 |
| PSV-99 | 30.9 ± 1.6 km/h (observed 35.9) | 32.0 ± 1.1 km/h (observed 33.4) | 30.9 km/h |

## Reading it

- A naive interval around the observed value inherits the whole error-induced bias, which for
  distance KPIs is many half-widths (column *Naive*): the bias correction is the core of the card.
- The extrapolation assumes a smooth response of each KPI to the error variance. The linear step
  cannot follow a curved response (distance of slow phases grows like σ, threshold KPIs respond
  convexly); the quadratic one follows curvature but adds Monte Carlo variance. Whatever the
  raw coverage, the leave-one-match-out factor shows how much the half-width must grow to reach
  90 %; it is fitted on the other matches only.
- Everything is conditional on the error model: on real tracking the card is only as good as the
  error process measured on SN-GSR (stationary, independent of motion, 5 fps, smoothed-label
  reference; see `reports/kpi_budget_spectral.md`).
