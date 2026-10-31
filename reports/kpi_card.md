# KPI uncertainty card: a consistency check on IDSSE (T4.4)

> **Consistency check, not validation.** The "observed" tracks are clean IDSSE tracking plus errors
> simulated by the same model the card assumes (fitted on SN-GSR dev, `reports/kpi_budget_spectral.md`).
> Coverage here checks the card's arithmetic and its noise-on-noise approximation; it says nothing about
> whether the error model is right for real broadcast tracking.

- **Command**: `uv run python scripts/kpi_card.py --matches J03WMX J03WN1 J03WOH J03WOY --n-draws 20 --n-outer 8 --n-inner 8`
- **Git commit**: `796a452`; date 2026-10-31T18:47:01; runtime 627.4 s
- **Error model** per camera: `matern32:copula:white` from `reports/error_process_temporal_dev.json` (`predfoot.kpi.noise.fit_error_model`).
- **Data**: IDSSE (Bassek, Rein, Weber & Memmert 2025, CC BY 4.0), 4 matches (J03WMX, J03WN1, J03WOH, J03WOY), 80 outfield player-matches with ≥ 45 min; 20 error draws per card, each at two doses; nested bootstrap with 8 pseudo-observations of 8 draws each; coverage intervals: 1000 bootstrap resamples over matches; seed 20261031.

## The card

From the observed track and the camera's error model only (`predfoot.kpi.card`): fresh errors are
drawn from the model and added to the observed track at one and two extra doses of error variance,
and the 10 fps KPI pipeline runs on each copy (SIMEX, Cook & Stefanski 1994). The **linear**
estimate removes the change caused by one extra dose (noise on noise); the **quadratic** one
extrapolates the parabola through zero, one and two extra doses back to no error. Estimates and
intervals that leave the KPI's support (all these KPIs are ≥ 0) are projected onto it. The
interval is estimate ± t·s: s² is the sampling variance of the estimate over the observed error,
from a nested parametric bootstrap (the same extrapolation on pseudo-observations, the observed
track plus one more model draw), plus the Monte Carlo variance of the draws, combined draw by draw
since both doses reuse each draw. The spread of one dose's effect, the previous choice of s,
understates the sampling variance of KPIs that jump with the track: the extrapolation amplifies
their noise. The intervals are model-based and uncalibrated, hence *nominal* 90 %. The **naive**
interval is the observed value ± t·(spread of one dose), without bias correction. The truth is the
clean track through the same 10 fps pipeline.

## Coverage of the clean truth

Coverage of nominal 90 % intervals across player-matches [95 % CI over matches]: quadratic and
linear with the nested-bootstrap variance, quadratic with the spread of one dose (previous
half-width), and naive. LOMO: the quadratic interval with a leave-one-match-out calibrated
half-width (median scale factor in brackets). Truncated: share of quadratic estimates projected
onto the support. Bias: relative error of the KPI summed over player-matches, before (observed)
and after correction (estimate). The observed bias comes from errors simulated by the model of
`reports/kpi_budget_spectral.md`, whose 5 fps check against SN-GSR fails (section 2 there): it
overstates the KPI bias of predfoot's real tracking and is not predfoot's KPI error.

| Camera | KPI | Quadratic | Quadratic, one-dose spread | Linear | Naive | Quadratic, LOMO-calibrated | Median half-width (rel.) | Truncated | Bias, observed | Bias after linear | Bias after quadratic |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | Distance | 0 % [0, 5]† | 0 % [0, 5]† | 0 % [0, 5]† | 0 % [0, 5]† | 89 % (×6.08) | 123 m (1 %) | 0 % | +41.8 % | +11.2 % | +4.9 % |
| none | HI distance (≥ 20 km/h) | 16 % [10, 22] | 4 % [1, 5] | 0 % [0, 5]† | 0 % [0, 5]† | 86 % (×2.80) | 142 m (22 %) | 6 % | +194.9 % | -77.2 % | -34.6 % |
| none | Sprint distance (> 25 km/h) | 59 % [44, 68] | 46 % [34, 56] | 1 % [0, 4] | 0 % [0, 5]† | 89 % (×2.03) | 110 m (79 %) | 2 % | +363.4 % | -97.4 % | +48.2 % |
| none | HSR efforts | 94 % [88, 99] | 56 % [50, 62] | 75 % [70, 80] | 56 % [51, 62] | 90 % (×0.89) | 15.4 (40 %) | 0 % | -14.1 % | -18.3 % | -10.9 % |
| none | Sprint efforts | 91 % [90, 94] | 51 % [42, 60] | 85 % [78, 92] | 61 % [55, 69] | 90 % (×0.98) | 7.4 (100 %) | 6 % | -26.0 % | -29.8 % | -21.8 % |
| none | PSV-99 | 75 % [65, 82] | 45 % [35, 51] | 65 % [58, 71] | 0 % [0, 5]† | 89 % (×1.47) | 3.2 km/h (11 %) | 0 % | +21.6 % | +7.2 % | +7.6 % |
| ptz | Distance | 36 % [15, 69] | 24 % [4, 56] | 0 % [0, 5]† | 0 % [0, 5]† | 85 % (×2.96) | 85 m (1 %) | 0 % | +19.3 % | +3.9 % | +1.2 % |
| ptz | HI distance (≥ 20 km/h) | 55 % [32, 66] | 32 % [14, 46] | 20 % [16, 24] | 0 % [0, 5]† | 89 % (×1.97) | 91 m (12 %) | 0 % | +60.7 % | -15.9 % | +9.9 % |
| ptz | Sprint distance (> 25 km/h) | 21 % [10, 32] | 12 % [8, 15] | 49 % [37, 68] | 0 % [0, 5]† | 89 % (×2.46) | 59 m (40 %) | 0 % | +101.8 % | -31.7 % | +51.4 % |
| ptz | HSR efforts | 90 % [86, 94] | 59 % [55, 63] | 89 % [82, 94] | 91 % [87, 95] | 91 % (×1.05) | 15.2 (39 %) | 0 % | -3.1 % | -9.1 % | -7.1 % |
| ptz | Sprint efforts | 94 % [88, 100] | 55 % [44, 66] | 89 % [80, 98] | 86 % [78, 93] | 92 % (×0.92) | 6.1 (79 %) | 5 % | -10.2 % | -15.4 % | -12.4 % |
| ptz | PSV-99 | 86 % [72, 100] | 70 % [58, 84] | 86 % [74, 98] | 2 % [0, 5] | 89 % (×1.17) | 2.0 km/h (7 %) | 0 % | +10.3 % | +2.2 % | +2.9 % |

† The bootstrap over matches collapsed to a point (every interval covers, or none does):
Clopper-Pearson interval over player-matches instead, which ignores the clustering by match.

![Cards for HI distance](figures/kpi_card.png)

## Example cards

**J03WOY, home #5**

| KPI | Card, none camera (nominal 90 %) | Card, ptz camera (nominal 90 %) | Clean truth |
|---|---|---|---|
| Distance | 10447 ± 143 m (observed 14473) | 10047 ± 126 m (observed 12035) | 9947 m |
| HI distance (≥ 20 km/h) | 253 ± 172 m (observed 1918) | 647 ± 105 m (observed 953) | 558 m |
| Sprint distance (> 25 km/h) | 227 ± 126 m (observed 667) | 299 ± 59 m (observed 297) | 157 m |
| HSR efforts | 23.3 ± 18.4 (observed 25.0) | 19.7 ± 14.0 (observed 25.0) | 29.0 |
| Sprint efforts | 5.7 [0.0, 11.9] (observed 5.0) | 9.2 ± 6.9 (observed 8.0) | 9.0 |
| PSV-99 | 29.6 ± 2.9 km/h (observed 33.7) | 31.7 ± 1.6 km/h (observed 33.2) | 30.3 km/h |

**J03WOH, home #34**

| KPI | Card, none camera (nominal 90 %) | Card, ptz camera (nominal 90 %) | Clean truth |
|---|---|---|---|
| Distance | 10520 ± 153 m (observed 14363) | 10288 ± 93 m (observed 12150) | 10100 m |
| HI distance (≥ 20 km/h) | 290 ± 134 m (observed 2088) | 825 ± 78 m (observed 1147) | 685 m |
| Sprint distance (> 25 km/h) | 275 ± 111 m (observed 783) | 289 ± 97 m (observed 355) | 149 m |
| HSR efforts | 38.3 ± 18.2 (observed 34.0) | 31.4 ± 17.3 (observed 37.0) | 37.0 |
| Sprint efforts | 14.5 ± 8.6 (observed 9.0) | 7.2 ± 6.3 (observed 7.0) | 10.0 |
| PSV-99 | 29.1 ± 2.4 km/h (observed 34.1) | 27.8 ± 1.3 km/h (observed 30.5) | 27.7 km/h |

**J03WMX, away #7**

| KPI | Card, none camera (nominal 90 %) | Card, ptz camera (nominal 90 %) | Clean truth |
|---|---|---|---|
| Distance | 12137 ± 158 m (observed 16073) | 11723 ± 114 m (observed 13619) | 11675 m |
| HI distance (≥ 20 km/h) | 623 ± 118 m (observed 2761) | 1099 ± 106 m (observed 1581) | 1066 m |
| Sprint distance (> 25 km/h) | 268 ± 181 m (observed 1060) | 291 ± 67 m (observed 446) | 163 m |
| HSR efforts | 51.5 ± 14.1 (observed 47.0) | 61.2 ± 15.3 (observed 55.0) | 58.0 |
| Sprint efforts | 5.9 ± 5.2 (observed 6.0) | 14.4 ± 7.6 (observed 11.0) | 11.0 |
| PSV-99 | 30.2 ± 2.1 km/h (observed 34.0) | 27.8 ± 1.6 km/h (observed 30.6) | 27.2 km/h |

## Reading it

- A naive interval around the observed value inherits the whole error-induced bias, which for
  distance KPIs is many half-widths (column *Naive*): the bias correction is the core of the card.
- The extrapolation assumes a smooth response of each KPI to the error variance. The linear step
  cannot follow a curved response (distance of slow phases grows like σ, threshold KPIs respond
  convexly); the quadratic one follows curvature, at the price of a larger sampling variance: it
  multiplies the observed value by 3, and for KPIs that jump with the track the averaged doses do
  not cancel that noise. Compare the columns *Quadratic* and *Quadratic, one-dose spread*.
- Where coverage stays below 90 % with the nested-bootstrap variance, the remaining extrapolation
  bias is the cause (compare *Bias after quadratic* with the half-width). The leave-one-match-out
  factor shows how much the half-width must grow to reach 90 %; it is fitted on the other matches
  only, on simulated errors, so it does not carry over to real tracking as such.
- Everything is conditional on the error model: on real tracking the card is only as good as the
  error process measured on SN-GSR (stationary, independent of motion, 5 fps, smoothed-label
  reference; see `reports/kpi_budget_spectral.md`).
