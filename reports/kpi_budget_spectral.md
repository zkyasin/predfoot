# KPI error budget with a spectral error model (T4.3)

> **Simulation, not measurement.** Every KPI error below comes from errors *simulated* by a model
> fitted to predfoot's error process on SN-GSR dev (5 fps), injected into clean IDSSE tracking. The
> only measured KPI numbers are the SN-GSR proxy values quoted in section 2.

- **Command**: `uv run python scripts/kpi_spectral.py `
- **Git commit**: `5a726fe`; date 2026-10-31T17:25:14; runtime 260.9 s
- **Error process**: `reports/error_process_temporal_dev.json` (from `uv run python scripts/error_process.py runs/20261031-153059_c3861c7_ptz/players.parquet --compare runs/20261031-152537_c3861c7_none/players.parquet --name ptz --compare-name none`), errors against the smoothed labels, frames one in five (5 fps), per axis (x along the touchlines, y across) and part.
- **IDSSE**: proxy check on J03WMX; budget on 7 matches (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), 142 player-matches (outfield, ≥ 45 min). IDSSE: Bassek, Rein, Weber & Memmert 2025, CC BY 4.0.
- **Seed** 20261031, 1000 bootstrap resamples over matches; one noise draw per condition and match.

## 1. Error model

Per axis and part, a stationary process with autocorrelation ρ(τ) = w_n·ν(τ) + w_m·k(τ/ℓ) + w_s·k(τ/T) (`predfoot.kpi.noise`): a nugget ν plus a main and a slow scale of one kernel k, fitted by weighted least squares to the five measured lags (0.2-1.0 s). Scales are kept at or above the 0.2 s sampling interval: anything faster is not resolved by the measurement and is carried by the nugget, whose shape at 25 Hz is an assumption (section 3). Two marginals: **Gaussian** with the measured σ, or the **measured marginal** through a Gaussian copula, ρ being then fitted to the ACF of the normal scores. The measured errors are heavy-tailed: a Gaussian with the measured σ makes the typical error about twice too large (robust σ = 0.36 to 0.68 of the SD, kurtosis 11 to 35 on these tables).

Largest |ACF residual| over the five lags (model minus measurement; normal-score ACF for the copula fits):

| Camera | Part | Axis | single OU through 0.2 s (previous) | nugget + exponential (gauss) | nugget + Matérn-3/2 (gauss) | nugget + Matérn-5/2 (gauss) | nugget + squared exponential (gauss) | nugget + exponential (copula) | nugget + Matérn-3/2 (copula) | nugget + Matérn-5/2 (copula) | nugget + squared exponential (copula) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | common | x | 0.256 | 0.007 | 0.005 | 0.005 | 0.004 | 0.004 | 0.002 | 0.002 | 0.001 |
| none | common | y | 0.233 | 0.005 | 0.005 | 0.005 | 0.003 | 0.004 | 0.003 | 0.003 | 0.002 |
| none | individual | x | 0.237 | 0.009 | 0.008 | 0.007 | 0.005 | 0.009 | 0.007 | 0.006 | 0.005 |
| none | individual | y | 0.222 | 0.009 | 0.007 | 0.007 | 0.005 | 0.009 | 0.007 | 0.007 | 0.006 |
| ptz | common | x | 0.065 | 0.003 | 0.002 | 0.002 | 0.004 | 0.002 | 0.001 | 0.002 | 0.003 |
| ptz | common | y | 0.062 | 0.025 | 0.001 | 0.002 | 0.003 | 0.020 | 0.001 | 0.002 | 0.003 |
| ptz | individual | x | 0.224 | 0.012 | 0.010 | 0.009 | 0.008 | 0.012 | 0.010 | 0.009 | 0.008 |
| ptz | individual | y | 0.259 | 0.017 | 0.016 | 0.015 | 0.014 | 0.016 | 0.015 | 0.014 | 0.014 |

Chosen model: **nugget + Matérn-3/2, measured marginal, white nugget**. Parameters (σ in m, scales in s; the last column is the RMS of the time derivative of a Gaussian process with the same σ and ρ, in m/s, infinite when the nugget is white or the kernel exponential):

| Camera | Part | Axis | σ | w_n | ℓ | w_s | T | RMS d/dt |
|---|---|---|---|---|---|---|---|---|
| none | common | x | 0.542 | 0.32 | 0.20 | 0.58 | 1.4 | ∞ |
| none | common | y | 0.580 | 0.32 | 0.38 | 0.46 | 1.6 | ∞ |
| none | individual | x | 0.535 | 0.41 | 0.20 | 0.41 | 1.5 | ∞ |
| none | individual | y | 0.652 | 0.48 | 0.20 | 0.33 | 1.7 | ∞ |
| ptz | common | x | 0.375 | 0.08 | 0.39 | 0.65 | 1.8 | ∞ |
| ptz | common | y | 0.476 | 0.05 | 0.47 | 0.55 | 1.8 | ∞ |
| ptz | individual | x | 0.326 | 0.20 | 0.20 | 0.57 | 1.4 | ∞ |
| ptz | individual | y | 0.396 | 0.26 | 0.20 | 0.48 | 1.8 | ∞ |

## 2. Simulation against measurement: the KPI proxy at 5 fps

The SN-GSR proxy compares predfoot's positions with the smoothed labels; on IDSSE the same code compares clean tracks plus simulated errors with the clean tracks, on runs drawn from the measured run lengths. At 5 fps a model is only seen through its correlations at multiples of 0.2 s and its marginal, so this check tests the fitted lags, the marginal and the transfer from SN-GSR clips to IDSSE matches, **not** the 25 Hz shape: models that differ only above 2.5 Hz give the same proxy.

| Reference tracks | Mean speed (m/s) | Share of distance ≥ 5.56 m/s | Runs | Hours |
|---|---|---|---|---|
| SN-GSR smoothed labels (measured) | 2.29 | 13.9 % | 5433 | 13.2 |
| IDSSE clean, all frames | 1.82 | 6.9 % | 14160 | 34.9 |
| IDSSE clean, ball in play | 2.37 | 8.9 % | 9858 | 20.2 |

Simulated values are *all frames / ball in play* (speed error: RMS / p90 of |error|, all frames · ball in play):

| Camera and model | Distance bias | HSR-distance bias | Speed error (m/s) |
|---|---|---|---|
| **none: measured on SN-GSR** | +27.1 % | +67.9 % | 1.50 / 1.54 |
| none: single OU through 0.2 s (previous) | +99.0 % / +64.4 % | +235.4 % / +207.3 % | 1.77 / 2.68 · 1.75 / 2.62 |
| none: nugget + Matérn-3/2, Gaussian, white nugget | +81.6 % / +52.4 % | +146.3 % / +131.7 % | 1.48 / 2.18 · 1.47 / 2.15 |
| none: nugget + exponential, measured marginal, white nugget | +75.7 % / +49.0 % | +167.3 % / +146.4 % | 1.52 / 2.18 · 1.52 / 2.14 |
| none: nugget + Matérn-3/2, measured marginal, white nugget | +75.2 % / +48.4 % | +168.2 % / +144.5 % | 1.51 / 2.18 · 1.50 / 2.12 |
| none: nugget + squared exponential, measured marginal, white nugget | +75.7 % / +48.9 % | +166.4 % / +142.6 % | 1.51 / 2.18 · 1.50 / 2.14 |
| **ptz: measured on SN-GSR** | +3.9 % | +9.5 % | 0.79 / 0.88 |
| ptz: single OU through 0.2 s (previous) | +37.3 % / +21.9 % | +49.9 % / +48.0 % | 0.93 / 1.45 · 0.92 / 1.43 |
| ptz: nugget + Matérn-3/2, Gaussian, white nugget | +33.2 % / +19.3 % | +40.1 % / +38.9 % | 0.85 / 1.32 · 0.85 / 1.32 |
| ptz: nugget + exponential, measured marginal, white nugget | +31.5 % / +18.6 % | +46.3 % / +42.5 % | 0.87 / 1.28 · 0.86 / 1.25 |
| ptz: nugget + Matérn-3/2, measured marginal, white nugget | +32.0 % / +18.5 % | +47.2 % / +42.4 % | 0.88 / 1.31 · 0.86 / 1.27 |
| ptz: nugget + squared exponential, measured marginal, white nugget | +32.0 % / +19.1 % | +49.8 % / +46.1 % | 0.89 / 1.30 · 0.87 / 1.29 |

What the simulation does not reproduce: the SN-GSR reference is the smoothed labels, which carry their own smoothing error, while IDSSE's is the clean tracking; real errors are heteroscedastic (larger far from the camera, along the viewing direction) and can correlate with motion (fast pans during fast play), while the simulated ones are stationary and independent of motion; and the two data sets move differently (table above). Nothing was tuned on the proxy: models are fitted to the ACF and the marginal only.

## 3. Budget at the export rate (10 fps pipeline)

Errors are sampled at 25 Hz and the 10 fps pipeline runs as for exported tracking. The nugget (0.05 to 0.48 of the variance) is invisible at 5 fps; two shapes bound what it can do at 25 Hz: **white** (independent frames, power spread to 12.5 Hz, mostly removed by the 10 fps smoothing after aliasing) and **band-limited** (flat spectrum up to 2.5 Hz: the smoothest process still uncorrelated at 0.2 s, with all its power where the 1 s smoothing lets speed noise through). The kernels bracket the resolved part between its measured lags, from exponential (rough) to squared exponential (smooth).

### Chosen model (nugget + Matérn-3/2, measured marginal, white nugget) against the clean 25 Hz reference

| Camera | KPI | Ref. mean | MAPE % [95 % CI] | Bias % [95 % CI] | Spearman [95 % CI] |
|---|---|---|---|---|---|
| none | Distance | 9646.8 | 46.11 [44.15, 48.77] | +45.95 [+43.96, +48.57] | 0.985 [0.976, 0.988] |
| none | HI dist. ≥ 20 | 735.9 | 253.85 [238.32, 268.41] | +226.65 [+217.75, +235.79] | 0.779 [0.726, 0.824] |
| none | Dist. > 25 | 180.1 | 587.10 [527.96, 651.22] | +416.50 [+388.60, +445.92] | 0.663 [0.620, 0.719] |
| none | HSR efforts | 38.3 | 17.32 [15.62, 18.75] | -15.95 [-17.52, -14.21] | 0.908 [0.870, 0.927] |
| none | Sprint efforts | 8.8 | 38.31 [36.50, 40.43] | -32.16 [-35.29, -28.52] | 0.768 [0.679, 0.832] |
| none | PSV-99 | 28.5 | 23.69 [22.67, 24.83] | +23.42 [+22.39, +24.55] | 0.715 [0.618, 0.792] |
| ptz | Distance | 9646.8 | 22.74 [21.76, 24.15] | +22.66 [+21.70, +24.05] | 0.993 [0.987, 0.995] |
| ptz | HI dist. ≥ 20 | 735.9 | 83.85 [78.50, 89.42] | +75.38 [+72.32, +78.88] | 0.918 [0.887, 0.941] |
| ptz | Dist. > 25 | 180.1 | 166.90 [151.36, 185.53] | +122.15 [+112.65, +132.53] | 0.879 [0.849, 0.910] |
| ptz | HSR efforts | 38.3 | 9.27 [7.88, 10.41] | -4.29 [-5.85, -2.54] | 0.944 [0.893, 0.968] |
| ptz | Sprint efforts | 8.8 | 23.52 [20.71, 26.48] | -16.64 [-19.53, -13.91] | 0.898 [0.858, 0.926] |
| ptz | PSV-99 | 28.5 | 11.64 [10.69, 12.72] | +11.48 [+10.54, +12.56] | 0.794 [0.705, 0.860] |

### Sensitivity: bias % of each KPI under every model

Ghost: distance per minute of a motionless player through the 10 fps pipeline (one player's full error).

| Model | Ghost (m/min) | Distance | HI dist. ≥ 20 | Dist. > 25 | HSR efforts | Sprint efforts | PSV-99 |
|---|---|---|---|---|---|---|---|
| clean 10 fps (control) | – | -0.1 | -0.2 | -0.2 | -0.5 | +0.0 | -0.1 |
| none: single OU through 0.2 s (previous) | 155.0 | +74.1 | +443.2 | +806.3 | +3.9 | -10.6 | +34.1 |
| none: nugget + Matérn-3/2, Gaussian, white nugget | 117.6 | +48.7 | +198.8 | +327.3 | -14.0 | -27.6 | +21.2 |
| none: nugget + exponential, measured marginal, white nugget | 113.0 | +47.0 | +235.3 | +436.5 | -14.7 | -30.9 | +24.1 |
| none: nugget + exponential, measured marginal, bandlimited nugget | 127.9 | +56.8 | +347.5 | +721.8 | -15.7 | -30.2 | +34.5 |
| none: nugget + Matérn-3/2, measured marginal, white nugget | 110.6 | +45.9 | +226.6 | +416.5 | -16.0 | -32.2 | +23.4 |
| none: nugget + Matérn-3/2, measured marginal, bandlimited nugget | 127.8 | +58.2 | +364.1 | +768.5 | -16.4 | -31.2 | +36.5 |
| none: nugget + Matérn-5/2, measured marginal, white nugget | 111.8 | +45.8 | +222.6 | +408.1 | -15.7 | -29.4 | +22.7 |
| none: nugget + Matérn-5/2, measured marginal, bandlimited nugget | 127.2 | +58.1 | +367.9 | +786.9 | -16.1 | -27.3 | +36.9 |
| none: nugget + squared exponential, measured marginal, white nugget | 108.8 | +45.4 | +219.9 | +401.7 | -15.1 | -33.4 | +22.4 |
| none: nugget + squared exponential, measured marginal, bandlimited nugget | 125.5 | +58.2 | +368.0 | +783.3 | -17.5 | -31.4 | +36.8 |
| ptz: single OU through 0.2 s (previous) | 84.0 | +28.6 | +83.8 | +127.6 | -0.1 | -11.4 | +12.8 |
| ptz: nugget + Matérn-3/2, Gaussian, white nugget | 73.8 | +22.9 | +60.6 | +90.7 | -4.1 | -9.8 | +10.1 |
| ptz: nugget + exponential, measured marginal, white nugget | 70.5 | +22.8 | +76.8 | +125.1 | -4.7 | -9.8 | +11.8 |
| ptz: nugget + exponential, measured marginal, bandlimited nugget | 72.6 | +24.1 | +84.4 | +139.9 | -6.3 | -16.5 | +13.4 |
| ptz: nugget + Matérn-3/2, measured marginal, white nugget | 69.3 | +22.7 | +75.4 | +122.1 | -4.3 | -16.6 | +11.5 |
| ptz: nugget + Matérn-3/2, measured marginal, bandlimited nugget | 75.7 | +25.1 | +91.5 | +151.9 | -6.6 | -15.8 | +14.1 |
| ptz: nugget + Matérn-5/2, measured marginal, white nugget | 70.5 | +22.3 | +74.1 | +122.2 | -5.4 | -11.0 | +12.3 |
| ptz: nugget + Matérn-5/2, measured marginal, bandlimited nugget | 75.2 | +25.4 | +95.0 | +161.8 | -7.2 | -16.3 | +15.0 |
| ptz: nugget + squared exponential, measured marginal, white nugget | 67.9 | +22.0 | +71.2 | +114.9 | -4.2 | -15.4 | +11.2 |
| ptz: nugget + squared exponential, measured marginal, bandlimited nugget | 75.7 | +25.4 | +93.9 | +159.2 | -7.4 | -15.0 | +15.0 |

Sensitivity bound: the range of the bias over the four kernels and both nugget shapes (same marginal as the chosen model), next to the Gaussian Matérn-3/2 and the previous single OU:

| Camera | KPI | Chosen | Range, kernels × nuggets | Gaussian Matérn-3/2 | Single OU |
|---|---|---|---|---|---|
| none | Distance | +45.9 | +45.4 to +58.2 | +48.7 | +74.1 |
| none | HI dist. ≥ 20 | +226.6 | +219.9 to +368.0 | +198.8 | +443.2 |
| none | Dist. > 25 | +416.5 | +401.7 to +786.9 | +327.3 | +806.3 |
| none | HSR efforts | -16.0 | -17.5 to -14.7 | -14.0 | +3.9 |
| none | Sprint efforts | -32.2 | -33.4 to -27.3 | -27.6 | -10.6 |
| none | PSV-99 | +23.4 | +22.4 to +36.9 | +21.2 | +34.1 |
| ptz | Distance | +22.7 | +22.0 to +25.4 | +22.9 | +28.6 |
| ptz | HI dist. ≥ 20 | +75.4 | +71.2 to +95.0 | +60.6 | +83.8 |
| ptz | Dist. > 25 | +122.1 | +114.9 to +161.8 | +90.7 | +127.6 |
| ptz | HSR efforts | -4.3 | -7.4 to -4.2 | -4.1 | -0.1 |
| ptz | Sprint efforts | -16.6 | -16.6 to -9.8 | -9.8 | -11.4 |
| ptz | PSV-99 | +11.5 | +11.2 to +15.0 | +10.1 | +12.8 |

All KPIs, chosen model, MAPE % / bias % against the clean 25 Hz reference:

| Model | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 | Dist. > 25 | HI dist. ≥ 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 46.1 / +45.9 | 18.3 / +18.2 | 107.9 / +99.5 | 184.7 / +165.1 | 587.1 / +416.5 | 253.9 / +226.6 | 17.3 / -16.0 | 38.3 / -32.2 | 23.7 / +23.4 | 61.6 / -49.0 | 80.5 / +0.7 |
| ptz | 22.7 / +22.7 | 13.7 / +13.7 | 46.3 / +43.0 | 67.2 / +60.2 | 166.9 / +122.1 | 83.9 / +75.4 | 9.3 / -4.3 | 23.5 / -16.6 | 11.6 / +11.5 | 59.4 / -35.6 | 57.0 / -19.1 |
| clean 10 fps | 0.1 / -0.1 | 0.0 / -0.0 | 0.2 / -0.1 | 0.4 / -0.2 | 0.8 / -0.2 | 0.3 / -0.2 | 0.7 / -0.5 | 0.9 / +0.0 | 0.1 / -0.1 | 12.3 / +2.2 | 13.2 / +8.6 |
