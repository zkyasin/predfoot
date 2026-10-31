# KPI error budget with a spectral error model (T4.3)

> **Simulation, not measurement.** Every KPI error below comes from errors *simulated* by a model
> fitted to predfoot's error process on SN-GSR dev (5 fps), injected into clean IDSSE tracking. The
> only measured KPI numbers are the SN-GSR proxy values quoted in section 2.

- **Command**: `uv run python scripts/kpi_spectral.py `
- **Git commit**: `b30e1e2`; date 2026-10-31T18:57:13; runtime 244.1 s
- **Error process**: `reports/error_process_temporal_dev.json` (from `uv run python scripts/error_process.py runs/20261031-153059_c3861c7_ptz/players.parquet --compare runs/20261031-152537_c3861c7_none/players.parquet --name ptz --compare-name none`), errors against the smoothed labels, frames one in five (5 fps), per axis (x along the touchlines, y across) and part.
- **IDSSE**: proxy check on J03WMX; budget on 7 matches (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), 142 player-matches (outfield, ≥ 45 min). IDSSE: Bassek, Rein, Weber & Memmert 2025, CC BY 4.0.
- **Seed** 20261031, 1000 bootstrap resamples over matches; one noise draw per condition and match.

## 1. Error model

Per axis and part, a stationary process with autocorrelation ρ(τ) = w_n·ν(τ) + w_m·k(τ/ℓ) + w_s·k(τ/T) (`predfoot.kpi.noise`): a nugget ν plus a main and a slow scale of one kernel k, fitted by weighted least squares to the five measured lags (0.2-1.0 s). Scales are kept at or above the 0.2 s sampling interval: anything faster is not resolved by the measurement and is carried by the nugget, whose shape at 25 Hz is an assumption (section 3). Two marginals: **Gaussian** with the measured σ, or the **measured marginal** through a Gaussian copula, ρ being then fitted to the ACF of the normal scores. The measured errors are heavy-tailed: robust σ (IQR / 1.349) = 0.45 to 0.68 of the SD and kurtosis 11 to 34 on these series, so a Gaussian with the measured σ makes the typical error 1.5 to 2.2 times too large.

Largest |ACF residual| over the five lags (model minus measurement; normal-score ACF for the copula fits):

| Camera | Part | Axis | single OU through 0.2 s (previous) | nugget + exponential (gauss) | nugget + Matérn-3/2 (gauss) | nugget + Matérn-5/2 (gauss) | nugget + squared exponential (gauss) | nugget + exponential (copula) | nugget + Matérn-3/2 (copula) | nugget + Matérn-5/2 (copula) | nugget + squared exponential (copula) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | common | x | 0.291 | 0.017 | 0.016 | 0.015 | 0.014 | 0.005 | 0.005 | 0.005 | 0.004 |
| none | common | y | 0.272 | 0.006 | 0.006 | 0.006 | 0.005 | 0.006 | 0.004 | 0.004 | 0.003 |
| none | individual | x | 0.343 | 0.008 | 0.007 | 0.006 | 0.006 | 0.007 | 0.006 | 0.005 | 0.003 |
| none | individual | y | 0.298 | 0.006 | 0.004 | 0.004 | 0.003 | 0.006 | 0.004 | 0.003 | 0.003 |
| ptz | common | x | 0.045 | 0.005 | 0.001 | 0.000 | 0.001 | 0.003 | 0.001 | 0.001 | 0.002 |
| ptz | common | y | 0.079 | 0.034 | 0.001 | 0.002 | 0.003 | 0.024 | 0.001 | 0.002 | 0.003 |
| ptz | individual | x | 0.208 | 0.005 | 0.004 | 0.005 | 0.006 | 0.008 | 0.006 | 0.005 | 0.005 |
| ptz | individual | y | 0.294 | 0.010 | 0.009 | 0.008 | 0.007 | 0.012 | 0.010 | 0.009 | 0.008 |

Chosen model: **nugget + Matérn-3/2, measured marginal, white nugget**. The kernel is a prior choice, not a fit result: the smooth kernels fit the five lags about as well as each other (table above), and Matérn-3/2 is the roughest of them with differentiable paths, as expected of the error of a smoothed camera with inertia. Parameters (σ in m, scales in s; the last column is the RMS of the time derivative of a Gaussian process with the same σ and ρ, in m/s, infinite when the nugget is white or the kernel exponential):

| Camera | Part | Axis | σ | w_n | ℓ | w_s | T | RMS d/dt |
|---|---|---|---|---|---|---|---|---|
| none | common | x | 0.542 | 0.27 | 0.33 | 0.62 | 1.8 | ∞ |
| none | common | y | 0.580 | 0.27 | 0.52 | 0.43 | 2.6 | ∞ |
| none | individual | x | 0.535 | 0.35 | 0.20 | 0.51 | 1.9 | ∞ |
| none | individual | y | 0.652 | 0.43 | 0.20 | 0.41 | 1.9 | ∞ |
| ptz | common | x | 0.375 | 0.05 | 0.49 | 0.66 | 2.4 | ∞ |
| ptz | common | y | 0.476 | 0.03 | 0.54 | 0.59 | 2.4 | ∞ |
| ptz | individual | x | 0.326 | 0.13 | 0.20 | 0.68 | 1.6 | ∞ |
| ptz | individual | y | 0.396 | 0.19 | 0.20 | 0.61 | 1.8 | ∞ |

Round trip at 5 fps: RMS and median |step| of the 0.2 s error steps (m), measured on SN-GSR dev and in 10 min of samples of the fitted processes (25 Hz, one sample in five): the Gaussian nugget + Matérn-3/2 and the chosen model. The ACF is pooled over series (`error_process.py`), so the Gaussian fit reproduces the measured step RMS up to its ACF residual; the copula fit matches the normal scores instead, so its step RMS can differ.

| Camera | Part | Axis | Step RMS: measured | Gaussian | chosen | Median |step|: measured | Gaussian | chosen |
|---|---|---|---|---|---|---|---|---|
| none | common | x | 0.441 | 0.443 | 0.481 | 0.109 | 0.298 | 0.185 |
| none | common | y | 0.458 | 0.458 | 0.485 | 0.179 | 0.309 | 0.256 |
| none | individual | x | 0.496 | 0.492 | 0.540 | 0.126 | 0.333 | 0.204 |
| none | individual | y | 0.674 | 0.673 | 0.688 | 0.199 | 0.455 | 0.291 |
| ptz | common | x | 0.179 | 0.180 | 0.201 | 0.053 | 0.122 | 0.079 |
| ptz | common | y | 0.190 | 0.191 | 0.225 | 0.080 | 0.129 | 0.112 |
| ptz | individual | x | 0.210 | 0.210 | 0.252 | 0.062 | 0.143 | 0.098 |
| ptz | individual | y | 0.307 | 0.307 | 0.329 | 0.101 | 0.206 | 0.151 |

Measured / chosen model: step RMS 0.83 to 0.98, median |step| 0.59 to 0.72. A median ratio below the RMS ratio means real steps are intermittent: mostly smaller than the model's, with rarer large jumps.

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
| none: single OU through 0.2 s (previous) | +89.0 % / +57.3 % | +207.4 % / +184.9 % | 1.68 / 2.57 · 1.65 / 2.50 |
| none: nugget + Matérn-3/2, Gaussian, white nugget | +68.7 % / +43.4 % | +109.2 % / +100.5 % | 1.31 / 1.95 · 1.31 / 1.93 |
| none: nugget + exponential, measured marginal, white nugget | +68.4 % / +44.0 % | +145.0 % / +128.3 % | 1.42 / 2.03 · 1.42 / 2.00 |
| none: nugget + Matérn-3/2, measured marginal, white nugget | +68.0 % / +43.4 % | +145.3 % / +126.2 % | 1.42 / 2.03 · 1.41 / 1.98 |
| none: nugget + squared exponential, measured marginal, white nugget | +68.4 % / +43.9 % | +144.3 % / +124.0 % | 1.42 / 2.03 · 1.41 / 2.00 |
| **ptz: measured on SN-GSR** | +3.9 % | +9.5 % | 0.79 / 0.88 |
| ptz: single OU through 0.2 s (previous) | +27.5 % / +15.6 % | +34.8 % / +33.7 % | 0.79 / 1.24 · 0.78 / 1.22 |
| ptz: nugget + Matérn-3/2, Gaussian, white nugget | +24.0 % / +13.5 % | +26.5 % / +25.8 % | 0.71 / 1.12 · 0.71 / 1.11 |
| ptz: nugget + exponential, measured marginal, white nugget | +26.0 % / +15.1 % | +35.8 % / +33.4 % | 0.78 / 1.15 · 0.78 / 1.13 |
| ptz: nugget + Matérn-3/2, measured marginal, white nugget | +26.5 % / +15.0 % | +37.6 % / +34.0 % | 0.80 / 1.18 · 0.78 / 1.14 |
| ptz: nugget + squared exponential, measured marginal, white nugget | +26.4 % / +15.6 % | +39.7 % / +37.4 % | 0.80 / 1.17 · 0.79 / 1.17 |

**Verdict.** The check passes when the chosen model's ball-in-play distance and HSR-distance biases are both within a factor of 2 of the measured ones.

| Camera | Distance bias: simulated / measured | HSR-distance bias: simulated / measured | Check |
|---|---|---|---|
| none | ×1.6 | ×1.9 | passes |
| ptz | ×3.9 | ×3.6 | **fails** |

Every simulated bias exceeds the measured one (×1.6 to ×3.9): the model overstates the 5 fps KPI cost even where the check passes.

What the simulation does not reproduce: the SN-GSR reference is the smoothed labels, which carry their own smoothing error, while IDSSE's is the clean tracking; real errors are heteroscedastic (larger far from the camera, along the viewing direction) and can correlate with motion (fast pans during fast play), while the simulated ones are stationary and independent of motion; and the two data sets move differently (table above). Nothing was tuned on the proxy: models are fitted to the ACF and the marginal only.

## 3. Budget at the export rate (10 fps pipeline)

> **Not predfoot's KPI error.** The check of section 2 fails for the ptz camera: the chosen model's simulated 5 fps proxy biases are several times the measured ones (ptz: distance ×3.9, HSR distance ×3.6, ball in play). For the none camera the check passes, but the simulated biases are still above the measured ones (none: distance ×1.6, HSR distance ×1.9). The numbers below are the budget of the fitted stationary error model; they overstate the KPI error of predfoot's tracking and must not be quoted as such.

Errors are sampled at 25 Hz and the 10 fps pipeline runs as for exported tracking. The nugget (0.03 to 0.43 of the variance) is invisible at 5 fps; two shapes bound what it can do at 25 Hz: **white** (independent frames, power spread to 12.5 Hz, mostly removed by the 10 fps smoothing after aliasing) and **band-limited** (flat spectrum up to 2.5 Hz: the smoothest process still uncorrelated at 0.2 s, with all its power where the 1 s smoothing lets speed noise through). The kernels bracket the resolved part between its measured lags, from exponential (rough) to squared exponential (smooth).

### Chosen model (nugget + Matérn-3/2, measured marginal, white nugget) against the clean 25 Hz reference

| Camera | KPI | Ref. mean | MAPE % [95 % CI] | Bias % [95 % CI] | Spearman [95 % CI] |
|---|---|---|---|---|---|
| none | Distance | 9646.8 | 41.28 [39.52, 43.66] | +41.14 [+39.35, +43.47] | 0.985 [0.976, 0.989] |
| none | HI dist. ≥ 20 | 735.9 | 215.44 [202.83, 228.06] | +192.71 [+184.92, +200.41] | 0.808 [0.764, 0.848] |
| none | Dist. > 25 | 180.1 | 493.46 [443.76, 546.92] | +350.91 [+327.23, +376.37] | 0.705 [0.674, 0.747] |
| none | HSR efforts | 38.3 | 16.97 [15.37, 18.59] | -14.79 [-16.60, -12.96] | 0.911 [0.874, 0.935] |
| none | Sprint efforts | 8.8 | 37.17 [35.20, 39.25] | -28.24 [-32.45, -23.96] | 0.776 [0.686, 0.843] |
| none | PSV-99 | 28.5 | 21.39 [20.45, 22.51] | +21.14 [+20.20, +22.25] | 0.740 [0.671, 0.804] |
| ptz | Distance | 9646.8 | 19.18 [18.40, 20.37] | +19.11 [+18.33, +20.29] | 0.994 [0.989, 0.995] |
| ptz | HI dist. ≥ 20 | 735.9 | 66.48 [62.09, 71.17] | +59.87 [+57.16, +62.87] | 0.936 [0.915, 0.950] |
| ptz | Dist. > 25 | 180.1 | 129.71 [117.23, 144.31] | +95.36 [+87.51, +104.18] | 0.910 [0.890, 0.930] |
| ptz | HSR efforts | 38.3 | 8.52 [7.65, 9.59] | -2.23 [-3.85, -0.71] | 0.947 [0.896, 0.969] |
| ptz | Sprint efforts | 8.8 | 21.20 [19.19, 23.65] | -13.12 [-15.67, -10.99] | 0.904 [0.879, 0.920] |
| ptz | PSV-99 | 28.5 | 9.70 [8.88, 10.65] | +9.57 [+8.76, +10.49] | 0.822 [0.747, 0.883] |

### Sensitivity: bias % of each KPI under every model

Ghost: distance per minute of a motionless player through the 10 fps pipeline (one player's full error).

| Model | Ghost (m/min) | Distance | HI dist. ≥ 20 | Dist. > 25 | HSR efforts | Sprint efforts | PSV-99 |
|---|---|---|---|---|---|---|---|
| clean 10 fps (control) | – | -0.1 | -0.2 | -0.2 | -0.5 | +0.0 | -0.1 |
| none: single OU through 0.2 s (previous) | 146.4 | +68.1 | +375.0 | +662.0 | +13.6 | -0.8 | +31.1 |
| none: nugget + Matérn-3/2, Gaussian, white nugget | 104.5 | +40.5 | +144.9 | +230.9 | -12.9 | -25.5 | +17.3 |
| none: nugget + exponential, measured marginal, white nugget | 105.1 | +42.2 | +201.2 | +368.9 | -13.7 | -29.1 | +21.8 |
| none: nugget + exponential, measured marginal, bandlimited nugget | 119.1 | +51.3 | +298.1 | +609.7 | -15.3 | -28.0 | +31.5 |
| none: nugget + Matérn-3/2, measured marginal, white nugget | 102.7 | +41.1 | +192.7 | +350.9 | -14.8 | -28.2 | +21.1 |
| none: nugget + Matérn-3/2, measured marginal, bandlimited nugget | 119.2 | +52.5 | +311.2 | +645.8 | -16.3 | -29.6 | +33.2 |
| none: nugget + Matérn-5/2, measured marginal, white nugget | 104.1 | +41.0 | +190.0 | +344.8 | -14.6 | -28.2 | +20.5 |
| none: nugget + Matérn-5/2, measured marginal, bandlimited nugget | 118.5 | +52.5 | +315.1 | +663.8 | -16.4 | -26.6 | +33.7 |
| none: nugget + squared exponential, measured marginal, white nugget | 100.7 | +40.8 | +188.5 | +341.0 | -14.3 | -30.6 | +20.2 |
| none: nugget + squared exponential, measured marginal, bandlimited nugget | 116.6 | +52.6 | +315.2 | +658.8 | -16.8 | -30.3 | +33.6 |
| ptz: single OU through 0.2 s (previous) | 70.6 | +21.5 | +55.1 | +82.1 | +1.4 | -6.4 | +9.0 |
| ptz: nugget + Matérn-3/2, Gaussian, white nugget | 61.6 | +16.9 | +39.7 | +58.9 | -1.2 | -5.2 | +7.2 |
| ptz: nugget + exponential, measured marginal, white nugget | 63.0 | +19.2 | +60.6 | +96.9 | -3.1 | -6.1 | +9.9 |
| ptz: nugget + exponential, measured marginal, bandlimited nugget | 64.7 | +20.1 | +65.2 | +106.2 | -4.0 | -12.9 | +11.1 |
| ptz: nugget + Matérn-3/2, measured marginal, white nugget | 62.3 | +19.1 | +59.9 | +95.4 | -2.2 | -13.1 | +9.6 |
| ptz: nugget + Matérn-3/2, measured marginal, bandlimited nugget | 67.5 | +20.9 | +70.5 | +114.3 | -4.4 | -13.5 | +11.4 |
| ptz: nugget + Matérn-5/2, measured marginal, white nugget | 63.1 | +18.8 | +59.0 | +96.3 | -2.7 | -7.4 | +10.4 |
| ptz: nugget + Matérn-5/2, measured marginal, bandlimited nugget | 66.8 | +21.1 | +73.6 | +123.3 | -4.9 | -13.8 | +12.3 |
| ptz: nugget + squared exponential, measured marginal, white nugget | 60.7 | +18.5 | +56.7 | +89.9 | -3.2 | -11.4 | +9.4 |
| ptz: nugget + squared exponential, measured marginal, bandlimited nugget | 67.2 | +21.1 | +72.1 | +120.1 | -5.6 | -12.5 | +12.2 |

Sensitivity bound: the range of the bias over the four kernels and both nugget shapes (same marginal as the chosen model), next to the Gaussian Matérn-3/2 and the previous single OU:

| Camera | KPI | Chosen | Range, kernels × nuggets | Gaussian Matérn-3/2 | Single OU |
|---|---|---|---|---|---|
| none | Distance | +41.1 | +40.8 to +52.6 | +40.5 | +68.1 |
| none | HI dist. ≥ 20 | +192.7 | +188.5 to +315.2 | +144.9 | +375.0 |
| none | Dist. > 25 | +350.9 | +341.0 to +663.8 | +230.9 | +662.0 |
| none | HSR efforts | -14.8 | -16.8 to -13.7 | -12.9 | +13.6 |
| none | Sprint efforts | -28.2 | -30.6 to -26.6 | -25.5 | -0.8 |
| none | PSV-99 | +21.1 | +20.2 to +33.7 | +17.3 | +31.1 |
| ptz | Distance | +19.1 | +18.5 to +21.1 | +16.9 | +21.5 |
| ptz | HI dist. ≥ 20 | +59.9 | +56.7 to +73.6 | +39.7 | +55.1 |
| ptz | Dist. > 25 | +95.4 | +89.9 to +123.3 | +58.9 | +82.1 |
| ptz | HSR efforts | -2.2 | -5.6 to -2.2 | -1.2 | +1.4 |
| ptz | Sprint efforts | -13.1 | -13.8 to -6.1 | -5.2 | -6.4 |
| ptz | PSV-99 | +9.6 | +9.4 to +12.3 | +7.2 | +9.0 |

All KPIs, chosen model, MAPE % / bias % against the clean 25 Hz reference:

| Model | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 | Dist. > 25 | HI dist. ≥ 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 41.3 / +41.1 | 17.8 / +17.7 | 94.6 / +87.3 | 157.8 / +141.5 | 493.5 / +350.9 | 215.4 / +192.7 | 17.0 / -14.8 | 37.2 / -28.2 | 21.4 / +21.1 | 64.4 / -52.6 | 78.6 / -16.2 |
| ptz | 19.2 / +19.1 | 12.1 / +12.1 | 38.1 / +35.5 | 53.8 / +48.4 | 129.7 / +95.4 | 66.5 / +59.9 | 8.5 / -2.2 | 21.2 / -13.1 | 9.7 / +9.6 | 56.8 / -37.7 | 60.9 / -32.4 |
| clean 10 fps | 0.1 / -0.1 | 0.0 / -0.0 | 0.2 / -0.1 | 0.4 / -0.2 | 0.8 / -0.2 | 0.3 / -0.2 | 0.7 / -0.5 | 0.9 / +0.0 | 0.1 / -0.1 | 12.3 / +2.2 | 13.2 / +8.6 |
