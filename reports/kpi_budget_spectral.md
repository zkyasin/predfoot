# KPI error budget with a spectral error model

> **Simulation, not measurement.** Every KPI error below comes from errors *simulated* by a model
> fitted to predfoot's error process on SN-GSR dev (5 fps), injected into clean IDSSE tracking. The
> only measured KPI numbers are the SN-GSR proxy values quoted in section 2.

- **Command**: `uv run python scripts/kpi_spectral.py `
- **Git commit**: `1bb22eb`; date 2026-10-31T19:10:25; runtime 239.4 s
- **Error process**: `reports/error_process_temporal_dev.json` (from `uv run python scripts/error_process.py runs/20261031-163641_bf28b35_ptz/players.parquet --compare runs/20261031-152537_c3861c7_none/players.parquet --name ptz --compare-name none`), errors against the smoothed labels, frames one in five (5 fps), per axis (x along the touchlines, y across) and part.
- **IDSSE**: proxy check on J03WMX; budget on 7 matches (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), 142 player-matches (outfield, ≥ 45 min). IDSSE: Bassek, Rein, Weber & Memmert 2025, CC BY 4.0.
- **Seed** 20261031, 1000 bootstrap resamples over matches; one noise draw per condition and match.

## 1. Error model

Per axis and part, a stationary process with autocorrelation ρ(τ) = w_n·ν(τ) + w_m·k(τ/ℓ) + w_s·k(τ/T) (`predfoot.kpi.noise`): a nugget ν plus a main and a slow scale of one kernel k, fitted by weighted least squares to the five measured lags (0.2-1.0 s). Scales are kept at or above the 0.2 s sampling interval: anything faster is not resolved by the measurement and is carried by the nugget, whose shape at 25 Hz is an assumption (section 3). Two marginals: **Gaussian** with the measured σ, or the **measured marginal** through a Gaussian copula, ρ being then fitted to the ACF of the normal scores. The measured errors are heavy-tailed: robust σ (IQR / 1.349) = 0.45 to 0.68 of the SD and kurtosis 11 to 31 on these series, so a Gaussian with the measured σ makes the typical error 1.5 to 2.2 times too large.

Largest |ACF residual| over the five lags (model minus measurement; normal-score ACF for the copula fits):

| Camera | Part | Axis | single OU through 0.2 s (previous) | nugget + exponential (gauss) | nugget + Matérn-3/2 (gauss) | nugget + Matérn-5/2 (gauss) | nugget + squared exponential (gauss) | nugget + exponential (copula) | nugget + Matérn-3/2 (copula) | nugget + Matérn-5/2 (copula) | nugget + squared exponential (copula) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | common | x | 0.294 | 0.015 | 0.013 | 0.013 | 0.011 | 0.005 | 0.004 | 0.004 | 0.003 |
| none | common | y | 0.273 | 0.006 | 0.005 | 0.005 | 0.004 | 0.006 | 0.004 | 0.004 | 0.003 |
| none | individual | x | 0.343 | 0.007 | 0.006 | 0.006 | 0.005 | 0.007 | 0.005 | 0.004 | 0.003 |
| none | individual | y | 0.297 | 0.005 | 0.003 | 0.002 | 0.002 | 0.006 | 0.004 | 0.003 | 0.002 |
| ptz | common | x | 0.107 | 0.046 | 0.000 | 0.001 | 0.002 | 0.012 | 0.000 | 0.001 | 0.002 |
| ptz | common | y | 0.128 | 0.059 | 0.001 | 0.001 | 0.002 | 0.041 | 0.001 | 0.001 | 0.002 |
| ptz | individual | x | 0.126 | 0.005 | 0.004 | 0.004 | 0.006 | 0.008 | 0.006 | 0.005 | 0.005 |
| ptz | individual | y | 0.245 | 0.010 | 0.008 | 0.007 | 0.006 | 0.012 | 0.010 | 0.009 | 0.008 |

Chosen model: **nugget + Matérn-3/2, measured marginal, white nugget**. The kernel is a prior choice, not a fit result: the smooth kernels fit the five lags about as well as each other (table above), and Matérn-3/2 is the roughest of them with differentiable paths, as expected of the error of a smoothed camera with inertia. Parameters (σ in m, scales in s; the last column is the RMS of the time derivative of a Gaussian process with the same σ and ρ, in m/s, infinite when the nugget is white or the kernel exponential):

| Camera | Part | Axis | σ | w_n | ℓ | w_s | T | RMS d/dt |
|---|---|---|---|---|---|---|---|---|
| none | common | x | 0.540 | 0.27 | 0.37 | 0.60 | 2.0 | ∞ |
| none | common | y | 0.578 | 0.27 | 0.52 | 0.43 | 2.8 | ∞ |
| none | individual | x | 0.534 | 0.35 | 0.20 | 0.51 | 1.9 | ∞ |
| none | individual | y | 0.650 | 0.43 | 0.20 | 0.41 | 1.9 | ∞ |
| ptz | common | x | 0.364 | 0.03 | 0.45 | 0.71 | 2.1 | ∞ |
| ptz | common | y | 0.472 | 0.02 | 0.56 | 0.57 | 2.5 | ∞ |
| ptz | individual | x | 0.318 | 0.10 | 0.20 | 0.70 | 1.6 | ∞ |
| ptz | individual | y | 0.380 | 0.16 | 0.20 | 0.62 | 1.8 | ∞ |

Round trip at 5 fps: RMS and median |step| of the 0.2 s error steps (m), measured on SN-GSR dev and in 10 min of samples of the fitted processes (25 Hz, one sample in five): the Gaussian nugget + Matérn-3/2 and the chosen model. The ACF is pooled over series (`error_process.py`), so the Gaussian fit reproduces the measured step RMS up to its ACF residual; the copula fit matches the normal scores instead, so its step RMS can differ.

| Camera | Part | Axis | Step RMS: measured | Gaussian | chosen | Median |step|: measured | Gaussian | chosen |
|---|---|---|---|---|---|---|---|---|
| none | common | x | 0.435 | 0.437 | 0.475 | 0.109 | 0.294 | 0.184 |
| none | common | y | 0.456 | 0.456 | 0.483 | 0.179 | 0.308 | 0.255 |
| none | individual | x | 0.494 | 0.490 | 0.538 | 0.126 | 0.332 | 0.203 |
| none | individual | y | 0.672 | 0.671 | 0.686 | 0.199 | 0.454 | 0.291 |
| ptz | common | x | 0.131 | 0.132 | 0.179 | 0.051 | 0.089 | 0.071 |
| ptz | common | y | 0.169 | 0.170 | 0.209 | 0.078 | 0.114 | 0.103 |
| ptz | individual | x | 0.181 | 0.181 | 0.235 | 0.061 | 0.123 | 0.093 |
| ptz | individual | y | 0.268 | 0.267 | 0.307 | 0.099 | 0.180 | 0.143 |

Measured / chosen model: step RMS 0.73 to 0.98, median |step| 0.59 to 0.75. A median ratio below the RMS ratio means real steps are intermittent: mostly smaller than the model's, with rarer large jumps.

## 2. Simulation against measurement: the KPI proxy at 5 fps

The SN-GSR proxy compares predfoot's positions with the smoothed labels; on IDSSE the same code compares clean tracks plus simulated errors with the clean tracks, on runs drawn from the measured run lengths. At 5 fps a model is only seen through its correlations at multiples of 0.2 s and its marginal, so this check tests the fitted lags, the marginal and the transfer from SN-GSR clips to IDSSE matches, **not** the 25 Hz shape: models that differ only above 2.5 Hz give the same proxy.

| Reference tracks | Mean speed (m/s) | Share of distance ≥ 5.56 m/s | Runs | Hours |
|---|---|---|---|---|
| SN-GSR smoothed labels (measured) | 2.29 | 13.9 % | 5409 | 13.2 |
| IDSSE clean, all frames | 1.82 | 6.9 % | 14118 | 34.9 |
| IDSSE clean, ball in play | 2.37 | 8.8 % | 9827 | 20.2 |

Simulated values are *all frames / ball in play* (speed error: RMS / p90 of |error|, all frames · ball in play):

| Camera and model | Distance bias | HSR-distance bias | Speed error (m/s) |
|---|---|---|---|
| **none: measured on SN-GSR** | +27.0 % | +67.2 % | 1.49 / 1.54 |
| none: single OU through 0.2 s (previous) | +88.4 % / +56.8 % | +205.4 % / +183.1 % | 1.67 / 2.56 · 1.64 / 2.49 |
| none: nugget + Matérn-3/2, Gaussian, white nugget | +68.6 % / +43.3 % | +109.8 % / +100.7 % | 1.31 / 1.95 · 1.31 / 1.93 |
| none: nugget + exponential, measured marginal, white nugget | +68.0 % / +43.6 % | +142.8 % / +126.4 % | 1.41 / 2.02 · 1.41 / 1.99 |
| none: nugget + Matérn-3/2, measured marginal, white nugget | +67.6 % / +43.1 % | +142.6 % / +124.2 % | 1.41 / 2.02 · 1.39 / 1.97 |
| none: nugget + squared exponential, measured marginal, white nugget | +68.0 % / +43.6 % | +143.7 % / +122.8 % | 1.41 / 2.03 · 1.40 / 1.98 |
| **ptz: measured on SN-GSR** | +2.7 % | +6.2 % | 0.71 / 0.85 |
| ptz: single OU through 0.2 s (previous) | +21.6 % / +11.9 % | +25.3 % / +24.8 % | 0.68 / 1.08 · 0.68 / 1.07 |
| ptz: nugget + Matérn-3/2, Gaussian, white nugget | +21.0 % / +11.6 % | +23.9 % / +23.3 % | 0.67 / 1.06 · 0.67 / 1.06 |
| ptz: nugget + exponential, measured marginal, white nugget | +24.2 % / +13.9 % | +31.5 % / +30.0 % | 0.75 / 1.10 · 0.74 / 1.09 |
| ptz: nugget + Matérn-3/2, measured marginal, white nugget | +24.7 % / +13.9 % | +34.5 % / +31.4 % | 0.77 / 1.14 · 0.75 / 1.11 |
| ptz: nugget + squared exponential, measured marginal, white nugget | +24.6 % / +14.4 % | +36.5 % / +34.3 % | 0.77 / 1.13 · 0.76 / 1.13 |

**Verdict.** The check passes when the chosen model's ball-in-play distance and HSR-distance biases are both within a factor of 2 of the measured ones.

| Camera | Distance bias: simulated / measured | HSR-distance bias: simulated / measured | Check |
|---|---|---|---|
| none | ×1.6 | ×1.8 | passes |
| ptz | ×5.1 | ×5.0 | **fails** |

Every simulated bias exceeds the measured one (×1.6 to ×5.1): the model overstates the 5 fps KPI cost even where the check passes.

What the simulation does not reproduce: the SN-GSR reference is the smoothed labels, which carry their own smoothing error, while IDSSE's is the clean tracking; real errors are heteroscedastic (larger far from the camera, along the viewing direction) and can correlate with motion (fast pans during fast play), while the simulated ones are stationary and independent of motion; and the two data sets move differently (table above). Nothing was tuned on the proxy: models are fitted to the ACF and the marginal only.

## 3. Budget at the export rate (10 fps pipeline)

> **Not predfoot's KPI error.** The check of section 2 fails for the ptz camera: the chosen model's simulated 5 fps proxy biases are several times the measured ones (ptz: distance ×5.1, HSR distance ×5.0, ball in play). For the none camera the check passes, but the simulated biases are still above the measured ones (none: distance ×1.6, HSR distance ×1.8). The numbers below are the budget of the fitted stationary error model; they overstate the KPI error of predfoot's tracking and must not be quoted as such.

Errors are sampled at 25 Hz and the 10 fps pipeline runs as for exported tracking. The nugget (0.02 to 0.43 of the variance) is invisible at 5 fps; two shapes bound what it can do at 25 Hz: **white** (independent frames, power spread to 12.5 Hz, mostly removed by the 10 fps smoothing after aliasing) and **band-limited** (flat spectrum up to 2.5 Hz: the smoothest process still uncorrelated at 0.2 s, with all its power where the 1 s smoothing lets speed noise through). The kernels bracket the resolved part between its measured lags, from exponential (rough) to squared exponential (smooth).

### Chosen model (nugget + Matérn-3/2, measured marginal, white nugget) against the clean 25 Hz reference

| Camera | KPI | Ref. mean | MAPE % [95 % CI] | Bias % [95 % CI] | Spearman [95 % CI] |
|---|---|---|---|---|---|
| none | Distance | 9646.8 | 41.10 [39.35, 43.45] | +40.96 [+39.19, +43.27] | 0.985 [0.976, 0.989] |
| none | HI dist. ≥ 20 | 735.9 | 213.37 [200.87, 225.82] | +190.88 [+183.23, +198.42] | 0.809 [0.765, 0.848] |
| none | Dist. > 25 | 180.1 | 487.68 [438.60, 540.55] | +346.89 [+323.21, +372.02] | 0.708 [0.679, 0.748] |
| none | HSR efforts | 38.3 | 16.89 [15.22, 18.52] | -14.76 [-16.60, -12.91] | 0.908 [0.867, 0.933] |
| none | Sprint efforts | 8.8 | 37.75 [35.49, 40.01] | -28.48 [-32.69, -24.39] | 0.772 [0.691, 0.835] |
| none | PSV-99 | 28.5 | 21.27 [20.32, 22.37] | +21.03 [+20.07, +22.12] | 0.734 [0.676, 0.799] |
| ptz | Distance | 9646.8 | 18.37 [17.61, 19.53] | +18.30 [+17.54, +19.46] | 0.994 [0.989, 0.995] |
| ptz | HI dist. ≥ 20 | 735.9 | 62.35 [58.29, 66.83] | +56.18 [+53.47, +59.10] | 0.941 [0.918, 0.955] |
| ptz | Dist. > 25 | 180.1 | 121.41 [109.60, 135.75] | +89.35 [+81.32, +97.87] | 0.915 [0.898, 0.933] |
| ptz | HSR efforts | 38.3 | 8.12 [7.37, 8.82] | -1.12 [-2.73, +0.29] | 0.948 [0.899, 0.968] |
| ptz | Sprint efforts | 8.8 | 20.21 [18.48, 22.32] | -12.64 [-15.27, -9.92] | 0.910 [0.874, 0.928] |
| ptz | PSV-99 | 28.5 | 9.29 [8.49, 10.18] | +9.17 [+8.38, +10.05] | 0.826 [0.751, 0.885] |

### Sensitivity: bias % of each KPI under every model

Ghost: distance per minute of a motionless player through the 10 fps pipeline (one player's full error).

| Model | Ghost (m/min) | Distance | HI dist. ≥ 20 | Dist. > 25 | HSR efforts | Sprint efforts | PSV-99 |
|---|---|---|---|---|---|---|---|
| clean 10 fps (control) | – | -0.1 | -0.2 | -0.2 | -0.5 | +0.0 | -0.1 |
| none: single OU through 0.2 s (previous) | 145.7 | +67.6 | +370.2 | +651.8 | +13.5 | -1.3 | +30.9 |
| none: nugget + Matérn-3/2, Gaussian, white nugget | 104.5 | +40.5 | +145.0 | +231.2 | -12.3 | -25.0 | +17.3 |
| none: nugget + exponential, measured marginal, white nugget | 105.0 | +42.1 | +199.9 | +365.5 | -13.7 | -28.7 | +21.7 |
| none: nugget + exponential, measured marginal, bandlimited nugget | 118.5 | +51.0 | +294.5 | +600.2 | -15.0 | -27.6 | +31.2 |
| none: nugget + Matérn-3/2, measured marginal, white nugget | 102.4 | +41.0 | +190.9 | +346.9 | -14.8 | -28.5 | +21.0 |
| none: nugget + Matérn-3/2, measured marginal, bandlimited nugget | 118.8 | +52.2 | +307.6 | +636.4 | -16.2 | -29.0 | +32.8 |
| none: nugget + Matérn-5/2, measured marginal, white nugget | 103.8 | +40.9 | +188.3 | +340.9 | -14.3 | -27.8 | +20.3 |
| none: nugget + Matérn-5/2, measured marginal, bandlimited nugget | 118.1 | +52.2 | +311.8 | +654.5 | -16.3 | -26.6 | +33.4 |
| none: nugget + squared exponential, measured marginal, white nugget | 100.4 | +40.6 | +186.8 | +337.0 | -14.0 | -30.7 | +20.1 |
| none: nugget + squared exponential, measured marginal, bandlimited nugget | 116.2 | +52.3 | +311.8 | +649.7 | -16.7 | -30.3 | +33.3 |
| ptz: single OU through 0.2 s (previous) | 61.2 | +16.9 | +39.4 | +57.8 | +0.1 | -6.4 | +6.7 |
| ptz: nugget + Matérn-3/2, Gaussian, white nugget | 59.1 | +15.8 | +36.0 | +53.2 | +0.7 | -4.6 | +6.8 |
| ptz: nugget + exponential, measured marginal, white nugget | 61.0 | +18.0 | +55.0 | +86.9 | -2.5 | -5.8 | +9.3 |
| ptz: nugget + exponential, measured marginal, bandlimited nugget | 62.1 | +18.6 | +57.5 | +92.3 | -3.1 | -11.2 | +10.0 |
| ptz: nugget + Matérn-3/2, measured marginal, white nugget | 60.7 | +18.3 | +56.2 | +89.3 | -1.1 | -12.6 | +9.2 |
| ptz: nugget + Matérn-3/2, measured marginal, bandlimited nugget | 64.8 | +19.6 | +63.4 | +102.1 | -3.1 | -12.0 | +10.4 |
| ptz: nugget + Matérn-5/2, measured marginal, white nugget | 61.5 | +18.0 | +55.4 | +90.3 | -2.2 | -6.6 | +9.9 |
| ptz: nugget + Matérn-5/2, measured marginal, bandlimited nugget | 64.2 | +19.8 | +66.5 | +110.4 | -3.6 | -11.8 | +11.4 |
| ptz: nugget + squared exponential, measured marginal, white nugget | 59.1 | +17.7 | +53.1 | +83.8 | -2.4 | -11.3 | +9.0 |
| ptz: nugget + squared exponential, measured marginal, bandlimited nugget | 64.8 | +19.8 | +64.9 | +107.1 | -4.5 | -11.2 | +11.3 |

Sensitivity bound: the range of the bias over the four kernels and both nugget shapes (same marginal as the chosen model), next to the Gaussian Matérn-3/2 and the previous single OU:

| Camera | KPI | Chosen | Range, kernels × nuggets | Gaussian Matérn-3/2 | Single OU |
|---|---|---|---|---|---|
| none | Distance | +41.0 | +40.6 to +52.3 | +40.5 | +67.6 |
| none | HI dist. ≥ 20 | +190.9 | +186.8 to +311.8 | +145.0 | +370.2 |
| none | Dist. > 25 | +346.9 | +337.0 to +654.5 | +231.2 | +651.8 |
| none | HSR efforts | -14.8 | -16.7 to -13.7 | -12.3 | +13.5 |
| none | Sprint efforts | -28.5 | -30.7 to -26.6 | -25.0 | -1.3 |
| none | PSV-99 | +21.0 | +20.1 to +33.4 | +17.3 | +30.9 |
| ptz | Distance | +18.3 | +17.7 to +19.8 | +15.8 | +16.9 |
| ptz | HI dist. ≥ 20 | +56.2 | +53.1 to +66.5 | +36.0 | +39.4 |
| ptz | Dist. > 25 | +89.3 | +83.8 to +110.4 | +53.2 | +57.8 |
| ptz | HSR efforts | -1.1 | -4.5 to -1.1 | +0.7 | +0.1 |
| ptz | Sprint efforts | -12.6 | -12.6 to -5.8 | -4.6 | -6.4 |
| ptz | PSV-99 | +9.2 | +9.0 to +11.4 | +6.8 | +6.7 |

All KPIs, chosen model, MAPE % / bias % against the clean 25 Hz reference:

| Model | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 | Dist. > 25 | HI dist. ≥ 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 41.1 / +41.0 | 17.8 / +17.7 | 94.1 / +86.8 | 156.6 / +140.3 | 487.7 / +346.9 | 213.4 / +190.9 | 16.9 / -14.8 | 37.8 / -28.5 | 21.3 / +21.0 | 64.9 / -52.6 | 78.7 / -16.9 |
| ptz | 18.4 / +18.3 | 11.8 / +11.7 | 36.1 / +33.6 | 50.6 / +45.4 | 121.4 / +89.3 | 62.4 / +56.2 | 8.1 / -1.1 | 20.2 / -12.6 | 9.3 / +9.2 | 57.3 / -34.5 | 64.7 / -33.1 |
| clean 10 fps | 0.1 / -0.1 | 0.0 / -0.0 | 0.2 / -0.1 | 0.4 / -0.2 | 0.8 / -0.2 | 0.3 / -0.2 | 0.7 / -0.5 | 0.9 / +0.0 | 0.1 / -0.1 | 12.3 / +2.2 | 13.2 / +8.6 |
