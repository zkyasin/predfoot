# KPI error budget: first sensitivity study

> **This is a sensitivity analysis on simulated position errors.** None of these numbers is a measured
> error of predfoot: the real error process (variance, temporal correlation, common-mode part, dependence
> on the predicted uncertainty) is measured on SN-GSR and injected in later reports. KPI
> definitions marked *assumption* below are ours, not SkillCorner's.

> **Measured condition.** The row *measured on SN-GSR* injects the error process
> measured by `scripts/error_process.py` (per-frame calibration, no temporal smoothing
> of the camera). Its σ is measured against the SN-GSR labels, which carry their own
> error, so it is an upper bound; its 25 Hz correlations come from 5 fps measurements
> under an exponential-decay assumption, which overstates the correlation of per-frame
> jitter. Read it as the KPI cost of calibrating each frame independently.

- **Command**: `uv run python scripts/kpi_sensitivity.py --matches J03WMX J03WN1 --n-boot 500 --measured 0.62 0.832 0.62 0.848 --report reports/kpi_budget_measured.md`
- **Git commit**: `07cfd8f`; date 2026-10-31T13:04:17; runtime 31.5 s
- **Data**: IDSSE (Bassek, Rein, Weber & Memmert 2025, CC BY 4.0, doi:10.6084/m9.figshare.28196177), 2 matches (J03WMX, J03WN1), DFL tracking at 25 Hz, from the Hugging Face mirror `pysport/idsse-data` (SHA-256 of the raw files at the end).
- **Population**: 39 player-matches: outfield players with at least 45 minutes on the pitch (goalkeepers excluded).
- **Config**: `configs/kpi.yaml` (SHA-256 `b5bcace96008`), seed 20261031, 500 bootstrap resamples.

## Summary

- **Gate G4** (clean data, 10 fps vs 25 Hz, MAPE < 0.5 %): passed by Distance, Dist. < 15, Dist. 15-20, Dist. 20-25 (HSR), HI dist. > 20, Sprint efforts, PSV-99; not passed by Dist. > 25 (sprint) 0.68 %, HSR efforts 0.97 %, Expl. acc. to HSR 14.34 %, Expl. acc. to sprint 9.77 %. Totals stay close (bias within ±0.62 % for every KPI except explosive accelerations): the misses come from discretisation at thresholds, where an effort near the minimum duration or an acceleration near 3 m/s² flips between the two pipelines, on small per-player counts (7.1 explosive accelerations to HSR per player-match on average).
- **Smoothing** is the main lever: with i.i.d. σ = 0.2 m, the distance error is 5.77 % with the 1 s window and 1.64 % with 2 s, but on clean data the 2 s window already moves explosive accelerations to HSR by +17.45 % (definitions and noise robustness pull in opposite directions).
- **Distance**: MAPE 1.68 % for i.i.d. σ = 0.1 m and 5.70 % for σ = 0.2 m, an overestimate (bias +5.66 % at 0.2 m). At σ = 0.2 m, slow errors (AR(1) ρ = 0.99, about 4 s) give 1.48 %, and errors correlated over about 0.4 s (ρ = 0.9) give 7.25 %, more than white noise: what matters is how much error power falls in the band the 1 s smoothing lets through, not correlation as such.
- **High-speed KPIs** react more: at i.i.d. σ = 0.1 m, distance above 25 km/h has MAPE 4.95 % and sprint efforts 11.90 % against the clean 10 fps control; PSV-99 stays within 0.75 % (bias +0.63 %).
- **Explosive accelerations** are the most fragile (a threshold on a second derivative): -54.55 % of the total at i.i.d. σ = 0.1 m and -76.36 % at 0.2 m, because noise breaks the runs of more than 0.7 s above 3 m/s².
- **Rankings** of players at i.i.d. σ = 0.2 m: Spearman 0.991 for HI distance, 0.897 for sprint efforts and 0.483 for explosive accelerations to HSR.

## Set-up

- **Cleaning**: the raw IDSSE feed holds the last position of a player who is not observed, then jumps to the re-acquired position; there are also a few identity swaps. Steps faster than 43.2 km/h are cut out of the source data before anything else (held samples, or the sample after the jump): 248 of 6194671 player-samples (0.004 %). Without this step each glitch becomes a fake sprint after smoothing, and the 25 Hz and 10 fps pipelines disagree on explosive accelerations by tens of percent.
- **Reference**: clean IDSSE positions at 25 Hz through the standard pipeline: Savitzky-Golay smoothing of positions over 1 s (order 2, 25 samples), velocity and acceleration by central differences that never cross a gap, then the KPIs.
- **Control (gate G4)**: the same clean positions resampled to 10 fps (linear interpolation between the two neighbouring 25 Hz samples) and smoothed over 11 samples (1.1 s). Gate G4 asks for MAPE < 0.5 %.
- **Injected errors**: Gaussian, isotropic, added to the 25 Hz positions before the 10 fps pipeline; σ is the marginal standard deviation per axis. AR(1) correlation ρ is per 25 Hz frame: ρ = 0.9 is a correlation time of 0.38 s, ρ = 0.99 of 4.0 s. The common-mode condition gives every player of a frame the same slow error (ρ = 0.99) plus an individual one (ρ = 0.9), each with half of the variance. Linear resampling averages two 25 Hz samples at every other 10 fps sample, which halves i.i.d. noise variance there; this is part of the pipeline being tested.
- **Statistics**: MAPE over player-matches with a positive reference; bias = relative error of the total over all player-matches; Spearman correlation of player rankings, on per-90-minute values (PSV-99 as is). 95 % intervals: percentile bootstrap resampling whole matches. With 2 matches the intervals are rough, and they do not include noise-seed variability (one noise draw per condition).

## KPI definitions

| KPI | Definition | Status |
|---|---|---|
| Distance | Sum of steps between consecutive smoothed 10 fps positions | standard |
| Speed bands | Distance of the steps whose speed is in < 15, 15-20, 20-25, > 25 km/h | 20 and 25 km/h published |
| HI distance | Distance at or above 20 km/h | published threshold |
| HSR / sprint efforts | Runs at or above 20 / 25 km/h lasting at least 1 s (crossings interpolated) | thresholds published; duration *assumption* |
| PSV-99 | 99th percentile of speed samples in runs above 15 km/h lasting at least 1 s | published wording; segmentation *assumption* |
| Explosive acc. to HSR / sprint | Acceleration phase starting below 9 km/h, reaching 20 / 25 km/h, with more than 0.7 s at 3 m/s² or more | criteria published; segmentation *assumption* |
| Smoothing | Savitzky-Golay over 1 s, order 2 | *assumption* |

Sources: SkillCorner, "New feature: explosive accelerations" (6 Feb 2025) and "SkillCorner releases new peak velocity metric" (11 Jun 2026); see `configs/kpi.yaml`.

## Gate G4: clean data through the 10 fps pipeline

| KPI | Ref. mean | MAPE % [95 % CI] | Bias % [95 % CI] | Spearman [95 % CI] | G4 |
|---|---|---|---|---|---|
| Distance | 9531.5 | 0.07 [0.05, 0.08] | -0.07 [-0.08, -0.05] | 1.000 [0.998, 1.000] | pass |
| Dist. < 15 | 7519.5 | 0.05 [0.03, 0.06] | -0.04 [-0.06, -0.03] | 1.000 [1.000, 1.000] | pass |
| Dist. 15-20 | 1304.7 | 0.26 [0.19, 0.33] | -0.15 [-0.17, -0.13] | 0.999 [0.998, 1.000] | pass |
| Dist. 20-25 (HSR) | 522.7 | 0.39 [0.33, 0.45] | -0.11 [-0.14, -0.09] | 1.000 [0.998, 1.000] | pass |
| Dist. > 25 (sprint) | 184.7 | 0.68 [0.66, 0.70] | -0.25 [-0.28, -0.21] | 0.999 [0.997, 1.000] | **fail** |
| HI dist. > 20 | 707.3 | 0.27 [0.24, 0.30] | -0.15 [-0.16, -0.14] | 1.000 [0.998, 1.000] | pass |
| HSR efforts | 37.2 | 0.97 [0.67, 1.29] | -0.62 [-0.72, -0.53] | 0.998 [0.993, 1.000] | **fail** |
| Sprint efforts | 9.2 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] | pass |
| PSV-99 | 28.9 | 0.16 [0.13, 0.20] | -0.14 [-0.19, -0.10] | 0.997 [0.991, 0.998] | pass |
| Expl. acc. to HSR | 7.1 | 14.34 [10.36, 18.11] | +0.00 [-2.63, +3.25] | 0.931 [0.898, 0.940] | **fail** |
| Expl. acc. to sprint | 2.1 | 9.77 [9.03, 10.56] | +6.10 [+0.00, +13.89] | 0.928 [0.905, 0.928] | **fail** |

## Sensitivity to position errors (against the 25 Hz reference)

### MAPE (%)

| Condition | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 (HSR) | Dist. > 25 (sprint) | HI dist. > 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean, 10 fps | 0.07 | 0.05 | 0.26 | 0.39 | 0.68 | 0.27 | 0.97 | 0.00 | 0.16 | 14.34 | 9.77 |
| i.i.d. σ=0.05 m | 0.47 | 0.52 | 0.60 | 1.12 | 1.71 | 0.83 | 3.88 | 7.09 | 0.39 | 30.12 | 40.65 |
| i.i.d. σ=0.1 m | 1.68 | 1.62 | 2.09 | 2.47 | 4.84 | 2.30 | 6.35 | 11.90 | 0.68 | 53.91 | 65.28 |
| i.i.d. σ=0.2 m | 5.70 | 4.99 | 8.32 | 8.57 | 18.60 | 10.26 | 15.02 | 24.60 | 2.09 | 79.47 | 92.95 |
| i.i.d. σ=0.5 m | 27.43 | 19.90 | 50.78 | 64.21 | 142.24 | 78.00 | 32.02 | 49.44 | 10.67 | 92.24 | 95.05 |
| AR(1) ρ=0.9, σ=0.05 m | 0.61 | 0.66 | 0.83 | 1.13 | 3.00 | 0.94 | 3.36 | 6.60 | 0.47 | 34.61 | 33.43 |
| AR(1) ρ=0.9, σ=0.1 m | 2.14 | 2.02 | 2.65 | 3.64 | 6.37 | 3.36 | 6.57 | 11.64 | 0.91 | 39.18 | 60.88 |
| AR(1) ρ=0.9, σ=0.2 m | 7.25 | 6.33 | 10.56 | 11.09 | 24.12 | 13.28 | 8.45 | 18.21 | 3.04 | 60.26 | 82.41 |
| AR(1) ρ=0.9, σ=0.5 m | 34.23 | 23.15 | 68.53 | 87.82 | 200.51 | 107.83 | 12.17 | 28.40 | 15.60 | 68.51 | 101.63 |
| AR(1) ρ=0.99, σ=0.05 m | 0.08 | 0.14 | 0.49 | 0.62 | 1.22 | 0.45 | 1.80 | 0.85 | 0.27 | 15.98 | 14.12 |
| AR(1) ρ=0.99, σ=0.1 m | 0.41 | 0.45 | 0.94 | 1.28 | 2.60 | 0.94 | 3.01 | 6.94 | 0.46 | 27.14 | 39.71 |
| AR(1) ρ=0.99, σ=0.2 m | 1.48 | 1.46 | 1.87 | 2.76 | 4.88 | 2.15 | 3.99 | 11.03 | 0.86 | 37.81 | 51.62 |
| AR(1) ρ=0.99, σ=0.5 m | 7.53 | 6.42 | 11.99 | 11.32 | 24.34 | 13.55 | 6.65 | 17.47 | 2.75 | 46.67 | 64.01 |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | 4.50 | 4.03 | 6.33 | 5.92 | 15.88 | 7.62 | 8.43 | 10.53 | 1.79 | 49.43 | 80.55 |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | 90.26 | 16.51 | 264.86 | 474.59 | 1541.73 | 671.79 | 12.31 | 31.85 | 38.66 | 158.71 | 498.67 |

### Bias (%, relative error of the total)

| Condition | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 (HSR) | Dist. > 25 (sprint) | HI dist. > 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean, 10 fps | -0.07 | -0.04 | -0.15 | -0.11 | -0.25 | -0.15 | -0.62 | +0.00 | -0.14 | +0.00 | +6.10 |
| i.i.d. σ=0.05 m | +0.47 | +0.51 | +0.21 | +0.48 | +0.70 | +0.54 | -0.62 | -2.51 | +0.03 | -25.09 | -36.59 |
| i.i.d. σ=0.1 m | +1.67 | +1.60 | +1.88 | +1.84 | +2.84 | +2.10 | -5.59 | -8.10 | +0.48 | -54.55 | -63.41 |
| i.i.d. σ=0.2 m | +5.66 | +4.91 | +7.96 | +7.67 | +14.08 | +9.34 | -14.35 | -19.55 | +2.06 | -76.36 | -87.80 |
| i.i.d. σ=0.5 m | +27.24 | +19.64 | +47.40 | +59.10 | +103.92 | +70.80 | -31.47 | -45.25 | +10.57 | -90.91 | -90.24 |
| AR(1) ρ=0.9, σ=0.05 m | +0.60 | +0.65 | +0.39 | +0.08 | +1.74 | +0.51 | -1.38 | -1.40 | +0.12 | -12.36 | -18.29 |
| AR(1) ρ=0.9, σ=0.1 m | +2.12 | +1.99 | +2.41 | +2.72 | +3.76 | +2.99 | -4.83 | -6.70 | +0.63 | -36.00 | -43.90 |
| AR(1) ρ=0.9, σ=0.2 m | +7.21 | +6.24 | +10.15 | +10.08 | +17.66 | +12.06 | -7.25 | -11.73 | +3.00 | -58.18 | -78.05 |
| AR(1) ρ=0.9, σ=0.5 m | +33.99 | +22.85 | +63.54 | +81.31 | +144.85 | +97.90 | -9.11 | -18.99 | +15.49 | -2.55 | +35.37 |
| AR(1) ρ=0.99, σ=0.05 m | +0.08 | +0.13 | -0.08 | -0.12 | -0.05 | -0.10 | -0.35 | -0.56 | -0.17 | -0.73 | -2.44 |
| AR(1) ρ=0.99, σ=0.1 m | +0.40 | +0.44 | +0.26 | +0.27 | +0.46 | +0.32 | -0.69 | +0.28 | -0.09 | -1.45 | +2.44 |
| AR(1) ρ=0.99, σ=0.2 m | +1.47 | +1.43 | +1.53 | +1.54 | +2.39 | +1.76 | -1.59 | -3.35 | +0.27 | -23.64 | -24.39 |
| AR(1) ρ=0.99, σ=0.5 m | +7.48 | +6.32 | +11.47 | +10.27 | +18.56 | +12.44 | -1.52 | -9.50 | +2.62 | -45.09 | -54.88 |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | +4.46 | +3.96 | +6.05 | +5.13 | +11.76 | +6.86 | -6.35 | -7.26 | +1.70 | -47.27 | -68.29 |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | +89.67 | +15.32 | +241.44 | +433.53 | +1071.64 | +600.11 | -1.66 | -19.55 | +38.39 | +76.36 | +385.37 |

### Spearman correlation of player rankings

| Condition | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 (HSR) | Dist. > 25 (sprint) | HI dist. > 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean, 10 fps | 1.000 | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 | 0.998 | 1.000 | 0.997 | 0.931 | 0.928 |
| i.i.d. σ=0.05 m | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.980 | 0.969 | 0.987 | 0.755 | 0.778 |
| i.i.d. σ=0.1 m | 0.998 | 0.999 | 0.997 | 0.992 | 0.991 | 0.998 | 0.972 | 0.959 | 0.980 | 0.534 | 0.726 |
| i.i.d. σ=0.2 m | 0.997 | 0.996 | 0.983 | 0.977 | 0.975 | 0.991 | 0.940 | 0.897 | 0.964 | 0.483 | 0.436 |
| i.i.d. σ=0.5 m | 0.992 | 0.895 | 0.919 | 0.789 | 0.859 | 0.888 | 0.887 | 0.873 | 0.938 | 0.305 | 0.185 |
| AR(1) ρ=0.9, σ=0.05 m | 0.999 | 1.000 | 0.999 | 0.999 | 0.997 | 0.998 | 0.983 | 0.971 | 0.989 | 0.723 | 0.853 |
| AR(1) ρ=0.9, σ=0.1 m | 0.998 | 0.997 | 0.996 | 0.995 | 0.992 | 0.999 | 0.956 | 0.940 | 0.981 | 0.622 | 0.683 |
| AR(1) ρ=0.9, σ=0.2 m | 0.997 | 0.994 | 0.978 | 0.975 | 0.962 | 0.990 | 0.957 | 0.948 | 0.952 | 0.508 | 0.442 |
| AR(1) ρ=0.9, σ=0.5 m | 0.993 | 0.675 | 0.900 | 0.754 | 0.802 | 0.859 | 0.948 | 0.887 | 0.868 | 0.101 | 0.068 |
| AR(1) ρ=0.99, σ=0.05 m | 1.000 | 1.000 | 0.999 | 0.999 | 0.999 | 1.000 | 0.991 | 0.999 | 0.993 | 0.907 | 0.936 |
| AR(1) ρ=0.99, σ=0.1 m | 1.000 | 0.999 | 0.998 | 0.999 | 0.995 | 0.996 | 0.989 | 0.970 | 0.982 | 0.857 | 0.881 |
| AR(1) ρ=0.99, σ=0.2 m | 0.998 | 0.998 | 0.995 | 0.992 | 0.993 | 0.997 | 0.982 | 0.942 | 0.972 | 0.677 | 0.860 |
| AR(1) ρ=0.99, σ=0.5 m | 0.997 | 0.995 | 0.976 | 0.951 | 0.964 | 0.984 | 0.957 | 0.922 | 0.943 | 0.491 | 0.373 |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | 0.997 | 0.999 | 0.986 | 0.988 | 0.983 | 0.997 | 0.965 | 0.957 | 0.967 | 0.653 | 0.506 |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | 0.978 | -0.748 | 0.800 | 0.478 | 0.405 | 0.581 | 0.853 | 0.839 | 0.754 | 0.204 | 0.177 |

Confidence intervals for every cell are in the appendix.

### MAPE (%) against the clean 10 fps control: the effect of the noise alone

Same noisy conditions, compared with clean data through the same 10 fps pipeline, so that the difference between the pipelines (gate G4 above) is not counted.

| Condition | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 (HSR) | Dist. > 25 (sprint) | HI dist. > 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| i.i.d. σ=0.05 m | 0.54 | 0.56 | 0.75 | 1.17 | 1.82 | 0.97 | 3.19 | 7.09 | 0.39 | 30.19 | 43.39 |
| i.i.d. σ=0.1 m | 1.75 | 1.67 | 2.26 | 2.54 | 4.95 | 2.45 | 5.90 | 11.90 | 0.75 | 52.94 | 65.86 |
| i.i.d. σ=0.2 m | 5.77 | 5.04 | 8.51 | 8.68 | 18.79 | 10.43 | 14.42 | 24.60 | 2.23 | 80.35 | 92.42 |
| i.i.d. σ=0.5 m | 27.52 | 19.96 | 51.06 | 64.41 | 142.64 | 78.27 | 31.61 | 49.44 | 10.82 | 92.03 | 93.65 |
| AR(1) ρ=0.9, σ=0.05 m | 0.67 | 0.70 | 0.88 | 1.19 | 3.19 | 0.97 | 3.58 | 6.60 | 0.48 | 32.38 | 37.06 |
| AR(1) ρ=0.9, σ=0.1 m | 2.21 | 2.07 | 2.82 | 3.72 | 6.58 | 3.50 | 6.81 | 11.64 | 0.95 | 44.93 | 63.39 |
| AR(1) ρ=0.9, σ=0.2 m | 7.32 | 6.37 | 10.76 | 11.23 | 24.33 | 13.45 | 7.81 | 18.21 | 3.19 | 59.74 | 83.93 |
| AR(1) ρ=0.9, σ=0.5 m | 34.32 | 23.21 | 68.84 | 88.05 | 201.06 | 108.15 | 11.87 | 28.40 | 15.76 | 73.97 | 98.49 |
| AR(1) ρ=0.99, σ=0.05 m | 0.15 | 0.17 | 0.48 | 0.73 | 1.33 | 0.44 | 1.86 | 0.85 | 0.23 | 8.31 | 10.94 |
| AR(1) ρ=0.99, σ=0.1 m | 0.47 | 0.49 | 0.93 | 1.34 | 2.57 | 1.01 | 2.61 | 6.94 | 0.48 | 24.93 | 37.45 |
| AR(1) ρ=0.99, σ=0.2 m | 1.55 | 1.50 | 1.95 | 2.87 | 5.04 | 2.31 | 3.73 | 11.03 | 0.87 | 35.25 | 50.29 |
| AR(1) ρ=0.99, σ=0.5 m | 7.60 | 6.46 | 12.19 | 11.49 | 24.54 | 13.72 | 6.87 | 17.47 | 2.85 | 44.52 | 68.78 |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | 4.57 | 4.07 | 6.52 | 6.10 | 16.11 | 7.78 | 7.81 | 10.53 | 1.91 | 49.71 | 79.97 |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | 90.38 | 16.55 | 265.60 | 475.41 | 1544.41 | 673.07 | 12.54 | 31.85 | 38.85 | 166.87 | 500.31 |

## Ghost distance of a motionless player

Closed form 60·f·σ·sqrt(π(1 - ρ)) m/min for raw positions (60·f·σ·√π for i.i.d. noise), checked by simulating 30 minutes at 25 Hz, and the same noise through the 10 fps pipeline.

| ρ | σ (m) | Closed form (m/min) | Simulated, raw 25 Hz | Through the 10 fps pipeline |
|---|---|---|---|---|
| 0.00 | 0.05 | 132.9 | 132.5 | 7.9 |
| 0.00 | 0.10 | 265.9 | 266.5 | 15.8 |
| 0.00 | 0.20 | 531.7 | 533.5 | 31.4 |
| 0.00 | 0.50 | 1329.3 | 1319.0 | 78.0 |
| 0.90 | 0.05 | 42.0 | 42.1 | 9.0 |
| 0.90 | 0.10 | 84.1 | 84.2 | 18.3 |
| 0.90 | 0.20 | 168.1 | 168.0 | 36.4 |
| 0.90 | 0.50 | 420.4 | 420.0 | 90.2 |
| 0.99 | 0.05 | 13.3 | 13.3 | 3.7 |
| 0.99 | 0.10 | 26.6 | 26.6 | 7.3 |
| 0.99 | 0.20 | 53.2 | 53.1 | 14.9 |
| 0.99 | 0.50 | 132.9 | 133.5 | 36.5 |

## Smoothing window

MAPE % / bias % at 10 fps. Rows "clean vs reference" show the cost of the window on clean data; rows with noise compare noisy and clean data through the same window, which isolates the effect of the noise (σ = 0.2 m).

| Window | Comparison | Distance | HI dist. > 20 | Sprint efforts | PSV-99 | Expl. acc. to HSR |
|---|---|---|---|---|---|---|
| 0.5 s | clean vs reference | 0.08 / +0.08 | 0.37 / +0.21 | 1.17 / -0.84 | 0.13 / -0.06 | 18.66 / -8.73 |
| 0.5 s | i.i.d. σ=0.2 m vs clean (same window) | 30.70 / +30.48 | 92.88 / +84.40 | 55.60 / -50.14 | 11.45 / +11.34 | 100.00 / -100.00 |
| 0.5 s | AR(1) ρ=0.99, σ=0.2 m vs clean (same window) | 3.22 / +3.20 | 4.97 / +4.59 | 16.10 / -8.17 | 1.13 / +1.00 | 79.37 / -75.70 |
| 1 s (default) | clean vs reference | 0.07 / -0.07 | 0.27 / -0.15 | 0.00 / +0.00 | 0.16 / -0.14 | 14.34 / +0.00 |
| 1 s (default) | i.i.d. σ=0.2 m vs clean (same window) | 5.77 / +5.73 | 10.43 / +9.51 | 24.60 / -19.55 | 2.23 / +2.21 | 80.35 / -76.36 |
| 1 s (default) | AR(1) ρ=0.99, σ=0.2 m vs clean (same window) | 1.55 / +1.54 | 2.31 / +1.91 | 11.03 / -3.35 | 0.87 / +0.41 | 35.25 / -23.64 |
| 2 s | clean vs reference | 0.68 / -0.68 | 2.08 / -2.02 | 3.95 / -2.23 | 0.39 / -0.36 | 31.58 / +17.45 |
| 2 s | i.i.d. σ=0.2 m vs clean (same window) | 1.64 / +1.63 | 2.44 / +2.00 | 9.75 / -3.71 | 0.74 / +0.58 | 61.39 / -57.59 |
| 2 s | AR(1) ρ=0.99, σ=0.2 m vs clean (same window) | 0.87 / +0.86 | 1.77 / +1.14 | 8.45 / +1.14 | 0.74 / +0.18 | 16.77 / -3.41 |
| none | clean vs reference | 0.15 / +0.15 | 0.61 / +0.35 | 1.79 / -2.23 | 0.15 / -0.02 | 18.49 / -12.73 |
| none | i.i.d. σ=0.2 m vs clean (same window) | 100.56 / +99.93 | 840.31 / +749.99 | 80.35 / -74.86 | 22.21 / +22.02 | 100.00 / -100.00 |
| none | AR(1) ρ=0.99, σ=0.2 m vs clean (same window) | 5.78 / +5.74 | 10.15 / +9.30 | 19.08 / -14.86 | 1.55 / +1.50 | 87.10 / -83.75 |

## Unpublished definitions (clean data, 10 fps)

Change of each KPI when one assumption changes, against the default configuration (bias % of the total, MAPE % across player-matches). These are the choices SkillCorner does not publish; they move the KPI values themselves, before any position error.

| Variant | KPIs that change |
|---|---|
| min_effort_0.5s | HSR efforts: +28.7 % (MAPE 29.9 %); Sprint efforts: +31.3 % (MAPE 38.9 %) |
| min_effort_2s | HSR efforts: -41.7 % (MAPE 42.7 %); Sprint efforts: -49.2 % (MAPE 50.7 %) |
| psv_run_peaks | PSV-99: +3.3 % (MAPE 3.4 %) |
| psv_min_activity_0.5s | PSV-99: -0.3 % (MAPE 0.3 %) |
| psv_min_activity_2s | PSV-99: +1.0 % (MAPE 1.0 %) |

## Reading these numbers

- Errors here are stationary, isotropic and independent of the camera; real broadcast errors grow with the distance to the camera, are larger along the viewing direction and depend on the calibration of each frame (later reports measure them and inject errors driven by the predicted uncertainty σ̂).
- The reference is IDSSE's own tracking, not ground truth; injected errors are added on top of it.
- Distance is robust to noise once positions are smoothed: for a moving player, noise lengthens each step only at second order. Speed-threshold KPIs (counts, PSV-99, explosive accelerations) react at first order, because noise moves speed across thresholds.

## Appendix: all comparisons with 95 % intervals

| Condition | KPI | n | MAPE % | Bias % | Spearman |
|---|---|---|---|---|---|
| clean, 10 fps | Distance | 39 | 0.07 [0.05, 0.08] | -0.07 [-0.08, -0.05] | 1.000 [0.998, 1.000] |
| clean, 10 fps | Dist. < 15 | 39 | 0.05 [0.03, 0.06] | -0.04 [-0.06, -0.03] | 1.000 [1.000, 1.000] |
| clean, 10 fps | Dist. 15-20 | 39 | 0.26 [0.19, 0.33] | -0.15 [-0.17, -0.13] | 0.999 [0.998, 1.000] |
| clean, 10 fps | Dist. 20-25 (HSR) | 39 | 0.39 [0.33, 0.45] | -0.11 [-0.14, -0.09] | 1.000 [0.998, 1.000] |
| clean, 10 fps | Dist. > 25 (sprint) | 39 | 0.68 [0.66, 0.70] | -0.25 [-0.28, -0.21] | 0.999 [0.997, 1.000] |
| clean, 10 fps | HI dist. > 20 | 39 | 0.27 [0.24, 0.30] | -0.15 [-0.16, -0.14] | 1.000 [0.998, 1.000] |
| clean, 10 fps | HSR efforts | 39 | 0.97 [0.67, 1.29] | -0.62 [-0.72, -0.53] | 0.998 [0.993, 1.000] |
| clean, 10 fps | Sprint efforts | 39 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| clean, 10 fps | PSV-99 | 39 | 0.16 [0.13, 0.20] | -0.14 [-0.19, -0.10] | 0.997 [0.991, 0.998] |
| clean, 10 fps | Expl. acc. to HSR | 39 | 14.34 [10.36, 18.11] | +0.00 [-2.63, +3.25] | 0.931 [0.898, 0.940] |
| clean, 10 fps | Expl. acc. to sprint | 39 | 9.77 [9.03, 10.56] | +6.10 [+0.00, +13.89] | 0.928 [0.905, 0.928] |
| i.i.d. σ=0.05 m | Distance | 39 | 0.47 [0.40, 0.55] | +0.47 [+0.40, +0.55] | 0.999 [0.998, 0.999] |
| i.i.d. σ=0.05 m | Dist. < 15 | 39 | 0.52 [0.38, 0.65] | +0.51 [+0.38, +0.65] | 0.999 [0.995, 1.000] |
| i.i.d. σ=0.05 m | Dist. 15-20 | 39 | 0.60 [0.47, 0.72] | +0.21 [+0.01, +0.36] | 0.999 [0.998, 1.000] |
| i.i.d. σ=0.05 m | Dist. 20-25 (HSR) | 39 | 1.12 [1.02, 1.22] | +0.48 [+0.24, +0.69] | 0.999 [0.999, 1.000] |
| i.i.d. σ=0.05 m | Dist. > 25 (sprint) | 39 | 1.71 [1.51, 1.90] | +0.70 [+0.67, +0.72] | 0.999 [0.995, 0.999] |
| i.i.d. σ=0.05 m | HI dist. > 20 | 39 | 0.83 [0.75, 0.91] | +0.54 [+0.37, +0.69] | 0.999 [0.996, 1.000] |
| i.i.d. σ=0.05 m | HSR efforts | 39 | 3.88 [3.45, 4.32] | -0.62 [-0.66, -0.58] | 0.980 [0.946, 0.981] |
| i.i.d. σ=0.05 m | Sprint efforts | 39 | 7.09 [5.66, 8.58] | -2.51 [-3.12, -1.81] | 0.969 [0.951, 0.981] |
| i.i.d. σ=0.05 m | PSV-99 | 39 | 0.39 [0.36, 0.43] | +0.03 [-0.02, +0.08] | 0.987 [0.968, 0.987] |
| i.i.d. σ=0.05 m | Expl. acc. to HSR | 39 | 30.12 [26.93, 33.13] | -25.09 [-29.27, -21.71] | 0.755 [0.667, 0.860] |
| i.i.d. σ=0.05 m | Expl. acc. to sprint | 39 | 40.65 [39.00, 42.19] | -36.59 [-44.44, -30.43] | 0.778 [0.676, 0.869] |
| i.i.d. σ=0.1 m | Distance | 39 | 1.68 [1.45, 1.93] | +1.67 [+1.45, +1.93] | 0.998 [0.994, 0.998] |
| i.i.d. σ=0.1 m | Dist. < 15 | 39 | 1.62 [1.24, 2.03] | +1.60 [+1.22, +2.03] | 0.999 [0.998, 0.999] |
| i.i.d. σ=0.1 m | Dist. 15-20 | 39 | 2.09 [1.85, 2.32] | +1.88 [+1.58, +2.12] | 0.997 [0.993, 0.997] |
| i.i.d. σ=0.1 m | Dist. 20-25 (HSR) | 39 | 2.47 [1.46, 3.43] | +1.84 [+0.90, +2.67] | 0.992 [0.991, 0.995] |
| i.i.d. σ=0.1 m | Dist. > 25 (sprint) | 39 | 4.84 [4.54, 5.13] | +2.84 [+2.55, +3.14] | 0.991 [0.967, 0.995] |
| i.i.d. σ=0.1 m | HI dist. > 20 | 39 | 2.30 [1.62, 2.95] | +2.10 [+1.50, +2.64] | 0.998 [0.995, 0.998] |
| i.i.d. σ=0.1 m | HSR efforts | 39 | 6.35 [6.07, 6.61] | -5.59 [-5.83, -5.33] | 0.972 [0.942, 0.973] |
| i.i.d. σ=0.1 m | Sprint efforts | 39 | 11.90 [11.76, 12.03] | -8.10 [-8.33, -7.83] | 0.959 [0.947, 0.959] |
| i.i.d. σ=0.1 m | PSV-99 | 39 | 0.68 [0.68, 0.69] | +0.48 [+0.45, +0.52] | 0.980 [0.932, 0.981] |
| i.i.d. σ=0.1 m | Expl. acc. to HSR | 39 | 53.91 [50.79, 57.21] | -54.55 [-58.54, -51.32] | 0.534 [0.411, 0.547] |
| i.i.d. σ=0.1 m | Expl. acc. to sprint | 39 | 65.28 [56.94, 73.09] | -63.41 [-72.22, -56.52] | 0.726 [0.552, 0.838] |
| i.i.d. σ=0.2 m | Distance | 39 | 5.70 [5.00, 6.43] | +5.66 [+4.99, +6.43] | 0.997 [0.991, 0.997] |
| i.i.d. σ=0.2 m | Dist. < 15 | 39 | 4.99 [3.89, 6.15] | +4.91 [+3.84, +6.14] | 0.996 [0.982, 0.996] |
| i.i.d. σ=0.2 m | Dist. 15-20 | 39 | 8.32 [7.58, 9.03] | +7.96 [+7.01, +8.72] | 0.983 [0.975, 0.983] |
| i.i.d. σ=0.2 m | Dist. 20-25 (HSR) | 39 | 8.57 [6.88, 10.17] | +7.67 [+6.61, +8.60] | 0.977 [0.968, 0.993] |
| i.i.d. σ=0.2 m | Dist. > 25 (sprint) | 39 | 18.60 [18.42, 18.78] | +14.08 [+13.57, +14.56] | 0.975 [0.935, 0.982] |
| i.i.d. σ=0.2 m | HI dist. > 20 | 39 | 10.26 [8.99, 11.46] | +9.34 [+8.48, +10.12] | 0.991 [0.991, 0.993] |
| i.i.d. σ=0.2 m | HSR efforts | 39 | 15.02 [15.01, 15.02] | -14.35 [-14.44, -14.27] | 0.940 [0.888, 0.963] |
| i.i.d. σ=0.2 m | Sprint efforts | 39 | 24.60 [20.67, 28.33] | -19.55 [-23.44, -15.06] | 0.897 [0.882, 0.897] |
| i.i.d. σ=0.2 m | PSV-99 | 39 | 2.09 [1.88, 2.30] | +2.06 [+1.84, +2.29] | 0.964 [0.934, 0.964] |
| i.i.d. σ=0.2 m | Expl. acc. to HSR | 39 | 79.47 [72.42, 86.91] | -76.36 [-83.74, -70.39] | 0.483 [0.433, 0.542] |
| i.i.d. σ=0.2 m | Expl. acc. to sprint | 39 | 92.95 [92.83, 93.06] | -87.80 [-94.44, -82.61] | 0.436 [0.203, 0.630] |
| i.i.d. σ=0.5 m | Distance | 39 | 27.43 [24.60, 30.41] | +27.24 [+24.53, +30.36] | 0.992 [0.986, 0.992] |
| i.i.d. σ=0.5 m | Dist. < 15 | 39 | 19.90 [15.41, 24.64] | +19.64 [+15.25, +24.66] | 0.895 [0.808, 0.895] |
| i.i.d. σ=0.5 m | Dist. 15-20 | 39 | 50.78 [48.85, 52.61] | +47.40 [+44.93, +49.38] | 0.919 [0.881, 0.919] |
| i.i.d. σ=0.5 m | Dist. 20-25 (HSR) | 39 | 64.21 [55.88, 72.12] | +59.10 [+52.47, +64.96] | 0.789 [0.735, 0.879] |
| i.i.d. σ=0.5 m | Dist. > 25 (sprint) | 39 | 142.24 [139.41, 144.94] | +103.92 [+98.76, +108.78] | 0.859 [0.825, 0.910] |
| i.i.d. σ=0.5 m | HI dist. > 20 | 39 | 78.00 [70.55, 85.07] | +70.80 [+64.86, +76.14] | 0.888 [0.888, 0.905] |
| i.i.d. σ=0.5 m | HSR efforts | 39 | 32.02 [31.35, 32.72] | -31.47 [-31.79, -31.12] | 0.887 [0.869, 0.898] |
| i.i.d. σ=0.5 m | Sprint efforts | 39 | 49.44 [44.37, 54.26] | -45.25 [-45.83, -44.58] | 0.873 [0.853, 0.873] |
| i.i.d. σ=0.5 m | PSV-99 | 39 | 10.67 [10.54, 10.81] | +10.57 [+10.43, +10.72] | 0.938 [0.914, 0.938] |
| i.i.d. σ=0.5 m | Expl. acc. to HSR | 39 | 92.24 [89.80, 94.81] | -90.91 [-92.68, -89.47] | 0.305 [0.166, 0.516] |
| i.i.d. σ=0.5 m | Expl. acc. to sprint | 39 | 95.05 [94.22, 95.83] | -90.24 [-91.67, -89.13] | 0.185 [0.038, 0.236] |
| AR(1) ρ=0.9, σ=0.05 m | Distance | 39 | 0.61 [0.52, 0.70] | +0.60 [+0.52, +0.71] | 0.999 [0.998, 0.999] |
| AR(1) ρ=0.9, σ=0.05 m | Dist. < 15 | 39 | 0.66 [0.53, 0.80] | +0.65 [+0.52, +0.80] | 1.000 [1.000, 1.000] |
| AR(1) ρ=0.9, σ=0.05 m | Dist. 15-20 | 39 | 0.83 [0.82, 0.85] | +0.39 [+0.21, +0.53] | 0.999 [0.996, 1.000] |
| AR(1) ρ=0.9, σ=0.05 m | Dist. 20-25 (HSR) | 39 | 1.13 [1.08, 1.18] | +0.08 [-0.18, +0.36] | 0.999 [0.998, 1.000] |
| AR(1) ρ=0.9, σ=0.05 m | Dist. > 25 (sprint) | 39 | 3.00 [2.38, 3.59] | +1.74 [+1.27, +2.18] | 0.997 [0.985, 0.998] |
| AR(1) ρ=0.9, σ=0.05 m | HI dist. > 20 | 39 | 0.94 [0.87, 1.02] | +0.51 [+0.43, +0.61] | 0.998 [0.995, 0.998] |
| AR(1) ρ=0.9, σ=0.05 m | HSR efforts | 39 | 3.36 [3.04, 3.69] | -1.38 [-1.46, -1.30] | 0.983 [0.983, 0.988] |
| AR(1) ρ=0.9, σ=0.05 m | Sprint efforts | 39 | 6.60 [4.63, 8.68] | -1.40 [-3.65, +1.20] | 0.971 [0.964, 0.974] |
| AR(1) ρ=0.9, σ=0.05 m | PSV-99 | 39 | 0.47 [0.45, 0.50] | +0.12 [+0.11, +0.13] | 0.989 [0.974, 0.989] |
| AR(1) ρ=0.9, σ=0.05 m | Expl. acc. to HSR | 39 | 34.61 [27.07, 42.57] | -12.36 [-20.33, -5.92] | 0.723 [0.695, 0.741] |
| AR(1) ρ=0.9, σ=0.05 m | Expl. acc. to sprint | 39 | 33.43 [20.39, 45.66] | -18.29 [-25.00, -13.04] | 0.853 [0.741, 0.927] |
| AR(1) ρ=0.9, σ=0.1 m | Distance | 39 | 2.14 [1.84, 2.46] | +2.12 [+1.83, +2.46] | 0.998 [0.992, 0.998] |
| AR(1) ρ=0.9, σ=0.1 m | Dist. < 15 | 39 | 2.02 [1.55, 2.52] | +1.99 [+1.53, +2.52] | 0.997 [0.989, 0.997] |
| AR(1) ρ=0.9, σ=0.1 m | Dist. 15-20 | 39 | 2.65 [2.45, 2.85] | +2.41 [+1.90, +2.81] | 0.996 [0.984, 0.998] |
| AR(1) ρ=0.9, σ=0.1 m | Dist. 20-25 (HSR) | 39 | 3.64 [3.26, 4.00] | +2.72 [+2.56, +2.87] | 0.995 [0.986, 0.996] |
| AR(1) ρ=0.9, σ=0.1 m | Dist. > 25 (sprint) | 39 | 6.37 [6.18, 6.55] | +3.76 [+3.39, +4.11] | 0.992 [0.982, 0.992] |
| AR(1) ρ=0.9, σ=0.1 m | HI dist. > 20 | 39 | 3.36 [3.01, 3.68] | +2.99 [+2.78, +3.19] | 0.999 [0.997, 0.999] |
| AR(1) ρ=0.9, σ=0.1 m | HSR efforts | 39 | 6.57 [6.55, 6.59] | -4.83 [-5.70, -3.89] | 0.956 [0.939, 0.961] |
| AR(1) ρ=0.9, σ=0.1 m | Sprint efforts | 39 | 11.64 [7.06, 15.99] | -6.70 [-8.85, -4.22] | 0.940 [0.876, 0.961] |
| AR(1) ρ=0.9, σ=0.1 m | PSV-99 | 39 | 0.91 [0.72, 1.10] | +0.63 [+0.50, +0.76] | 0.981 [0.958, 0.981] |
| AR(1) ρ=0.9, σ=0.1 m | Expl. acc. to HSR | 39 | 39.18 [38.38, 39.93] | -36.00 [-41.46, -31.58] | 0.622 [0.525, 0.738] |
| AR(1) ρ=0.9, σ=0.1 m | Expl. acc. to sprint | 39 | 60.88 [56.00, 65.45] | -43.90 [-52.78, -36.96] | 0.683 [0.492, 0.795] |
| AR(1) ρ=0.9, σ=0.2 m | Distance | 39 | 7.25 [6.39, 8.16] | +7.21 [+6.38, +8.16] | 0.997 [0.993, 0.997] |
| AR(1) ρ=0.9, σ=0.2 m | Dist. < 15 | 39 | 6.33 [4.96, 7.76] | +6.24 [+4.90, +7.77] | 0.994 [0.988, 0.994] |
| AR(1) ρ=0.9, σ=0.2 m | Dist. 15-20 | 39 | 10.56 [9.76, 11.33] | +10.15 [+9.03, +11.04] | 0.978 [0.963, 0.978] |
| AR(1) ρ=0.9, σ=0.2 m | Dist. 20-25 (HSR) | 39 | 11.09 [8.92, 13.15] | +10.08 [+8.49, +11.49] | 0.975 [0.975, 0.988] |
| AR(1) ρ=0.9, σ=0.2 m | Dist. > 25 (sprint) | 39 | 24.12 [23.42, 24.86] | +17.66 [+16.93, +18.34] | 0.962 [0.917, 0.975] |
| AR(1) ρ=0.9, σ=0.2 m | HI dist. > 20 | 39 | 13.28 [11.64, 14.83] | +12.06 [+10.75, +13.24] | 0.990 [0.989, 0.991] |
| AR(1) ρ=0.9, σ=0.2 m | HSR efforts | 39 | 8.45 [8.22, 8.68] | -7.25 [-8.65, -5.96] | 0.957 [0.940, 0.970] |
| AR(1) ρ=0.9, σ=0.2 m | Sprint efforts | 39 | 18.21 [16.57, 19.76] | -11.73 [-12.50, -10.84] | 0.948 [0.918, 0.948] |
| AR(1) ρ=0.9, σ=0.2 m | PSV-99 | 39 | 3.04 [2.80, 3.30] | +3.00 [+2.76, +3.25] | 0.952 [0.901, 0.952] |
| AR(1) ρ=0.9, σ=0.2 m | Expl. acc. to HSR | 39 | 60.26 [58.95, 61.63] | -58.18 [-58.54, -57.89] | 0.508 [0.323, 0.693] |
| AR(1) ρ=0.9, σ=0.2 m | Expl. acc. to sprint | 39 | 82.41 [74.83, 90.50] | -78.05 [-80.43, -75.00] | 0.442 [0.442, 0.484] |
| AR(1) ρ=0.9, σ=0.5 m | Distance | 39 | 34.23 [30.81, 37.84] | +33.99 [+30.71, +37.77] | 0.993 [0.986, 0.993] |
| AR(1) ρ=0.9, σ=0.5 m | Dist. < 15 | 39 | 23.15 [17.77, 28.81] | +22.85 [+17.60, +28.84] | 0.675 [0.532, 0.675] |
| AR(1) ρ=0.9, σ=0.5 m | Dist. 15-20 | 39 | 68.53 [67.27, 69.73] | +63.54 [+61.99, +64.79] | 0.900 [0.868, 0.900] |
| AR(1) ρ=0.9, σ=0.5 m | Dist. 20-25 (HSR) | 39 | 87.82 [77.13, 97.98] | +81.31 [+72.21, +89.34] | 0.754 [0.690, 0.839] |
| AR(1) ρ=0.9, σ=0.5 m | Dist. > 25 (sprint) | 39 | 200.51 [194.62, 206.10] | +144.85 [+136.60, +152.63] | 0.802 [0.754, 0.883] |
| AR(1) ρ=0.9, σ=0.5 m | HI dist. > 20 | 39 | 107.83 [97.63, 117.51] | +97.90 [+89.45, +105.49] | 0.859 [0.859, 0.932] |
| AR(1) ρ=0.9, σ=0.5 m | HSR efforts | 39 | 12.17 [11.20, 13.19] | -9.11 [-11.24, -7.15] | 0.948 [0.932, 0.948] |
| AR(1) ρ=0.9, σ=0.5 m | Sprint efforts | 39 | 28.40 [24.77, 32.21] | -18.99 [-22.40, -15.06] | 0.887 [0.842, 0.888] |
| AR(1) ρ=0.9, σ=0.5 m | PSV-99 | 39 | 15.60 [15.46, 15.74] | +15.49 [+15.34, +15.63] | 0.868 [0.811, 0.868] |
| AR(1) ρ=0.9, σ=0.5 m | Expl. acc. to HSR | 39 | 68.51 [51.45, 86.51] | -2.55 [-14.47, +12.20] | 0.101 [-0.221, 0.479] |
| AR(1) ρ=0.9, σ=0.5 m | Expl. acc. to sprint | 39 | 101.63 [100.78, 102.43] | +35.37 [+30.56, +39.13] | 0.068 [0.002, 0.115] |
| AR(1) ρ=0.99, σ=0.05 m | Distance | 39 | 0.08 [0.07, 0.09] | +0.08 [+0.07, +0.09] | 1.000 [1.000, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | Dist. < 15 | 39 | 0.14 [0.12, 0.17] | +0.13 [+0.09, +0.17] | 1.000 [0.998, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | Dist. 15-20 | 39 | 0.49 [0.46, 0.52] | -0.08 [-0.20, +0.01] | 0.999 [0.995, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | Dist. 20-25 (HSR) | 39 | 0.62 [0.56, 0.69] | -0.12 [-0.15, -0.09] | 0.999 [0.998, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | Dist. > 25 (sprint) | 39 | 1.22 [1.11, 1.33] | -0.05 [-0.20, +0.09] | 0.999 [0.995, 0.999] |
| AR(1) ρ=0.99, σ=0.05 m | HI dist. > 20 | 39 | 0.45 [0.41, 0.51] | -0.10 [-0.16, -0.04] | 1.000 [0.998, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | HSR efforts | 39 | 1.80 [1.26, 2.31] | -0.35 [-0.43, -0.26] | 0.991 [0.984, 0.991] |
| AR(1) ρ=0.99, σ=0.05 m | Sprint efforts | 39 | 0.85 [0.44, 1.23] | -0.56 [-0.60, -0.52] | 0.999 [0.998, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | PSV-99 | 39 | 0.27 [0.23, 0.31] | -0.17 [-0.20, -0.14] | 0.993 [0.986, 0.993] |
| AR(1) ρ=0.99, σ=0.05 m | Expl. acc. to HSR | 39 | 15.98 [12.19, 19.57] | -0.73 [-2.44, +0.66] | 0.907 [0.857, 0.959] |
| AR(1) ρ=0.99, σ=0.05 m | Expl. acc. to sprint | 39 | 14.12 [12.89, 15.28] | -2.44 [-11.11, +4.35] | 0.936 [0.908, 0.936] |
| AR(1) ρ=0.99, σ=0.1 m | Distance | 39 | 0.41 [0.34, 0.47] | +0.40 [+0.34, +0.48] | 1.000 [0.998, 1.000] |
| AR(1) ρ=0.99, σ=0.1 m | Dist. < 15 | 39 | 0.45 [0.35, 0.56] | +0.44 [+0.33, +0.56] | 0.999 [0.996, 0.999] |
| AR(1) ρ=0.99, σ=0.1 m | Dist. 15-20 | 39 | 0.94 [0.92, 0.95] | +0.26 [+0.06, +0.42] | 0.998 [0.993, 1.000] |
| AR(1) ρ=0.99, σ=0.1 m | Dist. 20-25 (HSR) | 39 | 1.28 [1.19, 1.38] | +0.27 [+0.13, +0.44] | 0.999 [0.996, 0.999] |
| AR(1) ρ=0.99, σ=0.1 m | Dist. > 25 (sprint) | 39 | 2.60 [2.50, 2.70] | +0.46 [+0.04, +0.86] | 0.995 [0.982, 0.995] |
| AR(1) ρ=0.99, σ=0.1 m | HI dist. > 20 | 39 | 0.94 [0.91, 0.96] | +0.32 [+0.31, +0.33] | 0.996 [0.995, 0.998] |
| AR(1) ρ=0.99, σ=0.1 m | HSR efforts | 39 | 3.01 [2.71, 3.29] | -0.69 [-1.19, -0.14] | 0.989 [0.985, 0.995] |
| AR(1) ρ=0.99, σ=0.1 m | Sprint efforts | 39 | 6.94 [5.49, 8.47] | +0.28 [-2.08, +3.01] | 0.970 [0.968, 0.983] |
| AR(1) ρ=0.99, σ=0.1 m | PSV-99 | 39 | 0.46 [0.37, 0.57] | -0.09 [-0.13, -0.04] | 0.982 [0.961, 0.982] |
| AR(1) ρ=0.99, σ=0.1 m | Expl. acc. to HSR | 39 | 27.14 [23.41, 30.68] | -1.45 [-6.50, +2.63] | 0.857 [0.830, 0.907] |
| AR(1) ρ=0.99, σ=0.1 m | Expl. acc. to sprint | 39 | 39.71 [36.33, 42.88] | +2.44 [-5.56, +8.70] | 0.881 [0.815, 0.930] |
| AR(1) ρ=0.99, σ=0.2 m | Distance | 39 | 1.48 [1.27, 1.70] | +1.47 [+1.27, +1.70] | 0.998 [0.992, 0.998] |
| AR(1) ρ=0.99, σ=0.2 m | Dist. < 15 | 39 | 1.46 [1.07, 1.86] | +1.43 [+1.06, +1.86] | 0.998 [0.993, 0.998] |
| AR(1) ρ=0.99, σ=0.2 m | Dist. 15-20 | 39 | 1.87 [1.45, 2.27] | +1.53 [+0.77, +2.13] | 0.995 [0.986, 0.995] |
| AR(1) ρ=0.99, σ=0.2 m | Dist. 20-25 (HSR) | 39 | 2.76 [2.39, 3.11] | +1.54 [+1.29, +1.77] | 0.992 [0.986, 0.993] |
| AR(1) ρ=0.99, σ=0.2 m | Dist. > 25 (sprint) | 39 | 4.88 [4.65, 5.11] | +2.39 [+2.25, +2.54] | 0.993 [0.977, 0.995] |
| AR(1) ρ=0.99, σ=0.2 m | HI dist. > 20 | 39 | 2.15 [1.79, 2.49] | +1.76 [+1.62, +1.89] | 0.997 [0.995, 0.997] |
| AR(1) ρ=0.99, σ=0.2 m | HSR efforts | 39 | 3.99 [3.91, 4.07] | -1.59 [-2.25, -0.86] | 0.982 [0.968, 0.982] |
| AR(1) ρ=0.99, σ=0.2 m | Sprint efforts | 39 | 11.03 [10.39, 11.64] | -3.35 [-4.82, -2.08] | 0.942 [0.942, 0.947] |
| AR(1) ρ=0.99, σ=0.2 m | PSV-99 | 39 | 0.86 [0.70, 1.03] | +0.27 [+0.10, +0.44] | 0.972 [0.946, 0.972] |
| AR(1) ρ=0.99, σ=0.2 m | Expl. acc. to HSR | 39 | 37.81 [34.48, 41.34] | -23.64 [-30.08, -18.42] | 0.677 [0.591, 0.772] |
| AR(1) ρ=0.99, σ=0.2 m | Expl. acc. to sprint | 39 | 51.62 [38.72, 63.72] | -24.39 [-30.56, -19.57] | 0.860 [0.770, 0.887] |
| AR(1) ρ=0.99, σ=0.5 m | Distance | 39 | 7.53 [6.62, 8.49] | +7.48 [+6.61, +8.49] | 0.997 [0.993, 0.997] |
| AR(1) ρ=0.99, σ=0.5 m | Dist. < 15 | 39 | 6.42 [4.93, 7.98] | +6.32 [+4.86, +7.99] | 0.995 [0.989, 0.995] |
| AR(1) ρ=0.99, σ=0.5 m | Dist. 15-20 | 39 | 11.99 [11.02, 12.91] | +11.47 [+10.27, +12.44] | 0.976 [0.953, 0.976] |
| AR(1) ρ=0.99, σ=0.5 m | Dist. 20-25 (HSR) | 39 | 11.32 [8.81, 13.70] | +10.27 [+8.02, +12.26] | 0.951 [0.951, 0.958] |
| AR(1) ρ=0.99, σ=0.5 m | Dist. > 25 (sprint) | 39 | 24.34 [23.09, 25.66] | +18.56 [+17.84, +19.24] | 0.964 [0.937, 0.982] |
| AR(1) ρ=0.99, σ=0.5 m | HI dist. > 20 | 39 | 13.55 [11.71, 15.30] | +12.44 [+10.65, +14.04] | 0.984 [0.979, 0.986] |
| AR(1) ρ=0.99, σ=0.5 m | HSR efforts | 39 | 6.65 [6.08, 7.19] | -1.52 [-3.46, +0.26] | 0.957 [0.942, 0.962] |
| AR(1) ρ=0.99, σ=0.5 m | Sprint efforts | 39 | 17.47 [13.82, 20.93] | -9.50 [-13.02, -5.42] | 0.922 [0.922, 0.948] |
| AR(1) ρ=0.99, σ=0.5 m | PSV-99 | 39 | 2.75 [2.45, 3.06] | +2.62 [+2.33, +2.92] | 0.943 [0.919, 0.953] |
| AR(1) ρ=0.99, σ=0.5 m | Expl. acc. to HSR | 39 | 46.67 [43.80, 49.39] | -45.09 [-46.05, -43.90] | 0.491 [0.279, 0.693] |
| AR(1) ρ=0.99, σ=0.5 m | Expl. acc. to sprint | 39 | 64.01 [56.72, 70.83] | -54.88 [-63.89, -47.83] | 0.373 [0.363, 0.373] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Distance | 39 | 4.50 [3.88, 5.15] | +4.46 [+3.87, +5.15] | 0.997 [0.991, 0.997] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Dist. < 15 | 39 | 4.03 [3.06, 5.04] | +3.96 [+3.01, +5.05] | 0.999 [0.997, 1.000] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Dist. 15-20 | 39 | 6.33 [5.47, 7.15] | +6.05 [+4.97, +6.91] | 0.986 [0.980, 0.989] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Dist. 20-25 (HSR) | 39 | 5.92 [4.82, 6.97] | +5.13 [+4.35, +5.83] | 0.988 [0.986, 0.992] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Dist. > 25 (sprint) | 39 | 15.88 [15.72, 16.06] | +11.76 [+11.17, +12.39] | 0.983 [0.956, 0.983] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | HI dist. > 20 | 39 | 7.62 [7.04, 8.17] | +6.86 [+6.50, +7.19] | 0.997 [0.995, 0.997] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | HSR efforts | 39 | 8.43 [8.32, 8.55] | -6.35 [-7.02, -5.62] | 0.965 [0.965, 0.966] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Sprint efforts | 39 | 10.53 [9.35, 11.66] | -7.26 [-8.85, -5.42] | 0.957 [0.952, 0.977] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | PSV-99 | 39 | 1.79 [1.64, 1.94] | +1.70 [+1.54, +1.87] | 0.967 [0.956, 0.967] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Expl. acc. to HSR | 39 | 49.43 [46.79, 51.93] | -47.27 [-48.78, -46.05] | 0.653 [0.544, 0.737] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Expl. acc. to sprint | 39 | 80.55 [80.17, 80.90] | -68.29 [-75.00, -63.04] | 0.506 [0.448, 0.568] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | Distance | 39 | 90.26 [82.59, 98.33] | +89.67 [+82.33, +98.14] | 0.978 [0.937, 0.978] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | Dist. < 15 | 39 | 16.51 [10.54, 22.78] | +15.32 [+8.72, +22.85] | -0.748 [-0.748, -0.705] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | Dist. 15-20 | 39 | 264.86 [236.64, 294.57] | +241.44 [+218.18, +270.38] | 0.800 [0.776, 0.800] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | Dist. 20-25 (HSR) | 39 | 474.59 [472.71, 476.38] | +433.53 [+432.42, +434.78] | 0.478 [0.269, 0.600] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | Dist. > 25 (sprint) | 39 | 1541.73 [1494.02, 1587.04] | +1071.64 [+1009.51, +1130.21] | 0.405 [0.405, 0.487] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | HI dist. > 20 | 39 | 671.79 [656.29, 686.51] | +600.11 [+588.62, +610.43] | 0.581 [0.534, 0.761] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | HSR efforts | 39 | 12.31 [11.11, 13.57] | -1.66 [-4.61, +1.06] | 0.853 [0.765, 0.922] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | Sprint efforts | 39 | 31.85 [28.86, 34.70] | -19.55 [-21.88, -16.87] | 0.839 [0.828, 0.841] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | PSV-99 | 39 | 38.66 [37.72, 39.54] | +38.39 [+37.44, +39.29] | 0.754 [0.732, 0.755] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | Expl. acc. to HSR | 39 | 158.71 [109.42, 210.73] | +76.36 [+44.08, +116.26] | 0.204 [-0.071, 0.496] |
| measured on SN-GSR (σ 0.62/0.62 m, ρ 0.832/0.848) | Expl. acc. to sprint | 39 | 498.67 [355.22, 633.16] | +385.37 [+293.48, +502.78] | 0.177 [0.048, 0.275] |

## Appendix: raw data files

| File | SHA-256 |
|---|---|
| `DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml` | `a884728647a564cf8b20e5223a1d04a5da6f6725e1e506c8651505327c197ba9` |
| `DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml` | `d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49` |
| `DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml` | `5490286436c86e6f80a538d2b1b379d0efd016e0defcaf594bc59d9ef5079f4c` |
| `DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml` | `56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec` |
