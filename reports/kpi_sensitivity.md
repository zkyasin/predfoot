# KPI error budget: first sensitivity study (T4.1)

> **This is a sensitivity analysis on simulated position errors.** None of these numbers is a measured
> error of predfoot: the real error process (variance, temporal correlation, common-mode part, dependence
> on the predicted uncertainty) is measured on SN-GSR later (ticket T4.2) and injected in T4.3. KPI
> definitions marked *assumption* below are ours, not SkillCorner's.

- **Command**: `uv run python scripts/kpi_sensitivity.py`
- **Git commit**: `f4e6967`; date 2026-10-31T02:20:55; runtime 229.4 s
- **Data**: IDSSE (Bassek, Rein, Weber & Memmert 2025, CC BY 4.0, doi:10.6084/m9.figshare.28196177), 7 matches (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), DFL tracking at 25 Hz, from the Hugging Face mirror `pysport/idsse-data` (SHA-256 of the raw files at the end).
- **Population**: 142 player-matches: outfield players with at least 45 minutes on the pitch (goalkeepers excluded).
- **Config**: `configs/kpi.yaml` (SHA-256 `b5bcace96008`), seed 20261031, 2000 bootstrap resamples.

## Summary

- **Gate G4** (clean data, 10 fps vs 25 Hz, MAPE < 0.5 %): passed by Distance, Dist. < 15, Dist. 15-20, Dist. 20-25 (HSR), HI dist. > 20, PSV-99; not passed by Dist. > 25 (sprint) 0.76 %, HSR efforts 0.71 %, Sprint efforts 0.90 %, Expl. acc. to HSR 12.34 %, Expl. acc. to sprint 13.20 %. Totals stay close (bias within ±0.52 % for every KPI except explosive accelerations): the misses come from discretisation at thresholds, where an effort near the minimum duration or an acceleration near 3 m/s² flips between the two pipelines, on small per-player counts (6.5 explosive accelerations to HSR per player-match on average).
- **Smoothing** is the main lever: with i.i.d. σ = 0.2 m, the distance error is 5.37 % with the 1 s window and 1.53 % with 2 s, but on clean data the 2 s window already moves explosive accelerations to HSR by +21.77 % (definitions and noise robustness pull in opposite directions).
- **Distance**: MAPE 1.56 % for i.i.d. σ = 0.1 m and 5.30 % for σ = 0.2 m, an overestimate (bias +5.28 % at 0.2 m). At σ = 0.2 m, slow errors (AR(1) ρ = 0.99, about 4 s) give 1.38 %, and errors correlated over about 0.4 s (ρ = 0.9) give 6.73 %, more than white noise: what matters is how much error power falls in the band the 1 s smoothing lets through, not correlation as such.
- **High-speed KPIs** react more: at i.i.d. σ = 0.1 m, distance above 25 km/h has MAPE 5.27 % and sprint efforts 14.19 % against the clean 10 fps control; PSV-99 stays within 0.77 % (bias +0.67 %).
- **Explosive accelerations** are the most fragile (a threshold on a second derivative): -54.11 % of the total at i.i.d. σ = 0.1 m and -81.86 % at 0.2 m, because noise breaks the runs of more than 0.7 s above 3 m/s².
- **Rankings** of players at i.i.d. σ = 0.2 m: Spearman 0.994 for HI distance, 0.927 for sprint efforts and 0.554 for explosive accelerations to HSR.

## Set-up

- **Cleaning**: the raw IDSSE feed holds the last position of a player who is not observed, then jumps to the re-acquired position; there are also a few identity swaps. Steps faster than 43.2 km/h are cut out of the source data before anything else (held samples, or the sample after the jump): 41893 of 21874234 player-samples (0.192 %). Without this step each glitch becomes a fake sprint after smoothing, and the 25 Hz and 10 fps pipelines disagree on explosive accelerations by tens of percent.
- **Reference**: clean IDSSE positions at 25 Hz through the standard pipeline: Savitzky-Golay smoothing of positions over 1 s (order 2, 25 samples), velocity and acceleration by central differences that never cross a gap, then the KPIs.
- **Control (gate G4)**: the same clean positions resampled to 10 fps (linear interpolation between the two neighbouring 25 Hz samples) and smoothed over 11 samples (1.1 s). Gate G4 asks for MAPE < 0.5 %.
- **Injected errors**: Gaussian, isotropic, added to the 25 Hz positions before the 10 fps pipeline; σ is the marginal standard deviation per axis. AR(1) correlation ρ is per 25 Hz frame: ρ = 0.9 is a correlation time of 0.38 s, ρ = 0.99 of 4.0 s. The common-mode condition gives every player of a frame the same slow error (ρ = 0.99) plus an individual one (ρ = 0.9), each with half of the variance. Linear resampling averages two 25 Hz samples at every other 10 fps sample, which halves i.i.d. noise variance there; this is part of the pipeline being tested.
- **Statistics**: MAPE over player-matches with a positive reference; bias = relative error of the total over all player-matches; Spearman correlation of player rankings, on per-90-minute values (PSV-99 as is). 95 % intervals: percentile bootstrap resampling whole matches. With 7 matches the intervals are rough, and they do not include noise-seed variability (one noise draw per condition).

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
| Distance | 9646.8 | 0.06 [0.06, 0.07] | -0.06 [-0.07, -0.06] | 1.000 [1.000, 1.000] | pass |
| Dist. < 15 | 7506.2 | 0.05 [0.04, 0.05] | -0.04 [-0.05, -0.03] | 1.000 [1.000, 1.000] | pass |
| Dist. 15-20 | 1404.6 | 0.22 [0.19, 0.26] | -0.11 [-0.14, -0.08] | 1.000 [1.000, 1.000] | pass |
| Dist. 20-25 (HSR) | 555.8 | 0.41 [0.37, 0.45] | -0.16 [-0.23, -0.10] | 1.000 [0.999, 1.000] | pass |
| Dist. > 25 (sprint) | 180.1 | 0.76 [0.62, 0.94] | -0.23 [-0.30, -0.17] | 1.000 [0.999, 1.000] | **fail** |
| HI dist. > 20 | 735.9 | 0.29 [0.26, 0.33] | -0.18 [-0.22, -0.14] | 1.000 [1.000, 1.000] | pass |
| HSR efforts | 38.3 | 0.71 [0.44, 0.98] | -0.52 [-0.64, -0.36] | 0.999 [0.996, 0.999] | **fail** |
| Sprint efforts | 8.8 | 0.90 [0.06, 2.43] | +0.00 [-0.24, +0.25] | 1.000 [0.998, 1.000] | **fail** |
| PSV-99 | 28.5 | 0.13 [0.11, 0.16] | -0.11 [-0.14, -0.08] | 1.000 [0.998, 1.000] | pass |
| Expl. acc. to HSR | 6.5 | 12.34 [9.40, 15.13] | +2.16 [-0.62, +5.43] | 0.947 [0.928, 0.963] | **fail** |
| Expl. acc. to sprint | 2.0 | 13.20 [8.99, 17.96] | +8.63 [+1.60, +16.67] | 0.933 [0.915, 0.948] | **fail** |

## Sensitivity to position errors (against the 25 Hz reference)

### MAPE (%)

| Condition | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 (HSR) | Dist. > 25 (sprint) | HI dist. > 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean, 10 fps | 0.06 | 0.05 | 0.22 | 0.41 | 0.76 | 0.29 | 0.71 | 0.90 | 0.13 | 12.34 | 13.20 |
| i.i.d. σ=0.05 m | 0.43 | 0.47 | 0.60 | 1.09 | 2.07 | 0.81 | 3.71 | 7.47 | 0.32 | 31.34 | 39.82 |
| i.i.d. σ=0.1 m | 1.56 | 1.47 | 1.86 | 2.45 | 5.12 | 2.44 | 6.23 | 13.89 | 0.70 | 52.35 | 65.41 |
| i.i.d. σ=0.2 m | 5.30 | 4.46 | 7.90 | 9.14 | 19.63 | 10.64 | 13.72 | 24.70 | 2.10 | 83.51 | 94.04 |
| i.i.d. σ=0.5 m | 25.68 | 17.59 | 47.28 | 66.59 | 154.02 | 80.76 | 31.99 | 47.95 | 10.93 | 87.22 | 88.52 |
| AR(1) ρ=0.9, σ=0.05 m | 0.56 | 0.58 | 0.80 | 1.34 | 2.99 | 0.97 | 3.71 | 8.93 | 0.43 | 31.25 | 38.34 |
| AR(1) ρ=0.9, σ=0.1 m | 1.98 | 1.81 | 2.62 | 3.52 | 6.81 | 3.42 | 5.77 | 12.25 | 1.01 | 39.70 | 61.88 |
| AR(1) ρ=0.9, σ=0.2 m | 6.73 | 5.58 | 10.37 | 11.79 | 26.56 | 13.88 | 8.64 | 22.88 | 3.32 | 61.53 | 83.17 |
| AR(1) ρ=0.9, σ=0.5 m | 32.08 | 20.39 | 63.05 | 90.23 | 220.63 | 111.45 | 12.31 | 29.56 | 16.21 | 85.49 | 109.12 |
| AR(1) ρ=0.99, σ=0.05 m | 0.07 | 0.12 | 0.45 | 0.76 | 1.53 | 0.52 | 2.16 | 3.92 | 0.25 | 20.48 | 23.61 |
| AR(1) ρ=0.99, σ=0.1 m | 0.38 | 0.42 | 0.81 | 1.39 | 2.79 | 1.02 | 3.07 | 8.42 | 0.42 | 25.74 | 35.48 |
| AR(1) ρ=0.99, σ=0.2 m | 1.38 | 1.31 | 1.82 | 2.70 | 5.07 | 2.28 | 4.21 | 12.06 | 0.83 | 36.05 | 48.29 |
| AR(1) ρ=0.99, σ=0.5 m | 6.97 | 5.73 | 10.99 | 12.38 | 27.88 | 14.58 | 7.62 | 18.49 | 3.00 | 49.12 | 70.98 |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | 4.19 | 3.63 | 5.93 | 6.90 | 16.33 | 7.98 | 7.72 | 15.08 | 1.93 | 56.17 | 77.49 |

### Bias (%, relative error of the total)

| Condition | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 (HSR) | Dist. > 25 (sprint) | HI dist. > 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean, 10 fps | -0.06 | -0.04 | -0.11 | -0.16 | -0.23 | -0.18 | -0.52 | +0.00 | -0.11 | +2.16 | +8.63 |
| i.i.d. σ=0.05 m | +0.43 | +0.46 | +0.20 | +0.49 | +0.81 | +0.57 | -1.75 | -1.28 | +0.11 | -20.37 | -28.06 |
| i.i.d. σ=0.1 m | +1.55 | +1.46 | +1.67 | +1.89 | +3.35 | +2.24 | -5.48 | -9.84 | +0.56 | -53.12 | -64.75 |
| i.i.d. σ=0.2 m | +5.28 | +4.44 | +7.43 | +8.10 | +14.81 | +9.74 | -13.32 | -20.80 | +2.07 | -81.47 | -91.73 |
| i.i.d. σ=0.5 m | +25.58 | +17.50 | +43.88 | +59.82 | +113.71 | +73.01 | -31.46 | -45.44 | +10.81 | -87.72 | -86.69 |
| AR(1) ρ=0.9, σ=0.05 m | +0.56 | +0.57 | +0.45 | +0.49 | +1.20 | +0.67 | -1.49 | -2.00 | +0.17 | -11.85 | -17.27 |
| AR(1) ρ=0.9, σ=0.1 m | +1.97 | +1.80 | +2.36 | +2.62 | +4.36 | +3.04 | -3.94 | -6.80 | +0.79 | -36.42 | -51.80 |
| AR(1) ρ=0.9, σ=0.2 m | +6.71 | +5.55 | +9.73 | +10.55 | +19.50 | +12.74 | -7.07 | -14.08 | +3.26 | -59.38 | -80.94 |
| AR(1) ρ=0.9, σ=0.5 m | +31.96 | +20.31 | +58.26 | +81.23 | +160.68 | +100.67 | -6.66 | -20.56 | +16.08 | +6.79 | +50.72 |
| AR(1) ρ=0.99, σ=0.05 m | +0.07 | +0.10 | -0.03 | +0.09 | -0.16 | +0.03 | -0.63 | -0.08 | -0.04 | +1.72 | +2.88 |
| AR(1) ρ=0.99, σ=0.1 m | +0.38 | +0.41 | +0.20 | +0.30 | +0.80 | +0.42 | -0.44 | +0.08 | -0.03 | -1.19 | -4.32 |
| AR(1) ρ=0.99, σ=0.2 m | +1.37 | +1.30 | +1.43 | +1.57 | +3.23 | +1.97 | -0.77 | -1.12 | +0.44 | -17.46 | -20.86 |
| AR(1) ρ=0.99, σ=0.5 m | +6.95 | +5.70 | +10.25 | +10.91 | +20.90 | +13.36 | -1.05 | -4.16 | +2.83 | -44.18 | -64.75 |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | +4.18 | +3.62 | +5.50 | +5.94 | +11.74 | +7.36 | -4.99 | -10.72 | +1.84 | -54.42 | -72.66 |

### Spearman correlation of player rankings

| Condition | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 (HSR) | Dist. > 25 (sprint) | HI dist. > 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean, 10 fps | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.947 | 0.933 |
| i.i.d. σ=0.05 m | 1.000 | 1.000 | 0.999 | 0.999 | 0.999 | 1.000 | 0.991 | 0.982 | 0.996 | 0.821 | 0.777 |
| i.i.d. σ=0.1 m | 0.999 | 0.999 | 0.997 | 0.997 | 0.997 | 0.999 | 0.984 | 0.958 | 0.994 | 0.690 | 0.552 |
| i.i.d. σ=0.2 m | 0.998 | 0.994 | 0.990 | 0.989 | 0.990 | 0.994 | 0.968 | 0.927 | 0.986 | 0.554 | 0.377 |
| i.i.d. σ=0.5 m | 0.993 | 0.800 | 0.925 | 0.867 | 0.878 | 0.910 | 0.880 | 0.827 | 0.949 | 0.121 | 0.084 |
| AR(1) ρ=0.9, σ=0.05 m | 1.000 | 0.999 | 0.999 | 0.999 | 0.998 | 0.999 | 0.989 | 0.980 | 0.995 | 0.809 | 0.815 |
| AR(1) ρ=0.9, σ=0.1 m | 0.999 | 0.998 | 0.997 | 0.996 | 0.994 | 0.998 | 0.977 | 0.958 | 0.988 | 0.733 | 0.547 |
| AR(1) ρ=0.9, σ=0.2 m | 0.998 | 0.990 | 0.985 | 0.985 | 0.982 | 0.992 | 0.966 | 0.935 | 0.969 | 0.467 | 0.313 |
| AR(1) ρ=0.9, σ=0.5 m | 0.991 | 0.553 | 0.907 | 0.812 | 0.827 | 0.869 | 0.922 | 0.875 | 0.879 | 0.182 | 0.182 |
| AR(1) ρ=0.99, σ=0.05 m | 1.000 | 1.000 | 0.999 | 0.999 | 0.999 | 1.000 | 0.995 | 0.992 | 0.998 | 0.929 | 0.911 |
| AR(1) ρ=0.99, σ=0.1 m | 1.000 | 0.999 | 0.999 | 0.998 | 0.998 | 0.999 | 0.992 | 0.980 | 0.995 | 0.898 | 0.875 |
| AR(1) ρ=0.99, σ=0.2 m | 0.999 | 0.999 | 0.997 | 0.996 | 0.996 | 0.999 | 0.984 | 0.965 | 0.988 | 0.763 | 0.760 |
| AR(1) ρ=0.99, σ=0.5 m | 0.997 | 0.991 | 0.983 | 0.978 | 0.979 | 0.989 | 0.960 | 0.907 | 0.959 | 0.560 | 0.347 |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | 0.998 | 0.995 | 0.991 | 0.990 | 0.990 | 0.995 | 0.970 | 0.945 | 0.980 | 0.647 | 0.561 |

Confidence intervals for every cell are in the appendix.

### MAPE (%) against the clean 10 fps control: the effect of the noise alone

Same noisy conditions, compared with clean data through the same 10 fps pipeline, so that the difference between the pipelines (gate G4 above) is not counted.

| Condition | Distance | Dist. < 15 | Dist. 15-20 | Dist. 20-25 (HSR) | Dist. > 25 (sprint) | HI dist. > 20 | HSR efforts | Sprint efforts | PSV-99 | Expl. acc. to HSR | Expl. acc. to sprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| i.i.d. σ=0.05 m | 0.50 | 0.51 | 0.67 | 1.13 | 2.17 | 0.95 | 3.40 | 6.69 | 0.35 | 30.79 | 42.70 |
| i.i.d. σ=0.1 m | 1.62 | 1.51 | 1.97 | 2.58 | 5.27 | 2.64 | 5.85 | 14.19 | 0.77 | 53.58 | 66.52 |
| i.i.d. σ=0.2 m | 5.37 | 4.51 | 8.03 | 9.31 | 19.85 | 10.85 | 13.26 | 25.01 | 2.21 | 83.99 | 94.63 |
| i.i.d. σ=0.5 m | 25.76 | 17.64 | 47.47 | 66.90 | 154.51 | 81.12 | 31.61 | 48.29 | 11.05 | 86.71 | 87.63 |
| AR(1) ρ=0.9, σ=0.05 m | 0.63 | 0.62 | 0.87 | 1.45 | 3.09 | 1.12 | 3.71 | 8.42 | 0.47 | 32.03 | 39.12 |
| AR(1) ρ=0.9, σ=0.1 m | 2.05 | 1.85 | 2.74 | 3.66 | 6.95 | 3.61 | 5.63 | 12.54 | 1.06 | 42.14 | 62.29 |
| AR(1) ρ=0.9, σ=0.2 m | 6.80 | 5.62 | 10.51 | 11.98 | 26.78 | 14.09 | 8.29 | 22.13 | 3.43 | 61.63 | 83.20 |
| AR(1) ρ=0.9, σ=0.5 m | 32.17 | 20.44 | 63.26 | 90.58 | 221.26 | 111.87 | 12.19 | 29.74 | 16.34 | 93.03 | 112.06 |
| AR(1) ρ=0.99, σ=0.05 m | 0.14 | 0.15 | 0.48 | 0.78 | 1.50 | 0.54 | 2.13 | 4.08 | 0.25 | 15.38 | 20.85 |
| AR(1) ρ=0.99, σ=0.1 m | 0.44 | 0.45 | 0.84 | 1.48 | 2.94 | 1.14 | 2.93 | 8.59 | 0.43 | 26.43 | 35.77 |
| AR(1) ρ=0.99, σ=0.2 m | 1.44 | 1.35 | 1.91 | 2.83 | 5.23 | 2.45 | 4.09 | 11.26 | 0.85 | 36.67 | 51.36 |
| AR(1) ρ=0.99, σ=0.5 m | 7.04 | 5.77 | 11.13 | 12.59 | 28.06 | 14.80 | 7.66 | 18.76 | 3.09 | 50.24 | 75.58 |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | 4.26 | 3.67 | 6.07 | 7.09 | 16.50 | 8.19 | 7.32 | 14.42 | 2.03 | 57.79 | 78.99 |

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
| 0.5 s | clean vs reference | 0.06 / +0.06 | 0.34 / +0.17 | 2.26 / -0.40 | 0.11 / -0.03 | 14.95 / -7.33 |
| 0.5 s | i.i.d. σ=0.2 m vs clean (same window) | 28.80 / +28.69 | 95.88 / +86.67 | 57.27 / -54.54 | 11.61 / +11.48 | 99.76 / -99.65 |
| 0.5 s | AR(1) ρ=0.99, σ=0.2 m vs clean (same window) | 2.99 / +2.98 | 5.39 / +4.96 | 16.69 / -10.04 | 1.33 / +1.22 | 74.52 / -75.35 |
| 1 s (default) | clean vs reference | 0.06 / -0.06 | 0.29 / -0.18 | 0.90 / +0.00 | 0.13 / -0.11 | 12.34 / +2.16 |
| 1 s (default) | i.i.d. σ=0.2 m vs clean (same window) | 5.37 / +5.35 | 10.85 / +9.94 | 25.01 / -20.80 | 2.21 / +2.17 | 83.99 / -81.86 |
| 1 s (default) | AR(1) ρ=0.99, σ=0.2 m vs clean (same window) | 1.44 / +1.44 | 2.45 / +2.16 | 11.26 / -1.12 | 0.85 / +0.55 | 36.67 / -19.20 |
| 2 s | clean vs reference | 0.64 / -0.64 | 2.04 / -1.94 | 6.21 / -1.12 | 0.40 / -0.33 | 35.24 / +21.77 |
| 2 s | i.i.d. σ=0.2 m vs clean (same window) | 1.53 / +1.52 | 2.41 / +2.11 | 11.16 / -4.77 | 0.71 / +0.56 | 57.90 / -57.79 |
| 2 s | AR(1) ρ=0.99, σ=0.2 m vs clean (same window) | 0.81 / +0.81 | 1.68 / +1.07 | 8.84 / +1.46 | 0.65 / +0.25 | 18.75 / -5.13 |
| none | clean vs reference | 0.12 / +0.12 | 0.55 / +0.31 | 2.65 / -0.72 | 0.13 / -0.01 | 16.05 / -10.24 |
| none | i.i.d. σ=0.2 m vs clean (same window) | 95.36 / +95.03 | 820.13 / +727.59 | 77.68 / -75.50 | 22.98 / +22.77 | 99.82 / -99.88 |
| none | AR(1) ρ=0.99, σ=0.2 m vs clean (same window) | 5.38 / +5.36 | 10.74 / +9.88 | 21.03 / -17.73 | 1.88 / +1.81 | 85.56 / -85.95 |

## Unpublished definitions (clean data, 10 fps)

Change of each KPI when one assumption changes, against the default configuration (bias % of the total, MAPE % across player-matches). These are the choices SkillCorner does not publish; they move the KPI values themselves, before any position error.

| Variant | KPIs that change |
|---|---|
| min_effort_0.5s | HSR efforts: +29.0 % (MAPE 30.3 %); Sprint efforts: +37.2 % (MAPE 47.7 %) |
| min_effort_2s | HSR efforts: -40.2 % (MAPE 40.6 %); Sprint efforts: -49.8 % (MAPE 50.0 %) |
| psv_run_peaks | PSV-99: +3.1 % (MAPE 3.4 %) |
| psv_min_activity_0.5s | PSV-99: -0.3 % (MAPE 0.3 %) |
| psv_min_activity_2s | PSV-99: +0.9 % (MAPE 0.9 %) |

## Reading these numbers

- Errors here are stationary, isotropic and independent of the camera; real broadcast errors grow with the distance to the camera, are larger along the viewing direction and depend on the calibration of each frame (T4.2 measures them, T4.3 injects errors driven by the predicted uncertainty σ̂).
- The reference is IDSSE's own tracking, not ground truth; injected errors are added on top of it.
- Distance is robust to noise once positions are smoothed: for a moving player, noise lengthens each step only at second order. Speed-threshold KPIs (counts, PSV-99, explosive accelerations) react at first order, because noise moves speed across thresholds.

## Appendix: all comparisons with 95 % intervals

| Condition | KPI | n | MAPE % | Bias % | Spearman |
|---|---|---|---|---|---|
| clean, 10 fps | Distance | 142 | 0.06 [0.06, 0.07] | -0.06 [-0.07, -0.06] | 1.000 [1.000, 1.000] |
| clean, 10 fps | Dist. < 15 | 142 | 0.05 [0.04, 0.05] | -0.04 [-0.05, -0.03] | 1.000 [1.000, 1.000] |
| clean, 10 fps | Dist. 15-20 | 142 | 0.22 [0.19, 0.26] | -0.11 [-0.14, -0.08] | 1.000 [1.000, 1.000] |
| clean, 10 fps | Dist. 20-25 (HSR) | 142 | 0.41 [0.37, 0.45] | -0.16 [-0.23, -0.10] | 1.000 [0.999, 1.000] |
| clean, 10 fps | Dist. > 25 (sprint) | 142 | 0.76 [0.62, 0.94] | -0.23 [-0.30, -0.17] | 1.000 [0.999, 1.000] |
| clean, 10 fps | HI dist. > 20 | 142 | 0.29 [0.26, 0.33] | -0.18 [-0.22, -0.14] | 1.000 [1.000, 1.000] |
| clean, 10 fps | HSR efforts | 142 | 0.71 [0.44, 0.98] | -0.52 [-0.64, -0.36] | 0.999 [0.996, 0.999] |
| clean, 10 fps | Sprint efforts | 142 | 0.90 [0.06, 2.43] | +0.00 [-0.24, +0.25] | 1.000 [0.998, 1.000] |
| clean, 10 fps | PSV-99 | 142 | 0.13 [0.11, 0.16] | -0.11 [-0.14, -0.08] | 1.000 [0.998, 1.000] |
| clean, 10 fps | Expl. acc. to HSR | 142 | 12.34 [9.40, 15.13] | +2.16 [-0.62, +5.43] | 0.947 [0.928, 0.963] |
| clean, 10 fps | Expl. acc. to sprint | 142 | 13.20 [8.99, 17.96] | +8.63 [+1.60, +16.67] | 0.933 [0.915, 0.948] |
| i.i.d. σ=0.05 m | Distance | 142 | 0.43 [0.39, 0.48] | +0.43 [+0.39, +0.48] | 1.000 [0.999, 1.000] |
| i.i.d. σ=0.05 m | Dist. < 15 | 142 | 0.47 [0.40, 0.54] | +0.46 [+0.40, +0.54] | 1.000 [0.999, 1.000] |
| i.i.d. σ=0.05 m | Dist. 15-20 | 142 | 0.60 [0.53, 0.66] | +0.20 [+0.13, +0.28] | 0.999 [0.998, 0.999] |
| i.i.d. σ=0.05 m | Dist. 20-25 (HSR) | 142 | 1.09 [0.98, 1.20] | +0.49 [+0.34, +0.64] | 0.999 [0.998, 0.999] |
| i.i.d. σ=0.05 m | Dist. > 25 (sprint) | 142 | 2.07 [1.71, 2.50] | +0.81 [+0.54, +1.08] | 0.999 [0.998, 0.999] |
| i.i.d. σ=0.05 m | HI dist. > 20 | 142 | 0.81 [0.74, 0.90] | +0.57 [+0.49, +0.64] | 1.000 [0.999, 1.000] |
| i.i.d. σ=0.05 m | HSR efforts | 142 | 3.71 [3.46, 3.98] | -1.75 [-2.59, -1.07] | 0.991 [0.980, 0.994] |
| i.i.d. σ=0.05 m | Sprint efforts | 142 | 7.47 [6.30, 8.97] | -1.28 [-2.31, -0.08] | 0.982 [0.970, 0.990] |
| i.i.d. σ=0.05 m | PSV-99 | 142 | 0.32 [0.28, 0.37] | +0.11 [+0.06, +0.17] | 0.996 [0.992, 0.997] |
| i.i.d. σ=0.05 m | Expl. acc. to HSR | 142 | 31.34 [28.41, 34.42] | -20.37 [-24.60, -16.11] | 0.821 [0.769, 0.850] |
| i.i.d. σ=0.05 m | Expl. acc. to sprint | 142 | 39.82 [36.19, 42.70] | -28.06 [-36.27, -17.69] | 0.777 [0.718, 0.837] |
| i.i.d. σ=0.1 m | Distance | 142 | 1.56 [1.45, 1.70] | +1.55 [+1.44, +1.70] | 0.999 [0.998, 0.999] |
| i.i.d. σ=0.1 m | Dist. < 15 | 142 | 1.47 [1.31, 1.68] | +1.46 [+1.29, +1.68] | 0.999 [0.997, 0.999] |
| i.i.d. σ=0.1 m | Dist. 15-20 | 142 | 1.86 [1.65, 2.07] | +1.67 [+1.49, +1.86] | 0.997 [0.992, 0.998] |
| i.i.d. σ=0.1 m | Dist. 20-25 (HSR) | 142 | 2.45 [2.06, 2.87] | +1.89 [+1.48, +2.29] | 0.997 [0.995, 0.997] |
| i.i.d. σ=0.1 m | Dist. > 25 (sprint) | 142 | 5.12 [4.63, 5.65] | +3.35 [+2.83, +3.91] | 0.997 [0.992, 0.998] |
| i.i.d. σ=0.1 m | HI dist. > 20 | 142 | 2.44 [2.14, 2.72] | +2.24 [+1.97, +2.48] | 0.999 [0.997, 0.999] |
| i.i.d. σ=0.1 m | HSR efforts | 142 | 6.23 [5.41, 7.29] | -5.48 [-6.67, -4.31] | 0.984 [0.969, 0.988] |
| i.i.d. σ=0.1 m | Sprint efforts | 142 | 13.89 [12.10, 15.64] | -9.84 [-11.48, -8.32] | 0.958 [0.939, 0.968] |
| i.i.d. σ=0.1 m | PSV-99 | 142 | 0.70 [0.65, 0.77] | +0.56 [+0.47, +0.68] | 0.994 [0.988, 0.995] |
| i.i.d. σ=0.1 m | Expl. acc. to HSR | 142 | 52.35 [48.06, 56.28] | -53.12 [-56.50, -49.61] | 0.690 [0.604, 0.749] |
| i.i.d. σ=0.1 m | Expl. acc. to sprint | 142 | 65.41 [61.35, 69.86] | -64.75 [-69.20, -60.07] | 0.552 [0.429, 0.675] |
| i.i.d. σ=0.2 m | Distance | 142 | 5.30 [5.00, 5.70] | +5.28 [+4.98, +5.70] | 0.998 [0.996, 0.999] |
| i.i.d. σ=0.2 m | Dist. < 15 | 142 | 4.46 [4.00, 5.09] | +4.44 [+3.97, +5.07] | 0.994 [0.989, 0.996] |
| i.i.d. σ=0.2 m | Dist. 15-20 | 142 | 7.90 [7.05, 8.76] | +7.43 [+6.66, +8.28] | 0.990 [0.982, 0.993] |
| i.i.d. σ=0.2 m | Dist. 20-25 (HSR) | 142 | 9.14 [8.22, 10.01] | +8.10 [+7.48, +8.65] | 0.989 [0.982, 0.991] |
| i.i.d. σ=0.2 m | Dist. > 25 (sprint) | 142 | 19.63 [17.99, 21.93] | +14.81 [+13.50, +16.19] | 0.990 [0.981, 0.993] |
| i.i.d. σ=0.2 m | HI dist. > 20 | 142 | 10.64 [9.89, 11.42] | +9.74 [+9.23, +10.22] | 0.994 [0.991, 0.995] |
| i.i.d. σ=0.2 m | HSR efforts | 142 | 13.72 [13.03, 14.45] | -13.32 [-14.11, -12.41] | 0.968 [0.947, 0.976] |
| i.i.d. σ=0.2 m | Sprint efforts | 142 | 24.70 [22.09, 27.47] | -20.80 [-22.75, -18.60] | 0.927 [0.904, 0.940] |
| i.i.d. σ=0.2 m | PSV-99 | 142 | 2.10 [1.97, 2.23] | +2.07 [+1.93, +2.21] | 0.986 [0.976, 0.988] |
| i.i.d. σ=0.2 m | Expl. acc. to HSR | 142 | 83.51 [79.73, 86.16] | -81.47 [-84.34, -77.25] | 0.554 [0.496, 0.617] |
| i.i.d. σ=0.2 m | Expl. acc. to sprint | 142 | 94.04 [92.25, 96.04] | -91.73 [-94.38, -88.06] | 0.377 [0.228, 0.512] |
| i.i.d. σ=0.5 m | Distance | 142 | 25.68 [24.44, 27.36] | +25.58 [+24.35, +27.26] | 0.993 [0.986, 0.995] |
| i.i.d. σ=0.5 m | Dist. < 15 | 142 | 17.59 [15.78, 20.02] | +17.50 [+15.67, +19.94] | 0.800 [0.729, 0.864] |
| i.i.d. σ=0.5 m | Dist. 15-20 | 142 | 47.28 [42.51, 51.93] | +43.88 [+39.81, +48.41] | 0.925 [0.884, 0.951] |
| i.i.d. σ=0.5 m | Dist. 20-25 (HSR) | 142 | 66.59 [61.53, 71.44] | +59.82 [+56.85, +62.60] | 0.867 [0.803, 0.917] |
| i.i.d. σ=0.5 m | Dist. > 25 (sprint) | 142 | 154.02 [141.29, 170.89] | +113.71 [+104.73, +123.60] | 0.878 [0.848, 0.903] |
| i.i.d. σ=0.5 m | HI dist. > 20 | 142 | 80.76 [75.72, 85.97] | +73.01 [+69.75, +75.73] | 0.910 [0.883, 0.934] |
| i.i.d. σ=0.5 m | HSR efforts | 142 | 31.99 [30.42, 33.11] | -31.46 [-32.69, -29.92] | 0.880 [0.841, 0.906] |
| i.i.d. σ=0.5 m | Sprint efforts | 142 | 47.95 [46.15, 50.27] | -45.44 [-46.80, -44.04] | 0.827 [0.758, 0.885] |
| i.i.d. σ=0.5 m | PSV-99 | 142 | 10.93 [10.51, 11.51] | +10.81 [+10.39, +11.37] | 0.949 [0.932, 0.957] |
| i.i.d. σ=0.5 m | Expl. acc. to HSR | 142 | 87.22 [84.34, 90.26] | -87.72 [-89.82, -85.24] | 0.121 [-0.051, 0.312] |
| i.i.d. σ=0.5 m | Expl. acc. to sprint | 142 | 88.52 [84.48, 92.37] | -86.69 [-88.85, -84.31] | 0.084 [-0.028, 0.170] |
| AR(1) ρ=0.9, σ=0.05 m | Distance | 142 | 0.56 [0.52, 0.62] | +0.56 [+0.51, +0.62] | 1.000 [0.999, 1.000] |
| AR(1) ρ=0.9, σ=0.05 m | Dist. < 15 | 142 | 0.58 [0.50, 0.67] | +0.57 [+0.49, +0.67] | 0.999 [0.999, 1.000] |
| AR(1) ρ=0.9, σ=0.05 m | Dist. 15-20 | 142 | 0.80 [0.72, 0.89] | +0.45 [+0.31, +0.58] | 0.999 [0.996, 0.999] |
| AR(1) ρ=0.9, σ=0.05 m | Dist. 20-25 (HSR) | 142 | 1.34 [1.23, 1.42] | +0.49 [+0.25, +0.68] | 0.999 [0.998, 0.999] |
| AR(1) ρ=0.9, σ=0.05 m | Dist. > 25 (sprint) | 142 | 2.99 [2.59, 3.40] | +1.20 [+0.75, +1.66] | 0.998 [0.995, 0.999] |
| AR(1) ρ=0.9, σ=0.05 m | HI dist. > 20 | 142 | 0.97 [0.92, 1.02] | +0.67 [+0.56, +0.76] | 0.999 [0.998, 1.000] |
| AR(1) ρ=0.9, σ=0.05 m | HSR efforts | 142 | 3.71 [2.91, 4.41] | -1.49 [-2.09, -0.89] | 0.989 [0.981, 0.992] |
| AR(1) ρ=0.9, σ=0.05 m | Sprint efforts | 142 | 8.93 [7.26, 10.61] | -2.00 [-3.65, -0.08] | 0.980 [0.968, 0.988] |
| AR(1) ρ=0.9, σ=0.05 m | PSV-99 | 142 | 0.43 [0.37, 0.49] | +0.17 [+0.12, +0.25] | 0.995 [0.991, 0.996] |
| AR(1) ρ=0.9, σ=0.05 m | Expl. acc. to HSR | 142 | 31.25 [26.49, 36.34] | -11.85 [-15.80, -8.03] | 0.809 [0.761, 0.850] |
| AR(1) ρ=0.9, σ=0.05 m | Expl. acc. to sprint | 142 | 38.34 [29.51, 45.73] | -17.27 [-22.48, -10.46] | 0.815 [0.764, 0.868] |
| AR(1) ρ=0.9, σ=0.1 m | Distance | 142 | 1.98 [1.84, 2.16] | +1.97 [+1.83, +2.16] | 0.999 [0.998, 0.999] |
| AR(1) ρ=0.9, σ=0.1 m | Dist. < 15 | 142 | 1.81 [1.61, 2.07] | +1.80 [+1.59, +2.06] | 0.998 [0.996, 0.998] |
| AR(1) ρ=0.9, σ=0.1 m | Dist. 15-20 | 142 | 2.62 [2.27, 2.98] | +2.36 [+2.04, +2.72] | 0.997 [0.992, 0.998] |
| AR(1) ρ=0.9, σ=0.1 m | Dist. 20-25 (HSR) | 142 | 3.52 [3.18, 3.87] | +2.62 [+2.09, +3.12] | 0.996 [0.992, 0.997] |
| AR(1) ρ=0.9, σ=0.1 m | Dist. > 25 (sprint) | 142 | 6.81 [5.41, 8.48] | +4.36 [+3.54, +5.25] | 0.994 [0.990, 0.996] |
| AR(1) ρ=0.9, σ=0.1 m | HI dist. > 20 | 142 | 3.42 [3.13, 3.71] | +3.04 [+2.78, +3.30] | 0.998 [0.995, 0.998] |
| AR(1) ρ=0.9, σ=0.1 m | HSR efforts | 142 | 5.77 [5.13, 6.39] | -3.94 [-4.70, -3.14] | 0.977 [0.952, 0.984] |
| AR(1) ρ=0.9, σ=0.1 m | Sprint efforts | 142 | 12.25 [10.00, 14.55] | -6.80 [-8.26, -5.44] | 0.958 [0.931, 0.975] |
| AR(1) ρ=0.9, σ=0.1 m | PSV-99 | 142 | 1.01 [0.89, 1.13] | +0.79 [+0.63, +0.98] | 0.988 [0.981, 0.990] |
| AR(1) ρ=0.9, σ=0.1 m | Expl. acc. to HSR | 142 | 39.70 [37.44, 42.31] | -36.42 [-39.50, -33.47] | 0.733 [0.658, 0.783] |
| AR(1) ρ=0.9, σ=0.1 m | Expl. acc. to sprint | 142 | 61.88 [57.52, 66.53] | -51.80 [-58.71, -44.85] | 0.547 [0.455, 0.648] |
| AR(1) ρ=0.9, σ=0.2 m | Distance | 142 | 6.73 [6.35, 7.25] | +6.71 [+6.33, +7.23] | 0.998 [0.995, 0.998] |
| AR(1) ρ=0.9, σ=0.2 m | Dist. < 15 | 142 | 5.58 [4.98, 6.38] | +5.55 [+4.95, +6.35] | 0.990 [0.984, 0.993] |
| AR(1) ρ=0.9, σ=0.2 m | Dist. 15-20 | 142 | 10.37 [9.28, 11.55] | +9.73 [+8.79, +10.83] | 0.985 [0.973, 0.989] |
| AR(1) ρ=0.9, σ=0.2 m | Dist. 20-25 (HSR) | 142 | 11.79 [10.61, 12.90] | +10.55 [+9.75, +11.33] | 0.985 [0.976, 0.988] |
| AR(1) ρ=0.9, σ=0.2 m | Dist. > 25 (sprint) | 142 | 26.56 [24.17, 30.19] | +19.50 [+17.32, +21.62] | 0.982 [0.967, 0.987] |
| AR(1) ρ=0.9, σ=0.2 m | HI dist. > 20 | 142 | 13.88 [12.92, 14.82] | +12.74 [+11.96, +13.50] | 0.992 [0.987, 0.993] |
| AR(1) ρ=0.9, σ=0.2 m | HSR efforts | 142 | 8.64 [7.87, 9.36] | -7.07 [-7.97, -6.26] | 0.966 [0.952, 0.970] |
| AR(1) ρ=0.9, σ=0.2 m | Sprint efforts | 142 | 22.88 [19.45, 27.19] | -14.08 [-15.89, -12.26] | 0.935 [0.912, 0.946] |
| AR(1) ρ=0.9, σ=0.2 m | PSV-99 | 142 | 3.32 [3.04, 3.66] | +3.26 [+3.00, +3.58] | 0.969 [0.955, 0.976] |
| AR(1) ρ=0.9, σ=0.2 m | Expl. acc. to HSR | 142 | 61.53 [59.04, 63.70] | -59.38 [-62.12, -56.49] | 0.467 [0.336, 0.595] |
| AR(1) ρ=0.9, σ=0.2 m | Expl. acc. to sprint | 142 | 83.17 [77.61, 88.20] | -80.94 [-85.37, -77.47] | 0.313 [0.155, 0.461] |
| AR(1) ρ=0.9, σ=0.5 m | Distance | 142 | 32.08 [30.57, 34.18] | +31.96 [+30.47, +34.07] | 0.991 [0.982, 0.994] |
| AR(1) ρ=0.9, σ=0.5 m | Dist. < 15 | 142 | 20.39 [18.30, 23.30] | +20.31 [+18.19, +23.20] | 0.553 [0.464, 0.632] |
| AR(1) ρ=0.9, σ=0.5 m | Dist. 15-20 | 142 | 63.05 [56.48, 69.49] | +58.26 [+52.71, +64.36] | 0.907 [0.866, 0.938] |
| AR(1) ρ=0.9, σ=0.5 m | Dist. 20-25 (HSR) | 142 | 90.23 [84.00, 96.31] | +81.23 [+77.26, +85.18] | 0.812 [0.745, 0.855] |
| AR(1) ρ=0.9, σ=0.5 m | Dist. > 25 (sprint) | 142 | 220.63 [200.89, 246.19] | +160.68 [+148.11, +174.20] | 0.827 [0.778, 0.873] |
| AR(1) ρ=0.9, σ=0.5 m | HI dist. > 20 | 142 | 111.45 [104.80, 117.91] | +100.67 [+96.35, +104.29] | 0.869 [0.840, 0.902] |
| AR(1) ρ=0.9, σ=0.5 m | HSR efforts | 142 | 12.31 [10.84, 14.05] | -6.66 [-8.77, -4.53] | 0.922 [0.882, 0.946] |
| AR(1) ρ=0.9, σ=0.5 m | Sprint efforts | 142 | 29.56 [26.80, 33.42] | -20.56 [-22.87, -17.83] | 0.875 [0.839, 0.896] |
| AR(1) ρ=0.9, σ=0.5 m | PSV-99 | 142 | 16.21 [15.71, 16.80] | +16.08 [+15.60, +16.66] | 0.879 [0.840, 0.912] |
| AR(1) ρ=0.9, σ=0.5 m | Expl. acc. to HSR | 142 | 85.49 [64.60, 107.17] | +6.79 [-3.57, +18.45] | 0.182 [0.017, 0.357] |
| AR(1) ρ=0.9, σ=0.5 m | Expl. acc. to sprint | 142 | 109.12 [92.99, 124.92] | +50.72 [+30.12, +77.39] | 0.182 [0.055, 0.329] |
| AR(1) ρ=0.99, σ=0.05 m | Distance | 142 | 0.07 [0.06, 0.09] | +0.07 [+0.06, +0.08] | 1.000 [1.000, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | Dist. < 15 | 142 | 0.12 [0.10, 0.14] | +0.10 [+0.08, +0.12] | 1.000 [0.999, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | Dist. 15-20 | 142 | 0.45 [0.41, 0.49] | -0.03 [-0.11, +0.04] | 0.999 [0.998, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | Dist. 20-25 (HSR) | 142 | 0.76 [0.68, 0.84] | +0.09 [-0.07, +0.29] | 0.999 [0.999, 0.999] |
| AR(1) ρ=0.99, σ=0.05 m | Dist. > 25 (sprint) | 142 | 1.53 [1.30, 1.77] | -0.16 [-0.43, +0.04] | 0.999 [0.998, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | HI dist. > 20 | 142 | 0.52 [0.46, 0.60] | +0.03 [-0.10, +0.19] | 1.000 [0.999, 1.000] |
| AR(1) ρ=0.99, σ=0.05 m | HSR efforts | 142 | 2.16 [1.81, 2.54] | -0.63 [-0.93, -0.35] | 0.995 [0.989, 0.996] |
| AR(1) ρ=0.99, σ=0.05 m | Sprint efforts | 142 | 3.92 [1.99, 6.08] | -0.08 [-0.82, +0.81] | 0.992 [0.983, 0.996] |
| AR(1) ρ=0.99, σ=0.05 m | PSV-99 | 142 | 0.25 [0.22, 0.28] | -0.04 [-0.11, +0.02] | 0.998 [0.995, 0.998] |
| AR(1) ρ=0.99, σ=0.05 m | Expl. acc. to HSR | 142 | 20.48 [15.88, 26.51] | +1.72 [-2.05, +6.55] | 0.929 [0.902, 0.948] |
| AR(1) ρ=0.99, σ=0.05 m | Expl. acc. to sprint | 142 | 23.61 [17.37, 29.91] | +2.88 [-3.39, +9.03] | 0.911 [0.872, 0.943] |
| AR(1) ρ=0.99, σ=0.1 m | Distance | 142 | 0.38 [0.34, 0.42] | +0.38 [+0.34, +0.42] | 1.000 [0.999, 1.000] |
| AR(1) ρ=0.99, σ=0.1 m | Dist. < 15 | 142 | 0.42 [0.38, 0.47] | +0.41 [+0.36, +0.46] | 0.999 [0.998, 1.000] |
| AR(1) ρ=0.99, σ=0.1 m | Dist. 15-20 | 142 | 0.81 [0.71, 0.89] | +0.20 [+0.12, +0.28] | 0.999 [0.997, 0.999] |
| AR(1) ρ=0.99, σ=0.1 m | Dist. 20-25 (HSR) | 142 | 1.39 [1.21, 1.60] | +0.30 [+0.19, +0.42] | 0.998 [0.997, 0.998] |
| AR(1) ρ=0.99, σ=0.1 m | Dist. > 25 (sprint) | 142 | 2.79 [2.43, 3.16] | +0.80 [+0.38, +1.16] | 0.998 [0.995, 0.999] |
| AR(1) ρ=0.99, σ=0.1 m | HI dist. > 20 | 142 | 1.02 [0.94, 1.11] | +0.42 [+0.34, +0.51] | 0.999 [0.998, 0.999] |
| AR(1) ρ=0.99, σ=0.1 m | HSR efforts | 142 | 3.07 [2.92, 3.19] | -0.44 [-1.11, +0.26] | 0.992 [0.985, 0.994] |
| AR(1) ρ=0.99, σ=0.1 m | Sprint efforts | 142 | 8.42 [6.37, 10.61] | +0.08 [-1.40, +1.45] | 0.980 [0.964, 0.990] |
| AR(1) ρ=0.99, σ=0.1 m | PSV-99 | 142 | 0.42 [0.35, 0.49] | -0.03 [-0.08, +0.02] | 0.995 [0.989, 0.996] |
| AR(1) ρ=0.99, σ=0.1 m | Expl. acc. to HSR | 142 | 25.74 [22.08, 29.00] | -1.19 [-4.95, +2.72] | 0.898 [0.861, 0.929] |
| AR(1) ρ=0.99, σ=0.1 m | Expl. acc. to sprint | 142 | 35.48 [30.18, 40.43] | -4.32 [-12.08, +5.47] | 0.875 [0.841, 0.909] |
| AR(1) ρ=0.99, σ=0.2 m | Distance | 142 | 1.38 [1.27, 1.50] | +1.37 [+1.27, +1.51] | 0.999 [0.998, 0.999] |
| AR(1) ρ=0.99, σ=0.2 m | Dist. < 15 | 142 | 1.31 [1.16, 1.51] | +1.30 [+1.15, +1.50] | 0.999 [0.997, 0.999] |
| AR(1) ρ=0.99, σ=0.2 m | Dist. 15-20 | 142 | 1.82 [1.55, 2.14] | +1.43 [+1.08, +1.81] | 0.997 [0.993, 0.997] |
| AR(1) ρ=0.99, σ=0.2 m | Dist. 20-25 (HSR) | 142 | 2.70 [2.45, 2.93] | +1.57 [+1.44, +1.70] | 0.996 [0.994, 0.996] |
| AR(1) ρ=0.99, σ=0.2 m | Dist. > 25 (sprint) | 142 | 5.07 [4.63, 5.61] | +3.23 [+2.67, +3.82] | 0.996 [0.993, 0.997] |
| AR(1) ρ=0.99, σ=0.2 m | HI dist. > 20 | 142 | 2.28 [2.07, 2.49] | +1.97 [+1.84, +2.09] | 0.999 [0.998, 0.999] |
| AR(1) ρ=0.99, σ=0.2 m | HSR efforts | 142 | 4.21 [3.72, 4.73] | -0.77 [-1.63, +0.15] | 0.984 [0.975, 0.987] |
| AR(1) ρ=0.99, σ=0.2 m | Sprint efforts | 142 | 12.06 [9.22, 15.89] | -1.12 [-2.63, +0.34] | 0.965 [0.950, 0.975] |
| AR(1) ρ=0.99, σ=0.2 m | PSV-99 | 142 | 0.83 [0.73, 0.95] | +0.44 [+0.28, +0.59] | 0.988 [0.980, 0.991] |
| AR(1) ρ=0.99, σ=0.2 m | Expl. acc. to HSR | 142 | 36.05 [28.26, 44.64] | -17.46 [-23.61, -11.23] | 0.763 [0.689, 0.832] |
| AR(1) ρ=0.99, σ=0.2 m | Expl. acc. to sprint | 142 | 48.29 [33.62, 58.91] | -20.86 [-27.48, -13.16] | 0.760 [0.595, 0.873] |
| AR(1) ρ=0.99, σ=0.5 m | Distance | 142 | 6.97 [6.59, 7.50] | +6.95 [+6.57, +7.49] | 0.997 [0.994, 0.998] |
| AR(1) ρ=0.99, σ=0.5 m | Dist. < 15 | 142 | 5.73 [5.13, 6.55] | +5.70 [+5.10, +6.52] | 0.991 [0.986, 0.994] |
| AR(1) ρ=0.99, σ=0.5 m | Dist. 15-20 | 142 | 10.99 [9.84, 12.19] | +10.25 [+9.22, +11.36] | 0.983 [0.968, 0.989] |
| AR(1) ρ=0.99, σ=0.5 m | Dist. 20-25 (HSR) | 142 | 12.38 [11.04, 13.52] | +10.91 [+9.94, +11.68] | 0.978 [0.965, 0.985] |
| AR(1) ρ=0.99, σ=0.5 m | Dist. > 25 (sprint) | 142 | 27.88 [25.38, 30.59] | +20.90 [+18.62, +23.22] | 0.979 [0.962, 0.986] |
| AR(1) ρ=0.99, σ=0.5 m | HI dist. > 20 | 142 | 14.58 [13.52, 15.51] | +13.36 [+12.40, +14.13] | 0.989 [0.982, 0.992] |
| AR(1) ρ=0.99, σ=0.5 m | HSR efforts | 142 | 7.62 [6.92, 8.28] | -1.05 [-2.44, +0.33] | 0.960 [0.927, 0.970] |
| AR(1) ρ=0.99, σ=0.5 m | Sprint efforts | 142 | 18.49 [16.01, 20.86] | -4.16 [-7.75, -1.15] | 0.907 [0.854, 0.946] |
| AR(1) ρ=0.99, σ=0.5 m | PSV-99 | 142 | 3.00 [2.73, 3.27] | +2.83 [+2.50, +3.15] | 0.959 [0.948, 0.968] |
| AR(1) ρ=0.99, σ=0.5 m | Expl. acc. to HSR | 142 | 49.12 [44.29, 53.98] | -44.18 [-45.79, -42.26] | 0.560 [0.449, 0.673] |
| AR(1) ρ=0.99, σ=0.5 m | Expl. acc. to sprint | 142 | 70.98 [65.05, 76.26] | -64.75 [-72.99, -56.15] | 0.347 [0.224, 0.450] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Distance | 142 | 4.19 [3.90, 4.56] | +4.18 [+3.89, +4.54] | 0.998 [0.996, 0.999] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Dist. < 15 | 142 | 3.63 [3.22, 4.16] | +3.62 [+3.19, +4.16] | 0.995 [0.991, 0.998] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Dist. 15-20 | 142 | 5.93 [5.42, 6.47] | +5.50 [+4.92, +6.10] | 0.991 [0.984, 0.994] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Dist. 20-25 (HSR) | 142 | 6.90 [6.08, 7.65] | +5.94 [+5.33, +6.50] | 0.990 [0.984, 0.992] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Dist. > 25 (sprint) | 142 | 16.33 [14.59, 18.86] | +11.74 [+10.92, +12.59] | 0.990 [0.981, 0.993] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | HI dist. > 20 | 142 | 7.98 [7.61, 8.35] | +7.36 [+7.02, +7.69] | 0.995 [0.991, 0.996] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | HSR efforts | 142 | 7.72 [7.14, 8.27] | -4.99 [-6.02, -3.42] | 0.970 [0.951, 0.977] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Sprint efforts | 142 | 15.08 [11.75, 19.79] | -10.72 [-12.82, -8.55] | 0.945 [0.927, 0.960] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | PSV-99 | 142 | 1.93 [1.72, 2.17] | +1.84 [+1.63, +2.11] | 0.980 [0.969, 0.982] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Expl. acc. to HSR | 142 | 56.17 [50.85, 62.49] | -54.42 [-59.02, -50.27] | 0.647 [0.538, 0.734] |
| common+indiv. σ=0.2 m (half common, ρ 0.99/0.9) | Expl. acc. to sprint | 142 | 77.49 [71.57, 80.98] | -72.66 [-77.86, -67.24] | 0.561 [0.455, 0.665] |

## Appendix: raw data files

| File | SHA-256 |
|---|---|
| `DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml` | `a884728647a564cf8b20e5223a1d04a5da6f6725e1e506c8651505327c197ba9` |
| `DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml` | `d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49` |
| `DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml` | `5490286436c86e6f80a538d2b1b379d0efd016e0defcaf594bc59d9ef5079f4c` |
| `DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml` | `56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec` |
| `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml` | `f23b3ee13b988b1ff9a0bce438c416dac47e71355bbc7ee1dab21aaf3e33c18c` |
| `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml` | `ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f` |
| `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml` | `4c497fac683eee8292901c18ca02c0be1d9609d0b9d4bd8aa092c61f9e0b19e9` |
| `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml` | `a228ef0e676d009cefff1580fcb721b53f598454eb120ec347e0a628a3a4646b` |
| `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml` | `566dd5e4c86e124ab90fcdba51100881f0436e22afa71832945a451a2d95a420` |
| `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml` | `845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1` |
| `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml` | `e976aa0c5dba862cfa898ea57ec226efce9d1bc04b23b8d775908db86aa6fa41` |
| `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml` | `cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f` |
| `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml` | `429b584e89df3e3e8b81f4bd0d004b38bd0b4298bb390b8fb43efccb60963e01` |
| `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml` | `70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269` |
