# Dev evaluation: calibration, positions and leave-one-game-out coverage

`uv run python scripts/eval_dev.py` · commit `ba55bc7` · per-player table in `runs/20261031-0151_ba55bc7` (not in git).

Dev clips with cached detections: 49 from 6 games. Frames: 7350 (one in five), calibrated 96.4%. Player-frames: 114153 (ground-truth boxes; truncated boxes and frames without a labelled camera excluded).

## Position error (metres)

| Reference | p50 | p90 | p99 |
|---|---|---|---|
| labels (`x_raw`, `y_raw`) | 0.59 | 2.03 | 5.79 |
| trusted line oracle (61% of player-frames) | 0.44 | 2.54 | 12.62 |

Calibration (mean distance of annotated line points to the projected template): p50 4.8 px, p90 16.6 px.

## LOGO coverage at 90 %, reference: labels

All held-out predictions (114153 player-frames, 49 clips): Mondrian ellipses cover **92.3%** [90.4%, 94.0%] with a median area of 9.09 m²; the constant circle covers 92.5% [90.7%, 94.5%] with 17.04 m². Uncalibrated Laplace ellipses cover 18.6%.

Coverage by distance (metres per pixel at the feet), Mondrian ellipses / constant circle:

| Near (< 0.036 m/px) | Middle | Far (> 0.050 m/px) |
|---|---|---|
| 92.8% / 91.7% | 91.4% / 95.3% | 92.7% / 90.5% |

Per held-out game (its quantiles come from the five other games):

| Held-out game | Clips | Player-frames | Raw Laplace | Conformal, pooled | Conformal, Mondrian [95 % CI] | Constant circle | Median radius pooled / Mondrian / constant (m) | Median area Mondrian / constant (m²) |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 2715 | 17.6% | 97.9% | 96.8% [96.8%, 96.8%] | 92.5% | 4.48 / 4.45 / 2.28 | 22.37 / 16.39 |
| 3 | 1 | 1915 | 1.7% | 57.3% | 80.6% [80.6%, 80.6%] | 100.0% | 0.99 / 1.24 / 2.31 | 1.50 / 16.71 |
| 4 | 18 | 43616 | 19.5% | 94.9% | 95.2% [93.7%, 97.0%] | 95.8% | 3.40 / 3.42 / 2.68 | 9.51 / 22.54 |
| 5 | 9 | 21577 | 19.7% | 95.2% | 95.5% [93.3%, 97.1%] | 91.5% | 3.58 / 3.83 / 2.33 | 13.64 / 17.04 |
| 6 | 19 | 41798 | 18.0% | 89.9% | 87.5% [83.8%, 90.6%] | 89.0% | 3.59 / 3.13 / 2.22 | 7.42 / 15.50 |
| 9 | 1 | 2532 | 19.3% | 99.8% | 99.2% [99.2%, 99.2%] | 95.9% | 3.50 / 2.99 / 2.29 | 7.51 / 16.53 |

## LOGO coverage at 90 %, reference: trusted line oracle

All held-out predictions (69180 player-frames, 49 clips): Mondrian ellipses cover **94.3%** [91.7%, 96.5%] with a median area of 15.28 m²; the constant circle covers 93.9% [91.0%, 95.9%] with 46.94 m². Uncalibrated Laplace ellipses cover 20.9%.

Coverage by distance (metres per pixel at the feet), Mondrian ellipses / constant circle:

| Near (< 0.036 m/px) | Middle | Far (> 0.049 m/px) |
|---|---|---|
| 93.9% / 91.1% | 95.0% / 94.8% | 93.9% / 95.7% |

Per held-out game (its quantiles come from the five other games):

| Held-out game | Clips | Player-frames | Raw Laplace | Conformal, pooled | Conformal, Mondrian [95 % CI] | Constant circle | Median radius pooled / Mondrian / constant (m) | Median area Mondrian / constant (m²) |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 77 | 20.8% | 100.0% | 100.0% [100.0%, 100.0%] | 97.4% | 8.55 / 6.42 / 3.78 | 67.84 / 44.84 |
| 3 | 1 | 1889 | 13.3% | 100.0% | 90.7% [90.7%, 90.7%] | 100.0% | 1.24 / 1.69 / 3.79 | 2.97 / 45.22 |
| 4 | 18 | 26637 | 18.9% | 91.2% | 91.6% [87.2%, 95.7%] | 92.4% | 3.13 / 3.39 / 3.86 | 9.12 / 46.70 |
| 5 | 9 | 11449 | 21.1% | 95.0% | 94.1% [89.4%, 97.5%] | 92.6% | 4.21 / 3.30 / 3.87 | 9.95 / 46.94 |
| 6 | 19 | 28277 | 23.7% | 98.0% | 98.3% [97.0%, 99.1%] | 96.4% | 6.24 / 6.80 / 4.69 | 32.03 / 69.00 |
| 9 | 1 | 851 | 3.2% | 45.2% | 57.0% [57.0%, 57.0%] | 59.2% | 3.44 / 4.34 / 3.47 | 15.29 / 37.78 |

Notes. Exploratory dev numbers, not the preregistered test. Frames are subsampled at 5 fps and
remain correlated within a clip, hence clip-level bootstrap intervals. Mondrian bins: metres per
pixel at the feet (3 bins) × number of RANSAC inliers (2 bins); small bins use the pooled
quantile.
