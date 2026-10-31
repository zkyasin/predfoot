# Dev evaluation: calibration, positions and leave-one-game-out coverage

`uv run python scripts/eval_dev.py` · commit `c3861c7` · per-player table in `runs/20261031-152537_c3861c7_none` (not in git).

Dev clips with cached detections: 115 from 6 games. Frames: 17250 (one in five), calibrated 95.5%. Player-frames: 254976 (ground-truth boxes; truncated boxes and frames without a labelled camera excluded).

## Position error (metres)

| Reference | p50 | p90 | p99 |
|---|---|---|---|
| labels (`x_raw`, `y_raw`) | 0.59 | 2.02 | 5.64 |
| trusted line oracle (61% of player-frames) | 0.46 | 2.50 | 13.26 |

Calibration (mean distance of annotated line points to the projected template): p50 4.6 px, p90 15.9 px.

## LOGO coverage at 90 %, reference: labels

All held-out predictions (254976 player-frames, 115 clips): Mondrian ellipses cover **90.4%** [88.8%, 91.8%] with a median area of 6.21 m²; the constant circle covers 90.9% [89.5%, 92.2%] with 14.56 m². Uncalibrated Laplace ellipses cover 17.9%.

Coverage by distance (metres per pixel at the feet), Mondrian ellipses / constant circle:

| Near (< 0.036 m/px) | Middle | Far (> 0.049 m/px) |
|---|---|---|
| 91.1% / 90.0% | 89.6% / 94.1% | 90.6% / 88.6% |

Per held-out game (its quantiles come from the five other games):

| Held-out game | Clips | Player-frames | Raw Laplace | Conformal, pooled | Conformal, Mondrian [95 % CI] | Constant circle | Median radius pooled / Mondrian / constant (m) | Median area Mondrian / constant (m²) |
|---|---|---|---|---|---|---|---|---|
| 2 | 18 | 33390 | 14.6% | 88.6% | 85.6% [78.8%, 92.0%] | 88.9% | 2.85 / 2.68 / 2.10 | 5.40 / 13.80 |
| 3 | 21 | 50992 | 15.8% | 91.8% | 93.3% [90.5%, 96.0%] | 93.9% | 2.35 / 2.48 / 2.21 | 6.36 / 15.31 |
| 4 | 18 | 43616 | 19.5% | 91.1% | 90.9% [88.0%, 93.5%] | 93.1% | 2.64 / 2.63 / 2.17 | 5.64 / 14.86 |
| 5 | 19 | 43321 | 20.9% | 92.1% | 92.6% [90.2%, 94.7%] | 91.9% | 2.76 / 2.83 / 2.15 | 6.89 / 14.56 |
| 6 | 19 | 41798 | 18.0% | 86.5% | 85.7% [81.8%, 89.3%] | 87.6% | 3.02 / 2.86 / 2.06 | 6.10 / 13.38 |
| 9 | 20 | 41859 | 18.4% | 94.3% | 92.7% [90.1%, 94.7%] | 88.9% | 3.41 / 3.05 / 2.09 | 6.80 / 13.75 |

## LOGO coverage at 90 %, reference: trusted line oracle

All held-out predictions (155643 player-frames, 115 clips): Mondrian ellipses cover **94.4%** [92.7%, 96.0%] with a median area of 12.58 m²; the constant circle covers 94.1% [92.4%, 95.3%] with 57.72 m². Uncalibrated Laplace ellipses cover 20.2%.

Coverage by distance (metres per pixel at the feet), Mondrian ellipses / constant circle:

| Near (< 0.037 m/px) | Middle | Far (> 0.049 m/px) |
|---|---|---|
| 94.6% / 91.0% | 94.7% / 95.3% | 94.0% / 96.0% |

Per held-out game (its quantiles come from the five other games):

| Held-out game | Clips | Player-frames | Raw Laplace | Conformal, pooled | Conformal, Mondrian [95 % CI] | Constant circle | Median radius pooled / Mondrian / constant (m) | Median area Mondrian / constant (m²) |
|---|---|---|---|---|---|---|---|---|
| 2 | 18 | 13999 | 14.2% | 89.2% | 92.3% [85.1%, 97.0%] | 87.0% | 3.13 / 4.56 / 3.28 | 13.86 / 33.87 |
| 3 | 21 | 36592 | 19.8% | 94.5% | 93.4% [89.5%, 96.7%] | 96.3% | 2.99 / 2.64 / 4.32 | 7.18 / 58.61 |
| 4 | 18 | 26637 | 18.9% | 91.3% | 91.3% [86.0%, 95.5%] | 92.9% | 3.19 / 3.20 / 4.13 | 8.28 / 53.71 |
| 5 | 19 | 24301 | 20.5% | 94.8% | 94.6% [92.3%, 96.8%] | 93.5% | 3.83 / 3.63 / 4.19 | 11.14 / 55.13 |
| 6 | 19 | 28277 | 23.7% | 96.6% | 97.6% [95.9%, 98.8%] | 96.1% | 4.45 / 5.08 / 4.43 | 17.77 / 61.58 |
| 9 | 20 | 25837 | 21.2% | 95.4% | 96.5% [93.6%, 98.6%] | 94.4% | 4.54 / 6.12 / 4.29 | 24.67 / 57.72 |

Notes. Exploratory dev numbers, not the preregistered test. Frames are subsampled at 5 fps and
remain correlated within a clip, hence clip-level bootstrap intervals. Mondrian bins: metres per
pixel at the feet (3 bins) × number of RANSAC inliers (2 bins), formed within each covariance
source when the camera is smoothed (smoother, PTZ projection, per-frame); small bins use the
pooled quantile of their source.
