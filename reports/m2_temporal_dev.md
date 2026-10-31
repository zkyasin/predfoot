# Dev evaluation: calibration, positions and leave-one-game-out coverage

`uv run python scripts/eval_dev.py --smooth ptz` · commit `c3861c7` · per-player table in `runs/20261031-153059_c3861c7_ptz` (not in git).

Dev clips with cached detections: 115 from 6 games. Frames: 17250 (one in five), calibrated 95.5%. Player-frames: 253017 (ground-truth boxes; truncated boxes and frames without a labelled camera excluded).

## Position error (metres)

| Reference | p50 | p90 | p99 |
|---|---|---|---|
| labels (`x_raw`, `y_raw`) | 0.46 | 1.48 | 4.20 |
| trusted line oracle (61% of player-frames) | 0.37 | 2.25 | 12.56 |

Calibration (mean distance of annotated line points to the projected template): p50 3.3 px, p90 7.4 px.

## LOGO coverage at 90 %, reference: labels

All held-out predictions (253017 player-frames, 115 clips): Mondrian ellipses cover **90.6%** [89.0%, 92.0%] with a median area of 5.34 m²; the constant circle covers 90.6% [88.8%, 92.4%] with 7.71 m². Uncalibrated Laplace ellipses cover 6.9%.

Coverage by distance (metres per pixel at the feet), Mondrian ellipses / constant circle:

| Near (< 0.036 m/px) | Middle | Far (> 0.049 m/px) |
|---|---|---|
| 90.9% / 90.7% | 90.0% / 92.7% | 90.9% / 88.3% |

Per held-out game (its quantiles come from the five other games):

| Held-out game | Clips | Player-frames | Raw Laplace | Conformal, pooled | Conformal, Mondrian [95 % CI] | Constant circle | Median radius pooled / Mondrian / constant (m) | Median area Mondrian / constant (m²) |
|---|---|---|---|---|---|---|---|---|
| 2 | 18 | 32284 | 7.9% | 88.6% | 86.7% [79.2%, 93.0%] | 87.3% | 2.87 / 2.52 / 1.50 | 5.06 / 7.03 |
| 3 | 21 | 50992 | 4.0% | 93.1% | 95.7% [94.1%, 97.3%] | 93.3% | 2.32 / 2.33 / 1.59 | 6.00 / 7.90 |
| 4 | 18 | 43428 | 8.2% | 92.1% | 92.1% [89.0%, 94.4%] | 92.8% | 2.73 / 2.48 / 1.57 | 5.43 / 7.72 |
| 5 | 19 | 43013 | 8.0% | 91.2% | 92.1% [89.5%, 94.4%] | 92.8% | 2.58 / 2.44 / 1.57 | 5.57 / 7.71 |
| 6 | 19 | 41947 | 8.3% | 85.6% | 83.7% [78.5%, 87.8%] | 87.0% | 2.84 / 2.40 / 1.49 | 4.71 / 6.93 |
| 9 | 20 | 41353 | 6.0% | 92.9% | 91.1% [88.2%, 93.4%] | 88.7% | 3.26 / 2.60 / 1.51 | 5.07 / 7.19 |

## LOGO coverage at 90 %, reference: trusted line oracle

All held-out predictions (155487 player-frames, 115 clips): Mondrian ellipses cover **94.4%** [92.9%, 96.0%] with a median area of 25.32 m²; the constant circle covers 94.0% [92.3%, 95.4%] with 53.54 m². Uncalibrated Laplace ellipses cover 7.4%.

Coverage by distance (metres per pixel at the feet), Mondrian ellipses / constant circle:

| Near (< 0.036 m/px) | Middle | Far (> 0.049 m/px) |
|---|---|---|
| 94.4% / 89.9% | 94.9% / 95.1% | 93.9% / 96.9% |

Per held-out game (its quantiles come from the five other games):

| Held-out game | Clips | Player-frames | Raw Laplace | Conformal, pooled | Conformal, Mondrian [95 % CI] | Constant circle | Median radius pooled / Mondrian / constant (m) | Median area Mondrian / constant (m²) |
|---|---|---|---|---|---|---|---|---|
| 2 | 18 | 13777 | 8.3% | 88.9% | 90.6% [81.9%, 96.4%] | 86.8% | 4.96 / 7.09 / 3.03 | 36.49 / 28.76 |
| 3 | 21 | 36592 | 5.7% | 95.7% | 93.7% [89.7%, 97.1%] | 96.6% | 4.43 / 2.24 / 4.15 | 5.91 / 54.05 |
| 4 | 18 | 26453 | 7.1% | 92.1% | 92.1% [87.0%, 95.9%] | 92.7% | 5.11 / 2.94 / 3.91 | 7.87 / 48.11 |
| 5 | 19 | 24256 | 7.7% | 94.0% | 94.3% [91.8%, 96.8%] | 93.0% | 5.18 / 2.93 / 3.94 | 8.52 / 48.69 |
| 6 | 19 | 28447 | 8.4% | 96.3% | 97.7% [96.1%, 98.8%] | 95.6% | 6.23 / 7.81 / 4.21 | 42.59 / 55.67 |
| 9 | 20 | 25962 | 8.3% | 94.3% | 96.4% [93.7%, 98.5%] | 94.7% | 6.41 / 8.93 / 4.13 | 55.67 / 53.54 |

Notes. Exploratory dev numbers, not the preregistered test. Frames are subsampled at 5 fps and
remain correlated within a clip, hence clip-level bootstrap intervals. Mondrian bins: metres per
pixel at the feet (3 bins) × number of RANSAC inliers (2 bins), formed within each covariance
source when the camera is smoothed (smoother, PTZ projection, per-frame); small bins use the
pooled quantile of their source.

## Temporal smoothing

Camera `ptz`, noise fitted per `fold`; smoother config digest `3cce83ece8e7` (β = 0). Label-free per clip: projection centre, PTZ projection, misfit gate, segments, deletions and the RTS smoother. Conformal quantiles are refitted on these outputs.

Noise parameters (fitted without labels; fold g excludes game g):

| Fold | Training clips | κ² | √q pan (°/s^1.5) | √q tilt (°/s^1.5) | √q log f (1/s^1.5) | NLL per term | NIS mean (truncated) / median | Evaluations |
|---|---|---|---|---|---|---|---|---|
| 2 | 97 | 1.86 | 2.14 | 0.476 | 0.065 | -21.21 | 3.90 / 2.55 | 146 |
| 3 | 94 | 1.99 | 2.29 | 0.535 | 0.072 | -20.86 | 3.88 / 2.55 | 132 |
| 4 | 97 | 1.94 | 2.17 | 0.521 | 0.065 | -21.07 | 3.85 / 2.48 | 148 |
| 5 | 96 | 1.94 | 2.06 | 0.508 | 0.065 | -21.11 | 3.89 / 2.50 | 148 |
| 6 | 96 | 1.98 | 2.18 | 0.542 | 0.067 | -21.01 | 3.88 / 2.48 | 161 |
| 9 | 95 | 1.95 | 2.08 | 0.479 | 0.066 | -21.15 | 3.88 / 2.53 | 145 |

Largest ratio of a fold's parameter to the median over folds (κ², q pan, q tilt, q log f): ×1.04, ×1.13, ×1.17, ×1.21 (target: within ×1.5).

Projection centres per game (own estimate reliable when its largest sd ≤ 1 m with ≥ 10 PTZ inliers; otherwise the game centre, refused when the clip's misfit scale at it exceeds 4× the reference; without a centre the clip passes through):

| Game | Reliable / clips | Median centre (m) | MAD (m) | Game centre used | Game centre refused | None |
|---|---|---|---|---|---|---|
| 2 | 14 / 18 | (-0.0, -49.6, 9.7) | (0.1, 0.3, 0.2) | 4 | 0 | 0 |
| 3 | 20 / 21 | (0.0, -59.6, 15.2) | (0.1, 0.1, 0.0) | 1 | 0 | 0 |
| 4 | 18 / 18 | (-0.1, -54.3, 11.0) | (0.0, 0.2, 0.1) | 0 | 0 | 0 |
| 5 | 18 / 19 | (0.0, -54.3, 11.1) | (0.1, 0.1, 0.1) | 1 | 0 | 0 |
| 6 | 18 / 19 | (-0.1, -50.0, 9.5) | (0.0, 0.2, 0.0) | 1 | 0 | 0 |
| 9 | 19 / 20 | (-0.1, -49.1, 8.6) | (0.1, 0.2, 0.0) | 1 | 0 | 0 |

Frame outcomes over 17250 cached frames (one per frame):

| smoothed | repaired | filled | projected | passthrough | duplicate | rejected | abstain |
|---|---|---|---|---|---|---|---|
| 90.7% | 1.4% | 0.9% | 0.1% | 2.1% | 0.0% | 1.1% | 3.6% |

Per clip (median / max): segments 1 / 11, deleted measurements 2 / 39, frames in short segments 0 / 6, frames in manoeuvre segments 0 / 29, misfit share 1.6% / 12.4%.

Position error (m) on the same 252515 player-frames (raw calibration OK), and jitter of the common-mode error (per-frame median over players) against the smoothed labels:

| Camera | Labels p50 | p90 | p99 | Line oracle p50 | p90 | p99 | Smoothed labels p50 | p90 | p99 | Common-mode step RMS (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| per-frame | 0.59 | 2.01 | 5.62 | 0.46 | 2.49 | 13.15 | 0.58 | 1.92 | 5.34 | 2.267 |
| ptz | 0.46 | 1.48 | 4.14 | 0.37 | 2.22 | 12.17 | 0.46 | 1.35 | 3.60 | 2.282 |

Common-mode step RMS ratio ptz / per-frame: pooled 1.01; per clip median 0.41, worst 6.05.

Filled frames (raw calibration failed, camera from the smoother): 1647 player-frames on 123 frames; error p50 | p90 | p99 against labels 0.40 | 1.14 | 2.00 m. Coverage (labels, same fold models): Mondrian 100.0%, constant circle 96.2% (n = 1647). Coverage (trusted line oracle, same fold models): Mondrian 99.8%, constant circle 72.5% (n = 967).

Coverage per output type (Mondrian bins within each covariance source: smoother, PTZ projection, per-frame), Mondrian ellipses / constant circle, and median Mondrian area:

| Reference | Type | Player-frames | Mondrian | Constant circle | Median area (m²) |
|---|---|---|---|---|---|
| labels | smoothed | 243508 | 90.5% | 90.9% | 5.27 |
| labels | repaired | 3780 | 97.0% | 83.3% | 21.08 |
| labels | projected | 234 | 99.1% | 47.9% | 60212.78 |
| labels | passthrough | 5495 | 91.2% | 81.2% | 5.38 |
| labels | filled | 1647 | 100.0% | 96.2% | 467.51 |
| trusted line oracle | smoothed | 150272 | 94.4% | 94.1% | 25.81 |
| trusted line oracle | repaired | 1867 | 98.7% | 92.0% | 265.57 |
| trusted line oracle | projected | 106 | 95.3% | 1.9% | 471177.10 |
| trusted line oracle | passthrough | 3242 | 92.6% | 94.1% | 3.95 |
| trusted line oracle | filled | 967 | 99.8% | 72.5% | 6213.10 |

Coverage per camera-speed tercile (image motion of a ground grid between output frames), Mondrian ellipses / constant circle:

| Reference | Slow | Middle | Fast |
|---|---|---|---|
| labels (edges 46, 193 px/s) | 87.3% / 86.5% | 92.5% / 92.9% | 91.8% / 92.5% |
| trusted line oracle (edges 42, 182 px/s) | 97.9% / 97.0% | 93.8% / 93.8% | 91.6% / 91.3% |

Line reprojection on 9964 frames with self-consistent line annotations (raw calibration OK): per-frame p50 3.9 px, p90 10.5 px, > 10 px 10.8%; output p50 3.0 px, p90 5.1 px, > 10 px 1.5%.

Published-position rate (output camera OK): 95.3% of labelled frames (per-frame calibration: 95.5%).

Runtime: 380 s (calibration or loading 47 s).

## Ablations and reading of the results

*Written after the runs above; the numbers come from `runs/20261031-152537_c3861c7_none` (per-frame,
`reports/m2_logo_dev.md`), `runs/20261031-153734_c3861c7_proj`, this run and
`runs/20261031-153928_c3861c7_ptz-all`, all on the same 115 dev clips with the same per-frame calibrations
(`--calib-from`).*

| Camera | Coverage, labels [95 % CI] | Median area, Mondrian / constant circle (m²) | Error vs labels p50 / p90 / p99 (m) | Line reprojection p90, > 10 px | Common-mode step, median per-clip ratio | Published |
|---|---|---|---|---|---|---|
| per-frame | 90.4% [88.8%, 91.8%] | 6.21 / 14.56 | 0.59 / 2.02 / 5.64 | 10.5 px, 10.8% | 1 | 95.5% |
| `proj` (fixed-centre PTZ projection, no smoothing) | 90.5% [88.8%, 91.9%] | 6.68 / 8.50 | 0.48 / 1.54 / 4.21 | 5.5 px, 2.5% | 0.54 | 95.5% |
| `ptz`, noise fitted per fold | 90.6% [89.0%, 92.0%] | 5.34 / 7.71 | 0.46 / 1.48 / 4.14 | 5.1 px, 1.5% | 0.41 | 95.3% |
| `ptz`, one noise fit on all games | 90.6% [89.0%, 92.0%] | 5.33 / 7.72 | 0.46 / 1.48 / 4.14 | 5.1 px, 1.5% | 0.41 | 95.3% |

1. **Most of the accuracy comes from the camera model, the temporal smoother adds consistency.** Holding
   the camera centre fixed per clip (8 → 4 free parameters per frame) already brings the error and the line
   reprojection most of the way; the RTS smoother then halves the frame-to-frame jitter of the common-mode
   error (median per-clip step ratio 0.41) and removes most of the KPI inflation (below). The line
   reprojection uses the annotated lines only, so it is independent of both position references.
2. **Coverage stays at 90 %, the regions tighten, and their edge over a constant circle shrinks.** Smoothing
   makes the errors more homogeneous: the constant circle needs 7.7 m² instead of 14.6 m², our ellipses
   5.3 m² instead of 6.2 m². Per held-out game, coverage ranges from 83.7 % (game 6) to 95.7 % (game 3),
   a wider spread than per-frame (85.6-93.3 %).
3. **The trusted line oracle is a per-frame reference, so it penalises a smooth camera.** Errors against it
   fall (p50 0.46 → 0.37 m) but the median Mondrian area doubles (12.6 → 25.3 m²), driven by games 2, 6
   and 9. The oracle is fitted frame by frame on the annotated lines; its own frame-level error does not
   shrink with smoothing while our smoothed covariance does, so its scores grow. The labels come from the
   annotators' camera, which is itself smooth over time. Neither reference is the truth: the
   preregistration should keep both and state this asymmetry.
4. **KPI cost (model-free proxy on SN-GSR, 5 fps, Savitzky-Golay 5/2, against the smoothed labels):**
   distance bias +27.1 % → **+3.9 %**, high-speed-running distance +67.9 % → **+9.5 %**, speed error RMS
   1.50 → 0.79 m/s; the raw labels' own bias against the same reference is +17.9 % and +43.4 %
   (`reports/error_process_temporal_dev.md`). Injecting the measured process into IDSSE as an AR(1)
   (Ornstein-Uhlenbeck) error at 25 Hz still gives about +30 % for the smoothed camera: an exponential
   autocorrelation extrapolated from 5 fps to 25 Hz puts roughness at frequencies that were never measured
   and that a smoothed camera does not have. That simulation is therefore not reported for the smoothed
   camera; a spectrum-aware error model, checked against this proxy, comes next.
5. **Remaining failure modes (tails, not the headline).**
   - An isolated frame whose camera does not fit the clip's PTZ model is passed through with its raw
     camera, which is often a calibration failure: SNGS-110 frame 386 (13 m), SNGS-024 frame 176 (265 m).
     Inside a segment, the smoother's state would be the better output.
   - A lone frame after a long failure gap can form its own short segment and be published although its raw
     calibration failed (SNGS-064 frame 666, 63 m).
   - These single jumps dominate the pooled step RMS (ratio 1.01) while the per-clip median ratio is 0.41:
     read the median.
   - Filled frames (raw calibration failed) are accurate (p90 1.14 m against the labels) but their regions
     are far too wide (median 468 m²), because the process noise is inflated 20-fold across failure gaps;
     the PTZ-projected source is too small for its own bins (234 player-frames) and gets absurd areas.
     Both are excluded from the headline, and both are follow-ups.
