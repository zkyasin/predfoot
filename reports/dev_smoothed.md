# Dev evaluation: calibration, positions and leave-one-game-out coverage

`uv run python scripts/eval_dev.py --smooth ptz` · commit `bf28b35` · per-player table in `runs/20261031-163641_bf28b35_ptz` (not in git).

Dev clips with cached detections: 115 from 6 games. Frames: 17250 (one in five), calibrated 95.5%. Player-frames: 252689 (ground-truth boxes; truncated boxes and frames without a labelled camera excluded).

## Position error (metres)

| Reference | p50 | p90 | p99 |
|---|---|---|---|
| labels (`x_raw`, `y_raw`) | 0.46 | 1.46 | 3.99 |
| trusted line oracle (62% of player-frames) | 0.36 | 2.20 | 11.80 |

Calibration (mean distance of annotated line points to the projected template): p50 3.3 px, p90 7.0 px.

## LOGO coverage at 90 %, reference: labels

All held-out predictions (252689 player-frames, 115 clips): Mondrian ellipses cover **90.6%** [89.0%, 92.1%] with a median area of 5.37 m²; the constant circle covers 90.5% [88.7%, 92.4%] with 7.44 m². Uncalibrated Laplace ellipses cover 7.2%.

Coverage by distance (metres per pixel at the feet), Mondrian ellipses / constant circle:

| Near (< 0.036 m/px) | Middle | Far (> 0.049 m/px) |
|---|---|---|
| 90.8% / 90.6% | 90.1% / 92.6% | 91.0% / 88.4% |

Per held-out game (its quantiles come from the five other games):

| Held-out game | Clips | Player-frames | Raw Laplace | Conformal, pooled | Conformal, Mondrian [95 % CI] | Constant circle | Median radius pooled / Mondrian / constant (m) | Median area Mondrian / constant (m²) |
|---|---|---|---|---|---|---|---|---|
| 2 | 18 | 32089 | 8.5% | 88.4% | 86.6% [78.9%, 92.8%] | 87.4% | 2.86 / 2.50 / 1.47 | 5.08 / 6.81 |
| 3 | 21 | 50975 | 4.2% | 93.0% | 95.6% [94.0%, 97.2%] | 93.2% | 2.31 / 2.33 / 1.56 | 6.01 / 7.66 |
| 4 | 18 | 43395 | 8.4% | 92.2% | 92.3% [89.2%, 94.6%] | 92.7% | 2.72 / 2.48 / 1.54 | 5.47 / 7.44 |
| 5 | 19 | 43032 | 8.5% | 91.2% | 91.9% [89.1%, 94.4%] | 92.7% | 2.58 / 2.43 / 1.54 | 5.57 / 7.48 |
| 6 | 19 | 41947 | 8.5% | 85.6% | 84.0% [78.8%, 88.0%] | 86.8% | 2.85 / 2.41 / 1.46 | 4.79 / 6.66 |
| 9 | 20 | 41251 | 5.9% | 93.0% | 91.2% [88.4%, 93.4%] | 88.9% | 3.26 / 2.60 / 1.50 | 5.10 / 7.03 |

## LOGO coverage at 90 %, reference: trusted line oracle

All held-out predictions (155409 player-frames, 115 clips): Mondrian ellipses cover **94.4%** [92.9%, 96.0%] with a median area of 26.16 m²; the constant circle covers 94.0% [92.3%, 95.4%] with 51.79 m². Uncalibrated Laplace ellipses cover 7.9%.

Coverage by distance (metres per pixel at the feet), Mondrian ellipses / constant circle:

| Near (< 0.036 m/px) | Middle | Far (> 0.049 m/px) |
|---|---|---|
| 94.5% / 89.9% | 95.0% / 95.0% | 93.9% / 97.0% |

Per held-out game (its quantiles come from the five other games):

| Held-out game | Clips | Player-frames | Raw Laplace | Conformal, pooled | Conformal, Mondrian [95 % CI] | Constant circle | Median radius pooled / Mondrian / constant (m) | Median area Mondrian / constant (m²) |
|---|---|---|---|---|---|---|---|---|
| 2 | 18 | 13728 | 8.9% | 89.0% | 90.6% [82.1%, 96.2%] | 87.1% | 4.98 / 7.13 / 2.98 | 37.36 / 27.85 |
| 3 | 21 | 36592 | 6.2% | 95.7% | 93.6% [89.3%, 97.0%] | 96.5% | 4.41 / 2.17 / 4.06 | 5.59 / 51.79 |
| 4 | 18 | 26452 | 7.7% | 92.1% | 92.2% [87.1%, 95.9%] | 92.6% | 5.04 / 3.03 / 3.77 | 8.48 / 44.65 |
| 5 | 19 | 24275 | 8.5% | 94.1% | 94.5% [91.9%, 97.0%] | 92.8% | 5.22 / 2.98 / 3.83 | 9.13 / 46.08 |
| 6 | 19 | 28447 | 8.7% | 96.2% | 97.7% [96.1%, 98.8%] | 95.5% | 6.20 / 7.68 / 4.10 | 41.21 / 52.74 |
| 9 | 20 | 25915 | 8.4% | 94.3% | 96.5% [93.8%, 98.5%] | 95.1% | 6.42 / 8.91 / 4.09 | 55.25 / 52.65 |

Notes. Exploratory dev numbers, not the preregistered test. Frames are subsampled at 5 fps and
remain correlated within a clip, hence clip-level bootstrap intervals. Mondrian bins: metres per
pixel at the feet (3 bins) × number of RANSAC inliers (2 bins); small bins use the pooled
quantile of their source. With smoothing, quantiles are fitted within each covariance source (smoother; gap, one pooled quantile; single frame); a source with fewer than 10 calibration clips uses the model fitted on all sources.

## Temporal smoothing

Camera `ptz`, noise fitted per `fold`; smoother config digest `ad7d4fc89136` (β = 0). Label-free per clip: projection centre, PTZ projection, misfit gate, segments, deletions and the RTS smoother. Conformal quantiles are refitted on these outputs.

Noise parameters (fitted without labels; fold g excludes game g):

| Fold | Training clips | κ² | √q pan (°/s^1.5) | √q tilt (°/s^1.5) | √q log f (1/s^1.5) | Gap factor λ (innovations after a gap, their median NIS) | NLL per term | NIS mean (truncated) / median | Evaluations |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 97 | 1.88 | 2.20 | 0.495 | 0.068 | 1.43 (259, 2.74) | -21.26 | 3.90 / 2.56 | 222 |
| 3 | 94 | 2.00 | 2.34 | 0.553 | 0.078 | 1.44 (271, 2.84) | -20.91 | 3.88 / 2.56 | 243 |
| 4 | 97 | 1.91 | 2.25 | 0.542 | 0.069 | 1.58 (264, 2.91) | -21.12 | 3.86 / 2.50 | 224 |
| 5 | 96 | 1.94 | 2.13 | 0.535 | 0.069 | 1.65 (254, 2.96) | -21.15 | 3.88 / 2.49 | 240 |
| 6 | 96 | 1.99 | 2.24 | 0.559 | 0.071 | 1.44 (286, 2.96) | -21.07 | 3.89 / 2.51 | 223 |
| 9 | 95 | 1.94 | 2.13 | 0.504 | 0.069 | 1.17 (268, 2.54) | -21.22 | 3.90 / 2.56 | 248 |

Largest ratio of a fold's parameter to the median over folds (κ², q pan, q tilt, q log f, λ): ×1.03, ×1.11, ×1.18, ×1.26, ×1.22 (target: within ×1.5).

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

| smoothed | repaired | filled | bridged | projected | passthrough | duplicate | rejected | unchecked | abstain |
|---|---|---|---|---|---|---|---|---|---|
| 90.6% | 1.5% | 0.9% | 1.6% | 0.0% | 0.5% | 0.0% | 1.2% | 0.1% | 3.6% |

Per clip (median / max): segments 1 / 7, deleted measurements 2 / 39, frames in short segments 0 / 6, frames in manoeuvre segments 0 / 15, misfit share 1.6% / 12.4%.

Position error (m) on the same 252188 player-frames (raw calibration OK), and jitter of the common-mode error (per-frame median over players) against the smoothed labels:

| Camera | Labels p50 | p90 | p99 | Line oracle p50 | p90 | p99 | Smoothed labels p50 | p90 | p99 | Common-mode step RMS (m) | RMS, errors < 10 m | Median abs. step (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| per-frame | 0.59 | 2.01 | 5.60 | 0.46 | 2.48 | 13.02 | 0.58 | 1.91 | 5.32 | 2.267 | 0.445 | 0.140 |
| ptz | 0.46 | 1.46 | 3.99 | 0.36 | 2.19 | 11.80 | 0.45 | 1.33 | 3.44 | 0.351 | 0.151 | 0.063 |

Common-mode step ratio ptz / per-frame, pooled over clips: RMS 0.15, RMS without errors of 10 m or more 0.34, median absolute step 0.45; RMS ratio per clip: median 0.34, worst 0.91. One failed frame makes steps of tens of metres that dominate the plain RMS: read the robust ratios.

Filled frames (raw calibration failed, camera from the smoother): 1647 player-frames on 123 frames; error p50 | p90 | p99 against labels 0.38 | 1.15 | 2.08 m. Coverage (labels, gap source): Mondrian 99.0%, constant circle 95.6% (n = 1647). Coverage (trusted line oracle, gap source): Mondrian 77.2%, constant circle 71.9% (n = 967).

Bridged frames (run of off-model frames of at most 0.5 s inside a segment, raw camera replaced by the smoothed state): 4212 player-frames on 266 frames; error p50 | p90 | p99 against labels 0.63 | 2.48 | 24.13 m per-frame, 0.43 | 1.28 | 3.00 m output.

Coverage per output type (quantiles fitted within each covariance source: smoother for smoothed and repaired frames, gap for filled and bridged frames, single frame for the others), Mondrian ellipses / constant circle, and median Mondrian area:

| Reference | Type | Player-frames | Mondrian | Constant circle | Median area (m²) |
|---|---|---|---|---|---|
| labels | smoothed | 243248 | 90.5% | 90.6% | 5.30 |
| labels | repaired | 3858 | 97.3% | 83.9% | 21.75 |
| labels | bridged | 4300 | 90.4% | 93.0% | 5.25 |
| labels | passthrough | 1283 | 92.5% | 78.3% | 9.09 |
| labels | filled | 1647 | 99.0% | 95.6% | 17.70 |
| trusted line oracle | smoothed | 150248 | 94.3% | 94.0% | 25.61 |
| trusted line oracle | repaired | 1862 | 98.1% | 91.8% | 260.07 |
| trusted line oracle | bridged | 2691 | 97.0% | 97.6% | 24.37 |
| trusted line oracle | passthrough | 608 | 95.4% | 96.2% | 5.74 |
| trusted line oracle | filled | 967 | 77.2% | 71.9% | 75.48 |

Coverage per camera-speed tercile (image motion of a ground grid between output frames), Mondrian ellipses / constant circle:

| Reference | Slow | Middle | Fast |
|---|---|---|---|
| labels (edges 44, 187 px/s) | 87.5% / 86.2% | 92.6% / 92.8% | 91.6% / 92.7% |
| trusted line oracle (edges 40, 175 px/s) | 98.0% / 97.1% | 93.8% / 93.6% | 91.5% / 91.4% |

Line reprojection on 9950 frames with self-consistent line annotations (raw calibration OK): per-frame p50 3.9 px, p90 10.4 px, > 10 px 10.7%; output p50 3.0 px, p90 4.9 px, > 10 px 1.0%.

Published-position rate (output camera OK): 95.1% of labelled frames (per-frame calibration: 95.5%).

Runtime: 404 s (calibration or loading 47 s).

## Ablations and reading of the results

*Written after the runs above; the numbers come from `runs/20261031-152537_c3861c7_none` (per-frame,
`reports/dev_per_frame.md`), `runs/20261031-165523_bf28b35_proj`, this run and
`runs/20261031-165719_bf28b35_ptz-all`, all on the same 115 dev clips with the same per-frame calibrations
(`--calib-from`).*

| Camera | Coverage, labels [95 % CI] | Median area, Mondrian / constant circle (m²) | Error vs labels p50 / p90 / p99 (m) | Line reprojection p90, > 10 px | Common-mode step RMS ratio (errors < 10 m) | Published |
|---|---|---|---|---|---|---|
| per-frame | 90.4% [88.8%, 91.8%] | 6.21 / 14.56 | 0.59 / 2.02 / 5.64 | 10.4 px, 10.7% | 1 | 95.5% |
| `proj` (fixed-centre PTZ projection, no smoothing) | 90.5% [88.8%, 91.9%] | 6.67 / 8.50 | 0.48 / 1.54 / 4.21 | 5.5 px, 2.5% | 0.54 | 95.5% |
| `ptz`, noise fitted per fold | 90.6% [89.0%, 92.1%] | 5.37 / 7.44 | 0.46 / 1.46 / 3.99 | 4.9 px, 1.0% | 0.34 | 95.1% |
| `ptz`, one noise fit on all games | 90.6% [89.0%, 92.1%] | 5.38 / 7.46 | 0.46 / 1.46 / 3.99 | 4.9 px, 1.0% | 0.34 | 95.1% |

1. **Most of the accuracy comes from the camera model, the temporal smoother adds consistency.** Holding
   the camera centre fixed per clip (8 → 4 free parameters per frame) already brings the error and the line
   reprojection most of the way; the RTS smoother then divides the frame-to-frame jitter of the common-mode
   error by about three (step RMS ratio 0.34, median absolute step ratio 0.45) and removes most of the KPI
   inflation (below). The line reprojection uses the annotated lines only, so it is independent of both
   position references.
2. **Coverage stays at 90 %, the regions tighten, and their edge over a constant circle shrinks.** Smoothing
   makes the errors more homogeneous: the constant circle needs 7.4 m² instead of 14.6 m², our ellipses
   5.4 m² instead of 6.2 m². Per held-out game, coverage ranges from 84.0 % (game 6) to 95.6 % (game 3), a
   wider spread than per-frame (85.6-93.3 %).
3. **The trusted line oracle is a per-frame reference, so it penalises a smooth camera.** Errors against it
   fall (p50 0.46 → 0.36 m, p99 13.3 → 11.8 m) but the median Mondrian area doubles (12.6 → 26.2 m²), driven
   by games 2, 6 and 9. The oracle is fitted frame by frame on the annotated lines; its own frame-level error
   does not shrink with smoothing while our smoothed covariance does, so its scores grow. The labels come
   from the annotators' camera, which is itself smooth over time. Neither reference is the truth: the
   preregistration (`docs/preregistration.md`) keeps both and states this asymmetry.
4. **KPI cost (model-free proxy on SN-GSR, 5 fps, Savitzky-Golay 5/2, against the smoothed labels):**
   distance bias +27.0 % → **+2.7 %**, high-speed-running distance +67.2 % → **+6.2 %**, speed error RMS
   1.49 → 0.71 m/s; the raw labels' own bias against the same reference is +17.9 % and +43.1 %
   (`reports/error_process_temporal_dev.md`). Injecting the measured process into IDSSE as an AR(1)
   (Ornstein-Uhlenbeck) error at 25 Hz still gives about +30 % for the smoothed camera: an exponential
   autocorrelation extrapolated from 5 fps to 25 Hz puts roughness at frequencies that were never measured
   and that a smoothed camera does not have. That simulation is therefore not reported for the smoothed
   camera; the spectrum-aware error model checked against this proxy is in `reports/kpi_budget_spectral.md`.
5. **Failure handling.**
   - A brief run (≤ 0.5 s) of frames that the clip's PTZ model cannot explain, inside a segment, is a
     calibration failure rather than a cut to another camera: it gets the smoothed state (*bridged*, 266
     frames; per-frame errors p50 / p90 / p99 0.63 / 2.48 / 24.1 m become 0.43 / 1.28 / 3.00 m). Longer runs
     keep their raw camera (*passthrough*, 0.5 % of frames).
   - Frames of a segment too short for any innovation to test them abstain (*unchecked*, 19 frames).
   - The process-noise factor across failure gaps is fitted without labels: λ = 1.2-1.7 per fold instead of
     the a-priori 20. Filled frames (raw calibration failed) keep a p90 error of 1.15 m and their own pooled
     quantile (gap source): coverage 99.0 % against the labels with a median area of 17.7 m² (468 m² with
     λ = 20), but 77.2 % against the line oracle (967 player-frames).
   - Still open: about 19 isolated off-model frames at clip or segment edges pass through with their raw
     camera, 9 of them with a p90 error above 5 m. Filled frames are excluded from the headline coverage and
     reported here; bridged frames are in it (252689 = 243248 smoothed + 3858 repaired + 4300 bridged + 1283
     passthrough player-frames, table above). *Corrected after review: this note first said both were
     excluded.*
