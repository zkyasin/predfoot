# Preregistered test run

`uv run python scripts/run_test_once.py` · lock code commit `a46c15aaaea4` · preregistration SHA-256 `25bf24bf31da3027…` · lock SHA-256 `191bb06dbb75aba8…` · stage C started 2026-11-01T12:58:58+00:00.

Run once, as preregistered in `docs/preregistration.md`; every result is published, failed criteria included. Local outputs in `runs/test_once/` (not in git).

Clips: 49 from games 7, 8, 11; labelled frames (one in five): 7350. Intervals: 95 % percentile bootstrap over clips (2000 draws, seed 0).

## Success criteria (as preregistered)

| # | Endpoint | Result | Criterion | Verdict |
|---|---|---|---|---|
| P1 | Coverage of the 90 % Mondrian regions, PTZ-smoothed camera, SN-GSR raw labels, player-frames pooled | 93.1% [91.7%, 94.5%] | 95 % CI contains 90 % | **not met** |
| P2 | Median region area / constant circle at the same level, same camera and reference | 0.73 | below 1 | **met** |
| P3 | AUROC of gradient boosting − AUROC of `log_sigma_centre` (best single feature on dev), per-frame calibrations | +0.114 | above 0 | **met** |

## Coverage and size of the regions

Headline rows: player-frames whose raw calibration succeeded and for which the camera publishes a position (frames filled by the smoother are reported apart).

| Camera | Reference | Player-frames | Clips | Mondrian coverage [95 % CI] | Mondrian coverage, clips weighted equally [95 % CI] | Constant circle [95 % CI] | Uncalibrated Laplace | Median area (m²) | Constant circle area (m²) | Area ratio [95 % CI] | All players of a frame covered |
|---|---|---|---|---|---|---|---|---|---|---|---|
| per-frame camera | SN-GSR raw labels | 91268 | 49 | 90.7% [88.8%, 92.6%] | 90.6% [88.7%, 92.4%] | 92.1% [89.8%, 93.9%] | 17.9% | 4.69 | 13.98 | 0.34 [0.27, 0.43] | 73.9% |
| per-frame camera | trusted line oracle | 50908 | 48 | 90.2% [85.7%, 93.7%] | 89.1% [84.5%, 92.8%] | 90.2% [85.5%, 93.9%] | 16.4% | 9.83 | 51.12 | 0.19 [0.15, 0.24] | 81.2% |
| PTZ-smoothed camera | SN-GSR raw labels | 89886 | 49 | 93.1% [91.7%, 94.5%] | 93.0% [91.6%, 94.3%] | 94.1% [92.8%, 95.4%] | 8.0% | 5.17 | 7.03 | 0.73 [0.70, 0.77] | 76.6% |
| PTZ-smoothed camera | trusted line oracle | 49669 | 48 | 90.5% [85.8%, 94.2%] | 88.1% [82.8%, 92.3%] | 90.1% [85.1%, 94.0%] | 7.7% | 37.60 | 42.64 | 0.88 [0.59, 1.11] | 83.3% |

Raw-OK player-frames that the other camera publishes and this camera does not (outside its headline rows): the smoothed camera's abstentions, and feet that the per-frame camera's homography cannot place on the pitch. Coverage counting them as misses:

| Camera | Reference | Player-frames | Share | Frames | Coverage counting them as misses, player-frames pooled | Same, clips weighted equally |
|---|---|---|---|---|---|---|
| per-frame camera | SN-GSR raw labels | 186 | 0.2% | 14 | 90.5% | 90.4% |
| per-frame camera | trusted line oracle | 186 | 0.4% | 14 | 89.9% | 88.7% |
| PTZ-smoothed camera | SN-GSR raw labels | 1568 | 1.7% | 89 | 91.5% | 91.2% |
| PTZ-smoothed camera | trusted line oracle | 1425 | 2.8% | 79 | 87.9% | 85.8% |

Per test game (Mondrian coverage [95 % CI], player-frames):

| Camera | Reference | Game 7 | Game 8 | Game 11 |
|---|---|---|---|---|
| per-frame camera | SN-GSR raw labels | 89.8% [85.8%, 93.5%], n = 29526 | 89.3% [85.6%, 92.7%], n = 29545 | 92.9% [90.4%, 94.9%], n = 32197 |
| per-frame camera | trusted line oracle | 90.2% [81.8%, 95.3%], n = 12065 | 93.0% [86.0%, 97.2%], n = 17471 | 87.9% [79.3%, 93.6%], n = 21372 |
| PTZ-smoothed camera | SN-GSR raw labels | 89.8% [86.8%, 93.0%], n = 29440 | 93.2% [91.8%, 94.6%], n = 28268 | 96.0% [94.9%, 97.1%], n = 32178 |
| PTZ-smoothed camera | trusted line oracle | 88.2% [78.6%, 93.8%], n = 12047 | 92.2% [83.5%, 97.2%], n = 16269 | 90.5% [82.7%, 95.6%], n = 21353 |

Coverage by distance tercile (metres per pixel at the feet), Mondrian / constant circle:

| Camera | Reference | Near | Middle | Far |
|---|---|---|---|---|
| per-frame camera | SN-GSR raw labels | 90.6% / 88.7% | 89.2% / 95.8% | 92.4% / 91.8% |
| per-frame camera | trusted line oracle | 89.4% / 86.2% | 90.1% / 91.4% | 91.1% / 93.0% |
| PTZ-smoothed camera | SN-GSR raw labels | 90.2% / 91.9% | 94.2% / 95.9% | 94.9% / 94.6% |
| PTZ-smoothed camera | trusted line oracle | 88.3% / 85.1% | 92.9% / 91.6% | 90.2% / 93.6% |

Frames filled by the smoother (raw calibration failed), coverage Mondrian / constant circle:
- SN-GSR raw labels: 90.2% / 88.8% (n = 572).
- trusted line oracle: 53.4% / 42.6% (n = 373).

## Position error (m) on the same 89700 player-frames

| Camera | Labels p50 / p90 / p99 | Line oracle p50 / p90 / p99 | Smoothed labels p50 / p90 / p99 |
|---|---|---|---|
| per-frame camera | 0.51 / 1.82 / 6.87 | 0.41 / 4.04 / 15.67 | 0.51 / 1.73 / 6.51 |
| PTZ-smoothed camera | 0.40 / 1.16 / 4.00 | 0.35 / 3.57 / 14.67 | 0.40 / 1.05 / 3.30 |

## Failure detection (per-frame calibrations)

Frames with self-consistent lines and a calibration: 3839; failures (mean distance to the annotated lines > 10 px): 390.
Boosting − `log_sigma_centre`: +0.114 [+0.084, +0.150] (2000 bootstrap draws with both classes). Best frozen single feature on these frames, its dev sign kept (post hoc, stricter): `log_sigma_centre`, AUROC 0.797; boosting − it: +0.114.

| Detector | AUROC | Average precision | Failures removed at 5 % / 10 % abstention |
|---|---|---|---|
| boosting | 0.911 | 0.566 | 31.5% / 55.1% |
| logistic | 0.838 | 0.321 | 18.2% / 37.2% |
| log_sigma_centre | 0.797 | 0.275 | 14.6% / 34.1% |
| n_corr | 0.712 | 0.222 | 15.9% / 26.4% |
| rms_px | 0.707 | 0.241 | 19.0% / 28.2% |
| log_jump_px | 0.700 | 0.204 | 13.1% / 26.7% |
| n_kp | 0.682 | 0.222 | 15.6% / 23.6% |
| hull_area | 0.666 | 0.226 | 16.4% / 28.7% |
| kp_conf_mean | 0.663 | 0.207 | 14.9% / 23.1% |
| n_lines | 0.650 | 0.141 | 9.2% / 19.5% |
| log_s2 | 0.616 | 0.134 | 6.2% / 13.8% |
| log_cond | 0.470 | 0.243 | 20.0% / 27.4% |
| inlier_ratio | 0.408 | 0.082 | 1.8% / 5.1% |

| Game | Frames | Failure rate | AUROC boosting | AUROC best single (dev) |
|---|---|---|---|---|
| 7 | 945 | 13.8% | 0.896 | 0.783 |
| 8 | 1533 | 11.4% | 0.909 | 0.779 |
| 11 | 1361 | 6.3% | 0.929 | 0.807 |

## Residual failures of the smoothed camera

3741 frames with self-consistent lines where both cameras are calibrated: above 10 px, per-frame camera 9.7%, smoothed camera 2.4% (line distance p90 9.8 and 5.3 px). Apart: 38 frames filled by the smoother (above 10 px 7.9%) and 98 raw-OK frames where it abstains. Published-position rate: per-frame 91.6%, smoothed 90.8%.

| Output type | Frames | Above 10 px |
|---|---|---|
| smoothed | 3567 | 2.1% |
| repaired | 62 | 9.7% |
| filled | 38 | 7.9% |
| bridged | 76 | 0.0% |
| passthrough | 36 | 22.2% |

## KPI proxy (5 fps, against the smoothed labels, same input player-frames; each camera's runs break at its own errors above 10 m)

The noise floor is the raw labels' own bias against the smoothed labels, on the same runs.

| Camera | Runs | Distance bias | HSR-distance bias | Noise floor: distance / HSR-distance bias | Speed error RMS / p90 (m/s) |
|---|---|---|---|---|---|
| per-frame camera | 2226 | +26.1% | +72.4% | +16.8% / +39.2% | 1.61 / 1.48 |
| PTZ-smoothed camera | 2217 | +3.4% | +5.5% | +17.0% / +40.3% | 0.78 / 0.80 |
