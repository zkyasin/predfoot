# Position-error process on SN-GSR dev (T4.2)

`uv run python scripts/error_process.py runs/20261031-153059_c3861c7_ptz/players.parquet --compare runs/20261031-152537_c3861c7_none/players.parquet --name ptz --compare-name none` · commit `86382dc`

Errors of predfoot's positions (ground-truth boxes, our calibration), split into a common part
shared by the players of a frame (mostly the camera) and an individual part. σ is the marginal
standard deviation per axis; ACF the autocorrelation at 0.2-1.0 s (frames are one in five) of
the pooled process, 1 − γ(k)/σ² with γ the semi-variogram pooled over all series, so that
σ·√(2(1 − ACF(0.2 s))) is the lag-1 step RMS; ρ₂₅ the 25 Hz lag-one correlation implied by the
0.2 s and by the 1 s lag under an exponential decay.
Errors above 10 m are calibration failures, left to abstention, and excluded here. Frames filled by the smoother (raw calibration failed) are excluded.

Both tables restricted to the same 252515 player-frames.

## Reference: labels

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| none | 252037 | common | 0.618 | 0.57 / 0.48 / 0.42 / 0.39 / 0.35 | 0.571 | 0.895 – 0.958 |
| none | 252037 | individual | 0.613 | 0.49 / 0.43 / 0.38 / 0.36 / 0.33 | 0.617 | 0.868 – 0.957 |
| ptz | 252270 | common | 0.504 | 0.67 / 0.57 / 0.51 / 0.46 / 0.42 | 0.407 | 0.924 – 0.966 |
| ptz | 252270 | individual | 0.393 | 0.66 / 0.56 / 0.49 / 0.44 / 0.41 | 0.324 | 0.920 – 0.965 |

## Reference: trusted line oracle

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| none | 152598 | common | 0.952 | 0.79 / 0.72 / 0.68 / 0.65 / 0.60 | 0.618 | 0.954 – 0.979 |
| none | 152598 | individual | 0.621 | 0.59 / 0.51 / 0.47 / 0.45 / 0.42 | 0.563 | 0.900 – 0.966 |
| ptz | 152855 | common | 0.946 | 0.87 / 0.80 / 0.75 / 0.70 / 0.64 | 0.488 | 0.972 – 0.982 |
| ptz | 152855 | individual | 0.525 | 0.81 / 0.72 / 0.67 / 0.62 / 0.56 | 0.319 | 0.960 – 0.977 |

## Reference: smoothed labels

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| none | 252078 | common | 0.561 | 0.68 / 0.61 / 0.52 / 0.49 / 0.43 | 0.449 | 0.926 – 0.966 |
| none | 252078 | individual | 0.596 | 0.51 / 0.45 / 0.41 / 0.39 / 0.35 | 0.592 | 0.873 – 0.959 |
| ptz | 252292 | common | 0.429 | 0.91 / 0.82 / 0.73 / 0.65 / 0.59 | 0.185 | 0.981 – 0.979 |
| ptz | 252292 | individual | 0.363 | 0.74 / 0.64 / 0.59 / 0.53 / 0.48 | 0.263 | 0.941 – 0.971 |

## KPI proxy on SN-GSR (5 fps, Savitzky-Golay window 5, order 2)

| Camera | Runs | Samples | Distance bias vs smoothed labels | HSR-distance bias | Speed error RMS / p90 (m/s) |
|---|---|---|---|---|---|
| none | 5433 | 243683 | +27.1% | +67.9% | 1.50 / 1.54 |
| ptz | 5392 | 244116 | +3.9% | +9.5% | 0.79 / 0.88 |

Noise floor of the reference: the raw labels' own distance bias against the smoothed labels is +17.9% (HSR distance +43.4%; the smoothed labels cover 15086 m at high speed).

## Arguments for kpi_sensitivity.py (labels reference)

- none, 0.2 s lag: `--measured 0.61 0.868 0.62 0.895`
- none, 1 s lag: `--measured 0.61 0.957 0.62 0.958`
- ptz, 0.2 s lag: `--measured 0.39 0.920 0.50 0.924`
- ptz, 1 s lag: `--measured 0.39 0.965 0.50 0.966`

Caveats. (1) Errors against the labels include the labels' own error, so σ is an upper bound of
predfoot's; the smoothed labels remove the annotators' jitter but carry track-smoothing artefacts,
so use them for temporal metrics only. (2) The 25 Hz correlation is bracketed, not measured: the
two lags disagree when the process is not a single exponential (white jitter plus slow drift), and
only clips decoded at the full 25 fps (planned in T1.6) can settle it.
