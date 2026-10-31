# Position-error process on SN-GSR dev (T4.2)

`uv run python scripts/error_process.py runs/20261031-152537_c3861c7_none/players.parquet` · commit `1bb22eb`

Errors of predfoot's positions (ground-truth boxes, our calibration), split into a common part
shared by the players of a frame (mostly the camera) and an individual part. σ is the marginal
standard deviation per axis; ACF the autocorrelation at 0.2-1.0 s (frames are one in five) of
the pooled process, 1 − γ(k)/σ² with γ the semi-variogram pooled over all series, so that
σ·√(2(1 − ACF(0.2 s))) is the lag-1 step RMS; ρ₂₅ the 25 Hz lag-one correlation implied by the
0.2 s and by the 1 s lag under an exponential decay.
Errors above 10 m are calibration failures, left to abstention, and excluded here. Frames filled by the smoother (raw calibration failed) are excluded.

## Reference: labels

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| 20261031-152537_c3861c7_none | 254485 | common | 0.619 | 0.57 / 0.47 / 0.41 / 0.39 / 0.34 | 0.572 | 0.895 – 0.958 |
| 20261031-152537_c3861c7_none | 254485 | individual | 0.616 | 0.49 / 0.43 / 0.37 / 0.35 / 0.33 | 0.619 | 0.869 – 0.957 |

## Reference: trusted line oracle

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| 20261031-152537_c3861c7_none | 153142 | common | 0.953 | 0.79 / 0.72 / 0.68 / 0.65 / 0.60 | 0.621 | 0.953 – 0.980 |
| 20261031-152537_c3861c7_none | 153142 | individual | 0.622 | 0.59 / 0.52 / 0.47 / 0.45 / 0.42 | 0.564 | 0.900 – 0.966 |

## Reference: smoothed labels

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| 20261031-152537_c3861c7_none | 254524 | common | 0.562 | 0.68 / 0.60 / 0.52 / 0.48 / 0.43 | 0.450 | 0.926 – 0.966 |
| 20261031-152537_c3861c7_none | 254524 | individual | 0.599 | 0.51 / 0.45 / 0.40 / 0.38 / 0.35 | 0.594 | 0.873 – 0.959 |

## KPI proxy on SN-GSR (5 fps, Savitzky-Golay window 5, order 2)

| Camera | Runs | Samples | Distance bias vs smoothed labels | HSR-distance bias | Speed error RMS / p90 (m/s) |
|---|---|---|---|---|---|
| 20261031-152537_c3861c7_none | 5367 | 246091 | +27.4% | +67.8% | 1.50 / 1.55 |

Noise floor of the reference: the raw labels' own distance bias against the smoothed labels is +18.1% (HSR distance +43.4%; the smoothed labels cover 15270 m at high speed).

## Arguments for kpi_sensitivity.py (labels reference)

- 20261031-152537_c3861c7_none, 0.2 s lag: `--measured 0.62 0.869 0.62 0.895`
- 20261031-152537_c3861c7_none, 1 s lag: `--measured 0.62 0.957 0.62 0.958`

Caveats. (1) Errors against the labels include the labels' own error, so σ is an upper bound of
predfoot's; the smoothed labels remove the annotators' jitter but carry track-smoothing artefacts,
so use them for temporal metrics only. (2) The 25 Hz correlation is bracketed, not measured: the
two lags disagree when the process is not a single exponential (white jitter plus slow drift), and
only clips decoded at the full 25 fps (planned in T1.6) can settle it.
