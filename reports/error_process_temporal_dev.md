# Position-error process on SN-GSR dev

`uv run python scripts/error_process.py runs/20261031-163641_bf28b35_ptz/players.parquet --compare runs/20261031-152537_c3861c7_none/players.parquet --name ptz --compare-name none` · commit `1bb22eb`

Errors of predfoot's positions (ground-truth boxes, our calibration), split into a common part
shared by the players of a frame (mostly the camera) and an individual part. σ is the marginal
standard deviation per axis; ACF the autocorrelation at 0.2-1.0 s (frames are one in five) of
the pooled process, 1 − γ(k)/σ² with γ the semi-variogram pooled over all series, so that
σ·√(2(1 − ACF(0.2 s))) is the lag-1 step RMS; ρ₂₅ the 25 Hz lag-one correlation implied by the
0.2 s and by the 1 s lag under an exponential decay.
Errors above 10 m are calibration failures, left to abstention, and excluded here. Frames filled by the smoother (raw calibration failed) are excluded.

Both tables restricted to the same 252188 player-frames.

## Reference: labels

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| none | 251718 | common | 0.615 | 0.57 / 0.47 / 0.42 / 0.39 / 0.35 | 0.568 | 0.895 – 0.959 |
| none | 251718 | individual | 0.612 | 0.49 / 0.43 / 0.38 / 0.35 / 0.33 | 0.615 | 0.869 – 0.957 |
| ptz | 252051 | common | 0.498 | 0.69 / 0.58 / 0.52 / 0.47 / 0.42 | 0.393 | 0.928 – 0.966 |
| ptz | 252051 | individual | 0.382 | 0.70 / 0.59 / 0.51 / 0.46 / 0.43 | 0.297 | 0.931 – 0.966 |

## Reference: trusted line oracle

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| none | 152578 | common | 0.952 | 0.79 / 0.72 / 0.68 / 0.65 / 0.60 | 0.618 | 0.954 – 0.980 |
| none | 152578 | individual | 0.621 | 0.59 / 0.51 / 0.47 / 0.45 / 0.42 | 0.562 | 0.900 – 0.966 |
| ptz | 152908 | common | 0.942 | 0.87 / 0.80 / 0.75 / 0.71 / 0.64 | 0.478 | 0.973 – 0.982 |
| ptz | 152908 | individual | 0.517 | 0.84 / 0.74 / 0.69 / 0.64 / 0.58 | 0.290 | 0.966 – 0.978 |

## Reference: smoothed labels

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| none | 251760 | common | 0.559 | 0.68 / 0.61 / 0.52 / 0.49 / 0.43 | 0.445 | 0.926 – 0.967 |
| none | 251760 | individual | 0.595 | 0.51 / 0.45 / 0.41 / 0.38 / 0.35 | 0.590 | 0.873 – 0.959 |
| ptz | 252073 | common | 0.422 | 0.94 / 0.84 / 0.75 / 0.67 / 0.60 | 0.151 | 0.987 – 0.980 |
| ptz | 252073 | individual | 0.350 | 0.79 / 0.68 / 0.63 / 0.56 / 0.51 | 0.229 | 0.953 – 0.973 |

## KPI proxy on SN-GSR (5 fps, Savitzky-Golay window 5, order 2)

| Camera | Runs | Samples | Distance bias vs smoothed labels | HSR-distance bias | Speed error RMS / p90 (m/s) |
|---|---|---|---|---|---|
| none | 5409 | 243070 | +27.0% | +67.2% | 1.49 / 1.54 |
| ptz | 5352 | 243624 | +2.7% | +6.2% | 0.71 / 0.85 |

Noise floor of the reference: the raw labels' own distance bias against the smoothed labels is +17.9% (HSR distance +43.1%; the smoothed labels cover 15070 m at high speed).

## Arguments for kpi_sensitivity.py (labels reference)

- none, 0.2 s lag: `--measured 0.61 0.869 0.62 0.895`
- none, 1 s lag: `--measured 0.61 0.957 0.62 0.959`
- ptz, 0.2 s lag: `--measured 0.38 0.931 0.50 0.928`
- ptz, 1 s lag: `--measured 0.38 0.966 0.50 0.966`

Caveats. (1) Errors against the labels include the labels' own error, so σ is an upper bound of
predfoot's; the smoothed labels remove the annotators' jitter but carry track-smoothing artefacts,
so use them for temporal metrics only. (2) The 25 Hz correlation is bracketed, not measured: the
two lags disagree when the process is not a single exponential (white jitter plus slow drift), and
only clips decoded at the full 25 fps can settle it.
