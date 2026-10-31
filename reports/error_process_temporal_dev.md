# Position-error process on SN-GSR dev (T4.2)

`uv run python scripts/error_process.py runs/20261031-163641_bf28b35_ptz/players.parquet --compare /Users/ya-sin/Desktop/dsba-claude-projects/cv-football/wt/temporal/runs/20261031-152537_c3861c7_none/players.parquet` · commit `bf28b35`

Errors of predfoot's positions (ground-truth boxes, our calibration), split into a common part
shared by the players of a frame (mostly the camera) and an individual part. σ is the marginal
standard deviation per axis; ACF the autocorrelation at 0.2-1.0 s (frames are one in five);
ρ₂₅ the 25 Hz lag-one correlation implied by the 0.2 s and by the 1 s lag under an exponential
decay.
Errors above 10 m are calibration failures, left to abstention, and excluded here. Frames filled by the smoother (raw calibration failed) are excluded.

Both tables restricted to the same 252188 player-frames.

## Reference: labels

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| 20261031-152537_c3861c7_none | 251718 | common | 0.615 | 0.46 / 0.35 / 0.30 / 0.27 / 0.24 | 0.568 | 0.854 – 0.944 |
| 20261031-152537_c3861c7_none | 251718 | individual | 0.612 | 0.41 / 0.33 / 0.28 / 0.25 / 0.21 | 0.615 | 0.836 – 0.940 |
| 20261031-163641_bf28b35_ptz | 252051 | common | 0.498 | 0.58 / 0.47 / 0.40 / 0.35 / 0.30 | 0.393 | 0.898 – 0.953 |
| 20261031-163641_bf28b35_ptz | 252051 | individual | 0.382 | 0.61 / 0.50 / 0.42 / 0.36 / 0.31 | 0.297 | 0.905 – 0.955 |

## Reference: trusted line oracle

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| 20261031-152537_c3861c7_none | 152578 | common | 0.952 | 0.57 / 0.42 / 0.35 / 0.28 / 0.21 | 0.618 | 0.893 – 0.940 |
| 20261031-152537_c3861c7_none | 152578 | individual | 0.621 | 0.44 / 0.31 / 0.24 / 0.20 / 0.16 | 0.562 | 0.848 – 0.929 |
| 20261031-163641_bf28b35_ptz | 152908 | common | 0.942 | 0.77 / 0.60 / 0.49 / 0.39 / 0.31 | 0.478 | 0.948 – 0.954 |
| 20261031-163641_bf28b35_ptz | 152908 | individual | 0.517 | 0.77 / 0.61 / 0.50 / 0.41 / 0.33 | 0.290 | 0.949 – 0.957 |

## Reference: smoothed labels

| Camera | Player-frames | Part | σ (m) | ACF 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s | Lag-1 step RMS (m) | ρ₂₅ bracket |
|---|---|---|---|---|---|---|
| 20261031-152537_c3861c7_none | 251760 | common | 0.559 | 0.57 / 0.48 / 0.40 / 0.35 / 0.29 | 0.445 | 0.894 – 0.952 |
| 20261031-152537_c3861c7_none | 251760 | individual | 0.595 | 0.41 / 0.32 / 0.29 / 0.26 / 0.22 | 0.590 | 0.837 – 0.941 |
| 20261031-163641_bf28b35_ptz | 252073 | common | 0.422 | 0.88 / 0.75 / 0.62 / 0.53 / 0.45 | 0.151 | 0.975 – 0.969 |
| 20261031-163641_bf28b35_ptz | 252073 | individual | 0.350 | 0.63 / 0.50 / 0.45 / 0.39 / 0.33 | 0.229 | 0.913 – 0.957 |

## KPI proxy on SN-GSR (5 fps, Savitzky-Golay window 5, order 2)

| Camera | Runs | Samples | Distance bias vs smoothed labels | HSR-distance bias | Speed error RMS / p90 (m/s) |
|---|---|---|---|---|---|
| 20261031-152537_c3861c7_none | 5409 | 243070 | +27.0% | +67.2% | 1.49 / 1.54 |
| 20261031-163641_bf28b35_ptz | 5352 | 243624 | +2.7% | +6.2% | 0.71 / 0.85 |

Noise floor of the reference: the raw labels' own distance bias against the smoothed labels is +17.9% (HSR distance +43.1%; the smoothed labels cover 15070 m at high speed).

## Arguments for kpi_sensitivity.py (labels reference)

- 20261031-152537_c3861c7_none, 0.2 s lag: `--measured 0.61 0.836 0.62 0.854`
- 20261031-152537_c3861c7_none, 1 s lag: `--measured 0.61 0.940 0.62 0.944`
- 20261031-163641_bf28b35_ptz, 0.2 s lag: `--measured 0.38 0.905 0.50 0.898`
- 20261031-163641_bf28b35_ptz, 1 s lag: `--measured 0.38 0.955 0.50 0.953`

Caveats. (1) Errors against the labels include the labels' own error, so σ is an upper bound of
predfoot's; the smoothed labels remove the annotators' jitter but carry track-smoothing artefacts,
so use them for temporal metrics only. (2) The 25 Hz correlation is bracketed, not measured: the
two lags disagree when the process is not a single exponential (white jitter plus slow drift), and
only clips decoded at the full 25 fps (planned in T1.6) can settle it.
