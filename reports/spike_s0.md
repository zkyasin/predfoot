# Spike S0: calibration on real broadcast frames

`uv run python scripts/spike_s0.py <5 clips>` · commit `b8c89d1` · model `pnlcalib-sv@7ea78fa76aaf+d72f4ed71734`

Clips: SNGS-036, SNGS-056, SNGS-067, SNGS-093, SNGS-168. Frames: 750. Ground-truth boxes, truncated boxes excluded.

## Go/no-go criteria

| Criterion | Pass | Measured |
|---|---|---|
| Throughput ≥ 2 frames/s (both networks, M4 Pro, batch 1) | yes | 2.3 frames/s |
| ≥ 80 % of frames calibrated | yes | 99% |
| Error p90 ≥ 2 × floor (labels as reference) | no | 1.44 m vs floor 2.94 m (line oracle vs labels) |
| Spearman(σ̂, error) ≥ 0.3 and tercile ratio ≥ 2 (line-oracle reference) | yes | ρ = 0.54; p90 top/bottom σ̂ tercile 5.45 / 0.36 m |

## Position errors (metres)

| Calibration | Reference | p50 | p90 | Spearman(σ̂, error) |
|---|---|---|---|---|
| raw vs smoothed labels | labels | 0.12 | 0.44 | — |
| keypoints | labels | 0.60 | 1.44 | 0.31 |
| keypoints | line oracle | 0.31 | 3.19 | 0.54 |
| keypoints + lines | labels | 0.61 | 1.44 | 0.30 |
| keypoints + lines | line oracle | 0.32 | 3.15 | 0.54 |
| line oracle | labels | 0.54 | 2.94 | — |

## Calibration quality (pixels: mean distance of annotated line points to the projected template)

| Camera | p50 | p90 |
|---|---|---|
| keypoints | 4.6 | 17.3 |
| keypoints + lines | 4.5 | 17.2 |
| line oracle (fitted to these lines) | 2.0 | 2.4 |
| annotators' camera (homography implied by the position labels) | 3.9 | 10.7 |

## Per clip (keypoints + lines, labels as reference)

| Clip | Game | Frames | Calibrated | Line px p50 | Labels' camera px p50 |
|---|---|---|---|---|---|
| SNGS-036 | 2 | 150 | 100% | 15.9 | 9.6 |
| SNGS-056 | 3 | 150 | 100% | 4.0 | 4.7 |
| SNGS-067 | 4 | 150 | 100% | 4.6 | 3.2 |
| SNGS-093 | 5 | 150 | 95% | 3.6 | 2.6 |
| SNGS-168 | 9 | 150 | 100% | 4.7 | 2.7 |

Notes. The position labels fit a homography exactly (they are the annotators' camera applied to the
box bottoms), so their line reprojection error measures how far that camera is from the annotated
lines. σ̂ = √λmax(Σp) comes from the Laplace covariance before any conformal calibration; only its
ranking matters here. Frames within a clip are strongly correlated: treat these numbers as a
feasibility check, not results.
