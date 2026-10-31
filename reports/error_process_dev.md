# Position-error process on SN-GSR dev (T4.2)

`uv run python scripts/error_process.py ../geometry/runs/20261031-1200_7ae1b77/players.parquet` · commit `da2cf77`

Errors of predfoot's positions (ground-truth boxes, our calibration), split into a common part
shared by the players of a frame (mostly the camera) and an individual part. σ is the marginal
standard deviation per axis; ρ is the lag-one autocorrelation at 5 fps, converted to 25 Hz.
Errors above 10 m are calibration failures, left to abstention, and excluded here.

| Reference | Player-frames | Clips | σ common (m) | ρ common 5 fps → 25 Hz | σ individual (m) | ρ individual 5 fps → 25 Hz |
|---|---|---|---|---|---|---|
| labels | 254485 | 115 | 0.62 | 0.44 → 0.848 | 0.62 | 0.40 → 0.832 |
| trusted line oracle | 153142 | 115 | 0.95 | 0.49 → 0.866 | 0.62 | 0.39 → 0.829 |

Reading: a ρ close to 1 at 25 Hz means slow errors (a camera that is wrong for seconds at a time),
which the KPI study found to be the least harmful to distances; ρ ≈ 0.9 (about 0.4 s) was the most
harmful. These parameters are the input of the KPI budget with measured errors (T4.3).

Caveats. (1) Errors against the labels include the labels' own error, so σ is an upper bound of
predfoot's. (2) Each frame is calibrated independently, so part of the error is white jitter at any
frame rate: the exponential conversion from 5 fps to 25 Hz overstates the 25 Hz correlation of that
part. A direct measurement on clips decoded at the full 25 fps (planned in T1.6) will settle it.
