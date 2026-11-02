# Dev failure detection: flagging a wrong camera from the frame's own features

`uv run python scripts/eval_health.py runs/20261031-1200_7ae1b77/frames.parquet` · commit `7ae1b77`

Calibrated dev frames with self-consistent line annotations: 10024, from 115 clips and 6 games. Failures (mean distance to the annotated lines > 10 px): 11.0%.
All scores are out-of-game: each game is scored by detectors trained on the other games.

**Abstaining on the 10 % riskiest frames removes 50% of the failures with gradient boosting and 32% with logistic regression; the best single feature, `log_sigma_centre`, removes 29%.**

| Detector | AUROC | Average precision | AURC (failure rate) | AURC (line px, capped at 50) | Failures removed at 5 % / 10 % abstention |
|---|---|---|---|---|---|
| boosting | 0.890 | 0.558 | 0.021 | 3.87 | 31% / 50% |
| logistic | 0.791 | 0.306 | 0.037 | 4.25 | 16% / 32% |
| log_sigma_centre | 0.731 | 0.256 | 0.053 | 4.64 | 15% / 29% |
| log_jump_px | 0.706 | 0.230 | 0.057 | 4.78 | 15% / 25% |
| rms_px | 0.640 | 0.195 | 0.075 | 5.06 | 14% / 22% |
| hull_area | 0.630 | 0.210 | 0.082 | 5.41 | 15% / 26% |
| n_corr | 0.630 | 0.181 | 0.076 | 5.21 | 10% / 23% |
| n_kp | 0.605 | 0.182 | 0.090 | 5.50 | 11% / 24% |
| kp_conf_mean | 0.598 | 0.169 | 0.091 | 5.64 | 11% / 20% |
| log_s2 | 0.584 | 0.138 | 0.086 | 5.31 | 7% / 14% |
| n_lines | 0.579 | 0.128 | 0.077 | 5.23 | 10% / 17% |
| log_cond | 0.548 | 0.257 | 0.116 | 6.22 | 19% / 28% |
| inlier_ratio | 0.522 | 0.120 | 0.102 | 5.74 | 5% / 10% |

Per held-out game (logistic regression):

| Game | Frames | Failure rate | AUROC |
|---|---|---|---|
| 2 | 1049 | 13.3% | 0.722 |
| 3 | 2174 | 6.6% | 0.817 |
| 4 | 1500 | 11.7% | 0.790 |
| 5 | 1605 | 10.2% | 0.793 |
| 6 | 1923 | 12.1% | 0.802 |
| 9 | 1773 | 13.8% | 0.787 |

Notes. Exploratory dev numbers. Frames are one in five and correlated within a clip, so the
effective sample is closer to the number of clips than to the number of frames.
