# Dev failure detection: flagging a wrong camera from the frame's own features

`uv run python scripts/eval_health.py runs/20261031-0151_ba55bc7/frames.parquet` · commit `ba55bc7`

Calibrated dev frames with self-consistent line annotations: 4402, from 49 clips and 6 games. Failures (mean distance to the annotated lines > 10 px): 11.4%.
All scores are out-of-game: each game is scored by detectors trained on the other games.

**Abstaining on the 10 % riskiest frames removes 37% of the failures (logistic regression); the best single feature, `log_sigma_centre`, removes 29%.**

| Detector | AUROC | Average precision | AURC (failure rate) | AURC (line px, capped at 50) | Failures removed at 5 % / 10 % abstention |
|---|---|---|---|---|---|
| boosting | 0.854 | 0.477 | 0.028 | 4.09 | 27% / 43% |
| logistic | 0.803 | 0.348 | 0.040 | 4.25 | 18% / 37% |
| log_sigma_centre | 0.768 | 0.287 | 0.046 | 4.51 | 15% / 29% |
| log_jump_px | 0.704 | 0.229 | 0.058 | 4.81 | 14% / 24% |
| hull_area | 0.656 | 0.234 | 0.077 | 5.19 | 16% / 30% |
| rms_px | 0.624 | 0.194 | 0.082 | 5.23 | 12% / 20% |
| kp_conf_mean | 0.617 | 0.195 | 0.095 | 5.79 | 12% / 22% |
| log_s2 | 0.599 | 0.155 | 0.087 | 5.36 | 7% / 15% |
| n_corr | 0.585 | 0.168 | 0.097 | 5.66 | 11% / 19% |
| n_lines | 0.581 | 0.134 | 0.081 | 5.33 | 6% / 13% |
| n_kp | 0.558 | 0.157 | 0.108 | 5.94 | 11% / 19% |
| inlier_ratio | 0.519 | 0.122 | 0.112 | 6.05 | 5% / 10% |
| log_cond | 0.432 | 0.135 | 0.162 | 6.67 | 7% / 12% |

Per held-out game (logistic regression):

| Game | Frames | Failure rate | AUROC |
|---|---|---|---|
| 4 | 1500 | 11.7% | 0.797 |
| 5 | 780 | 11.7% | 0.826 |
| 6 | 1923 | 12.1% | 0.786 |
| 9 | 46 | 10.9% | 0.849 |

Notes. Exploratory dev numbers. Frames are one in five and correlated within a clip, so the
effective sample is closer to the number of clips than to the number of frames. The critic CNN
(ticket T3.3) has to beat the best row of this table to be worth its cost.
