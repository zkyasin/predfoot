# Preregistration (DRAFT, not frozen)

- Status: **draft for review**. Nothing here is binding until the file is frozen by
  `scripts/freeze_prereg.py` (commit hash quoted in the README). No test-game result exists yet.
- Locked test games: SN-GSR test split, games 7, 8 and 11 (49 clips). Development games: 2, 3, 4, 5, 6, 9.

## 1. Questions

1. **Coverage.** Do predfoot's 90 % position regions cover 90 % of the true positions on games never used for
   anything else?
2. **Sharpness.** At the same coverage, are they smaller than a constant-radius circle?
3. **Failure detection.** Do calibration-health features flag the frames whose camera is wrong better than the
   best single feature?

## 2. Data and references

- Frames: one in five (5 fps) of every clip; ground-truth boxes; truncated boxes, frames without a labelled
  camera and labels from a failed raw camera (raw and smoothed labels more than 5 m apart) are excluded.
- Two position references, reported side by side:
  - **R1, line oracle (primary)**: the homography fitted to the annotated pitch lines of the frame, used only
    where it fits those lines within 3 px. Rationale: the SN-GSR position labels come from an annotators'
    camera that sits 3.9 px (median) from the annotated lines on dev, so they cannot referee sub-metre errors.
  - **R2, SN-GSR raw labels (secondary)**: `bbox_pitch_raw`, as published.
- Unit of analysis: the clip. Confidence intervals resample clips (bootstrap, 2,000 draws).

## 3. Method under test (frozen at the commit that freezes this file)

- PnLCalib keypoint and line networks (published CC BY 4.0 weights, no fine-tuning).
- Calibration: MSAC RANSAC, soft-L1 Levenberg-Marquardt on keypoints and line points, Laplace covariance with
  Huber's scale correction, noise prior σ₀ = 2 px with 10 degrees of freedom.
- Positions: delta-method covariance Σp of the foot point.
- Conformal: grouped split conformal, α = 0.10, score d² = eᵀ Σp⁻¹ e, Mondrian bins over metres per pixel at
  the feet (3 bins) × RANSAC inliers (2 bins), minimum 200 scores and 3 clips per bin, pooled fallback.
  Quantiles for the test run are fitted on all six dev games.
- Health: logistic regression on the features of `predfoot.health.features`, trained on all dev games;
  abstention threshold set on dev to keep 95 % of frames.

## 4. Primary endpoints

| # | Endpoint | Reference | Success criterion |
|---|---|---|---|
| P1 | Marginal coverage of the Mondrian regions on the test games | R1 | 95 % CI contains 0.90 |
| P2 | Median region area vs constant circle, at the same nominal level | R1 | Ratio < 1 |
| P3 | AUROC of the failure detector on test frames (failure: mean distance to the annotated lines > 10 px, frames with self-consistent lines) | lines | Above the best single feature |

Secondary: everything with R2; coverage by distance tercile; simultaneous (per-frame) coverage; per-game
coverage; the share of failures removed at 5 % and 10 % abstention.

## 5. Procedure

1. Freeze this file (`scripts/freeze_prereg.py` records its hash and the code commit).
2. `scripts/run_test_once.py` refuses to run without the freeze, and refuses a second run.
3. Every result of the test run is published, including failed criteria.

## 6. Known limitations, stated in advance

- Six development games and three test games from one league and one broadcast style: no claim beyond that.
- Frames are correlated within a clip; the effective sample is closer to the number of clips (49 on test).
- R1 exists for about half of the player-frames (frames whose annotated lines are self-consistent).
