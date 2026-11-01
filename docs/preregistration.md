# Preregistration of the test run

- Version: 1
- Status: final. The owner settled every choice on 2026-11-01. The text binds from the commit that freezes
  it: `scripts/freeze_prereg.py` writes `docs/preregistration.lock.json` (SHA-256 of this file and of the
  frozen artifacts, code commit, date) and prints the line that README.md quotes.
- Prior exposure to the test split, stated exactly. At M0 the SN-GSR labels of every split, test included
  (v1.3), were downloaded (`scripts/download_data.py sn-gsr-labels`) and parsed by `scripts/build_index.py`
  into `index/clips.parquet` (one row per clip, test clips included) and `index/test/` (the test split's
  per-frame tables, kept apart so that nothing reads them by accident). Only the split's structure was
  reported (`reports/data_audit.md`: 49 clips; games 7, 8 and 11 with 16, 19 and 14 clips; 750 frames each).
  Since then no analysis has read a test label: `index/test/` has not been opened, no test frame has been
  downloaded and no test result exists. Nothing in the lock can prove this; stage C's claim to be the only
  reading of the test labels rests on nobody opening `index/test/` outside it.
- Locked test games: SN-GSR test split, games 7, 8 and 11 (49 clips). Development games: 2, 3, 4, 5, 6 and 9
  (115 clips).

## 1. Questions

1. **Coverage.** Do predfoot's 90 % position regions cover 90 % of the true positions on games never used for
   anything else?
2. **Sharpness.** At the same nominal level, are they smaller than a constant-radius circle?
3. **Failure detection.** Do the per-frame calibration-health features, combined by gradient boosting, flag the
   frames whose camera is wrong better than the best single feature?
4. **KPI cost (secondary).** How much do the per-frame and the smoothed cameras bias distance and
   high-speed-running distance?

## 2. Data and references

- Frames: one in five (5 fps) of every test clip, downloaded and passed through the keypoint networks before
  any label is read. Positions come from the ground-truth boxes (feet at the bottom centre), so the test
  measures calibration and uncertainty, not detection. Excluded, as on dev: truncated boxes, frames without
  labelled positions, and labels from a failed raw camera (raw and smoothed labels more than 5 m apart).
- **R1, SN-GSR raw labels (`bbox_pitch_raw`): primary reference**.
- **R2, trusted line oracle: secondary reference**. The homography fitted to the annotated pitch lines of the
  frame, used only where it fits those lines within 3 px (about 60 % of the player-frames on dev: 61 % for the
  per-frame camera, 62 % for the smoothed one).
- The two references are not symmetric, and neither is exact:
  - R1 comes from the annotators' camera, which sits 3.9 px (median) from the annotated lines on five dev clips
    (`reports/spike_s0.md`, 2.6 to 9.6 px per clip): coverage against R1 mixes predfoot's error with the
    labels' own.
  - R2 is fitted frame by frame. Its own jitter counts as error, more so for a camera that does not jitter: on
    dev both cameras over-cover against R2 (94.4 % [92.9, 96.0] for the smoothed camera, 94.4 % [92.7, 96.0]
    for the per-frame one), and the smoothed camera's regions calibrated against R2 are twice those of the
    per-frame camera (median 26.2 m² against 12.6 m²), whereas against R1 they are smaller (5.4 against
    6.2 m²). R2 therefore cannot referee the smoothing.
  - R2 has gross failures of its own: fitting the annotated lines within 3 px does not make the homography
    right where the lines are few or nearly parallel. On dev, 1.6 % of the per-frame camera's R2
    player-frames (449 frames) and 1.4 % of the smoothed camera's (366 frames) are more than 10 m from R2,
    while the camera agrees with R1 (median error 0.79 and 0.34 m on those rows). R2's error tails (p99) and
    part of its coverage therefore measure R2's failures as well as the camera's; R2 is not a gross-error
    check.
- Smoothed labels (`bbox_pitch`) are used three ways: as the reference of the KPI proxy (temporal metrics);
  in the exclusion rule (raw and smoothed labels more than 5 m apart); and as a third reference of the
  position errors (secondary analysis 7).

## 3. Method under test (frozen at the commit that freezes this file)

- PnLCalib keypoint and line networks (published CC BY 4.0 weights, no fine-tuning), run once into a detection
  cache kept apart from the dev cache.
- Per-frame calibration: MSAC RANSAC seeded by (clip, frame), soft-L1 Levenberg-Marquardt on keypoints and
  line points, Laplace covariance with Huber's scale correction, noise prior σ₀ = 2 px with 10 degrees of
  freedom.
- **Camera under test: the PTZ-smoothed camera**: the fixed-centre pan-tilt-zoom Kalman/RTS smoother of
  `predfoot.calib.temporal`, its noise parameters (κ², q pan, q tilt, q log f, gap factor λ) fitted without
  labels on all six dev games and frozen; projection centres per test clip, and the game centre per test game,
  estimated from the test clips' own per-frame calibrations, without labels. **Comparison arm: the per-frame
  camera.** Both arms come from the same detections and the same per-frame calibrations, and each runs once.
- Positions: delta-method covariance Σp of the foot point.
- Conformal regions: grouped split conformal, α = 0.10, score d² = eᵀ Σp⁻¹ e, every clip with the same total
  weight and the level raised to (1 − α)(1 + 1/G) for G clips. Mondrian bins over metres per pixel at the feet
  (3 bins) × RANSAC inliers (2 bins), with at least 200 scores and 3 clips per bin, else the pooled quantile.
  For the smoothed camera, quantiles are fitted within each covariance source (smoother: smoothed and
  repaired frames; gap: filled and bridged frames, one pooled quantile; frame: single-frame cameras); a
  source with fewer than 10 calibration clips uses the model fitted on all sources. One set of models per
  camera and reference, fitted on all six dev games (the construction of a leave-one-game-out fold of
  `scripts/eval_dev.py` with every dev game on the calibration side) and frozen.
- Constant circle: radius equal to the grouped conformal quantile of the error norms on all six dev games,
  same camera and reference, frozen.
- **Failure detector: gradient boosting** on the 11 per-frame calibration-health features of
  `predfoot.health.features` (depth 3, learning rate 0.05, 300 iterations, balanced class weights, early
  stopping off), trained on the per-frame calibrations of all six dev games and frozen as plain parameters.
  Baseline: the single feature with the best out-of-game AUROC on dev (by the dev numbers, `log_sigma_centre`,
  0.731), its sign fixed on dev, frozen with it. Logistic regression and every single feature are frozen too,
  for the secondary table.

## 4. Primary endpoints and success criteria

| # | Endpoint | Camera | Reference | Success criterion |
|---|---|---|---|---|
| P1 | Marginal coverage of the 90 % Mondrian regions on the test games, player-frames pooled | PTZ-smoothed | R1 | The 95 % CI contains 0.90 |
| P2 | Median region area / area of the constant circle at the same nominal level | PTZ-smoothed | R1 | Ratio below 1 |
| P3 | AUROC of the frozen gradient boosting minus AUROC of the frozen best single feature, on the per-frame calibrations of the test frames with self-consistent lines (failure: mean distance to the annotated lines above 10 px) | per-frame | annotated lines | Difference above 0 |

P1 and P2 use the headline rows of their camera: player-frames whose raw calibration succeeded and for which
that camera publishes a position. The two cameras' headline rows differ slightly. The smoothed camera abstains
on some raw-OK frames (the smoother rejects the frame or cannot check it): on dev, 2,788 player-frames that the
per-frame camera publishes (1.1 %, in 168 frames), with larger per-frame errors (p90 2.72 m against 2.02 m).
The per-frame camera publishes no position for feet that its homography cannot place on the pitch (above its
horizon): on dev, 501 player-frames that the smoothed camera publishes (0.2 %, in 39 frames). For each camera
and reference, the raw-OK player-frames that the other camera publishes and this one does not are counted and
reported next to its coverage, with that coverage also computed counting them as misses (player-frames pooled,
and clips weighted equally); for the primary camera and reference this is P1's coverage counting them as
misses. Frames that only the smoother fills (raw calibration failed) are reported apart. P2 and P3 are decided
on the point estimate; their intervals are reported.

P1 pools the headline player-frames, as every dev coverage quoted here does. The grouped conformal quantile
weighs each calibration clip equally, so the coverage with every clip weighted equally (the mean of the clips'
own coverage rates) is the quantity its construction targets; it is reported next to P1 for every camera and
reference (secondary analysis 1) but decides nothing. On dev the two agree against R1 (90.4 % pooled against
90.5 % with clips weighted equally, per-frame camera) and differ against R2 (94.4 % against 91.6 %), where a
clip's coverage rises with its number of trusted player-frames. Pooling is also the stricter test here:
small clips weigh less, and on dev its intervals are the narrower ones (against R1, per-frame camera, [88.8,
91.8] against [88.8, 92.0]).

## 5. Secondary analyses

All are reported; none is a success criterion.

1. P1 and P2 for the per-frame camera (comparison arm), and for both cameras against R2; for every camera and
   reference, the raw-OK player-frames that the other camera publishes and this one does not, with the
   coverage counting them as misses, and the coverage with every clip weighted equally, with its
   clip-bootstrap interval.
2. Coverage of the constant circle; coverage of the uncalibrated Laplace regions.
3. Coverage by distance tercile (metres per pixel at the feet, terciles of the test rows).
4. Simultaneous coverage: the share of frames in which every player is inside their region (the marginal
   regions, without a simultaneous calibration).
5. Coverage per test game, with clip-bootstrap intervals.
6. Frames filled by the smoother (raw calibration failed): coverage reported apart.
7. Position error p50 / p90 / p99 of both cameras on the same player-frames, against R1, R2 and the smoothed
   labels.
8. Failure detection: AUROC, average precision and the share of failures removed at 5 % and 10 % abstention
   for every frozen detector; clip-bootstrap interval of the AUROC difference of P3; AUROC per game; and the
   best single feature on the test frames (post hoc, a stricter comparison than P3).
9. Residual failures of the smoothed camera, descriptive: the share of frames with self-consistent lines
   whose camera is more than 10 px from the annotated lines, for both cameras on the same frames (raw and
   output cameras both calibrated, as on dev); frames filled by the smoother and raw-OK frames where it
   abstains counted apart; the share per output type (smoothed, repaired, filled, bridged, passthrough,
   duplicate) over every published frame of the smoothed camera; published-position rates.
10. **KPI proxy**: the model-free proxy of `predfoot.kpi.proxy` (5 fps, Savitzky-Golay window 5, order 2, runs
    of at least 8 frames with every error against R1 below 10 m), with the distance bias and the
    high-speed-running (≥ 20 km/h) distance bias against the smoothed labels, for both cameras from the same
    input player-frames. Each camera's runs break at its own errors above 10 m, so the two cameras' runs can
    differ slightly (their counts are reported). The raw labels' own bias against the smoothed labels is the
    noise floor of that reference.

## 6. Unit of analysis and intervals

- Unit: the clip. Frames are correlated within a clip, so the effective sample is closer to the number of
  clips (49) than to the number of player-frames.
- Intervals: 95 % percentile bootstrap over clips, 2,000 draws, one generator seeded with 0 for the whole
  evaluation, used in a fixed order. Each draw resamples the 49 test clips with replacement, across the three
  games: the number of clips per game varies between draws, and games are never resampled. A per-game interval
  resamples that game's clips. The dev intervals quoted here were drawn the same way, over the dev clips.

## 7. Procedure

1. The owner reviewed this text and settled every choice (2026-11-01).
2. `scripts/freeze_prereg.py` (clean working tree only) fits every frozen artifact on dev only (smoother
   noise, conformal models, detectors) into `artifacts/prereg/v1/`, writes `docs/preregistration.lock.json`
   and prints the line to quote in README.md. The lock records the SHA-256 of this file and of every
   artifact, the code commit, the configuration digests, the detection model version of the dev caches, the
   SHA-256 of the network weights, the interpreter and package versions (torch included), and the test clip
   list (count and digest of the (clip id, game) pairs of `index/clips.parquet`). The artifacts and the lock
   are committed together, in one commit that holds nothing else (the freeze commit).
3. `scripts/run_test_once.py` refuses to run unless the lock matches: the hashes of this file and of the
   artifacts; the lock anchored in its freeze commit (lock, preregistration and artifacts tracked, unchanged
   since that commit, which adds only them to the lock's code commit); no change under `src/`, `scripts/`,
   `configs/`, `pyproject.toml`, `uv.lock` or `.python-version` since the lock commit; a clean working tree
   (no tracked file hiding its changes, no ignored file under those paths); the interpreter and package
   versions of the freeze; and the pinned clip list. A re-freeze (lock and artifacts deleted, then frozen and
   committed again) stays possible until stage C has started; every re-freeze is visible in git history.
   Stage A (test frames at 5 fps, weights checked against the lock, keypoint networks into a cache of its own)
   and stage B (per-frame calibration, smoothing with the frozen noise parameters) read no label and resume
   after a crash. Their outputs are tied to the lock: a re-freeze starts them from nothing, and stage C
   refuses outputs of another lock. Stage C, the only stage that reads test labels, first checks and loads
   every label-free input and checks that both label tables exist with the columns it reads (their schemas,
   no label), asks the owner to confirm, then writes its markers before reading any label, and runs once: a
   second stage C is refused, even after a crash. A crash or an interruption after the markers (an exception, Ctrl-C, SIGTERM or SIGHUP)
   still writes `reports/test_run.md`, naming the failure, with every result computed before it; after a hard
   kill (SIGKILL) or a power loss only the markers remain, and the owner writes the failure into
   `reports/test_run.md` by hand.
4. Every result goes into `reports/test_run.md`, failed criteria included. Nothing is re-run or re-analysed
   after stage C; any deviation from this text is listed in the report.

## 8. Known limitations, stated in advance

- Three test games from one league and one broadcast style: no claim beyond that.
- Between-game spread: on dev, the smoothed camera's coverage per held-out game ranges from 84.0 % (game 6)
  to 95.6 % (game 3). With three test games, one atypical game can move the pooled coverage by several points,
  and P1 can fail through between-game variability alone.
- The clip bootstrap resamples the test clips across the three games, never the games themselves: its interval
  describes these three games, not a new one. Three games cannot estimate the variability between games, so
  the interval is narrower than the uncertainty about a new game.
- Both references carry their own errors (section 2): coverage against R1 includes the labels' error; R2
  covers about 60 % of the player-frames, penalises a smooth camera and fails grossly on about 1.5 % of its
  player-frames.
- The single evaluation is enforced by markers in one data folder (`PREDFOOT_DATA_DIR/test_once/`) and one
  checkout (`runs/test_once/`), and by `reports/test_run.md` once it is committed. A fresh clone with
  another data folder, before that report is committed, would not be refused: across clones the guarantee
  rests on committing the report (written even when stage C fails, by hand after a hard kill or a power loss)
  and on the owner.
- The KPI proxy's reference, the smoothed labels, carries track-smoothing artefacts; it measures the cost of
  jitter, not absolute KPI accuracy.
- For orientation only (dev, leave-one-game-out, not a prediction): coverage against R1 90.6 % [89.0, 92.1]
  for the smoothed camera and 90.4 % [88.8, 91.8] for the per-frame camera; median areas 5.4 and 6.2 m²
  against constant circles of 7.4 and 14.6 m²; failure-detection AUROC 0.89 for gradient boosting against
  0.73 for the best single feature; KPI proxy distance bias +2.7 % against +27.0 %, high-speed-running
  distance bias +6.2 % against +67.2 %.
