# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/) and
the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- SoccerTrack v2 demo lane (T5.D1): `soccertrack-v2` download source (raw, MOT boxes, MOT clip of one match),
  `predfoot.io.soccertrack` loaders with conversion to the internal frame, virtual broadcast cameras inside the
  panorama (`predfoot.demo.virtual_camera`), `scripts/render_demo.py` (side-by-side demo video, still, GIF)
  and `reports/soccertrack_v2.md`: the released calibration is metres off on the far half, so SoccerTrack v2
  is a demo-pixel source, not a metric benchmark.
- Project skeleton: uv, pre-commit, CI, licence check, package layout, coordinate conventions, data contracts,
  ADR-0001 (conventions) and ADR-0002 (licences and data).
- SkillCorner-format export: record writer and reader with the uncertainty extension fields, image corners
  clipped like SkillCorner's, minimal `match.json` that kloppy loads, 25 → 10 fps export by interpolation.
- Frame JSON Schema 1.0 and semantic validator; `predfoot validate` (also for genuine SkillCorner files).
- Contract tests against kloppy and a three-record excerpt of SkillCorner open data (MIT).
- Minimap with 90 % ellipses and greyed-out abstentions.
- `predfoot run`: stub tracer bullet processed in resumable blocks (byte-identical after a kill).
- CPU Docker image and a workflow that runs the tracer bullet in it.
- Data access: SN-GSR labels and frames over HTTP range requests, SHA-256 manifest, Parquet index (clips,
  frames, persons, lines) in the internal frame, data audit; the test split is kept apart.
- Geometry: pitch template with PnLCalib keypoint/line ids and SoccerNet line classes, normalised DLT,
  horizon-aware back-projection, analytic Jacobians, pinhole broadcast cameras for simulation.
- Calibration: MSAC RANSAC, soft-L1 Levenberg-Marquardt on keypoints and points on lines, Laplace covariance
  with Huber's scale correction and a shrunk per-frame noise estimate.
- PnLCalib keypoint and line networks behind our own preprocessing and decoding; per-clip detection cache.
- Uncertainty: delta-method position covariances, F/χ² region quantiles, grouped split conformal with
  Mondrian bins and a simultaneous per-frame variant.
- Health: per-frame calibration features, selective-prediction metrics, failure-detection evaluation.
- Physical KPIs on IDSSE (distance, speed bands, HSR/sprint efforts, PSV-99, explosive accelerations),
  error injection and a sensitivity study; measured error-process estimation on SN-GSR.
- Real pipeline stages for `predfoot run` (cached detections → calibration → calibrated ellipses).
- Scripts: spike S0, dev evaluation with leave-one-game-out coverage, failure detection, error process.
- Preregistration draft for the locked test run.
- Temporal camera smoothing (PTZ-RTS): fixed-centre pan-tilt-zoom camera model, label-free projection centre
  per clip (game centre when the clip's own is unreliable, refused when it does not explain the clip), Kalman
  filter and RTS smoother inside shot segments; outlier deletion under one process-noise model (shrink, then
  release from a clean core), manoeuvre segments and single-step manoeuvres for camera moves faster than the
  fitted noise, deleted frames without a smoothed state abstain; noise fitted without labels per
  leave-one-game-out fold; `eval_dev.py --smooth none|proj|ptz` (`--calib-from`, `--fit-scope`, `--beta`)
  with Mondrian bins per covariance source, `error_process.py --compare` with a model-free KPI proxy,
  optional smoothing in the real stages (with an optional game centre).
