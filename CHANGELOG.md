# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/) and
the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
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
- HTTP API, `predfoot serve` (extra `serve`): `/healthz`; `/v1/localize`, pitch positions with calibrated
  covariances and 90 % radii from a camera and its 8 × 8 Σθ, nulls above the horizon, with the Mondrian
  conformal regions of the pipeline when the server loads a model (`--conformal`, fitted by
  `scripts/fit_conformal.py`); `/v1/calibrate`, one uploaded frame → PnLCalib keypoints and lines →
  calibration and image corners, with size and content checks before decoding, a bounded queue on worker
  threads of its own, and a 503 when the networks cannot load or the queue is full; `/metrics` for
  Prometheus.
- ONNX export of both keypoint networks, installed only once it passes a parity check against PyTorch
  (`scripts/export_onnx.py`, extra `export`; heatmaps within 1e-4, the same detections, decoded keypoints
  within 0.5 px) and a latency benchmark on PyTorch CPU, MPS and CUDA and ONNX Runtime CPU, CoreML and CUDA
  (`scripts/bench_latency.py`, `reports/latency_m4.md`).

### Fixed
- Temporal smoothing tails: a run of misfit frames lasting at most 0.5 s strictly inside a segment gets the
  smoothed state (flag `bridged`) instead of its raw camera; raw-OK frames of a segment too short for any
  innovation to test them abstain (flag `unchecked`) instead of publishing their PTZ projection; the
  process-noise factor across failure gaps is fitted without labels (`NoiseParams.gap`), apart from the fixed
  manoeuvre factor. `eval_dev.py` calibrates filled and bridged frames in their own covariance source with one
  pooled quantile, counts PTZ-projected frames as single-frame calibrations, gives a source its own quantiles
  only with at least 10 calibration clips, and reports robust common-mode step statistics.
