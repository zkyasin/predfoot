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
- Spectral KPI error budget (T4.3) and KPI uncertainty card (T4.4): `predfoot.kpi.noise` (nugget + one or two
  scales of an exponential, Matérn-3/2, Matérn-5/2 or squared-exponential kernel, Gaussian or Gaussian-copula
  marginal, exact circulant-embedding sampling, weighted least-squares fit to the 5 fps ACF);
  `error_process.py --json` (per-axis σ, ACF and normal-score ACF pooled over series through the
  semi-variogram, so that a fitted process reproduces the measured step variances; step RMS and median,
  marginal quantiles and kurtosis, run lengths); `kpi_spectral.py` (5 fps step round trip, simulation of the
  SN-GSR KPI proxy on IDSSE with an explicit verdict, 10 fps budget with kernel × nugget sensitivity) and
  `predfoot.kpi.card` with `kpi_card.py` (linear and quadratic SIMEX bias correction, estimates projected onto
  the KPI's support, variance from a nested parametric bootstrap, nominal intervals with their coverage on
  IDSSE and a leave-one-match-out calibration). Finding: the PTZ camera's simulated 5 fps proxy fails its
  check against SN-GSR (distance bias +15 % ball in play against +3.9 % measured, HSR distance +34 % against
  +9.5 %; the per-frame camera is within a factor of two, but above too). Fitted Gaussian processes reproduce
  the measured step RMS, yet the measured median 0.2 s step is only 0.59 to 0.72 of the chosen model's: real
  steps are intermittent, which no stationary model fitted to σ, ACF and marginal captures, so the 10 fps
  budget (PTZ distance +19 %, HI distance +60 %) overstates predfoot's KPI error. On IDSSE with simulated
  errors (PTZ camera, 4 matches), the card's nested-bootstrap intervals cover effort counts and PSV-99 at
  about their nominal 90 % (86 to 94 %, against 55 to 70 % with the spread of one dose); distance KPIs stay
  below (21 to 55 %) because of the remaining extrapolation bias.
