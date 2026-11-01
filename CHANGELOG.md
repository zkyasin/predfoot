# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/) and
the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- SoccerTrack v2 demo lane (T5.D1): `soccertrack-v2` download source (raw, MOT boxes, MOT clip of one match),
  `predfoot.io.soccertrack` loaders with conversion to the internal frame, virtual broadcast cameras inside the
  panorama (`predfoot.demo.virtual_camera`), `scripts/render_demo.py` (side-by-side demo video, still, GIF)
  and `reports/soccertrack_v2.md`: the released calibration is metres off on the far half, so SoccerTrack v2
  is a demo-pixel source, not a metric benchmark. The demo's noise parameters and conformal quantiles are
  versioned in `src/predfoot/demo/dev_calibration.json` (`scripts/fit_demo_calibration.py`), so it runs
  without SN-GSR data; `--no-smooth` uses the per-frame quantiles.
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
- Preregistration draft for the locked test run (now final, below).
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
- QA report: `predfoot run` writes `qa_report.html` next to `tracking.jsonl`, and `predfoot qa OUT` rebuilds it
  from the outputs of a run. One self-contained file (inline CSS and SVG, no script, no external reference)
  with the run summary, the validation of the outputs and consistency checks, a timeline of calibration
  health, abstentions and p90 radii, the distribution of the radii, the worst frames and field definitions.
- Docker: a `serve` target next to the default `cli` one (API extra, port 8000, `HEALTHCHECK` on `/healthz`,
  numeric non-root user); the `docker` workflow builds and smoke-tests both; a `release` workflow pushes both
  images to `ghcr.io/zkyasin/predfoot` on `v*` tags, with version and commit tags and OCI labels.
- Golden test of the real stages on committed synthetic detections (`scripts/make_golden.py --update`
  regenerates the expected outputs), detection files as JSON (`predfoot-detections/1`) and a generator of
  network-style detections from a known camera.
- `predfoot run --detections FILE [--conformal MODEL]` runs the real stages (calibration from the detections,
  delta-method regions, Mondrian conformal scaling) without torch; `python -m predfoot.io.compare` compares a
  run with golden outputs, one written decimal of slack on numbers. The Docker and release workflows run the
  golden check inside the images.
- `docker/smoke-cli.sh` and `docker/smoke-serve.sh`: the smoke tests of the images, shared by the `docker` and
  `release` workflows and runnable locally.
- `scripts/run_real_clip.py` writes its inputs next to its outputs (`input_clip.json`, `detections.json`
  re-keyed by clip frame index, `conformal.json`), so `predfoot run --detections` reruns a dev clip from files.
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
  check against SN-GSR (distance bias +13.9 % ball in play against +2.7 % measured, HSR distance +31.4 %
  against +6.2 %; the per-frame camera is within a factor of two, but above too). Fitted Gaussian processes
  reproduce the measured step RMS, yet the measured median 0.2 s step is only 0.59 to 0.75 of the chosen
  model's: real steps are intermittent, which no stationary model fitted to σ, ACF and marginal captures, so
  the 10 fps budget (PTZ distance +18 %, HI distance +56 %) overstates predfoot's KPI error. On IDSSE with
  simulated errors (PTZ camera, 4 matches), the card's nested-bootstrap intervals cover effort counts and
  PSV-99 at their nominal 90 % (90 to 91 %, against 57 to 69 % with the spread of one dose); distance KPIs
  stay below (32 to 51 %) because of the remaining extrapolation bias, and only leave-one-match-out
  calibrated intervals reach about 90 % for every KPI.
- Portfolio README with the development results of both cameras, coverage with player-frames pooled and with
  each clip weighted equally, and what the evidence does not show. `scripts/render_minimap_demo.py` animates
  one dev clip, per-frame against PTZ-smoothed camera (`assets/minimap_demo.gif`, abstentions greyed, frames
  without position labels hatched); `scripts/make_figures.py` draws the dev reliability (coverage minus the
  nominal level), region area, coverage by distance and by clip size, and the KPI proxy
  (`reports/dev_figures.md`, `reports/minimap_demo.md`). An index of the reports, serving and Docker details
  in `docs/serving.md` and `docs/docker.md`, and ARCHITECTURE.md brought up to date. `eval_dev.logo` takes the
  level α and returns each held-out row's Mondrian quantile.
- Preregistered test run (T6.1). `docs/preregistration.md`, version 1, final: SN-GSR raw labels as the
  primary reference, the PTZ-smoothed camera under test with the per-frame camera as the comparison arm,
  coverage (player-frames pooled, clips weighted equally reported next to it), region size against a constant
  circle, gradient boosting against the best dev single feature, KPI proxy biases as a secondary endpoint.
  `scripts/freeze_prereg.py` fits every frozen artifact on the six dev games (conformal regions per camera
  and reference, constant circles, smoother noise, failure detectors exported as JSON and checked against
  scikit-learn) and writes `docs/preregistration.lock.json` (SHA-256 of the text and the artifacts, code
  commit, configuration digests, interpreter and packages, network weights, detection model version, pinned
  test clip list). `scripts/run_test_once.py` runs stage A (test frames, keypoint networks) and stage B
  (per-frame calibration, smoothing with the frozen noise) without labels and resumably, then stage C once
  behind markers. Every stage refuses what does not match the lock: an edited text or artifact, a change of
  code or dependencies since the freeze, a dirty tree, another environment or clip list, outputs of another
  lock. Stage C checks every label-free input and asks before its claim, saves the primary results as soon as
  they are computed, and writes `reports/test_run.md` even after a crash or an interruption. `--dry-run`
  rehearses the whole chain on dev clips with a scratch freeze.

### Changed
- The dev evaluation core moved from `scripts/eval_dev.py` into `predfoot.eval.positions`, the failure
  detectors into `predfoot.health.detector` and the players-table KPI proxy into `predfoot.kpi.proxy`; the
  scripts give identical results. The gradient boosting of the failure detector runs with early stopping
  off: scikit-learn switches it on above 10,000 rows, which the six-game fit of the freeze reaches and no dev
  fold does.

### Fixed
- Temporal smoothing tails: a run of misfit frames lasting at most 0.5 s strictly inside a segment gets the
  smoothed state (flag `bridged`) instead of its raw camera; raw-OK frames of a segment too short for any
  innovation to test them abstain (flag `unchecked`) instead of publishing their PTZ projection; the
  process-noise factor across failure gaps is fitted without labels (`NoiseParams.gap`), apart from the fixed
  manoeuvre factor. `eval_dev.py` calibrates filled and bridged frames in their own covariance source with one
  pooled quantile, counts PTZ-projected frames as single-frame calibrations, gives a source its own quantiles
  only with at least 10 calibration clips, and reports robust common-mode step statistics.
- The export rate entered the config hash as given, so `RunConfig()` (10) and `predfoot run` (10.0) hashed the
  same configuration differently, and a run resumed through the other path recomputed every block.
- `meta.json` described every stage as a stub and the conformal model as null even when the real stages ran
  with a Mondrian model; stages now describe themselves, and the real ones record their model (hash, alpha,
  pooled quantile, edges, per-bin quantiles).
- The release could push the `serve` image without ever starting its server: its smoke test only ran CLI
  commands, and nothing tied a tag to a green `docker` run. The release now runs the same scripts as the
  `docker` workflow, the HTTP checks included, before it logs in to the registry.
- The QA timeline shaded a whole column as soon as one of its frames abstained: on a full match (90 frames per
  column), 1 % of scattered abstentions shaded 90 % of the chart. The shading is now as high as the share of
  abstained frames, and columns are as wide as the frames they hold.
- `predfoot run --detections` ran detections keyed by image number (as the detection cache keys them) with
  the cameras of other frames and no warning. Detections that name frames beyond the clip are refused before
  anything runs, a file without frame 0 gets a warning, and the 0-based key is documented.
- The `serve` image's `HEALTHCHECK` went through the proxy variables the Docker CLI can pass to containers, and
  a command that only added options to `serve` listened on 127.0.0.1 alone. The probe now ignores proxies, and
  `predfoot serve` takes its host and port from `PREDFOOT_HOST` and `PREDFOOT_PORT`, set in the image.
- A conformal model with an infinite bin edge passed the start-up checks, then made `meta.json` unwritable
  after the whole computation; it is now refused when it is loaded.
- The non-root checks of the Docker workflows passed when `docker run` itself failed, and the `docker` workflow
  did not run when `README.md`, `LICENSE` or `ATTRIBUTION.md`, which the images copy, changed.
- `predfoot qa` never returned on an absurd `export_fps` in `meta.json` (it now falls back to 10 fps outside
  1e-3 to 1e3) and crashed on a line nested thousands deep, which the validator now reports as invalid JSON.
- ATTRIBUTION.md credited SoccerNet Calibration 2023 for a health critic that does not exist; nothing uses
  that dataset.
