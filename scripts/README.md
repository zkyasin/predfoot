# Scripts

Command-line entry points, run with `uv run python scripts/<name>.py --help`. Each report in `reports/` names
the script, the commit and the data behind it.

Data:

- `download_data.py`: download the datasets and weights into `PREDFOOT_DATA_DIR`.
- `verify_data.py`: check every downloaded file against `MANIFEST.sha256`.
- `build_index.py`: build the SN-GSR Parquet index (clips, persons, lines) from the labels.
- `data_audit.py`: audit that index → `reports/data_audit.md`.
- `prepare_idsse.py`: download the IDSSE tracking data and parse each match into Parquet.
- `cache_keypoints.py`: cache the keypoint and line detections of every downloaded clip;
  `modal_cache_keypoints.py` does the same on Modal GPUs.

Development (SN-GSR dev games):

- `spike_s0.py`: first calibration on real broadcast frames → `reports/spike_s0.md`.
- `eval_dev.py`: positions, uncertainty and leave-one-game-out coverage, with or without camera smoothing →
  `reports/m2_logo_dev.md`, `reports/m2_temporal_dev.md`.
- `eval_health.py`: failure detection from calibration features → `reports/m3_health_dev.md`.
- `make_figures.py`: development figures → `reports/dev_figures.md`.
- `run_real_clip.py`: `predfoot run` with the real stages on one cached dev clip.

Preregistered test:

- `freeze_prereg.py`: fit the frozen artifacts on dev only and write the lock.
- `run_test_once.py`: the single evaluation on the locked test games → `reports/test_run.md`.

Physical stats (KPI) error budget:

- `error_process.py`: the position-error process and the model-free KPI proxy →
  `reports/error_process_dev.md`, `reports/error_process_temporal_dev.md`.
- `kpi_sensitivity.py`: first sensitivity study on IDSSE with simulated errors → `reports/kpi_sensitivity.md`,
  `reports/kpi_budget_measured.md`.
- `kpi_spectral.py`: spectral error model and KPI budget → `reports/kpi_budget_spectral.md`.
- `kpi_card.py`: per-player KPI uncertainty cards → `reports/kpi_card.md`.

Demo, visuals and showcase page:

- `render_demo.py`: virtual broadcast demo on SoccerTrack v2 pixels (`assets/demo.jpg`).
- `soccertrack_feasibility.py`: SoccerTrack v2 as a benchmark → `reports/soccertrack_v2.md`.
- `fit_demo_calibration.py`: rebuild the demo's uncertainty calibration from two dev runs.
- `render_minimap_demo.py`: side-by-side minimap animation of one dev clip → `reports/minimap_demo.md`.
- `render_hero.py`: the showcase page's hero loop → `reports/showcase_visuals.md`.
- `showcase_percentiles.py`: the error percentiles drawn by the page's accuracy chart.
- `build_site.py`, `deploy_space.py`: build the page into `site/dist/` and upload it to the Hugging Face Space.

Serving and engineering:

- `export_onnx.py`: export the keypoint networks to ONNX and check parity with PyTorch.
- `bench_latency.py`: network latency on this machine → `reports/latency_m4.md`.
- `fit_conformal.py`: fit the conformal model that `predfoot serve --conformal` loads.
- `make_golden.py`, `make_synthetic_clip.py`: fixtures of the golden test and of the tracer bullet.
- `check_licenses.py`: fail on a dependency licence the project cannot ship with.
