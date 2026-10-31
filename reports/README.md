# Reports

Generated markdown reports with aggregate numbers only (no frames, no per-frame labels). Each report states the
command, the git commit and the data it was produced from. All SN-GSR numbers are development results on the
six dev games; the test games stay locked until the preregistered run (`docs/preregistration.md`).

Calibration and position uncertainty (SN-GSR dev):

- `spike_s0.md`: first calibration on real broadcast frames, and why two position references are needed.
- `m2_logo_dev.md`: per-frame camera: position errors, leave-one-game-out coverage of the 90 % regions.
- `m2_temporal_dev.md`: the same with the PTZ-smoothed camera, the smoother's diagnostics and the ablations.
- `dev_figures.md`: coverage at the 50, 80, 90 and 95 % levels, pooled over player-frames and with every
  clip weighted equally, region size against a constant circle, coverage by distance and by clip size, and
  the KPI proxy, with the figures in `figures/`.
- `minimap_demo.md`: how the clip and window of `assets/minimap_demo.gif` were chosen, and which frames it
  leaves unevaluated (no labels).
- `m3_health_dev.md`: failure detection from calibration-health features.

Error process and KPI budget:

- `error_process_dev.md`, `error_process_temporal_dev.md` (+ `.json`): the position-error process and the
  model-free KPI proxy measured on SN-GSR dev.
- `kpi_sensitivity.md`, `kpi_budget_measured.md`: first KPI sensitivity studies on IDSSE with simulated errors
  (superseded by the spectral budget).
- `kpi_budget_spectral.md`: KPI budget on IDSSE with a spectral error model; a simulation, whose 5 fps check
  against the measured proxy fails for the smoothed camera.
- `kpi_card.md`: per-player KPI uncertainty cards, a consistency check on simulated errors.

Data and engineering:

- `data_audit.md`: SN-GSR structure and label checks.
- `soccertrack_v2.md`: SoccerTrack v2 files, virtual broadcast views and their feasibility as a benchmark.
- `latency_m4.md`: keypoint network latency on an Apple M4 Pro.
