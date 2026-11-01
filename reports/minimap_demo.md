# Minimap demo: per-frame against PTZ-smoothed camera on one dev clip

`uv run python scripts/render_minimap_demo.py --none-run runs/20261031-152537_c3861c7_none --ptz-run runs/20261031-163641_bf28b35_ptz` · commit `e9e9524` · writes `minimap_demo.gif` (705 kB, 75 frames at 5 fps) and `minimap_demo.png` (86 kB).

Exploratory SN-GSR dev data, ground-truth boxes. Each player's 90 % region uses the Mondrian quantile of its bin fitted on the five games other than the clip's (leave one game out). Only pitch coordinates are drawn (ADR-0002).

**Clip SNGS-081 (game 5).** Chosen by rule, to show the smoothing: among the third of the 115 dev clips whose smoothed camera moves fastest, the one where smoothing cuts the frame-to-frame jitter of the common-mode error most (robust step RMS against the smoothed labels, errors of 10 m or more left out). It is not a typical clip:

| Quantity | SNGS-081 | Median over the dev clips |
|---|---|---|
| Image speed of the smoothed camera (px/s, median) | 228 | 121 |
| Common-mode step RMS, per-frame camera (m) | 0.52 | 0.39 |
| Common-mode step RMS, smoothed camera (m) | 0.11 | 0.13 |
| Step RMS ratio, smoothed / per-frame | 0.21 | 0.35 |
| Line reprojection p90, per-frame camera (px) | 9.6 | 12.1 |
| Line reprojection p90, smoothed camera (px) | 5.1 | 5.6 |
| Frames above 10 px, per-frame camera | 8.3% | 14.7% |
| Frames above 10 px, smoothed camera | 0.7% | 0.7% |

**Window 6.6-21.4 s**: the 15 s where the per-frame camera's common-mode error jitters most.

| Camera | Common-mode step RMS (m) | Labelled positions inside their 90 % region | Frames without positions (grey) | Frames not evaluated (hatched) |
|---|---|---|---|---|
| per-frame camera | 0.73 | 96.4% of 990 | 5 of 75 | 2 of 75 |
| PTZ-smoothed camera | 0.12 | 93.1% of 1006 | 4 of 75 | 2 of 75 |

Not evaluated: the camera published positions, but SN-GSR has no usable position label on the frame, so eval_dev.py wrote no player row and nothing checks them. Line reprojection of those frames (annotated lines to the projected pitch): per-frame camera t = 15.2 s, 534.9 px; t = 19.0 s, 4.8 px. PTZ-smoothed camera t = 15.2 s, 534.9 px; t = 19.0 s, 2.8 px.

**Still** (t = 7.4 s, the frame of the window where the per-frame camera's median error is largest): per-frame camera 18 / 18 inside, median area 143.2 m²; PTZ-smoothed camera 18 / 18 inside, median area 4.1 m².
