# Attribution

predfoot is Apache-2.0. It uses the following third-party data and models. Apart from the SkillCorner
excerpt and the SoccerTrack v2 still below, none of them is redistributed in this repository; each is
downloaded at runtime.

| Resource | Use | Licence | Credit |
|---|---|---|---|
| PnLCalib keypoint and line weights (`SV_kp`, `SV_lines`) | Pitch keypoint and line heatmaps | CC BY 4.0 | M. Gutiérrez-Pérez and A. Agudo, PnLCalib, https://github.com/mguti97/PnLCalib |
| HRNet model definition (`cls_hrnet.py`, as shipped with PnLCalib) | Network architecture | MIT | Microsoft HRNet authors |
| SoccerNet Game State Reconstruction (SN-GSR-2025) | Evaluation only; frames never shown publicly. Results appear as metrics, plots and minimaps: `assets/minimap_demo.gif` and `.png` draw predfoot's positions and the labelled positions of 15 s of dev clip SNGS-081 on a pitch diagram, no pixels | Labels tagged GPL-3.0; frames not redistributable | Somers et al., SoccerNet Game State Reconstruction, CVPRW 2024 |
| SoccerTrack v2 (atomscott/soccertrack-v2, release v1.2, https://huggingface.co/datasets/atomscott/soccertrack-v2) | Public demo pixels, out-of-domain tests; `assets/demo.jpg` is a virtual view of match 117093 (MOT clip, frame 25) with predfoot overlays | CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ | Scott et al., SoccerTrack v2 (arXiv:2508.01802); frames reprojected to a virtual camera, cropped, downscaled and overlaid |
| IDSSE / DFL tracking data | KPI error budget | CC BY 4.0 | Bassek, Rein, Weber and Memmert (2025), doi:10.6084/m9.figshare.28196177 |
| SkillCorner open data | Output schema reference and format tests; `tests/fixtures/skillcorner_opendata_excerpt.jsonl` holds three unmodified records (frames 0, 30 and 210) of match 1874553 | MIT, Copyright (c) SkillCorner | SkillCorner, https://github.com/SkillCorner/opendata |

predfoot is not affiliated with SkillCorner, SoccerNet, PnLCalib or the SoccerTrack authors.
