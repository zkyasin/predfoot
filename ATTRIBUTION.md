# Attribution

predfoot is Apache-2.0. It uses the following third-party data and models. Apart from the SkillCorner
excerpt below, none of them is redistributed in this repository; each is downloaded at runtime.

| Resource | Use | Licence | Credit |
|---|---|---|---|
| PnLCalib keypoint and line weights (`SV_kp`, `SV_lines`) | Pitch keypoint and line heatmaps | CC BY 4.0 | M. Gutiérrez-Pérez and A. Agudo, PnLCalib, https://github.com/mguti97/PnLCalib |
| HRNet model definition (`cls_hrnet.py`, as shipped with PnLCalib) | Network architecture | MIT | Microsoft HRNet authors |
| SoccerNet Game State Reconstruction (SN-GSR-2025) | Evaluation only, never shown publicly | Labels tagged GPL-3.0; frames not redistributable | Somers et al., SoccerNet Game State Reconstruction, CVPRW 2024 |
| SoccerNet Calibration 2023 | Evaluation and training of the health critic | SoccerNet terms | Cioppa et al., SoccerNet |
| SoccerTrack v2 | Public demo pixels, out-of-domain tests | CC BY 4.0 | SoccerTrack v2 authors (arXiv:2508.01802); frames cropped, downscaled and overlaid |
| IDSSE / DFL tracking data | KPI error budget | CC BY 4.0 | Bassek, Rein, Weber and Memmert (2025), doi:10.6084/m9.figshare.28196177 |
| SkillCorner open data | Output schema reference and format tests; `tests/fixtures/skillcorner_opendata_excerpt.jsonl` holds three unmodified records (frames 0, 30 and 210) of match 1874553 | MIT, Copyright (c) SkillCorner | SkillCorner, https://github.com/SkillCorner/opendata |

predfoot is not affiliated with SkillCorner, SoccerNet, PnLCalib or the SoccerTrack authors.
