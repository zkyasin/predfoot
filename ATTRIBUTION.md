# Attribution

predfoot is Apache-2.0. It uses the following third-party data, models and libraries. Apart from the HRNet
model definition (a modified copy in `src/predfoot/keypoints/hrnet_mit.py`), the SkillCorner excerpt, the
SoccerTrack v2 images and clips and the showcase page's three JavaScript libraries listed below, none of them
is redistributed in this repository; each is downloaded at runtime. The SN-GSR files listed below are pitch
diagrams drawn from its labels, with no pixels.

| Resource | Use | Licence | Credit |
|---|---|---|---|
| PnLCalib keypoint and line weights (`SV_kp`, `SV_lines`) | Pitch keypoint and line heatmaps | CC BY 4.0 | M. Gutiérrez-Pérez and A. Agudo, PnLCalib, https://github.com/mguti97/PnLCalib |
| HRNet model definition (`cls_hrnet.py`, as shipped with PnLCalib) | Network architecture; a modified copy is in `src/predfoot/keypoints/hrnet_mit.py` | MIT, Copyright (c) 2019 Microsoft Corporation | Microsoft HRNet authors, https://github.com/HRNet/HRNet-Image-Classification; licence text in the file's header |
| SoccerNet Game State Reconstruction (SN-GSR-2025) | Evaluation only; frames never shown publicly. Results appear as metrics, plots and minimaps: `assets/minimap_demo.gif` and `.png` draw predfoot's positions and the labelled positions of 15 s of dev clip SNGS-081 on a pitch diagram, no pixels; `assets/showcase/smoothing.mp4`, `smoothing-tall.mp4`, their `.jpg` posters and `smoothing.gif` draw the same 15 s for the showcase page and the README | Labels tagged GPL-3.0; frames not redistributable | Somers et al., SoccerNet Game State Reconstruction, CVPRW 2024 |
| SoccerTrack v2 (atomscott/soccertrack-v2, release v1.2, https://huggingface.co/datasets/atomscott/soccertrack-v2) | Public demo pixels, out-of-domain tests; `assets/demo.jpg` is a virtual view of match 117093 (MOT clip, frame 25) with predfoot overlays; `assets/showcase/hero.mp4` and `hero.jpg` are virtual views of the same clip, frames 355-524 | CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ | Scott et al., SoccerTrack v2 (arXiv:2508.01802); frames reprojected to a virtual camera, cropped, downscaled and overlaid |
| D3 7.9.0 (`assets/showcase/vendor/d3.min.js`) | Figures of the showcase page | ISC | Mike Bostock, https://d3js.org; licence text in `assets/showcase/vendor/LICENSES.txt` |
| Motion 12.43.0 (`assets/showcase/vendor/motion.js`) | Animations of the showcase page | MIT | Motion B.V., https://motion.dev; licence text in `assets/showcase/vendor/LICENSES.txt` |
| Lenis 1.3.26 (`assets/showcase/vendor/lenis.min.js`) | Smooth scrolling of the showcase page | MIT | darkroom.engineering, https://github.com/darkroomengineering/lenis; licence text in `assets/showcase/vendor/LICENSES.txt` |
| IDSSE / DFL tracking data | KPI error budget | CC BY 4.0 | Bassek, Rein, Weber and Memmert (2025), doi:10.6084/m9.figshare.28196177 |
| SkillCorner open data | Output schema reference and format tests; `tests/fixtures/skillcorner_opendata_excerpt.jsonl` holds three unmodified records (frames 0, 30 and 210) of match 1874553 | MIT, Copyright (c) SkillCorner | SkillCorner, https://github.com/SkillCorner/opendata |

predfoot is not affiliated with SkillCorner, SoccerNet, PnLCalib or the SoccerTrack authors.
