# SoccerTrack v2: demo pixels and feasibility as a quasi-ground-truth benchmark (T5.D1)

`uv run python scripts/soccertrack_feasibility.py --match <m>` and
`uv run python scripts/render_demo.py --match 117093 --start 0 --frames 10 --step 5` · commit `638c7ea` ·
data: SoccerTrack v2 release v1.2 (CC BY 4.0), `raw/`, `mot/<m>.txt` and `mot/clips/<m>.mp4` of matches
117092 (M1, night match) and 117093 (M2, day match). Outputs in `runs/` (not in git).

**Verdict.** The demo works: virtual broadcast views of the panorama are rendered with a known camera and
SoccerTrack pixels can be published. As a metric benchmark the dataset is **not** quasi-exact. Its released
calibration misses its own 65 annotated keypoints by 23 px (M1) and 8 px (M2) median in the video. On the far
half of the pitch one video pixel is about 0.2 m in depth, so that is 2.5 to 3.3 m median on the pitch. A
virtual camera is exact for the released *model*, not for the real pitch. Usable references are the annotated
keypoints mapped into each view (true correspondences up to annotation noise) and, at best, the near half of
the pitch. This is risk R8 of the plan: the benchmark idea is dropped and SoccerTrack stays a demo-pixel source.

## What the files are

| Item | Finding (117092 / 117093) |
|---|---|
| MOT clip | 3840 × 1906 / 4096 × 1080, H.264, 25 fps, 6000 / 5976 frames. The same pixel space as the half-match panoramas (`CHANGES_v1_1.md`). |
| Video frame | **Not** a single fisheye image: a stitch of two wide-angle cameras (on 117092 a seam is visible at mid-width on the top border), with strong barrel curvature. Night match under floodlights (M1); daylight (M2). |
| `mapx.npy`, `mapy.npy` | OpenCV fixed-point remap tables (`int16` (h, w, 2) integer coordinates + `uint16` 32 × 32 interpolation index) from **rectified** pixels to **video** pixels, same size as the video. Decoded with `cv2.convertMaps`, they equal `cv2.fisheye.initUndistortRectifyMap(K, D, I, Knew, size)` to 0.022 px (the 1/32 px grid). 71.5 % / 60.9 % of the rectified pixels fall inside the video. |
| `camera_intrinsics.npz` | Fisheye `K` (anisotropic: fx 1365 / 1591, fy 946 / 1667 px), `D` = (k1, k2, 0, 0), rectified intrinsics `Knew` (fx 241 / 496 px), and one pose `rvecs`, `tvecs` fitted on the 65 keypoints (provider RMS 30.6 / 8.6 px). `rvecs` and `tvecs` are pickled object arrays: predfoot reads them with an unpickler that only accepts numbers. |
| Camera centre (internal frame) | (0.4, −40.5, 13.5) m / (−1.4, −50.3, 7.1) m: at the halfway line, 6.5 m / 16.3 m behind the near touchline, 13.5 m / 7.1 m high. |
| `padding_info.csv` | Time, not space: where the match clock starts in the half-match video (ms). Not needed for the MOT clips. |
| `homography.npy` | Corner-origin pitch metres → **rectified** pixels (SoccerTrack y towards the near touchline). Against the rectified keypoints: median 23.5 / 2.8 px, p90 179 / 18.7 px, only 16 / 33 of 65 points under 3 px (a RANSAC fit that keeps a subset). On the pitch: 4.7 / 1.8 m median. |
| `keypoints.json` | 65 points in **video** pixels: 21 per touchline every 5.5 m (36 of them are not marked on the pitch and were placed by eye along the line), plus goal lines, boxes, centre line and centre spot (29 landmarks). |
| MOT boxes | Video pixels, top-left `x, y, w, h`, 0-based frames that match the clip's decoded frames; 22 boxes per frame (players and goalkeepers), median box height 41 / 46 px. Foot point = bottom centre (`x + w/2`, `y + h`). 99.6 % of the foot points land within 3 m of the pitch under the released model. |

Pitch frame: the raw files use a corner origin with y towards the near touchline;
`predfoot.io.soccertrack` converts to the internal frame, (x − 52.5, 34 − y), with z up (the corner frame
completed by z = x × y points down), through `predfoot.io.conventions`.

## Is the rectified panorama a pinhole image?

Not to the precision a metric benchmark needs. Residuals at the 65 keypoints (video pixels unless stated):

| Model | 117092 median / p90 | 117093 median / p90 |
|---|---|---|
| Released fisheye model (K, D, pose) | 23.3 / 47.5 px | 8.2 / 11.7 px |
| Same, back-projected to the pitch | 3.31 / 8.02 m | 2.47 / 5.02 m |
| ... far touchline / near touchline points (median) | 5.67 / 0.80 m | 3.67 / 0.63 m |
| ... the 29 marked landmarks only (median) | 4.34 m | 3.32 m |
| One homography on all 65 rectified points, leave-one-out | 26.1 / 51.2 px | 8.5 / 14.3 px |
| Local homographies on thirds of the pitch, leave-one-out (median per third) | 22.4, 24.1, 15.4 px | 5.4, 3.9, 5.3 px |
| One global homography on the same points (median per third) | 21.4, 32.1, 22.0 px | 9.4, 7.5, 8.6 px |
| Released model + smooth residual field (thin-plate spline), leave-one-out | 8.7 / 30.6 px | 2.5 / 6.5 px |

Metres per video pixel (√|det J|) under the released model: 0.014 / 0.017 at the near touchline, 0.086 / 0.090
at the centre spot, 0.19 / 0.20 at the far touchline. On M1 the released model puts the centre spot 47.5 px
below its annotation, about 8 m in depth.

So:

- On M2 the rectified panorama is pinhole-like only **locally**: local homographies halve the global
  residual (≈ 4-5 px against ≈ 8-9 px), and a smooth correction field reaches 2.5 px median. On M1 (night,
  poorer annotation and model) nothing gets below about 9 px.
- Rectified pixels are a poor unit: `Knew` has fx 241 px against 1365 px for the video at its centre, so near
  the middle one rectified pixel covers about 5.7 video pixels. That is why the virtual views are rendered from
  the video frame, not from the rectified image.
- Virtual pinhole views are exact homographies of the **released model** of the pitch, by construction (unit
  tests: round trips to 0.05 px, a rendered pitch point lands within 0.35 px of its projection). They are not
  exact homographies of the real pitch: the real lines in a view sit where the video puts them, off the
  reference camera's lines by the model error multiplied by the zoom. Even a perfect 2D model would leave the
  far half at about 0.2 m of depth per video pixel, so 2-3 px of annotation noise is already 0.4-0.6 m there.

## Virtual broadcast camera

`predfoot.demo.virtual_camera`: a roll-free PTZ camera (square pixels, principal point at the centre of a
1920 × 1080 frame) at the released projection centre C, so H_virt = K_virt R_virt M(C) =
(K_virt R_virt Rᵀ Knew⁻¹) H_pano. One `cv2.remap` renders it from the video frame. The remap table is built
in two steps: virtual pixel → rectified pixel (rotation homography), then rectified pixel → video pixel (the
dataset tables, sampled bilinearly). The camera aims at the players' centroid, from the MOT foot points under
the released model, 6 m nearer the camera, smoothed over 1.5 s. It zooms so that the 10-90 % horizontal spread
of the players fills 75 % of the half-width, with a horizontal field of view kept between 25° and 45°. MOT foot
points and annotated keypoints are mapped into each view exactly (the remap is a known bijection).

## predfoot on virtual views: smoke frames

Ten frames (0, 5, …, 45 of each clip, 0.2 s apart), network inference on each. PnLCalib keypoints and lines →
robust solver → PTZ smoother (noise parameters reused from the dev run, the median of its six label-free fold
fits, κ² ≈ 1.94, not refitted) → delta-method positions of the MOT foot points. The 90 % regions use Mondrian
conformal quantiles fitted on the SN-GSR dev players table (`runs/20261031-153059_c3861c7_ptz/players.parquet`,
reference: labels; bins: metres per pixel at the feet (3) × RANSAC inliers (2), per covariance source, as in
`scripts/eval_dev.py`). **This transfers a calibration from SN-GSR broadcast frames to SoccerTrack virtual
views, a domain shift**: there is no SoccerTrack data to recalibrate on.

- **117092 (night):** PnLCalib finds almost nothing (keypoint heatmap maximum 0.07 against a 0.34
  threshold; 0-3 keypoints even after a gamma or CLAHE boost; no line ends). All 10 frames abstain. The
  released model misses the annotated keypoints in these views by 46 px (6.3 m) median.
- **117093 (day):** all 10 frames calibrate, but with only 7-11 inliers (RMS 1.7-4.0 px). The smoother needs
  10 accepted frames to estimate a projection centre, did not get it, and passed all 10 frames through
  unsmoothed. The regions therefore come from the per-frame (`frame`) covariance source, whose dev quantiles
  are large (q̂ = 68 in the bin of these frames, against χ²₂(0.90) = 4.6).

| 117093, 10 frames | Ours | Reference camera (released model) |
|---|---|---|
| Error at the annotated keypoints in the views (160 point-views, all on the far touchline), px, median / p90 | 29.6 / 92.7 | 11.4 / 15.2 |
| Same on the pitch, m, median / p90 | 7.97 / 11.45 | 4.18 / 5.52 |
| Player positions, ours against the reference camera (173 player-frames), m, median / p90 | 3.67 / 8.91 | — |
| Coverage of our 90 % regions by the reference positions | 52.0 % (near half 92.7 %, n = 55; far half 33.1 %, n = 118) | — |
| Median calibrated 90 % radius | 4.69 m | — |

Read these numbers as a smoke check, not a benchmark. There are ten correlated frames, all in the first 2 s of
one clip, and the field-of-view range and aim offset were chosen by looking at these same frames. The keypoints
visible in the views all lie on the far touchline (15 of the 16 were placed by eye), where depth resolution is worst. The reference
itself is metres off on the far half, so the far-half coverage cannot separate our error from its error.
What the numbers do show: (1) PnLCalib, trained on SoccerNet broadcast, does not transfer to the night
panorama and transfers weakly to the day one (7-11 inliers against a median of 11 over the dev player-frames, on 1920 × 1080
broadcast frames); (2) the dev conformal quantiles do not cover the far half of these views. A demo on these
pixels must therefore show abstentions and wide regions honestly.

## What it would take to make it a benchmark

- A better pitch → video model than the released one: per-camera models on each side of the seam, or the
  released model plus a residual field fitted on the keypoints (2.5 px leave-one-out on M2). Then render the
  views so that the pitch plane follows H_virt exactly.
- More and better keypoints: line points annotated densely in the video instead of positions placed by eye
  along the touchlines.
- An evaluation restricted to the near half, where 1 px is 0.02-0.09 m.
- A keypoint model fine-tuned or adapted to panorama views (day and night), for predfoot to calibrate at all.
