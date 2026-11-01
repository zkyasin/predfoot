# Showcase visuals

The animations in `assets/showcase/`, made for the showcase page and the README. Every file is under 900 kB.

| File | Size | Format | Made by |
|---|---|---|---|
| `hero.mp4` | 713 kB | H.264 (High, yuv420p, `+faststart`, no audio), 720 × 850, 25 fps, 170 frames, 6.8 s loop | `scripts/render_hero.py` at `93ad599` |
| `hero.jpg` | 124 kB | JPEG 720 × 850, poster: the middle frame (clip frame 440) | same run |
| `smoothing.mp4` | 456 kB | H.264 (High, yuv420p, `+faststart`, no audio), 960 × 374, 5 fps, 75 frames, 15.0 s: the two pitches side by side | `scripts/render_minimap_demo.py --style page` at `9ce6da5` |
| `smoothing.jpg` | 39 kB | JPEG 960 × 374, poster: the still of `reports/minimap_demo.md` (t = 7.4 s) | same run |
| `smoothing-tall.mp4` | 532 kB | H.264 as above, 640 × 992: the two pitches stacked, for phones | same run |
| `smoothing-tall.jpg` | 54 kB | JPEG 640 × 992, the same still | same run |
| `smoothing.gif` | 598 kB | GIF 880 × 440, 75 frames at 200 ms, 15.0 s (72 frames stored: Pillow merges identical frames that follow each other), for the README | `scripts/render_minimap_demo.py --style plain` at `9ce6da5` |

## `hero.mp4`, `hero.jpg`: predfoot on public footage

```bash
uv run python scripts/render_hero.py --start 13.0 --seconds 8 --show-from 14.2 --crf 29 \
    --detections runs/20261101-180608_39f1560_hero/detections.json \
    --mp4 assets/showcase/hero.mp4 --poster assets/showcase/hero.jpg
```

**What it shows.** SoccerTrack v2 match 117093, the daytime match (Scott et al., CC BY 4.0), MOT clip from 14.2 to
21.0 s (frames 355-524). At the top, a virtual broadcast camera follows the play inside the dataset's fixed
panorama. It is rendered at every clip frame, the top 92 of 1080 rows are cut where the panorama has no pixels,
and it is scaled down to 720 px. The view shows predfoot's pitch lines in aqua and each player's feet as amber dots.
The feet are the bottom centre of the dataset's annotated boxes: predfoot does not detect players. At the bottom,
the same players are drawn on a top-down 105 × 68 m pitch, each with its 90 % zone in the page's pink, and the
lighter area outlined in aqua is the ground the camera sees. 15 to 20 players are placed per frame. The only text
is the credit "SoccerTrack v2 (CC BY 4.0)".

**How.** PnLCalib, with its public CC BY 4.0 weights and not retrained, runs on MPS on 41 keyframes, one clip frame
in five from 13.0 s (frames 325-525). Then come the robust solver and the PTZ smoother, with the noise parameters
of the SN-GSR dev evaluation. Over the frames shown, 34 are keyframes and 136 lie between keyframes. All of them
carry the smoother's state; none holds a neighbour's camera and none abstains. A frame between keyframes gets the
smoother's own estimate at its time: a cubic Hermite interpolation of the smoothed pan, tilt, roll and log focal
and of their rates. That is the RTS smoother's mean under its constant-velocity model. The covariance is
interpolated linearly between the two keyframes, which is an approximation. The zones use the Mondrian quantiles
of the SN-GSR dev evaluation (`src/predfoot/demo/dev_calibration.json`, smoother model). Their median 90 % radius
is 3.0 m and their p90 is 4.3 m. The keyframe detections come from an earlier run of the same script on the same
views (largest difference in camera parameters: 0). The final render reused them through `--detections`, so it ran
no network itself. The loop was rendered again at `93ad599`, from the same detections, only to draw the zones in
pink instead of the players' amber; nothing else changed.

**What it does not show.**

- **It is not evidence of accuracy.** There is no reference to score against: SoccerTrack's own released
  calibration is metres off (`reports/soccertrack_v2.md`), and the zones borrow SN-GSR quantiles across a domain
  shift. Suggested caption: *Illustration on public footage (SoccerTrack v2, Scott et al., CC BY 4.0): frames
  reprojected to a virtual camera, cropped, downscaled and overlaid. Not a measurement.*
- **The window was picked to look good.** A geometry-only scan of the clip, with no network and no labels, found
  the 8 s spans where the virtual camera pans most; the processed span starts at 13.0 s. Its first six keyframes
  (13.0-14.0 s, view turned towards the left penalty area) fail to calibrate, with 1-3 inliers. The loop starts at
  14.2 s, after them.
- **The smoothing is fragile on these views.** Calibrations rest on only 7-14 inliers. On this span the smoother
  found a reliable projection centre: 0.5 m standard deviation, 0.7 m from the panorama's true centre. A discarded
  second trial (14.2-22.2 s, whose views differ slightly because the virtual camera follows a different span) found
  none (1.3 m standard deviation, over the 1 m limit). Every one of its 41 per-frame cameras was then passed
  through, which left no smoothed camera for the frames in between. That trial ran the network on 41 more
  keyframes. Each render stays within the script's cap of 60 keyframes (`MAX_KEYFRAMES`).
- One keyframe (frame 400) was bridged. Its zone uses the smoother model (`keyframe_sources` in the script).
  `predfoot.demo.calibration.cov_sources` would give it the per-frame model, which is roughly ten times too small
  for a smoothed covariance. That function predates bridging, and `scripts/render_demo.py` still uses it.
- The loop restarts with a cut. Detection, tracking and team identity are not part of predfoot.

## `smoothing.mp4`, `smoothing-tall.mp4`, `.jpg`, `smoothing.gif`: frame by frame against smoothed over time

```bash
uv run python scripts/render_minimap_demo.py --none-run runs/20261031-152537_c3861c7_none \
    --ptz-run runs/20261031-163641_bf28b35_ptz --style page     # the page's two clips and their posters
uv run python scripts/render_minimap_demo.py --none-run runs/20261031-152537_c3861c7_none \
    --ptz-run runs/20261031-163641_bf28b35_ptz --style plain    # the README's GIF
```

The run tables are those of `reports/minimap_demo.md`, made by the dev evaluation at `c3861c7` and `bf28b35`.

**What it shows.** SN-GSR dev clip SNGS-081 (game 5), from 6.6 to 21.4 s, one frame in five, played in real
time. Only top-down pitches are drawn, never SoccerNet pixels. Each panel shows the estimated positions with
their shaded 90 % zones; the quantiles were fitted on the five other dev games, leaving the clip's game out.

- Page style (`smoothing.mp4`, `smoothing-tall.mp4`): the per-frame camera in the page's grey (left, or top on
  phones) and the PTZ smoother in the page's pink (right, or bottom), on the mown pitch of the hero's map. The
  annotated true positions are small white dots. The video carries no title or legend: the page labels it in
  HTML, in a bar of the same green above the clip.
- Plain style (`smoothing.gif`, the README): "Frame by frame" on the left in blue, "Smoothed over time" on the
  right in orange, true positions as small black dots, and a three-item legend with no counts, areas or
  subtitles.

Both keep the honest states. "No position" (dark, or grey) marks a frame where the camera published nothing:
14.4-15.0 s for both cameras, plus 19.4 s for the per-frame one. Two frames published positions that no
position label can check (hatched): at 19.0 s both cameras sit 4.8 and 2.8 px from the annotated pitch lines,
and the veil reads "not checked"; at 15.2 s both are 534.9 px off, and the veil reads "camera lost" (the rule:
more than 10 px, the evaluation's failure threshold).

**What it does not show.** The clip and the window were chosen to show the effect, by the rule of
`reports/minimap_demo.md`, which gives the numbers. The clip is the one, among the fastest-moving third of the
115 dev clips, where smoothing cuts the frame-to-frame jitter most. The window is the 15 s where the per-frame
camera jitters most. It is not a typical clip, and it comes from development data, not from the preregistered
test. Positions come from annotated boxes. The smoothed camera fails too: at 15.2 s it loses the pitch with the
per-frame camera, and the veil hides those positions rather than drawing them. The poster is the window's worst
frame for the per-frame camera. The report style is unchanged: regenerating it gives a GIF and a PNG
byte-identical to `assets/minimap_demo.*`.

## Legibility check

A scratch page embedded both videos and was captured by headless Chrome at 1280 px, with the hero in a 480 px
column and the smoothing animation about 1000 px wide. It was also captured in a 375 px viewport: an iframe was
used, since the headless window will not go below about 500 px. At 343 px of content width, the hero's lines,
feet and zones remain clear and its credit is about 7 px high. The side-by-side smoothing clip was too small
there (each pitch about 165 px wide), so phones get the stacked clip instead: each pitch is 343 px wide, and
the page's own text labels it at full size.
