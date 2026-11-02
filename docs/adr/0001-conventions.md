# ADR-0001: Conventions and data contracts

- Status: accepted
- Date: 2026-10-31

## Context

predfoot reads SoccerNet labels, SoccerTrack v2 videos and IDSSE tracking, and writes SkillCorner-style
JSONL. Each source has its own axes, units and frame rates. Mixing them silently is the most likely way to get
plausible-looking but wrong results (a mirrored pitch, ellipses tilted the wrong way, 1-2 px biases).

What was checked on 2026-10-30:
- **SkillCorner open data** (match 1874553): records have `frame`, `timestamp`, `period`, `ball_data`,
  `possession`, `image_corners_projection`, `player_data` (`x`, `y`, `player_id`, `is_detected`); 10 fps; the
  top image corners project to y = +39 (clipped 5 m beyond the far touchline), the bottom ones to y ≈ −31 to −35.
- **SN-GSR** (clip SNGS-056): `bbox_pitch` of a player low in the image (near the camera) has y = +16.9; pitch
  lines are polylines in coordinates normalised to [0, 1]; 6 of the 26 line classes are goal posts and
  crossbars (not on the ground plane).
- **SoccerTrack v2** README: "y towards the touchline nearer the camera".
- kloppy's SkillCorner reader parses the JSONL line by line and reads `frame["ball_data"]` directly.

## Decision

| Topic | Convention |
|---|---|
| Pitch frame | SkillCorner's: origin at the centre spot, metres, x towards the right-hand goal of the main camera, y towards the far touchline. Default pitch 105 × 68 m, configurable. |
| SoccerNet / SoccerTrack v2 → internal | `(x, y) ↦ (x, −y)`; covariances `Σ ↦ D Σ D`, D = diag(1, −1). Only in `predfoot/io/`, with `predfoot.io.conventions`. |
| Pixels | Original frame (1920 × 1080 for SN-GSR), OpenCV convention. SoccerNet normalised points are multiplied by width and height, with no half-pixel offset (not verified; the possible error is at most 0.5 px). Network inputs are resized (960 × 540 for PnLCalib) and mapped back in one helper. |
| Ground plane | Only planar template elements enter a homography; goal posts and crossbars are excluded. |
| Time | Native frame index (25 fps for SN-GSR) inside the pipeline; exports at 10 fps with `timestamp` `HH:MM:SS.ss`. |
| Identifiers | `clip_id` (e.g. `SNGS-056`), `game_id` (integer from the labels' `info`), `track_id`. Exports use `player_id = track_id` (anonymous) and ship a minimal `match.json`. |
| Export record | Always carries `ball_data` and `possession` (null values when unknown) so kloppy can load it. Corners are clipped like SkillCorner's and null above the horizon. |
| Randomness | Seeds derived from `(clip_id, frame)` for RANSAC and Monte-Carlo. |
| Data location | `PREDFOOT_DATA_DIR` (shared between git worktrees), never inside git. |

### Contracts

Shared dataclasses live in `predfoot/contracts.py`: `KeypointObs`, `LinePointObs`, `CalibResult` (H with
H[2,2] = 1, θ = first 8 entries row-major, `cov_theta` 8×8), `FootObs`, `Localization`.

Parquet tables under `PREDFOOT_DATA_DIR/index/` (built by `scripts/build_index.py`):
- `clips`: every clip, test included (clip_id, split, game_id, action_class, n_frames, width, height);
- `frames`: per-frame label flags (`has_labeled_person`, `has_labeled_pitch`, `has_labeled_camera`);
- `persons`: clip_id, frame, track_id, role, team, jersey, bbox_x, bbox_y, bbox_w, bbox_h, foot_u, foot_v (px),
  x, y (smoothed ground truth) and x_raw, y_raw (raw ground truth) in the internal frame, `truncated`;
- `lines`: clip_id, frame, line_class, is_planar, point_idx, u, v (px).

The default `frames`, `persons` and `lines` tables hold dev clips only; the test split's go to `index/test/`,
so nothing reads the locked games by accident. Frames with `has_labeled_camera = false` have no ground-truth
positions and are dropped from evaluations.

## Consequences

Every conversion has one implementation and one test. Exports match SkillCorner's axes, so a mirrored pitch
fails a unit test instead of a demo.
