# SN-GSR data audit

- Generated 2026-10-31 by `uv run python scripts/data_audit.py` at commit `44fc111`.
- Source: SN-GSR-2025 labels (v1.3) downloaded with `scripts/download_data.py sn-gsr-labels`, indexed with `scripts/build_index.py` (checksums in `MANIFEST.sha256`).
- Aggregate numbers only. The test split is locked (ADR-0002): only its structure is reported.
- Pitch positions are in the internal frame (y towards the far touchline), pixels in 1920 × 1080.

## Structure per split

| split | clips | games | clips_per_game | all_750_frames | image_size | labels_version | expected_games | matches_research |
|---|---|---|---|---|---|---|---|---|
| train | 57 | 4, 6, 9 | 4: 18, 6: 19, 9: 20 | True | 1920x1080 | 1.3 | 4, 6, 9 | True |
| valid | 58 | 2, 3, 5 | 2: 18, 3: 21, 5: 19 | True | 1920x1080 | 1.3 | 2, 3, 5 | True |
| test | 49 | 7, 8, 11 | 11: 14, 7: 16, 8: 19 | True | 1920x1080 | 1.3 | 7, 8, 11 | True |

Games per split match the research split (dev = train + valid, test locked).

Dev frames: 86,250; with labelled persons 86,250, pitch 86,250, camera 82,913. Frames without a labelled camera keep their boxes and pitch lines but have no ground-truth pitch positions (`x`, `y` null): evaluations must drop them (`frames.parquet` keeps the flags). They concentrate in a few clips (`camera_pct` below).

## Label-noise floor (dev)

Distance between the smoothed (`bbox_pitch`) and raw (`bbox_pitch_raw`) ground-truth foot position of the same box, over persons (ball excluded). It measures the frame-to-frame jitter of the annotators' camera, not its systematic error: a lower bound on ground-truth error.

- All dev person-frames (n = 1,351,764): p50 0.176 m · p90 0.666 m · p99 2.077 m · max 379.55 m.
- Per-clip p90: min 0.141 m · median 0.654 m · max 1.312 m.
- Gaps above 20 m: 312 person-frames in 46 frames of 10 clips; in 16 of these frames at least half of the persons exceed it, i.e. the raw camera of the whole frame failed. The floor is therefore quoted as quantiles; the maxima only flag such frames.

| game_id | clips | floor_p90_median_m | floor_p90_max_m | truncated_pct_median |
|---|---|---|---|---|
| 2 | 18 | 0.722 | 1.128 | 3.384 |
| 3 | 21 | 0.503 | 0.76 | 1.973 |
| 4 | 18 | 0.565 | 1.312 | 2.241 |
| 5 | 19 | 0.654 | 1.281 | 2.891 |
| 6 | 19 | 0.729 | 1.082 | 2.94 |
| 9 | 20 | 0.707 | 1.124 | 2.471 |

## Dev clips

`box_pct`: frames showing a penalty or goal area; `middle_pct`: frames showing the halfway line or centre circle; `side`: the goal more often in view; `box_h_median_px`: median person box height (a zoom proxy); `truncated_pct`: boxes within 2 px of an image border; `camera_pct`: frames with a labelled camera (hence ground-truth positions).

| clip_id | split | game_id | action_class | frames | camera_pct | persons/frame | truncated_pct | box_h_median_px | box_pct | middle_pct | side | planar_lines_per_frame | floor_p50_m | floor_p90_m | floor_p99_m | floor_max_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SNGS-021 | valid | 2 | Kick-off | 750 | 99.73 | 15.9 | 5.06 | 107.0 | 66.4 | 57.47 | left | 7.33 | 0.16 | 1.1 | 3.8 | 7.51 |
| SNGS-022 | valid | 2 | Foul | 750 | 96.27 | 13.26 | 2.72 | 117.0 | 54.27 | 69.73 | left | 5.02 | 0.18 | 0.61 | 2.04 | 6.05 |
| SNGS-023 | valid | 2 | Offside | 750 | 92.4 | 20.18 | 1.49 | 98.5 | 100.0 | 69.73 | right | 7.74 | 0.18 | 0.91 | 3.29 | 10.89 |
| SNGS-024 | valid | 2 | Clearance | 750 | 84.27 | 11.08 | 3.37 | 114.0 | 73.2 | 63.87 | left | 5.81 | 0.26 | 0.9 | 2.16 | 8.74 |
| SNGS-025 | valid | 2 | Corner | 750 | 95.73 | 16.99 | 1.67 | 113.0 | 100.0 | 20.4 | left | 7.69 | 0.2 | 0.72 | 2.49 | 9.54 |
| SNGS-026 | valid | 2 | Shots on target | 750 | 95.6 | 11.38 | 4.8 | 115.0 | 97.47 | 21.47 | right | 7.65 | 0.23 | 0.9 | 2.31 | 18.17 |
| SNGS-027 | valid | 2 | Direct free-kick | 750 | 100.0 | 16.37 | 2.45 | 135.0 | 100.0 | 0.0 | left | 7.71 | 0.17 | 0.66 | 2.3 | 7.09 |
| SNGS-028 | valid | 2 | Shots off target | 750 | 90.13 | 10.7 | 2.43 | 126.0 | 82.0 | 22.8 | right | 6.75 | 0.17 | 0.64 | 2.14 | 7.52 |
| SNGS-029 | valid | 2 | Yellow card | 750 | 78.13 | 12.89 | 3.68 | 117.0 | 85.47 | 67.47 | right | 7.89 | 0.27 | 0.74 | 1.5 | 5.75 |
| SNGS-030 | valid | 2 | Direct free-kick | 750 | 99.87 | 14.26 | 3.4 | 124.0 | 100.0 | 0.0 | right | 8.92 | 0.22 | 0.73 | 1.53 | 3.65 |
| SNGS-031 | valid | 2 | Shots off target | 750 | 88.27 | 14.97 | 4.06 | 120.0 | 100.0 | 28.27 | left | 7.74 | 0.27 | 1.01 | 2.76 | 19.55 |
| SNGS-032 | valid | 2 | Clearance | 750 | 90.53 | 13.46 | 4.0 | 96.0 | 92.0 | 58.27 | left | 8.2 | 0.28 | 1.08 | 3.64 | 11.17 |
| SNGS-033 | valid | 2 | Foul | 750 | 98.27 | 10.61 | 10.36 | 92.0 | 98.27 | 62.53 | right | 9.91 | 0.14 | 0.61 | 1.41 | 11.12 |
| SNGS-034 | valid | 2 | Goal | 750 | 74.67 | 19.05 | 0.94 | 105.0 | 100.0 | 0.0 | right | 7.71 | 0.12 | 0.47 | 1.52 | 6.95 |
| SNGS-035 | valid | 2 | Kick-off | 750 | 49.87 | 16.24 | 3.68 | 100.0 | 67.47 | 100.0 | left | 6.35 | 0.2 | 0.59 | 1.15 | 2.65 |
| SNGS-036 | valid | 2 | Substitution | 750 | 99.47 | 18.69 | 2.57 | 89.0 | 100.0 | 71.87 | left | 9.39 | 0.1 | 0.44 | 1.13 | 3.6 |
| SNGS-037 | valid | 2 | Shots on target | 750 | 89.47 | 13.06 | 3.4 | 116.0 | 84.8 | 31.47 | right | 6.81 | 0.28 | 1.13 | 2.76 | 9.0 |
| SNGS-038 | valid | 2 | Offside | 750 | 99.87 | 14.3 | 1.49 | 103.0 | 100.0 | 0.0 | right | 6.98 | 0.17 | 0.55 | 1.23 | 3.01 |
| SNGS-039 | valid | 3 | Shots off target | 750 | 100.0 | 19.83 | 2.47 | 81.0 | 100.0 | 49.33 | left | 9.45 | 0.15 | 0.53 | 1.24 | 4.54 |
| SNGS-040 | valid | 3 | Corner | 750 | 99.47 | 18.67 | 1.85 | 75.0 | 89.73 | 61.87 | right | 7.51 | 0.24 | 0.73 | 1.73 | 5.5 |
| SNGS-041 | valid | 3 | Clearance | 750 | 97.87 | 15.26 | 6.94 | 88.0 | 52.4 | 100.0 | left | 6.01 | 0.19 | 0.52 | 1.16 | 3.51 |
| SNGS-042 | valid | 3 | Direct free-kick | 750 | 100.0 | 19.09 | 1.76 | 79.0 | 100.0 | 49.73 | right | 9.73 | 0.14 | 0.49 | 1.26 | 2.86 |
| SNGS-043 | valid | 3 | Goal | 750 | 100.0 | 17.09 | 1.97 | 79.0 | 94.93 | 73.73 | right | 7.81 | 0.19 | 0.59 | 1.38 | 4.36 |
| SNGS-044 | valid | 3 | Yellow card | 750 | 100.0 | 19.09 | 1.67 | 89.0 | 75.73 | 87.87 | right | 7.26 | 0.13 | 0.6 | 1.61 | 3.9 |
| SNGS-045 | valid | 3 | Offside | 750 | 78.8 | 16.09 | 5.3 | 94.0 | 94.4 | 77.07 | left | 8.37 | 0.18 | 0.52 | 1.2 | 3.17 |
| SNGS-046 | valid | 3 | Foul | 750 | 100.0 | 17.84 | 2.06 | 89.0 | 84.67 | 60.13 | left | 8.02 | 0.2 | 0.63 | 1.67 | 5.39 |
| SNGS-047 | valid | 3 | Goal | 750 | 98.93 | 16.99 | 1.7 | 89.0 | 77.33 | 48.13 | left | 6.2 | 0.16 | 0.45 | 1.11 | 3.84 |
| SNGS-048 | valid | 3 | Kick-off | 750 | 100.0 | 19.99 | 0.74 | 69.0 | 100.0 | 83.87 | right | 9.52 | 0.12 | 0.38 | 0.9 | 2.61 |
| SNGS-049 | valid | 3 | Kick-off | 750 | 100.0 | 20.24 | 1.46 | 62.0 | 90.67 | 80.4 | right | 8.28 | 0.14 | 0.5 | 1.18 | 4.97 |
| SNGS-050 | valid | 3 | Corner | 750 | 98.67 | 21.34 | 0.41 | 94.0 | 100.0 | 40.67 | left | 7.35 | 0.16 | 0.5 | 1.39 | 3.73 |
| SNGS-051 | valid | 3 | Clearance | 750 | 99.87 | 15.66 | 1.84 | 82.0 | 52.13 | 100.0 | right | 5.72 | 0.14 | 0.35 | 1.05 | 4.26 |
| SNGS-052 | valid | 3 | Shots off target | 750 | 99.87 | 18.46 | 3.47 | 82.0 | 87.07 | 69.33 | right | 7.84 | 0.16 | 0.49 | 1.22 | 4.39 |
| SNGS-053 | valid | 3 | Offside | 750 | 100.0 | 18.74 | 1.89 | 85.0 | 97.87 | 51.6 | left | 10.31 | 0.11 | 0.42 | 1.05 | 3.31 |
| SNGS-054 | valid | 3 | Goal | 750 | 100.0 | 14.97 | 3.96 | 97.0 | 79.73 | 59.33 | right | 6.69 | 0.21 | 0.65 | 1.74 | 4.76 |
| SNGS-055 | valid | 3 | Foul | 750 | 73.87 | 15.22 | 4.07 | 94.0 | 60.53 | 100.0 | right | 4.17 | 0.23 | 0.76 | 4.58 | 85.87 |
| SNGS-056 | valid | 3 | Goal | 750 | 100.0 | 13.52 | 5.75 | 141.0 | 100.0 | 0.0 | right | 8.7 | 0.03 | 0.14 | 0.55 | 2.88 |
| SNGS-057 | valid | 3 | Direct free-kick | 750 | 95.2 | 18.67 | 5.19 | 88.0 | 52.0 | 65.47 | left | 5.74 | 0.16 | 0.44 | 0.88 | 2.24 |
| SNGS-058 | valid | 3 | Goal | 750 | 97.87 | 15.35 | 6.16 | 93.0 | 73.07 | 57.2 | right | 6.33 | 0.21 | 0.66 | 1.76 | 4.73 |
| SNGS-059 | valid | 3 | Yellow card | 750 | 100.0 | 17.62 | 1.73 | 86.0 | 100.0 | 47.2 | right | 9.73 | 0.13 | 0.46 | 0.99 | 5.58 |
| SNGS-060 | train | 4 | Kick-off | 750 | 99.73 | 17.07 | 2.33 | 88.0 | 83.73 | 100.0 | right | 7.86 | 0.18 | 0.55 | 1.27 | 5.37 |
| SNGS-061 | train | 4 | Shots on target | 750 | 94.0 | 13.93 | 3.67 | 106.0 | 88.0 | 36.93 | right | 7.7 | 0.25 | 0.79 | 2.11 | 21.61 |
| SNGS-062 | train | 4 | Foul | 750 | 98.8 | 16.84 | 1.99 | 97.0 | 76.4 | 80.27 | right | 5.98 | 0.16 | 0.6 | 1.96 | 6.98 |
| SNGS-063 | train | 4 | Offside | 750 | 98.93 | 15.3 | 4.43 | 96.0 | 51.33 | 77.07 | left | 4.71 | 0.14 | 0.53 | 1.73 | 6.95 |
| SNGS-064 | train | 4 | Shots off target | 750 | 100.0 | 20.85 | 1.61 | 97.0 | 100.0 | 47.07 | left | 8.91 | 0.13 | 0.58 | 1.67 | 6.88 |
| SNGS-065 | train | 4 | Clearance | 750 | 100.0 | 17.35 | 1.55 | 82.0 | 41.87 | 100.0 | left | 5.72 | 0.12 | 0.6 | 1.58 | 7.79 |
| SNGS-066 | train | 4 | Direct free-kick | 750 | 100.0 | 20.03 | 2.01 | 86.0 | 75.07 | 61.07 | left | 7.67 | 0.21 | 0.92 | 2.42 | 7.09 |
| SNGS-067 | train | 4 | Corner | 750 | 100.0 | 18.71 | 2.74 | 82.0 | 97.2 | 48.4 | right | 8.68 | 0.14 | 0.48 | 1.09 | 2.79 |
| SNGS-068 | train | 4 | Yellow card | 750 | 98.93 | 16.9 | 2.62 | 99.0 | 81.2 | 59.73 | right | 7.93 | 0.22 | 1.07 | 4.87 | 49.02 |
| SNGS-069 | train | 4 | Kick-off | 750 | 89.07 | 16.08 | 1.04 | 105.0 | 53.33 | 68.53 | left | 4.19 | 0.11 | 0.32 | 0.9 | 15.61 |
| SNGS-070 | train | 4 | Shots on target | 750 | 95.07 | 14.15 | 2.97 | 99.0 | 66.0 | 68.4 | right | 6.09 | 0.2 | 0.8 | 1.95 | 5.65 |
| SNGS-071 | train | 4 | Direct free-kick | 750 | 100.0 | 19.07 | 1.0 | 120.0 | 100.0 | 0.0 | right | 8.24 | 0.23 | 0.74 | 1.67 | 7.43 |
| SNGS-072 | train | 4 | Yellow card | 750 | 80.4 | 16.52 | 4.44 | 113.0 | 67.33 | 77.07 | right | 4.89 | 0.22 | 1.31 | 8.72 | 22.09 |
| SNGS-073 | train | 4 | Shots off target | 750 | 100.0 | 17.95 | 2.15 | 82.0 | 98.27 | 68.4 | right | 9.15 | 0.14 | 0.48 | 1.76 | 8.7 |
| SNGS-074 | train | 4 | Clearance | 750 | 97.6 | 20.14 | 1.59 | 88.0 | 88.0 | 83.87 | right | 7.42 | 0.13 | 0.52 | 1.48 | 5.48 |
| SNGS-075 | train | 4 | Corner | 750 | 100.0 | 18.62 | 0.88 | 112.0 | 100.0 | 0.0 | right | 8.68 | 0.13 | 0.44 | 1.06 | 4.89 |
| SNGS-076 | train | 4 | Substitution | 750 | 100.0 | 24.13 | 4.25 | 84.0 | 100.0 | 100.0 | right | 10.93 | 0.1 | 0.43 | 11.09 | 24.69 |
| SNGS-077 | train | 4 | Foul | 750 | 78.8 | 18.28 | 7.35 | 100.0 | 100.0 | 65.33 | left | 8.89 | 0.12 | 0.45 | 1.26 | 5.68 |
| SNGS-078 | valid | 5 | Clearance | 750 | 96.8 | 12.96 | 5.38 | 101.0 | 90.0 | 49.2 | left | 7.77 | 0.17 | 0.57 | 1.58 | 8.49 |
| SNGS-079 | valid | 5 | Corner | 750 | 98.67 | 16.92 | 1.8 | 85.0 | 90.27 | 56.53 | right | 7.89 | 0.18 | 0.6 | 2.11 | 6.76 |
| SNGS-080 | valid | 5 | Yellow card | 750 | 71.07 | 14.02 | 3.86 | 120.0 | 100.0 | 15.6 | right | 6.62 | 0.26 | 0.71 | 1.58 | 3.86 |
| SNGS-081 | valid | 5 | Foul | 750 | 99.2 | 16.18 | 2.88 | 101.0 | 57.07 | 76.67 | right | 5.74 | 0.16 | 0.59 | 1.78 | 16.69 |
| SNGS-082 | valid | 5 | Shots on target | 750 | 100.0 | 18.45 | 2.89 | 85.0 | 85.73 | 81.6 | right | 7.77 | 0.17 | 0.65 | 1.82 | 5.56 |
| SNGS-083 | valid | 5 | Shots off target | 750 | 96.67 | 15.42 | 3.85 | 79.0 | 92.13 | 67.33 | left | 8.86 | 0.19 | 0.65 | 1.82 | 4.44 |
| SNGS-084 | valid | 5 | Goal | 750 | 92.67 | 15.89 | 4.61 | 75.0 | 96.4 | 62.13 | left | 8.25 | 0.18 | 0.66 | 1.94 | 4.63 |
| SNGS-085 | valid | 5 | Kick-off | 750 | 96.27 | 17.1 | 5.81 | 104.0 | 50.13 | 96.67 | right | 5.01 | 0.15 | 0.73 | 3.41 | 9.32 |
| SNGS-086 | valid | 5 | Corner | 750 | 100.0 | 19.63 | 1.77 | 121.0 | 100.0 | 5.6 | left | 8.53 | 0.22 | 0.81 | 2.04 | 6.71 |
| SNGS-087 | valid | 5 | Goal | 750 | 95.33 | 14.12 | 1.73 | 108.0 | 86.4 | 44.93 | left | 6.51 | 0.18 | 0.69 | 1.8 | 6.02 |
| SNGS-088 | valid | 5 | Offside | 750 | 99.33 | 17.51 | 1.95 | 80.0 | 87.6 | 78.4 | right | 8.73 | 0.21 | 0.69 | 1.61 | 6.06 |
| SNGS-089 | valid | 5 | Yellow card | 750 | 100.0 | 16.97 | 2.33 | 93.0 | 97.33 | 88.0 | left | 8.09 | 0.28 | 1.28 | 21.09 | 66.43 |
| SNGS-090 | valid | 5 | Shots on target | 750 | 100.0 | 14.42 | 3.36 | 93.0 | 100.0 | 18.27 | right | 9.03 | 0.14 | 0.42 | 1.24 | 5.05 |
| SNGS-091 | valid | 5 | Clearance | 750 | 100.0 | 17.05 | 1.76 | 82.0 | 100.0 | 92.8 | left | 10.96 | 0.22 | 0.85 | 2.15 | 5.68 |
| SNGS-092 | valid | 5 | Foul | 750 | 99.87 | 17.12 | 3.78 | 103.0 | 95.6 | 80.13 | left | 8.17 | 0.2 | 0.76 | 2.94 | 43.31 |
| SNGS-093 | valid | 5 | Direct free-kick | 750 | 97.73 | 15.46 | 2.84 | 119.0 | 89.07 | 56.93 | left | 7.28 | 0.16 | 0.5 | 1.22 | 7.35 |
| SNGS-094 | valid | 5 | Goal | 750 | 99.47 | 17.26 | 2.93 | 109.0 | 78.53 | 63.33 | left | 6.71 | 0.16 | 0.61 | 1.6 | 6.31 |
| SNGS-095 | valid | 5 | Kick-off | 750 | 100.0 | 13.05 | 3.19 | 82.0 | 65.07 | 57.73 | right | 7.43 | 0.16 | 0.48 | 1.16 | 4.71 |
| SNGS-096 | valid | 5 | Substitution | 750 | 98.8 | 21.02 | 1.39 | 71.0 | 100.0 | 100.0 | right | 11.04 | 0.17 | 0.61 | 1.42 | 6.38 |
| SNGS-097 | train | 6 | Shots on target | 750 | 100.0 | 16.13 | 4.18 | 109.0 | 100.0 | 15.2 | right | 9.83 | 0.21 | 0.65 | 1.55 | 4.55 |
| SNGS-098 | train | 6 | Clearance | 750 | 98.53 | 11.85 | 7.36 | 113.0 | 89.47 | 38.4 | left | 7.32 | 0.18 | 0.64 | 1.97 | 12.02 |
| SNGS-099 | train | 6 | Yellow card | 750 | 100.0 | 13.08 | 4.94 | 101.0 | 98.67 | 18.53 | right | 8.55 | 0.22 | 0.67 | 1.68 | 6.04 |
| SNGS-100 | train | 6 | Direct free-kick | 750 | 98.0 | 16.62 | 2.78 | 102.0 | 85.33 | 31.73 | left | 7.04 | 0.22 | 0.83 | 2.19 | 6.22 |
| SNGS-101 | train | 6 | Goal | 750 | 98.67 | 10.7 | 5.15 | 105.0 | 78.0 | 51.33 | right | 6.29 | 0.2 | 0.74 | 2.26 | 12.08 |
| SNGS-102 | train | 6 | Kick-off | 750 | 98.67 | 16.72 | 1.82 | 76.0 | 89.07 | 100.0 | right | 8.56 | 0.2 | 0.8 | 1.74 | 6.06 |
| SNGS-103 | train | 6 | Corner | 750 | 99.87 | 19.36 | 1.36 | 112.0 | 72.0 | 36.53 | left | 6.85 | 0.18 | 0.7 | 2.06 | 9.48 |
| SNGS-104 | train | 6 | Goal | 750 | 98.53 | 18.62 | 1.2 | 99.0 | 100.0 | 51.2 | right | 7.65 | 0.2 | 0.97 | 2.72 | 8.16 |
| SNGS-105 | train | 6 | Shots off target | 750 | 99.73 | 16.12 | 2.81 | 95.0 | 100.0 | 31.87 | right | 9.36 | 0.24 | 0.71 | 1.69 | 8.58 |
| SNGS-106 | train | 6 | Goal | 750 | 100.0 | 15.63 | 2.56 | 83.0 | 100.0 | 0.0 | left | 8.75 | 0.14 | 0.45 | 1.2 | 4.84 |
| SNGS-107 | train | 6 | Kick-off | 750 | 97.47 | 15.77 | 2.51 | 72.0 | 95.2 | 66.27 | right | 7.82 | 0.13 | 0.57 | 1.76 | 9.97 |
| SNGS-108 | train | 6 | Clearance | 750 | 99.6 | 14.05 | 3.48 | 103.0 | 76.0 | 37.33 | left | 7.5 | 0.21 | 0.88 | 2.69 | 5.53 |
| SNGS-109 | train | 6 | Direct free-kick | 750 | 95.33 | 17.35 | 2.39 | 103.0 | 70.93 | 65.2 | left | 5.97 | 0.21 | 0.8 | 2.41 | 27.45 |
| SNGS-110 | train | 6 | Corner | 750 | 94.93 | 16.87 | 2.94 | 108.0 | 70.13 | 47.6 | left | 6.43 | 0.2 | 0.73 | 1.89 | 6.93 |
| SNGS-111 | train | 6 | Yellow card | 750 | 99.07 | 14.29 | 3.46 | 105.0 | 92.53 | 34.93 | left | 7.93 | 0.17 | 0.71 | 2.31 | 8.07 |
| SNGS-112 | train | 6 | Shots on target | 750 | 100.0 | 14.8 | 3.54 | 86.0 | 65.6 | 50.13 | right | 7.29 | 0.16 | 0.5 | 1.28 | 3.37 |
| SNGS-113 | train | 6 | Substitution | 750 | 98.93 | 20.01 | 3.07 | 70.0 | 100.0 | 100.0 | right | 8.87 | 0.16 | 0.78 | 1.76 | 4.48 |
| SNGS-114 | train | 6 | Foul | 750 | 100.0 | 12.28 | 4.3 | 107.0 | 49.6 | 73.73 | left | 5.81 | 0.17 | 1.08 | 3.07 | 42.95 |
| SNGS-115 | train | 6 | Shots off target | 750 | 91.2 | 16.54 | 1.42 | 108.0 | 100.0 | 11.6 | right | 7.77 | 0.25 | 0.85 | 2.26 | 19.63 |
| SNGS-151 | train | 9 | Kick-off | 750 | 96.0 | 13.17 | 3.8 | 103.0 | 52.8 | 78.93 | right | 6.24 | 0.16 | 0.58 | 1.96 | 7.1 |
| SNGS-152 | train | 9 | Offside | 750 | 99.73 | 20.31 | 0.71 | 105.0 | 100.0 | 0.0 | right | 7.0 | 0.21 | 0.87 | 2.32 | 6.96 |
| SNGS-153 | train | 9 | Goal | 750 | 99.87 | 12.46 | 2.05 | 105.0 | 87.33 | 57.2 | right | 7.93 | 0.21 | 0.75 | 2.15 | 7.64 |
| SNGS-154 | train | 9 | Corner | 750 | 92.0 | 16.41 | 1.89 | 125.0 | 88.67 | 31.2 | left | 6.78 | 0.27 | 0.89 | 1.99 | 11.89 |
| SNGS-155 | train | 9 | Shots off target | 750 | 96.53 | 13.74 | 4.56 | 121.0 | 100.0 | 38.67 | left | 7.3 | 0.2 | 0.74 | 2.8 | 6.84 |
| SNGS-156 | train | 9 | Direct free-kick | 750 | 99.87 | 16.34 | 1.16 | 116.0 | 100.0 | 0.0 | left | 8.05 | 0.17 | 0.59 | 1.19 | 5.95 |
| SNGS-157 | train | 9 | Clearance | 750 | 96.4 | 12.93 | 4.25 | 99.0 | 90.8 | 40.13 | left | 8.22 | 0.25 | 0.9 | 2.43 | 8.03 |
| SNGS-158 | train | 9 | Foul | 750 | 98.53 | 13.23 | 2.39 | 97.0 | 84.8 | 46.93 | right | 7.66 | 0.14 | 0.68 | 1.6 | 8.27 |
| SNGS-159 | train | 9 | Shots on target | 750 | 94.4 | 15.75 | 3.37 | 97.0 | 100.0 | 28.93 | left | 7.76 | 0.26 | 0.85 | 1.95 | 11.18 |
| SNGS-160 | train | 9 | Kick-off | 750 | 99.33 | 13.27 | 1.89 | 91.0 | 50.27 | 88.0 | left | 5.39 | 0.17 | 0.62 | 1.37 | 4.63 |
| SNGS-161 | train | 9 | Shots off target | 750 | 92.27 | 13.66 | 4.0 | 108.0 | 71.07 | 40.53 | left | 5.92 | 0.25 | 0.85 | 2.22 | 11.23 |
| SNGS-162 | train | 9 | Direct free-kick | 750 | 99.6 | 17.11 | 2.4 | 112.0 | 96.4 | 40.67 | right | 7.98 | 0.38 | 1.12 | 2.75 | 8.25 |
| SNGS-163 | train | 9 | Offside | 750 | 93.73 | 13.12 | 1.72 | 95.0 | 60.13 | 56.8 | left | 5.59 | 0.2 | 0.66 | 1.58 | 5.74 |
| SNGS-164 | train | 9 | Foul | 750 | 99.6 | 17.22 | 1.82 | 88.0 | 100.0 | 75.87 | left | 9.29 | 0.28 | 0.95 | 2.69 | 8.92 |
| SNGS-165 | train | 9 | Goal | 750 | 99.73 | 14.37 | 3.01 | 103.0 | 99.33 | 14.4 | left | 8.58 | 0.12 | 0.65 | 2.35 | 5.9 |
| SNGS-166 | train | 9 | Substitution | 750 | 99.2 | 16.61 | 3.75 | 80.0 | 98.13 | 100.0 | right | 8.86 | 0.09 | 0.48 | 1.3 | 5.12 |
| SNGS-167 | train | 9 | Corner | 750 | 94.13 | 16.24 | 2.67 | 102.0 | 76.53 | 47.87 | right | 6.44 | 0.19 | 0.66 | 1.82 | 9.21 |
| SNGS-168 | train | 9 | Yellow card | 750 | 99.87 | 17.42 | 2.54 | 88.0 | 40.8 | 100.0 | right | 4.39 | 0.18 | 0.46 | 1.15 | 4.06 |
| SNGS-169 | train | 9 | Clearance | 750 | 99.6 | 14.62 | 4.41 | 93.0 | 86.8 | 37.87 | right | 7.61 | 0.24 | 0.89 | 11.34 | 379.55 |
| SNGS-170 | train | 9 | Shots on target | 750 | 100.0 | 16.08 | 1.84 | 89.0 | 96.13 | 17.6 | left | 8.45 | 0.17 | 0.64 | 1.84 | 9.7 |

All dev clips have 750 frames: True. Action classes: Goal 14, Kick-off 12, Clearance 12, Foul 11, Corner 11, Direct free-kick 11, Shots off target 11, Shots on target 10, Yellow card 10, Offside 8, Substitution 5.

## Line classes (dev)

`frames_pct`: share of dev frames where the class is annotated. Non-planar classes (goal posts and crossbars) never enter a homography; the circles are planar arcs.

| line_class | is_planar | frames_pct | points_per_instance | known |
|---|---|---|---|---|
| Side line top | True | 91.5 | 5.4 | True |
| Circle central | True | 53.9 | 43.4 | True |
| Big rect. right main | True | 46.3 | 4.0 | True |
| Big rect. right top | True | 45.3 | 3.1 | True |
| Circle right | True | 42.0 | 18.3 | True |
| Big rect. left main | True | 41.8 | 4.2 | True |
| Big rect. left top | True | 40.8 | 3.0 | True |
| Side line right | True | 40.2 | 4.6 | True |
| Middle line | True | 38.9 | 3.5 | True |
| Circle left | True | 36.6 | 19.3 | True |
| Side line bottom | True | 35.6 | 4.1 | True |
| Side line left | True | 35.4 | 4.8 | True |
| Small rect. right main | True | 34.1 | 3.1 | True |
| Small rect. right top | True | 33.9 | 2.8 | True |
| Small rect. left main | True | 29.4 | 3.3 | True |
| Small rect. left top | True | 29.2 | 2.8 | True |
| Goal right crossbar | False | 27.5 | 2.6 | True |
| Goal right post left | False | 27.5 | 2.5 | True |
| Goal left crossbar | False | 24.1 | 2.6 | True |
| Goal left post right | False | 24.0 | 2.5 | True |
| Goal right post right | False | 22.4 | 2.5 | True |
| Big rect. right bottom | True | 22.1 | 2.8 | True |
| Small rect. right bottom | True | 21.1 | 2.8 | True |
| Goal left post left | False | 20.8 | 2.4 | True |
| Small rect. left bottom | True | 20.4 | 2.9 | True |
| Big rect. left bottom | True | 19.2 | 3.0 | True |

## Anomalies (dev)

- Person-frames more than 5 m outside a 105 × 68 m pitch (smoothed or raw): 364 in 38 clip(s), largest excess 336.1 m.

| clip_id | person_frames | tracks | max_excess_m |
|---|---|---|---|
| SNGS-050 | 61 | 2 | 6.1 |
| SNGS-053 | 49 | 1 | 6.0 |
| SNGS-163 | 40 | 1 | 6.6 |
| SNGS-089 | 26 | 6 | 19.7 |
| SNGS-169 | 21 | 20 | 336.1 |
| SNGS-084 | 17 | 1 | 7.3 |
| SNGS-103 | 16 | 1 | 6.8 |
| SNGS-086 | 12 | 1 | 6.1 |
| SNGS-162 | 12 | 1 | 6.4 |
| SNGS-091 | 11 | 1 | 6.5 |

- Person-frames without a pitch position: 48,072.
- Duplicate (clip, frame, track_id): 0.
- Line points more than 2 px outside the image: 3,550.
- Roles: player 1,224,471, referee 123,393, goalkeeper 50,127, other 1,845.
