"""Synthetic broadcast cameras for tests and the tracer-bullet fixture.

A pinhole camera without distortion, placed in the internal pitch frame (x towards the right-hand goal, y
towards the far touchline, z up) and aimed at a target point. Its ground-plane homography maps pitch points
(x, y, 1) to pixels (u, v, 1), OpenCV convention (pixel centres on integers, so the principal point is
((W-1)/2, (H-1)/2)). The calibration module will provide real cameras; this one only needs to be plausible
and exact.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from predfoot.contracts import FloatArray


def look_at_homography(
    camera_xyz: ArrayLike, target_xyz: ArrayLike, focal_px: float, image_size: tuple[int, int]
) -> FloatArray:
    """Pitch → image homography of a level camera (no roll) at ``camera_xyz`` aimed at ``target_xyz``.

    Returns H normalised so that H[2, 2] = 1, as in :class:`predfoot.contracts.CalibResult`.
    """
    c = np.asarray(camera_xyz, dtype=np.float64)
    fwd = np.asarray(target_xyz, dtype=np.float64) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])  # image +u
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("a vertical optical axis has no defined image right")
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)  # image +v
    rot = np.stack([right, down, fwd])  # world → camera
    width, height = image_size
    k = np.array([[focal_px, 0.0, (width - 1) / 2], [0.0, focal_px, (height - 1) / 2], [0.0, 0.0, 1.0]])
    h = k @ np.column_stack([rot[:, 0], rot[:, 1], -rot @ c])
    return h / h[2, 2]


def project_ground(H: FloatArray, xy: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Project pitch points of shape (N, 2) to pixels; also return the homogeneous w (> 0 in front)."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H, dtype=np.float64).T
    return q[:, :2] / q[:, 2:3], q[:, 2]


# --- the tracer-bullet clip ------------------------------------------------------------------------

CAMERA_XYZ = (0.0, -45.0, 18.0)
FOCAL_PX = 1800.0
IMAGE_SIZE = (1920, 1080)
#: The camera aims at y = 20 m: the ground-plane horizon then sits about 41 px below the top of the image,
#: so a person standing in the stand above it has no ground position (exercises the w ≤ 0 rejection).
AIM_Y = 20.0
SPECTATOR_TRACK_ID = 99


def _rounded(value: float, digits: int = 10) -> float:
    return float(f"{value:.{digits}g}")


def make_broadcast_clip(
    *,
    seed: int = 20261031,
    n_frames: int = 75,
    fps: float = 25.0,
    failed_frames: range = range(40, 50),
    spectator_frames: range = range(10, 36),
) -> tuple[dict[str, Any], dict[tuple[int, int], FloatArray]]:
    """A ``predfoot-clip/1`` clip from a panning broadcast camera, and the true positions.

    22 players (tracks 1-11 home, 12-22 away) move smoothly (≤ 8 m/s); the camera at (0, -45, 18) m pans from
    x = -12 to +12 m. Only feet inside the image are listed. Calibration "fails" (null homography) on
    ``failed_frames``; on ``spectator_frames`` a person in the stand (track 99) is detected above the horizon.
    Returns (clip as a JSON-ready dict, {(native frame, track id): true position}).
    """
    rng = np.random.default_rng(seed)
    n_players = 22
    home = np.column_stack([rng.uniform(-40.0, 5.0, 11), rng.uniform(-25.0, 25.0, 11)])
    away = np.column_stack([rng.uniform(-5.0, 40.0, 11), rng.uniform(-25.0, 25.0, 11)])
    base = np.vstack([home, away])
    velocity = rng.uniform(-3.0, 3.0, size=(n_players, 2))  # m/s
    amplitude = rng.uniform(0.5, 2.0, size=(n_players, 2))  # m
    omega = rng.uniform(0.5, 1.5, size=(n_players, 2))  # rad/s
    phase = rng.uniform(0.0, 2 * np.pi, size=(n_players, 2))
    width, height = IMAGE_SIZE
    duration = (n_frames - 1) / fps
    frames = []
    truth: dict[tuple[int, int], FloatArray] = {}
    for k in range(n_frames):
        t = k / fps
        s = t / duration if duration > 0 else 0.0
        aim_x = -12.0 + 24.0 * (3 * s**2 - 2 * s**3)  # smooth pan
        H = look_at_homography(CAMERA_XYZ, (aim_x, AIM_Y, 0.0), FOCAL_PX, IMAGE_SIZE)
        positions = base + velocity * t + amplitude * np.sin(omega * t + phase)
        uv, w = project_ground(H, positions)
        feet = []
        for i in range(n_players):
            u, v = uv[i]
            if w[i] > 0 and 0.0 <= u <= width - 1 and 0.0 <= v <= height - 1:
                feet.append({"track_id": i + 1, "u": round(float(u), 3), "v": round(float(v), 3)})
                truth[(k, i + 1)] = positions[i]
        if k in spectator_frames:
            spectator = np.array([400.0 + 12.0 * k, 18.0])
            assert np.linalg.solve(H, [*spectator, 1.0])[2] < 0, "the spectator must be above the horizon"
            feet.append({"track_id": SPECTATOR_TRACK_ID, "u": spectator[0], "v": spectator[1]})
        homography = None if k in failed_frames else [[_rounded(v) for v in row] for row in H.tolist()]
        frames.append({"frame": k, "homography": homography, "feet": feet})
    clip = {
        "format": "predfoot-clip/1",
        "clip_id": "synthetic-broadcast-001",
        "fps": fps,
        "image_size": list(IMAGE_SIZE),
        "pitch": {"length": 105.0, "width": 68.0},
        "teams": {str(i): ("home" if i <= 11 else "away") for i in range(1, n_players + 1)},
        "frames": frames,
    }
    return clip, truth


def dumps_clip(clip: dict[str, Any]) -> str:
    """Clip JSON with one frame per line, so diffs of the fixture stay readable."""
    head = {k: v for k, v in clip.items() if k != "frames"}
    lines = ["{"]
    lines += [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    lines.append('  "frames": [')
    lines.append(",\n".join(f"    {json.dumps(f)}" for f in clip["frames"]))
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"
