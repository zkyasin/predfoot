"""Turn decoded network detections into calibration evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from predfoot.calib.lines import intersection_correspondences
from predfoot.calib.solver import LineObservation
from predfoot.geometry.pitch import pnlcalib_keypoints, pnlcalib_lines

FloatArray = NDArray[np.float64]

_KP = pnlcalib_keypoints()
_SEGMENTS = pnlcalib_lines()
#: An intersection or an end of a detected line counts as half a keypoint.
LINE_WEIGHT = 0.5


def observations_from_detections(
    det: dict[str, Any], use_lines: bool = True
) -> tuple[FloatArray, FloatArray, FloatArray, list[LineObservation]]:
    """Pitch ↔ image evidence of one frame.

    ``det`` holds ``kp_ids``, ``kp_uv``, ``kp_conf``, ``line_ids`` and ``line_uv`` (see
    predfoot.keypoints.cache). Returns point correspondences (planar keypoints, plus intersections of detected
    lines), their weights (keypoint confidence, :data:`LINE_WEIGHT` for intersections) and the detected line
    ends as point-on-line observations.
    """
    ids = [int(k) for k in det["kp_ids"]]
    planar = np.array([_KP[k][2] for k in ids], dtype=bool)
    xy = np.array([_KP[k][:2] for k in ids], dtype=np.float64).reshape(-1, 2)[planar]
    uv = np.asarray(det["kp_uv"], dtype=np.float64).reshape(-1, 2)[planar]
    w = np.asarray(det["kp_conf"], dtype=np.float64)[planar]
    lines: list[LineObservation] = []
    if use_lines and len(det["line_ids"]):
        named = {_SEGMENTS[int(k)].name: det["line_uv"][i] for i, k in enumerate(det["line_ids"])}
        lxy, luv = intersection_correspondences(named)
        if len(lxy):
            xy, uv = np.vstack([xy, lxy]), np.vstack([uv, luv])
            w = np.concatenate([w, np.full(len(lxy), LINE_WEIGHT)])
        for i, k in enumerate(det["line_ids"]):
            seg = _SEGMENTS[int(k)]
            if seg.planar:
                lines.append(
                    LineObservation(seg.a[:2], seg.b[:2], np.asarray(det["line_uv"][i]), LINE_WEIGHT)
                )
    return xy, uv, w, lines
