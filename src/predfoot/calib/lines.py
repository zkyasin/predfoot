"""Pitch lines as calibration evidence: fitted image lines, their intersections, and reprojection error.

Works the same for annotated lines (the line oracle used as a label-noise floor) and detected lines (extra
correspondences next to the keypoints).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from predfoot.geometry import homography as hg
from predfoot.geometry.pitch import Segment, sn_line_elements

FloatArray = NDArray[np.float64]
_ELEMENTS = sn_line_elements()
_PITCH_MARGIN = (55.0, 37.0)  # template intersections must fall on or next to the pitch


def fit_image_line(points: ArrayLike) -> FloatArray:
    """Total-least-squares image line (a, b, c) with a² + b² = 1 through points (m ≥ 2, 2)."""
    p = np.asarray(points, dtype=np.float64)
    centre = p.mean(axis=0)
    normal = np.linalg.svd(p - centre)[2][-1]
    return np.array([normal[0], normal[1], -normal @ centre])


def _straight_planar(name: str) -> Segment | None:
    el = _ELEMENTS.get(name)
    return el if isinstance(el, Segment) and el.planar else None


def intersection_correspondences(lines: Mapping[str, ArrayLike]) -> tuple[FloatArray, FloatArray]:
    """Pitch ↔ image correspondences at the intersections of pairs of straight planar lines.

    ``lines`` maps SoccerNet line class names to image points on that line (≥ 2 points, pixels). Only pairs
    whose template lines cross on or near the pitch are used. Returns (xy (k, 2) metres, uv (k, 2) pixels).
    """
    fitted = {
        name: (seg, fit_image_line(pts))
        for name, pts in lines.items()
        if (seg := _straight_planar(name)) is not None and len(np.asarray(pts)) >= 2
    }
    names = sorted(fitted)
    xy, uv = [], []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            (sa, la), (sb, lb) = fitted[a], fitted[b]
            da = np.subtract(sa.b[:2], sa.a[:2])
            db = np.subtract(sb.b[:2], sb.a[:2])
            cross = da[0] * db[1] - da[1] * db[0]
            if abs(cross) < 1e-6:
                continue  # parallel on the pitch
            t = np.linalg.solve(np.array([[da[0], -db[0]], [da[1], -db[1]]]), np.subtract(sb.a[:2], sa.a[:2]))
            p = np.add(sa.a[:2], t[0] * da)
            if abs(p[0]) > _PITCH_MARGIN[0] or abs(p[1]) > _PITCH_MARGIN[1]:
                continue
            q = np.cross(la, lb)
            if abs(q[2]) < 1e-9:
                continue  # parallel in the image
            xy.append(p)
            uv.append(q[:2] / q[2])
    return np.array(xy, dtype=np.float64).reshape(-1, 2), np.array(uv, dtype=np.float64).reshape(-1, 2)


def line_reprojection_px(H: ArrayLike | None, lines: Mapping[str, ArrayLike]) -> float:
    """Mean distance (px) from points of straight planar lines to the template lines projected by ``H``."""
    if H is None:
        return float("nan")
    dists: list[float] = []
    for name, pts in lines.items():
        seg = _straight_planar(name)
        if seg is None:
            continue
        line = hg.image_line(H, seg.a, seg.b)
        p = np.asarray(pts, dtype=np.float64)
        dists.extend(np.abs(np.c_[p, np.ones(len(p))] @ line).tolist())
    return float(np.mean(dists)) if dists else float("nan")


def line_oracle(lines: Mapping[str, ArrayLike]) -> FloatArray | None:
    """Homography from the intersections of annotated lines, or None with fewer than 4 intersections."""
    xy, uv = intersection_correspondences(lines)
    if len(xy) < 4:
        return None
    try:
        return hg.dlt(xy, uv)
    except np.linalg.LinAlgError:
        return None
