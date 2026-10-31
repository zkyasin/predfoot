"""Pitch template in the internal frame (ADR-0001): metres, origin at the centre spot, x towards the
right-hand goal of the main camera, y towards the far touchline.

Dimensions follow the Laws of the Game. The keypoint and line ids match the output channels of the PnLCalib
networks (their order is an interoperability fact, ADR-0002); coordinates are recomputed here from the
dimensions, except eight points on the circles that only exist as published coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16  # 18.32 / 2
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_WIDTH = 20.16  # 40.32 / 2
GOAL_HALF_WIDTH = 3.66  # 7.32 / 2
GOAL_HEIGHT = 2.44
PENALTY_SPOT_DISTANCE = 11.0
CIRCLE_RADIUS = 9.15


@dataclass(frozen=True)
class PitchDims:
    """Pitch size in metres. SN-GSR assumes 105 × 68; SkillCorner match files give the real size."""

    length: float = 105.0
    width: float = 68.0

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2


@dataclass(frozen=True)
class Segment:
    """A straight template element between two 3D points (z up)."""

    name: str
    a: tuple[float, float, float]
    b: tuple[float, float, float]

    @property
    def planar(self) -> bool:
        return self.a[2] == 0.0 and self.b[2] == 0.0


@dataclass(frozen=True)
class Arc:
    """A circular ground element: centre, radius and angular range in degrees (counter-clockwise from +x)."""

    name: str
    centre: tuple[float, float]
    radius: float
    start_deg: float
    end_deg: float

    def sample(self, n: int = 64) -> FloatArray:
        """Points along the arc, shape (n, 2)."""
        ang = np.deg2rad(np.linspace(self.start_deg, self.end_deg, n))
        cx, cy = self.centre
        return np.stack([cx + self.radius * np.cos(ang), cy + self.radius * np.sin(ang)], axis=-1)


def _corner_frame_to_internal(x: float, y: float, dims: PitchDims) -> tuple[float, float]:
    """PnLCalib/SoccerNet corner frame (x ∈ [0, L], y ∈ [0, W], y = 0 on the far touchline) → internal."""
    return x - dims.half_length, dims.half_width - y


def pnlcalib_keypoints(dims: PitchDims | None = None) -> dict[int, tuple[float, float, bool]]:
    """Ground positions targeted by the 57 channels of the PnLCalib keypoint network.

    Returns ``{id: (x, y, planar)}`` with 1-based ids. Four channels (12, 15, 16, 19) mark the tops of the
    goal posts: their ground projection is listed for reference, but they are not on the ground plane.
    """
    d = dims or PitchDims()
    L, W = d.length, d.width
    cy = W / 2
    bx, by = PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_WIDTH
    sx, sy = GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH
    gy, r, ps = GOAL_HALF_WIDTH, CIRCLE_RADIUS, PENALTY_SPOT_DISTANCE
    arc_dx = float(np.sqrt(r**2 - (bx - ps) ** 2))  # penalty arc meets the penalty-area line at cy ± arc_dx
    diag = r / np.sqrt(2.0)
    corner = {
        1: (0, 0), 2: (L / 2, 0), 3: (L, 0),
        4: (0, cy - by), 5: (bx, cy - by), 6: (L - bx, cy - by), 7: (L, cy - by),
        8: (0, cy - sy), 9: (sx, cy - sy), 10: (L - sx, cy - sy), 11: (L, cy - sy),
        12: (0, cy - gy), 13: (0, cy - gy), 14: (L, cy - gy), 15: (L, cy - gy),
        16: (0, cy + gy), 17: (0, cy + gy), 18: (L, cy + gy), 19: (L, cy + gy),
        20: (0, cy + sy), 21: (sx, cy + sy), 22: (L - sx, cy + sy), 23: (L, cy + sy),
        24: (0, cy + by), 25: (bx, cy + by), 26: (L - bx, cy + by), 27: (L, cy + by),
        28: (0, W), 29: (L / 2, W), 30: (L, W),
        31: (bx, cy - arc_dx), 32: (L / 2, cy - r), 33: (L - bx, cy - arc_dx),
        34: (bx, cy + arc_dx), 35: (L / 2, cy + r), 36: (L - bx, cy + arc_dx),
        # Published coordinates of points on the penalty arcs and the centre circle (no closed form given).
        37: (19.99, 32.29), 38: (43.68, 31.53), 39: (61.31, 31.53), 40: (85.0, 32.29),
        41: (19.99, 35.70), 42: (43.68, 36.46), 43: (61.31, 36.46), 44: (85.0, 35.70),
        45: (ps, cy), 46: (bx, cy), 47: (ps + r, cy),
        48: (L / 2 - diag, cy - diag), 49: (L / 2 + diag, cy - diag),
        50: (L / 2 - r, cy), 51: (L / 2, cy), 52: (L / 2 + r, cy),
        53: (L / 2 - diag, cy + diag), 54: (L / 2 + diag, cy + diag),
        55: (L - ps - r, cy), 56: (L - bx, cy), 57: (L - ps, cy),
    }  # fmt: skip
    elevated = {12, 15, 16, 19}
    return {
        k: (*_corner_frame_to_internal(float(x), float(y), d), k not in elevated)
        for k, (x, y) in corner.items()
    }


def pnlcalib_lines(dims: PitchDims | None = None) -> dict[int, Segment]:
    """The 23 straight elements predicted by the PnLCalib line network, ids 1-based, in the internal frame.

    Lines 7-12 are goal posts and crossbars (not planar).
    """
    d = dims or PitchDims()
    hl, hw = d.half_length, d.half_width
    bx, by = hl - PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_WIDTH
    sx, sy = hl - GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH
    gy, gz = GOAL_HALF_WIDTH, GOAL_HEIGHT

    def seg(name: str, a: tuple[float, ...], b: tuple[float, ...]) -> Segment:
        a3 = (float(a[0]), float(a[1]), float(a[2]) if len(a) > 2 else 0.0)
        b3 = (float(b[0]), float(b[1]), float(b[2]) if len(b) > 2 else 0.0)
        return Segment(name, a3, b3)

    return {
        1: seg("Big rect. left bottom", (-hl, -by), (-bx, -by)),
        2: seg("Big rect. left main", (-bx, by), (-bx, -by)),
        3: seg("Big rect. left top", (-bx, by), (-hl, by)),
        4: seg("Big rect. right bottom", (bx, -by), (hl, -by)),
        5: seg("Big rect. right main", (bx, by), (bx, -by)),
        6: seg("Big rect. right top", (bx, by), (hl, by)),
        7: seg("Goal left crossbar", (-hl, -gy, gz), (-hl, gy, gz)),
        8: seg("Goal left post right", (-hl, -gy, 0), (-hl, -gy, gz)),
        9: seg("Goal left post left", (-hl, gy, 0), (-hl, gy, gz)),
        10: seg("Goal right crossbar", (hl, -gy, gz), (hl, gy, gz)),
        11: seg("Goal right post left", (hl, gy, 0), (hl, gy, gz)),
        12: seg("Goal right post right", (hl, -gy, 0), (hl, -gy, gz)),
        13: seg("Middle line", (0, hw), (0, -hw)),
        14: seg("Side line bottom", (-hl, -hw), (hl, -hw)),
        15: seg("Side line left", (-hl, hw), (-hl, -hw)),
        16: seg("Side line right", (hl, hw), (hl, -hw)),
        17: seg("Side line top", (-hl, hw), (hl, hw)),
        18: seg("Small rect. left bottom", (-hl, -sy), (-sx, -sy)),
        19: seg("Small rect. left main", (-sx, -sy), (-sx, sy)),
        20: seg("Small rect. left top", (-sx, sy), (-hl, sy)),
        21: seg("Small rect. right bottom", (sx, -sy), (hl, -sy)),
        22: seg("Small rect. right main", (sx, -sy), (sx, sy)),
        23: seg("Small rect. right top", (sx, sy), (hl, sy)),
    }


def sn_line_elements(dims: PitchDims | None = None) -> dict[str, Segment | Arc]:
    """Template element for each of the 26 SoccerNet line classes (SN-GSR, SN-Calibration), internal frame."""
    d = dims or PitchDims()
    hl = d.half_length
    elements: dict[str, Segment | Arc] = {s.name: s for s in pnlcalib_lines(d).values()}
    spot = hl - PENALTY_SPOT_DISTANCE
    half_angle = float(np.degrees(np.arccos((PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE) / CIRCLE_RADIUS)))
    elements["Circle central"] = Arc("Circle central", (0.0, 0.0), CIRCLE_RADIUS, 0.0, 360.0)
    elements["Circle left"] = Arc("Circle left", (-spot, 0.0), CIRCLE_RADIUS, -half_angle, half_angle)
    elements["Circle right"] = Arc(
        "Circle right", (spot, 0.0), CIRCLE_RADIUS, 180 - half_angle, 180 + half_angle
    )
    return elements


def is_planar_sn_line(name: str) -> bool:
    """True for SoccerNet line classes lying on the ground (all but goal posts and crossbars)."""
    return not name.startswith("Goal ")
