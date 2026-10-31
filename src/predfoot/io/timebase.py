"""Time base of the exports (ADR-0001).

The pipeline works on native frame indices (25 fps for SN-GSR); SkillCorner-style exports are on a regular
10 fps grid with ``timestamp`` written ``HH:MM:SS.ss``. This module maps one onto the other.

When the native rate is not a multiple of the export rate (25 → 10 fps), an export instant falls between
two native frames. Picking the nearest native frame would shift samples by up to 20 ms, which is a 20 % error
on a speed measured over 0.1 s, so export frames are interpolated linearly instead
(:func:`interpolate_localization`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction

from predfoot.contracts import Localization

_TIMESTAMP = re.compile(r"^(\d{2}):([0-5]\d):([0-5]\d)\.(\d{2})$")


def format_timestamp(seconds: float) -> str:
    """Format a time in seconds as SkillCorner does: ``HH:MM:SS.ss`` (centiseconds)."""
    if seconds < 0:
        raise ValueError(f"negative time: {seconds}")
    centis = round(seconds * 100)
    hours, rest = divmod(centis, 360_000)
    minutes, rest = divmod(rest, 6_000)
    secs, cs = divmod(rest, 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def parse_timestamp(text: str) -> float:
    """Inverse of :func:`format_timestamp`."""
    match = _TIMESTAMP.match(text)
    if match is None:
        raise ValueError(f"timestamp {text!r} is not HH:MM:SS.ss")
    h, m, s, cs = (int(g) for g in match.groups())
    return 3600 * h + 60 * m + s + cs / 100


@dataclass(frozen=True)
class ExportSlot:
    """Native frames behind one export frame: ``i0`` weighted ``1 - w`` plus ``i1`` weighted ``w``.

    ``w == 0`` (and ``i1 == i0``) when the export instant coincides with a native frame.
    """

    export_frame: int
    i0: int
    i1: int
    w: float


def export_schedule(n_native: int, native_fps: float, export_fps: float) -> list[ExportSlot]:
    """Export frames covering native frames ``0 … n_native - 1``, at ``export_fps``.

    Export frame k is at t = k / export_fps. Rates are converted to fractions from their decimal text, so
    29.97 fps is handled exactly and frame indices never drift over a long clip. The last export frame is the
    last instant that does not need a native frame beyond the clip (no extrapolation).
    """
    if n_native <= 0:
        raise ValueError("a clip needs at least one native frame")
    if native_fps <= 0 or export_fps <= 0:
        raise ValueError("frame rates must be positive")
    ratio = Fraction(str(native_fps)) / Fraction(str(export_fps))  # native frames per export frame
    n_export = math.floor((n_native - 1) / ratio) + 1
    slots = []
    for k in range(n_export):
        s = k * ratio
        i0 = math.floor(s)
        frac = s - i0
        slots.append(ExportSlot(export_frame=k, i0=i0, i1=i0 if frac == 0 else i0 + 1, w=float(frac)))
    return slots


def interpolate_localization(a: Localization, b: Localization, w: float) -> Localization:
    """Localization at an instant between two native frames, ``(1 - w)·a + w·b``.

    Position, covariance and 90 % radius are all interpolated linearly. For the uncertainty this assumes that
    the position errors of two frames 40 ms apart are perfectly correlated (both come from nearly the same
    camera estimate): then the interpolated covariance is exactly that of the interpolated error, and the
    interpolated radius keeps its coverage. With independent errors both would be conservative. A convex
    combination of positive semi-definite matrices stays positive semi-definite.

    If either side has no position (abstention, point behind the horizon), the result has none either.
    """
    if a.track_id != b.track_id:
        raise ValueError(f"cannot interpolate track {a.track_id} with track {b.track_id}")
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"interpolation weight {w} is outside [0, 1]")
    if w == 0.0:
        return a
    detected = a.is_detected and b.is_detected
    if a.xy is None or b.xy is None:
        return Localization(track_id=a.track_id, xy=None, is_detected=detected)
    cov = None if a.cov_xy is None or b.cov_xy is None else (1.0 - w) * a.cov_xy + w * b.cov_xy
    radius = (
        None
        if a.p90_radius_m is None or b.p90_radius_m is None
        else (1.0 - w) * a.p90_radius_m + w * b.p90_radius_m
    )
    return Localization(
        track_id=a.track_id,
        xy=(1.0 - w) * a.xy + w * b.xy,
        cov_xy=cov,
        p90_radius_m=radius,
        is_detected=detected,
    )
