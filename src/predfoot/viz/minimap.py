"""Minimap: player positions with their 90 % regions, drawn in the internal pitch frame (ADR-0001).

x points right (towards the right-hand goal of the main camera) and y points up (towards the far touchline),
so the picture is the pitch as seen from the main camera's stand, from above. Each covariance is drawn as its
90 % ellipse, the set {p : (p - x̂)ᵀ Σ⁻¹ (p - x̂) ≤ χ²₂(0.90)}; with predfoot's calibrated ``pos_cov_m2``
that ellipse is the prediction region. Abstained frames are greyed out and show no positions.

Colours come from a palette checked for colour-vision deficiencies (blue/orange, every pair) on a light
surface; pitch markings stay recessive.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, Patch, Polygon, Rectangle
from numpy.typing import ArrayLike

from predfoot.io.files import atomic_write_bytes
from predfoot.io.skillcorner import DEFAULT_PITCH_LENGTH_M, DEFAULT_PITCH_WIDTH_M, Team, TrackingFrame

#: χ²₂ quantile at 0.90, -2 ln(0.1) ≈ 4.605: the squared Mahalanobis radius of a 2-D 90 % ellipse.
CHI2_2_090 = -2.0 * math.log(0.10)

SURFACE = "#fcfcfb"
PITCH_LINE = "#c3c2b7"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
FOOTPRINT = "#f0efec"
VEIL = "#e1e0d9"
TEAM_COLOURS: dict[str | None, str] = {"home": "#2a78d6", "away": "#eb6834", None: TEXT_SECONDARY}
TEAM_LABELS: dict[str | None, str] = {"home": "home", "away": "away", None: "team unknown"}

# FIFA Laws of the Game, metres.
CENTRE_CIRCLE_R = 9.15
PENALTY_AREA = (16.5, 40.32)  # depth, width
GOAL_AREA = (5.5, 18.32)
PENALTY_SPOT = 11.0


def ellipse_axes(cov: ArrayLike, quantile: float = CHI2_2_090) -> tuple[float, float, float]:
    """Full axis lengths and orientation of the ellipse pᵀ Σ⁻¹ p = ``quantile``.

    Returns (width, height, angle): width along the major axis, angle in degrees counter-clockwise from +x,
    in (-90, 90] (matplotlib's ``Ellipse`` convention when the y axis points up).
    """
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    vals = np.clip(vals, 0.0, None)  # rounding can leave -1e-5 on a degenerate covariance
    vx, vy = vecs[:, 1]
    angle = math.degrees(math.atan2(vy, vx))
    if angle <= -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    return 2.0 * math.sqrt(quantile * vals[1]), 2.0 * math.sqrt(quantile * vals[0]), angle


def select_panels(frames: Sequence[TrackingFrame], n_panels: int) -> list[int]:
    """Indices of up to ``n_panels`` evenly spaced frames, always including an abstained one if any."""
    m = len(frames)
    if m <= n_panels:
        return list(range(m))
    picks = sorted({round(v) for v in np.linspace(0, m - 1, n_panels)})
    abstained = [i for i, f in enumerate(frames) if f.abstain]
    if abstained and not any(frames[i].abstain for i in picks):
        interior = range(1, len(picks) - 1) if len(picks) > 2 else range(len(picks))
        nearest = min(interior, key=lambda k: abs(picks[k] - abstained[0]))
        picks[nearest] = abstained[0]
        picks = sorted(set(picks))
    return picks


def _outline(patch: Patch) -> Patch:
    patch.set(fill=False, edgecolor=PITCH_LINE, linewidth=0.8, zorder=1)
    return patch


def draw_pitch(
    ax: Axes, length: float = DEFAULT_PITCH_LENGTH_M, width: float = DEFAULT_PITCH_WIDTH_M
) -> None:
    """Pitch markings centred on the centre spot, in metres."""
    half_l, half_w = length / 2, width / 2
    ax.add_patch(_outline(Rectangle((-half_l, -half_w), length, width)))
    ax.plot([0.0, 0.0], [-half_w, half_w], color=PITCH_LINE, linewidth=0.8, zorder=1)
    ax.add_patch(_outline(Circle((0.0, 0.0), CENTRE_CIRCLE_R)))
    for sign in (-1.0, 1.0):
        goal_line = sign * half_l
        for depth, box_width in (PENALTY_AREA, GOAL_AREA):
            x0 = goal_line if sign < 0 else goal_line - depth
            ax.add_patch(_outline(Rectangle((x0, -box_width / 2), depth, box_width)))
        spot = goal_line - sign * PENALTY_SPOT
        ax.plot([spot], [0.0], marker="o", markersize=1.5, color=PITCH_LINE, zorder=1)
        # The penalty arc is the part of the 9.15 m circle around the spot outside the penalty area.
        half_angle = math.degrees(math.acos((PENALTY_AREA[0] - PENALTY_SPOT) / CENTRE_CIRCLE_R))
        centre_angle = 0.0 if sign < 0 else 180.0
        ax.add_patch(
            Arc(
                (spot, 0.0),
                2 * CENTRE_CIRCLE_R,
                2 * CENTRE_CIRCLE_R,
                theta1=centre_angle - half_angle,
                theta2=centre_angle + half_angle,
                edgecolor=PITCH_LINE,
                linewidth=0.8,
                zorder=1,
            )
        )
    ax.plot([0.0], [0.0], marker="o", markersize=1.5, color=PITCH_LINE, zorder=1)


def draw_frame(
    ax: Axes,
    frame: TrackingFrame,
    *,
    pitch_length: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width: float = DEFAULT_PITCH_WIDTH_M,
    team_of: Mapping[int, Team] | None = None,
) -> None:
    """One frame: camera footprint, pitch, 90 % ellipses and positions; a grey veil if the frame abstains."""
    team_of = team_of or {}
    ax.set_facecolor(SURFACE)
    if frame.corners is not None and not frame.abstain:
        ax.add_patch(
            Polygon(
                frame.corners,
                closed=True,
                facecolor=FOOTPRINT,
                edgecolor=PITCH_LINE,
                linewidth=0.8,
                linestyle="--",
                zorder=0.5,
                gid="footprint",
            )
        )
    draw_pitch(ax, pitch_length, pitch_width)
    n_located = 0
    for loc in frame.players:
        if loc.xy is None:
            continue
        n_located += 1
        colour = TEAM_COLOURS[team_of.get(loc.track_id)]
        x, y = (float(v) for v in loc.xy)
        if loc.cov_xy is not None:
            width, height, angle = ellipse_axes(loc.cov_xy)
            ax.add_patch(
                Ellipse(
                    (x, y),
                    width,
                    height,
                    angle=angle,
                    facecolor=colour,
                    alpha=0.18,
                    edgecolor="none",
                    zorder=2,
                    gid="p90",
                )
            )
            ax.add_patch(
                Ellipse((x, y), width, height, angle=angle, fill=False, edgecolor=colour, lw=0.8, zorder=2)
            )
        ax.plot([x], [y], marker="o", markersize=4.5, color=colour, markeredgecolor=SURFACE, zorder=3)
    half_l, half_w = pitch_length / 2, pitch_width / 2
    ax.set_xlim(-half_l - 5.0, half_l + 5.0)
    ax.set_ylim(-half_w - 8.0, half_w + 6.0)  # footprints reach about y = -45 below, are clipped at +39 above
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(VEIL)
    status = "abstained" if frame.abstain else f"{n_located} located"
    health = "" if frame.calib_health is None else f" · health {frame.calib_health:.2f}"
    stamp = f" · {frame.timestamp}" if frame.timestamp else ""
    ax.set_title(f"frame {frame.frame}{stamp} · {status}{health}", fontsize=9, color=TEXT, loc="left")
    if frame.abstain:
        ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=ax.transAxes,
                facecolor=VEIL,
                alpha=0.85,
                edgecolor="none",
                zorder=4,
                gid="abstain-veil",
            )
        )
        ax.text(
            0.5,
            0.5,
            "abstained",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=TEXT_SECONDARY,
            zorder=5,
        )


def render_minimap(
    frames: Sequence[TrackingFrame],
    path: Path,
    *,
    n_panels: int = 6,
    pitch_length: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width: float = DEFAULT_PITCH_WIDTH_M,
    team_of: Mapping[int, Team] | None = None,
    title: str | None = None,
    dpi: int = 100,
) -> Path:
    """Draw evenly spaced frames of a clip in a grid and write a PNG (atomically, byte-stable)."""
    if not frames:
        raise ValueError("no frames to draw")
    picks = select_panels(frames, n_panels)
    ncols = min(3, len(picks))
    nrows = math.ceil(len(picks) / ncols)
    fig = Figure(figsize=(4.6 * ncols, 3.5 * nrows + 0.9), dpi=dpi, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax, i in zip(axes.flat, picks, strict=False):
        draw_frame(ax, frames[i], pitch_length=pitch_length, pitch_width=pitch_width, team_of=team_of)
    for ax in axes.flat[len(picks) :]:
        ax.set_visible(False)
    present = {(team_of or {}).get(loc.track_id) for f in frames for loc in f.players if loc.xy is not None}
    teams_seen = [t for t in TEAM_COLOURS if t in present]  # home, away, unknown
    handles: list[Patch | Line2D] = [
        Line2D([], [], marker="o", linestyle="none", color=TEAM_COLOURS[t], label=TEAM_LABELS[t])
        for t in teams_seen
    ]
    handles += [
        Patch(facecolor=TEXT_SECONDARY, alpha=0.18, edgecolor=TEXT_SECONDARY, label="90 % region"),
        Patch(facecolor=FOOTPRINT, edgecolor=PITCH_LINE, linestyle="--", label="camera footprint"),
        Patch(facecolor=VEIL, edgecolor="none", label="abstained frame"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False, fontsize=9)
    heading = title or "predfoot minimap"
    fig.suptitle(rf"{heading} · 90 % position regions ($\chi^2_2$ = 4.605)", color=TEXT, fontsize=11)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=SURFACE, metadata={"Software": None})
    atomic_write_bytes(Path(path), buffer.getvalue())
    return Path(path)
