"""Hero loop of the showcase page: predfoot on real public footage, SoccerTrack v2 match 117093.

    uv run python scripts/render_hero.py --start 13.0 --seconds 8 --show-from 14.2 \
        --mp4 assets/showcase/hero.mp4 --poster assets/showcase/hero.jpg

(reports/showcase_visuals.md: how the committed loop was made and what it does not show.)

What it shows, without words. Top: a virtual broadcast camera that follows the play inside SoccerTrack v2's
fixed panorama (CC BY 4.0, Scott et al.; :mod:`predfoot.demo.virtual_camera`, as scripts/render_demo.py),
rendered at every clip frame, with predfoot's calibrated pitch lines and each player's feet (the bottom
centre of SoccerTrack's annotated MOT boxes: predfoot does not detect players). Bottom: the same players on a
top-down pitch, each with its calibrated 90 % zone, and the part of the pitch the camera sees. The only text
is the credit; the page around the video explains it.

How. PnLCalib runs on keyframes only, one clip frame in five as in the evaluation, at most
:data:`MAX_KEYFRAMES`; then the robust solver and the PTZ smoother with the noise parameters of the SN-GSR dev
evaluation (``src/predfoot/demo/dev_calibration.json``). A keyframe gets the camera the smoother publishes for
it. A frame between two keyframes whose cameras are smoothed states of one segment gets the smoother's
estimate at its own time: the cubic Hermite interpolation of the smoothed pan, tilt, roll and log focal and
of their rates, which is the RTS smoother's mean under its constant-velocity model; the covariance is
interpolated linearly between the two keyframes (an approximation). Any other frame between keyframes keeps
the nearest keyframe's camera ("held", counted in metrics.json). The 90 % zones use the Mondrian quantiles of
the dev evaluation, as render_demo.py does: a calibration transferred from SN-GSR broadcast frames to these
views. A player whose frame has no camera keeps a grey ring at the feet and gets no position and no zone.

The loop illustrates the output on real pixels; it is not evidence of accuracy (reports/soccertrack_v2.md:
the dataset's own calibration is metres off, so there is no reference to score against here).

Writes runs/<date>_<sha>_hero/ (metrics.json and detections.json, the keyframe detections) and the MP4 and
poster asked for. ``--detections`` reuses the detections an earlier run saved for the same views (same match,
virtual camera settings and followed span): the network only runs on the keyframes they lack.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any

import cv2
import matplotlib
import numpy as np
import polars as pl
import typer
from PIL import Image, ImageDraw, ImageFont

from predfoot.calib import temporal as tm
from predfoot.calib.observations import observations_from_detections
from predfoot.calib.solver import SolverConfig
from predfoot.calib.solver import calibrate as solve
from predfoot.contracts import CalibResult
from predfoot.demo import calibration as dc
from predfoot.demo import virtual_camera as vc
from predfoot.geometry import homography as hg
from predfoot.geometry import ptz
from predfoot.io import soccertrack as st
from predfoot.paths import REPO_ROOT, data_dir
from predfoot.uncertainty.conformal import MondrianConformal
from predfoot.uncertainty.delta import CHI2_2_90, position_covariance


def load_script(name: str) -> ModuleType:
    """A sibling script as a module (scripts/ is not a package), shared with any earlier import of it."""
    key = f"scripts_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


demo = load_script("render_demo")

app = typer.Typer(add_completion=False)

CLIP_FPS = 25.0
KEY_STEP = 5  # one clip frame in five, as the evaluation
MAX_KEYFRAMES = 60
CREDIT = "SoccerTrack v2 (CC BY 4.0)"
#: Flags whose output camera is the smoother's state (the frames a Hermite step may join).
SMOOTHED_FLAGS = ("smoothed", "repaired", "filled", "bridged")
FLAG_NAMES = ("smoothed", "repaired", "filled", "bridged", "projected", "passthrough", "duplicate",
              "rejected", "unchecked")  # fmt: skip

# Colours, BGR. Aqua is predfoot's camera (lines in the view, footprint on the map); amber is the players.
CAMERA_BGR = (232, 212, 95)  # #5fd4e8
PLAYER_BGR = (61, 194, 255)  # #ffc23d
OUTLINE_BGR = (24, 24, 24)
ABSTAIN_BGR = (190, 190, 190)
STRIPES_BGR = ((58, 89, 35), (64, 96, 40))  # #23593a, #286040: mown bands of the map's pitch
SURROUND_BGR = (46, 68, 27)  # #1b442e
SEPARATOR_BGR = (20, 30, 14)


# --- the smoothed camera between keyframes -----------------------------------------------------------------


def hermite(
    x0: np.ndarray, v0: np.ndarray, x1: np.ndarray, v1: np.ndarray, dt: float, s: float
) -> np.ndarray:
    """Cubic Hermite interpolation at fraction ``s`` ∈ [0, 1] of a step of ``dt`` seconds between states
    (value, rate) ``(x0, v0)`` and ``(x1, v1)``."""
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return np.asarray(h00 * x0 + h10 * dt * v0 + h01 * x1 + h11 * dt * v1, dtype=np.float64)


@dataclass(frozen=True)
class Smoothed:
    """What the PTZ smoother made of the keyframes, with the projection centre it used."""

    results: list[CalibResult]
    flags: tm.FrameFlags
    track: tm.SmoothedTrack
    centre: np.ndarray | None


def smooth_keyframes(
    frames: np.ndarray, raw: list[CalibResult], duplicate: np.ndarray, noise: tm.NoiseParams
) -> Smoothed:
    """:func:`predfoot.calib.temporal.smooth_results` step by step (same calls, same order), keeping the
    projection centre, which the cameras between keyframes need."""
    cfg = tm.SmootherConfig()
    own = tm.estimate_centre(raw, duplicate, cfg)
    meas, _ = tm.measure_at_centre("clip", frames, raw, duplicate, own, None, cfg)
    track = tm.smooth_clip(meas, noise, cfg) if meas.accepted.any() else tm.empty_track(meas, noise, cfg)
    results, flags = tm.to_calib_results(raw, meas, track, noise, cfg)
    return Smoothed(results, flags, track, None if meas.centre is None else meas.centre.centre)


@dataclass(frozen=True)
class Camera:
    """The camera predfoot publishes at one frame (keyframe or not)."""

    H: np.ndarray
    cov_theta: np.ndarray
    source: str  # covariance source of the conformal model: smoother, projected or frame
    n_inliers: int
    how: str  # keyframe, interpolated or held


def is_smoothed(flags: tm.FrameFlags, k: int) -> bool:
    return any(bool(getattr(flags, name)[k]) for name in SMOOTHED_FLAGS) and not bool(flags.duplicate[k])


def keyframe_sources(flags: tm.FrameFlags) -> list[str]:
    """Covariance source of each keyframe's camera: :func:`predfoot.demo.calibration.cov_sources`, except that
    a bridged frame takes the ``smoother`` model, as a filled one does there: both carry the smoothed state
    across a missing measurement (that rule predates bridging and would score the smoothed covariance of a
    bridged frame with the per-frame quantiles, about ten times too small)."""
    return ["smoother" if flags.bridged[k] and not flags.duplicate[k] else src
            for k, src in enumerate(dc.cov_sources(flags))]  # fmt: skip


def camera_at(frame: int, keys: np.ndarray, sm: Smoothed, sources: list[str]) -> Camera | None:
    """predfoot's camera at a clip ``frame`` inside the keyframe span (None: it abstains there)."""
    j = int(np.searchsorted(keys, frame))
    if j < len(keys) and keys[j] == frame:
        r = sm.results[j]
        if not (r.ok and r.H is not None and r.cov_theta is not None):
            return None
        return Camera(r.H, np.asarray(r.cov_theta), sources[j], r.n_inliers, "keyframe")
    if j == 0 or j == len(keys):
        raise ValueError(f"frame {frame} is outside the keyframes {keys[0]}-{keys[-1]}")
    a, b = j - 1, j
    s = (frame - keys[a]) / (keys[b] - keys[a])
    near = a if s <= 0.5 else b
    seg = sm.flags.segment
    if (
        sm.centre is not None
        and is_smoothed(sm.flags, a)
        and is_smoothed(sm.flags, b)
        and seg[a] == seg[b] >= 0
        and np.isfinite(sm.track.mean[[a, b]]).all()
    ):
        dt = (keys[b] - keys[a]) / CLIP_FPS
        ma, mb = sm.track.mean[a], sm.track.mean[b]
        phi = hermite(ma[:4], ma[4:], mb[:4], mb[4:], dt, s)
        p = (1 - s) * sm.track.cov[a, :4, :4] + s * sm.track.cov[b, :4, :4]
        pp = tm.SmootherConfig().principal_point
        jac = ptz.jacobian_theta_phi(phi, sm.centre, pp)
        cov = jac @ p @ jac.T
        n_inl = sm.results[near].n_inliers
        return Camera(
            ptz.homography(phi, sm.centre, pp), 0.5 * (cov + cov.T), "smoother", n_inl, "interpolated"
        )
    held = camera_at(int(keys[near]), keys, sm, sources)
    return None if held is None else Camera(held.H, held.cov_theta, held.source, held.n_inliers, "held")


@dataclass(frozen=True)
class Placed:
    """Players placed on the pitch by one camera: position (n, 2) and calibrated 90 % covariance (n, 2, 2),
    for the feet whose ``valid`` is set."""

    xy: np.ndarray
    cov: np.ndarray
    valid: np.ndarray

    def radius(self) -> np.ndarray:
        """Largest 90 % radius of each zone (m)."""
        return np.sqrt(CHI2_2_90 * np.linalg.eigvalsh(self.cov)[:, -1])


def place(cam: Camera, feet: np.ndarray, model: MondrianConformal) -> Placed:
    """Delta-method positions of foot pixels (n, 2) and their calibrated 90 % covariance, as render_demo.py:
    Σ_cal = Σ q̂ / χ²₂(0.90), q̂ the Mondrian quantile of the player's bin (metres per pixel, inliers)."""
    if not len(feet):
        return Placed(np.zeros((0, 2)), np.zeros((0, 2, 2)), np.zeros(0, dtype=bool))
    pred, cov, valid = position_covariance(cam.H, cam.cov_theta, feet)
    mpp = np.sqrt(np.abs(np.linalg.det(hg.jacobian_backproject_uv(cam.H, feet))))
    qhat = model.quantile_for({"mpp": mpp, "n_inliers": np.full(len(feet), float(cam.n_inliers))})
    cov_cal = cov * (np.asarray(qhat, dtype=np.float64) / CHI2_2_90)[:, None, None]
    ok = np.asarray(valid, dtype=bool) & np.isfinite(pred).all(axis=1) & np.isfinite(cov_cal).all(axis=(1, 2))
    return Placed(np.asarray(pred, dtype=np.float64), cov_cal, ok)


# --- drawing -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PitchMap:
    """Top-down pitch in the internal frame (far touchline at the top, as seen from the main stand)."""

    width: int
    margin_m: tuple[float, float] = (2.5, 2.5)
    length: float = 105.0
    breadth: float = 68.0

    @property
    def scale(self) -> float:
        """Pixels per metre."""
        return self.width / (self.length + 2 * self.margin_m[0])

    @property
    def height(self) -> int:
        return round((self.breadth + 2 * self.margin_m[1]) * self.scale)

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        """Pitch points (n, 2), metres → map pixels (pixel centres on integers)."""
        p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        u = (p[:, 0] + self.length / 2 + self.margin_m[0]) * self.scale - 0.5
        v = (self.breadth / 2 + self.margin_m[1] - p[:, 1]) * self.scale - 0.5
        return np.column_stack([u, v])

    def ground(self, pad: int = 0) -> np.ndarray:
        """Pitch coordinates (m) of every pixel centre of the map grown by ``pad`` pixels, (h, w, 2)."""
        u = np.arange(-pad, self.width + pad, dtype=np.float64)
        v = np.arange(-pad, self.height + pad, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        x = (uu + 0.5) / self.scale - self.length / 2 - self.margin_m[0]
        y = self.breadth / 2 + self.margin_m[1] - (vv + 0.5) / self.scale
        return np.stack([x, y], axis=-1)

    def background(self) -> np.ndarray:
        """Surround, mown bands and markings (the markings blended at 60 %)."""
        img = np.empty((self.height, self.width, 3), np.uint8)
        img[:] = SURROUND_BGR
        g = self.ground()
        on = (np.abs(g[..., 0]) <= self.length / 2) & (np.abs(g[..., 1]) <= self.breadth / 2)
        band = np.floor((g[..., 0] + self.length / 2) / (self.length / 14)).astype(int) % 2
        for b in (0, 1):
            img[on & (band == b)] = STRIPES_BGR[b]
        lines = img.copy()
        for line in demo.pitch_polylines():
            pts = np.round(self.to_px(line) * 4).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(lines, [pts], False, (255, 255, 255), 2, cv2.LINE_AA, shift=2)
        return cv2.addWeighted(lines, 0.6, img, 0.4, 0)

    def ellipse(self, cov_m: np.ndarray) -> tuple[tuple[float, float], float]:
        """Half axes (px) and angle (degrees, cv2 convention) of the 90 % zone of a calibrated covariance."""
        j = np.diag([self.scale, -self.scale])
        vals, vecs = np.linalg.eigh(j @ cov_m @ j)
        half = np.sqrt(CHI2_2_90 * np.clip(vals, 0.0, None))
        return (float(half[1]), float(half[0])), math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))


def view_to_display(scale: float, crop_top: int) -> np.ndarray:
    """Pixels of the 1920 × 1080 view → pixels of the displayed view (top ``crop_top`` rows removed, then
    resized by ``scale`` with pixel centres kept aligned)."""
    off = 0.5 * scale - 0.5
    return np.array([[scale, 0.0, off], [0.0, scale, off - scale * crop_top], [0.0, 0.0, 1.0]])


def footprint(grid: np.ndarray, H_disp: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Mask (h, w) of the map pixels whose ground point the displayed view sees."""
    q = grid @ H_disp.T
    front = q[..., 2] > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    return np.asarray(front & (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1]), dtype=bool)


def blend(img: np.ndarray, layer: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    out = img.astype(np.float32) * (1 - a) + layer.astype(np.float32) * a
    return np.asarray(np.clip(out + 0.5, 0, 255), dtype=np.uint8)


def draw_map(
    pmap: PitchMap,
    background: np.ndarray,
    grid_padded: np.ndarray,
    pad: int,
    cam_disp: np.ndarray | None,
    view_size: tuple[int, int],
    placed: Placed | None,
) -> np.ndarray:
    """One frame of the map: the camera's footprint, the players' 90 % zones (their union shaded) and their
    positions."""
    img = cv2.copyMakeBorder(background, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    if cam_disp is not None:
        mask = footprint(grid_padded, cam_disp, view_size)
        img = blend(img, np.full_like(img, 255), 0.13 * mask.astype(np.float32))
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge = img.copy()
        cv2.drawContours(edge, contours, -1, CAMERA_BGR, 2, cv2.LINE_AA)
        img = cv2.addWeighted(edge, 0.85, img, 0.15, 0)
    img = img[pad:-pad, pad:-pad].copy() if pad else img
    if placed is None or not placed.valid.any():
        return img
    xy, cov = placed.xy[placed.valid], placed.cov[placed.valid]
    centres = pmap.to_px(xy)
    zones = np.zeros(img.shape[:2], np.uint8)
    shapes = []
    for c, s in zip(centres, cov, strict=True):
        axes, angle = pmap.ellipse(s)
        axes = (min(axes[0], 4 * pmap.width), min(axes[1], 4 * pmap.width))
        ci = (round(c[0] * 16), round(c[1] * 16))
        ai = (max(round(axes[0] * 16), 1), max(round(axes[1] * 16), 1))
        shapes.append((ci, ai, angle))
        cv2.ellipse(zones, ci, ai, angle, 0, 360, 255, -1, cv2.LINE_AA, shift=4)
    img = blend(img, np.full_like(img, PLAYER_BGR), 0.32 * zones.astype(np.float32) / 255)
    rims = img.copy()
    for ci, ai, angle in shapes:
        cv2.ellipse(rims, ci, ai, angle, 0, 360, PLAYER_BGR, 1, cv2.LINE_AA, shift=4)
    img = cv2.addWeighted(rims, 0.8, img, 0.2, 0)
    for c in centres:
        ci = (round(c[0] * 16), round(c[1] * 16))
        cv2.circle(img, ci, 5 * 16, OUTLINE_BGR, -1, cv2.LINE_AA, shift=4)
        cv2.circle(img, ci, 4 * 16, PLAYER_BGR, -1, cv2.LINE_AA, shift=4)
    return img


def draw_view(
    view_disp: np.ndarray, cam_disp: np.ndarray | None, feet_disp: np.ndarray, line_px: int = 3
) -> np.ndarray:
    """The displayed view with predfoot's pitch lines and the players' feet (grey rings without a camera)."""
    img = view_disp.copy()
    if cam_disp is not None:
        lines = img.copy()
        demo.draw_pitch_lines(lines, cam_disp, CAMERA_BGR, line_px)
        img = cv2.addWeighted(lines, 0.9, img, 0.1, 0)
    for p in feet_disp:
        ci = (round(p[0] * 16), round(p[1] * 16))
        if cam_disp is None:
            cv2.circle(img, ci, 4 * 16, ABSTAIN_BGR, 2, cv2.LINE_AA, shift=4)
        else:
            cv2.circle(img, ci, 6 * 16, OUTLINE_BGR, -1, cv2.LINE_AA, shift=4)
            cv2.circle(img, ci, 5 * 16, PLAYER_BGR, -1, cv2.LINE_AA, shift=4)
    return img


def credit_sprite(text: str, height_px: int) -> tuple[np.ndarray, np.ndarray]:
    """The credit as a BGR patch and its alpha (h, w), light text with a soft dark shadow."""
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), height_px)
    left, top, right, bottom = font.getbbox(text)
    w, h = right - left + 8, bottom - top + 8
    alpha = Image.new("L", (w, h), 0)
    shadow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(shadow).text((4 - left + 1, 4 - top + 1), text, font=font, fill=200)
    ImageDraw.Draw(alpha).text((4 - left, 4 - top), text, font=font, fill=235)
    a_text = np.asarray(alpha, dtype=np.float32) / 255
    a_shadow = cv2.GaussianBlur(np.asarray(shadow, dtype=np.float32) / 255, (0, 0), 1.0)
    a = np.clip(a_text + a_shadow * (1 - a_text), 0, 1)  # light text over its dark shadow
    grey = 245 * a_text / np.maximum(a, 1e-6)
    return np.repeat(np.clip(grey, 0, 255).astype(np.uint8)[..., None], 3, axis=2), a


def stamp(img: np.ndarray, sprite: tuple[np.ndarray, np.ndarray], right: int, bottom: int) -> None:
    """Alpha-blend the sprite with its bottom-right corner at (right, bottom), in place."""
    patch, a = sprite
    h, w = a.shape
    region = img[bottom - h : bottom, right - w : right]
    region[:] = blend(region, patch, a)


def stack(view: np.ndarray, pitch: np.ndarray, gap: int = 3) -> np.ndarray:
    """The view above the map, separated by a thin dark rule; one row is dropped if the height is odd."""
    rule = np.full((gap, view.shape[1], 3), SEPARATOR_BGR, np.uint8)
    out = np.concatenate([view, rule, pitch], axis=0)
    return out[: out.shape[0] - out.shape[0] % 2]


# --- inputs ------------------------------------------------------------------------------------------------


def key_frames(first: int, n: int, step: int = KEY_STEP) -> np.ndarray:
    """Keyframes from ``first`` every ``step`` frames, enough for every one of the ``n`` frames to lie
    between two of them (the last keyframe may come after the last frame shown)."""
    return first + step * np.arange(math.ceil((n - 1) / step) + 1)


def crop_rows(phis: np.ndarray, pano: vc.PanoramaCamera, factor: int = 4, limit: float = 0.3) -> int:
    """Rows to cut from the top of every view so that no view shows the black area above the panorama's
    coverage: the lowest uncovered row within the top ``limit`` of any view, on views rendered ``factor``
    times smaller, plus one coarse row."""
    worst = 0
    for phi in phis:
        small = phi.copy()
        small[3] -= math.log(factor)
        view = vc.VirtualView(small, (vc.BROADCAST_SIZE[0] // factor, vc.BROADCAST_SIZE[1] // factor))
        tx, ty = (t[: round(limit * view.size[1])] for t in vc.remap_tables(view, pano))
        w, h = pano.size
        covered = (tx >= 0) & (tx <= w - 1) & (ty >= 0) & (ty <= h - 1)
        rows = np.flatnonzero(~covered.all(axis=1))
        if len(rows):
            worst = max(worst, int(rows[-1]) + 2)
    return worst * factor


FOLLOW_PAD = 50  # frames of play the virtual camera also follows before and after the keyframes (2 s)


def follow_span(
    mot: pl.DataFrame, pano: vc.PanoramaCamera, follow: vc.FollowConfig, lo: int, hi: int
) -> dict[int, np.ndarray]:
    """PTZ parameters of the virtual camera at clip frames ``lo``…``hi``: it follows the play over that span
    (aim and zoom smoothed over time, see :func:`predfoot.demo.virtual_camera.follow_action`)."""
    span = mot.filter(pl.col("frame").is_between(lo, hi))
    frames = span["frame"].to_numpy()
    xy = hg.backproject(pano.homography(), pano.video_to_rectified(st.foot_points(span)))[0]
    cut = np.searchsorted(frames, np.arange(lo, hi + 2))
    feet = [xy[cut[k] : cut[k + 1]] for k in range(hi - lo + 1)]
    phis = vc.follow_action(feet, pano.calib.C, fps=CLIP_FPS, config=follow)
    return {lo + k: phi for k, phi in enumerate(phis)}


def view_key(match: str, follow: vc.FollowConfig, lo: int, hi: int) -> dict[str, Any]:
    """What fixes the virtual views, hence the pixels of their detections: match, camera settings, span."""
    return json.loads(json.dumps({"match": match, "follow": follow.__dict__, "span": [lo, hi]}))


def save_detections(path: Path, key: dict[str, Any], dets: dict[int, dict[str, Any]]) -> None:
    rows = [{"frame": f} | {k: np.asarray(v).tolist() for k, v in d.items()} for f, d in sorted(dets.items())]
    path.write_text(json.dumps({"views": key, "detections": rows}))


def load_detections(path: Path, key: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Detections saved by :func:`save_detections`, by frame; ValueError when they belong to other views."""
    d = json.loads(path.read_text())
    if d.get("views") != key:
        raise ValueError(f"{path} holds detections of other views: {d.get('views')}, not {key}")
    shapes = {"kp_ids": (-1,), "kp_uv": (-1, 2), "kp_conf": (-1,), "line_ids": (-1,), "line_uv": (-1, 2, 2)}
    return {
        int(row["frame"]): {
            k: np.asarray(row[k], dtype=np.int64 if k.endswith("ids") else np.float64).reshape(shape)
            for k, shape in shapes.items()
        }
        for row in d["detections"]
    }


def read_frames(path: Path, first: int, last: int) -> Any:
    """Decoded frames ``first``…``last`` of a video, in order (one seek, then sequential reads)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    try:
        for f in range(first, last + 1):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"cannot read frame {f} of {path}")
            yield f, img
    finally:
        cap.release()


@app.command()
def main(
    match: Annotated[
        str, typer.Option(help="SoccerTrack v2 match (raw/, mot/ and its MOT clip on disk).")
    ] = "117093",
    start: Annotated[float, typer.Option(help="First second processed, in the MOT clip (25 fps).")] = 13.0,
    seconds: Annotated[float, typer.Option(help="Length processed (s): keyframes, smoother.")] = 8.0,
    show_from: Annotated[
        float | None, typer.Option(help="First second of the loop (default: --start); it ends with the span.")
    ] = None,
    render_step: Annotated[int, typer.Option(help="Clip frames between frames of the MP4 (1: 25 fps).")] = 1,
    width: Annotated[int, typer.Option(help="Width of the MP4 (px).")] = 720,
    crf: Annotated[int, typer.Option(help="x264 constant rate factor (higher: smaller, blurrier).")] = 29,
    device: Annotated[str, typer.Option(help="Torch device for PnLCalib.")] = "auto",
    calibration: Annotated[
        Path, typer.Option(help="Dev noise parameters and conformal models (JSON).")
    ] = dc.DEFAULT_PATH,
    hfov: Annotated[tuple[float, float], typer.Option(help="Horizontal field of view range, degrees.")] = (
        25.0,
        45.0,
    ),
    aim_y: Annotated[
        float, typer.Option(help="Metres added to the aim point's y (negative: nearer).")
    ] = -6.0,
    detections: Annotated[
        Path | None, typer.Option(help="detections.json of an earlier run, reused for its keyframes.")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Run folder (default runs/<date>_<sha>_hero).")] = None,
    mp4: Annotated[Path | None, typer.Option(help="Write the loop here (H.264).")] = None,
    poster: Annotated[Path | None, typer.Option(help="Write the middle frame here (JPEG).")] = None,
    poster_quality: Annotated[int, typer.Option(help="JPEG quality of the poster.")] = 84,
    ffmpeg: Annotated[str | None, typer.Option(help="ffmpeg executable (default: the one on PATH).")] = None,
) -> None:
    t0 = time.perf_counter()
    weights = data_dir() / "weights" / "pnlcalib"
    needed = demo.missing_inputs(match, weights, calibration)
    missing = [p for p in needed if weights not in p.parents]
    if detections is not None and not detections.is_file():
        missing.append(detections)
    if missing:
        listing = "\n  ".join(map(str, missing))
        typer.echo(f"missing inputs, nothing was run:\n  {listing}\n{demo.DOWNLOAD_HINT.format(match=match)}",
                   err=True)  # fmt: skip
        raise typer.Exit(code=2)
    cal = dc.load(calibration)["ptz"]
    assert cal.noise is not None  # checked when the calibration file is loaded
    first, n = round(start * CLIP_FPS), round(seconds * CLIP_FPS)
    keys = key_frames(first, n)
    if len(keys) > MAX_KEYFRAMES:
        raise typer.BadParameter(f"{len(keys)} keyframes: at most {MAX_KEYFRAMES} (shorten --seconds)")
    first_shown = first if show_from is None else round(show_from * CLIP_FPS)
    if not first <= first_shown < first + n:
        raise typer.BadParameter(f"--show-from {show_from} is outside the span processed")
    shown = np.arange(first_shown, first + n, render_step)
    out = out or REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{demo.git_sha()}_hero"
    out.mkdir(parents=True, exist_ok=True)

    pano = vc.PanoramaCamera(st.load_fisheye(match), *st.load_rectify_maps(match))
    centre = pano.calib.C
    mot = st.load_mot(match)
    follow = vc.FollowConfig(hfov_deg=hfov, aim_offset_m=(0.0, aim_y))
    n_clip = int(mot["frame"].max()) + 1
    if keys[-1] >= n_clip:
        raise typer.BadParameter(f"the loop needs frames up to {keys[-1]}, the clip has {n_clip}")
    lo, hi = max(0, first - FOLLOW_PAD), min(n_clip - 1, int(keys[-1]) + FOLLOW_PAD)
    phis = follow_span(mot, pano, follow, lo, hi)
    views = view_key(match, follow, lo, hi)
    window = mot.filter(pl.col("frame").is_between(first, keys[-1]))
    feet_video = {int(f): st.foot_points(b) for (f,), b in window.partition_by("frame", as_dict=True).items()}
    crop = crop_rows(np.array([phis[int(f)] for f in shown]), pano)
    scale = width / vc.BROADCAST_SIZE[0]
    disp_size = (width, round((vc.BROADCAST_SIZE[1] - crop) * scale))
    to_disp = view_to_display(scale, crop)

    # Keyframes: detections (saved ones reused, the network on the others), robust calibration, PTZ smoother.
    cache = load_detections(detections, views) if detections is not None else {}
    todo = [int(f) for f in keys if int(f) not in cache]
    t_net = 0.0
    if todo:
        if any(weights in p.parents for p in needed):
            typer.echo(f"PnLCalib weights missing, needed for {len(todo)} keyframes:\n"
                       f"{demo.DOWNLOAD_HINT.format(match=match)}", err=True)  # fmt: skip
            raise typer.Exit(code=2)
        from predfoot.keypoints.pnlcalib import PnLCalibNetworks

        nets = PnLCalibNetworks(weights / "SV_kp", weights / "SV_lines", device=device)
        for f, img in read_frames(st.clip_path(match), todo[0], todo[-1]):
            if f in todo:
                rgb = cv2.cvtColor(
                    vc.render(img, vc.remap_tables(vc.VirtualView(phis[f]), pano)), cv2.COLOR_BGR2RGB
                )
                t = time.perf_counter()
                cache[f] = demo.detections_of(nets, rgb)
                t_net += time.perf_counter() - t
    save_detections(out / "detections.json", views, cache)
    dets = [cache[int(f)] for f in keys]
    raw: list[CalibResult] = []
    for f, det in zip(keys, dets, strict=True):
        xy, uv, w, lines = observations_from_detections(det)
        rng = np.random.default_rng(int(match) * 100_003 + int(f))
        raw.append(solve(xy, uv, rng, weights=w, config=SolverConfig(), lines=lines)[0])
    sm = smooth_keyframes(keys, raw, tm.duplicate_mask(dets), cal.noise)
    sources = keyframe_sources(sm.flags)

    # Every frame shown: render, camera, players, map.
    pmap = PitchMap(width)
    background = pmap.background()
    pad = 4
    grid = np.concatenate([pmap.ground(pad), np.ones((*pmap.ground(pad).shape[:2], 1))], axis=-1)
    sprite = credit_sprite(CREDIT, max(11, round(width / 46)))
    hows: list[str] = []
    radii: list[float] = []
    n_players: list[int] = []
    poster_frame = int(shown[len(shown) // 2])
    poster_img = None
    frame_h = disp_size[1] + 3 + pmap.height
    frame_h -= frame_h % 2  # as stack() does
    writer = demo.Mp4Writer(
        out / "hero.mp4", (width, frame_h), CLIP_FPS / render_step, crf=crf, ffmpeg=ffmpeg
    )
    with writer:
        shown_set = set(shown.tolist())
        for f, img in read_frames(st.clip_path(match), int(shown[0]), int(shown[-1])):
            if f not in shown_set:
                continue
            view = vc.VirtualView(phis[f])
            rendered = vc.render(img, vc.remap_tables(view, pano))
            view_disp = cv2.resize(rendered[crop:], disp_size, interpolation=cv2.INTER_AREA)
            feet = vc.video_to_view(feet_video.get(f, np.zeros((0, 2))), view, pano)
            feet_disp = hg.project(to_disp, feet) if len(feet) else np.zeros((0, 2))
            seen = demo.inside(feet_disp, disp_size)
            cam = camera_at(f, keys, sm, sources)
            hows.append("abstained" if cam is None else cam.how)
            cam_disp = None if cam is None else hg.normalise(to_disp @ cam.H)
            placed = None if cam is None else place(cam, feet[seen], cal.model_for(cam.source))
            if placed is not None:
                radii += placed.radius()[placed.valid].tolist()
                n_players.append(int(placed.valid.sum()))
            top = draw_view(view_disp, cam_disp, feet_disp[seen])
            stamp(top, sprite, width - 6, disp_size[1] - 4)
            bottom = draw_map(pmap, background, grid, pad, cam_disp, disp_size, placed)
            frame = stack(top, bottom)
            writer.write(frame)
            if f == poster_frame:
                poster_img = frame
        mp4_bytes = writer.close()

    if mp4 is not None:
        mp4.parent.mkdir(parents=True, exist_ok=True)
        mp4.write_bytes((out / "hero.mp4").read_bytes())
    poster_bytes = None
    if poster is not None and poster_img is not None:
        poster_bytes = demo.write_still(poster_img, poster, width=width, quality=poster_quality)
    counts = {h: hows.count(h) for h in ("keyframe", "interpolated", "held", "abstained")}
    size = [width, None if poster_img is None else int(poster_img.shape[0])]
    radius = {"p50": float(np.median(radii)), "p90": float(np.quantile(radii, 0.9))} if radii else None
    metrics = {
        "match": match, "commit": demo.git_sha(), "frames_processed": [first, first + n - 1],
        "frames_shown": [int(shown[0]), int(shown[-1])],
        "n_frames_shown": len(shown), "fps": CLIP_FPS / render_step, "keyframes": keys.tolist(),
        "n_keyframes_network": len(todo), "detections_reused_from": str(detections) if detections else None,
        "views": views, "crop_top_rows": crop, "size": size, "segment": sm.flags.segment.tolist(),
        "raw_calibrated": [r.ok for r in raw], "n_inliers": [r.n_inliers for r in raw],
        "rms_px": [r.rms_px for r in raw], "published": [r.ok for r in sm.results],
        "flags": {name: np.asarray(getattr(sm.flags, name), dtype=bool).tolist() for name in FLAG_NAMES},
        "cov_sources": sources, "centre": None if sm.centre is None else sm.centre.tolist(),
        "panorama_centre": np.asarray(centre).tolist(), "frames_by_camera": counts,
        "players_per_frame": {"min": min(n_players, default=0), "max": max(n_players, default=0)},
        "zone_radius_90_m": radius, "mp4_bytes": mp4_bytes, "poster_bytes": poster_bytes, "crf": crf,
        "network_seconds": t_net, "total_seconds": time.perf_counter() - t0,
        "notes": ["foot points are SoccerTrack's annotated MOT boxes, not detections",
                  "zones borrow the SN-GSR dev quantiles: an illustration, not a measured coverage"],
    }  # fmt: skip
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    keep = ("frames_by_camera", "n_keyframes_network", "zone_radius_90_m", "crop_top_rows", "size",
            "mp4_bytes", "poster_bytes", "total_seconds")  # fmt: skip
    typer.echo(json.dumps({k: metrics[k] for k in keep}, indent=2, default=float))
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
