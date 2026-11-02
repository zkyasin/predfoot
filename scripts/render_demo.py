"""Public demo on SoccerTrack v2 pixels (CC BY 4.0, Scott et al.): a virtual broadcast camera that follows the
play inside the fixed panorama, predfoot's camera and positions on it, and the minimap with 90 % regions.

    uv run python scripts/render_demo.py --match 117093 --start 0 --frames 10 --step 5 --fps 5 \\
        --still assets/demo.jpg --gif runs/demo.gif

Per rendered frame: the virtual view (PTZ camera at the panorama's centre, known pitch homography under the
released calibration: the reference camera) is rendered from the MOT clip frame with one remap; PnLCalib
keypoints and lines → robust calibration → by default the PTZ smoother → delta-method positions of the MOT
ground-truth foot points. The smoother's noise parameters and the conformal quantiles of the 90 % regions come
from the SN-GSR dev evaluation (``src/predfoot/demo/dev_calibration.json``, see
:mod:`predfoot.demo.calibration`), not refitted: that transfers a calibration from SN-GSR broadcast frames to
SoccerTrack virtual views, a domain shift, reported as such. ``--no-smooth`` uses the per-frame camera and
the quantiles fitted for it.

Two references are scored on the rendered frames: the reference camera (the released calibration, whose own
error is metres on the far half, see reports/soccertrack_v2.md) and the annotated keypoints mapped into each
view (true correspondences up to annotation noise; the released calibration was fitted on them).

Writes ``runs/<date>_<sha>_demo/``: demo.mp4 (virtual view | minimap), metrics.json, players.parquet,
keypoints.parquet. Every input is checked before anything runs; network inference runs once per rendered
frame.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import polars as pl
import typer
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from predfoot.calib.observations import observations_from_detections
from predfoot.calib.solver import SolverConfig
from predfoot.calib.solver import calibrate as solve
from predfoot.calib.temporal import SmootherConfig, duplicate_mask, smooth_results
from predfoot.contracts import CalibResult, Localization
from predfoot.demo import calibration as dc
from predfoot.demo import virtual_camera as vc
from predfoot.geometry import homography as hg
from predfoot.geometry.pitch import Arc, sn_line_elements
from predfoot.io import soccertrack as st
from predfoot.io.skillcorner import TrackingFrame
from predfoot.paths import REPO_ROOT, data_dir
from predfoot.uncertainty.delta import CHI2_2_90, mahalanobis2, position_covariance
from predfoot.viz.minimap import SURFACE, draw_frame

#: Foot-point colours (BGR) by calibrated 90 % radius: confident, wider, wide; grey when the frame abstains.
RADIUS_BINS_M = (1.0, 2.5)
COLOURS_BGR = ((80, 200, 60), (40, 170, 245), (60, 60, 230))
ABSTAIN_BGR = (150, 150, 150)
#: Camera overlays, in colours the painted (white) lines and the foot points do not use.
OURS_BGR = (255, 0, 255)  # magenta
REFERENCE_BGR = (255, 255, 0)  # cyan
CREDIT = "Pixels: SoccerTrack v2 (CC BY 4.0, Scott et al.), virtual view of the panorama"
FLAG_NAMES = ("smoothed", "repaired", "filled", "projected", "passthrough", "duplicate", "rejected")
CLIP_FPS = 25.0
DOWNLOAD_HINT = (
    "Download them with:\n"
    "  uv run python scripts/download_data.py soccertrack-v2 --match {match} --parts raw mot clip\n"
    "  uv run python scripts/download_data.py pnlcalib-weights"
)

app = typer.Typer(add_completion=False)


def git_sha() -> str:
    """Short commit of the checkout, with ``-dirty`` when tracked files have uncommitted changes."""

    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        return out.stdout.strip()

    sha = git("rev-parse", "--short", "HEAD")
    if not sha:
        return "nogit"
    return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def missing_inputs(match: str, weights: Path, calibration: Path) -> list[Path]:
    """Inputs of a demo run that are not on disk (checked before rendering and network inference)."""
    raw = ("camera_intrinsics.npz", "mapx.npy", "mapy.npy", "keypoints.json")
    needed = [st.raw_path(match, n) for n in raw] + [st.mot_path(match), st.clip_path(match)]
    needed += [weights / "SV_kp", weights / "SV_lines", calibration]
    return [p for p in needed if not p.is_file()]


def pitch_polylines() -> list[np.ndarray]:
    """Ground markings of the internal pitch as dense polylines (metres)."""
    out = []
    for element in sn_line_elements().values():
        if isinstance(element, Arc):
            out.append(element.sample(96))
        elif element.planar:
            t = np.linspace(0.0, 1.0, 40)[:, None]
            out.append(np.asarray(element.a[:2]) * (1 - t) + np.asarray(element.b[:2]) * t)
    return out


def draw_pitch_lines(img: np.ndarray, H: np.ndarray, colour: tuple[int, int, int], thickness: int) -> None:
    for line in pitch_polylines():
        q = np.column_stack([line, np.ones(len(line))]) @ H.T
        front = q[:, 2] > 0
        uv = q[:, :2] / np.where(front, q[:, 2], 1.0)[:, None]
        ok = front & (np.abs(uv) < 1e5).all(axis=1)
        for run in np.split(np.arange(len(uv)), np.flatnonzero(np.diff(ok.astype(int))) + 1):
            if ok[run[0]] and len(run) > 1:
                pts = np.round(uv[run] * 4).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, colour, thickness, cv2.LINE_AA, shift=2)


def inside(uv: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Mask of the finite pixels (n, 2) that lie inside an image of ``size`` (width, height)."""
    return np.isfinite(uv).all(axis=1) & (uv >= 0).all(axis=1) & (uv[:, 0] < size[0]) & (uv[:, 1] < size[1])


def keypoint_rows(
    frame: int,
    names: list[str],
    kp_xy: np.ndarray,
    kp_view: np.ndarray,
    size: tuple[int, int],
    H_ref: np.ndarray,
    H_ours: np.ndarray | None,
) -> list[dict[str, Any]]:
    """One row per annotated keypoint inside the view: error of the reference camera and of ours (None when
    our camera abstains, ``H_ours`` None), in view pixels and in pitch metres, with the keypoint's group."""
    groups, marked = st.keypoint_groups(kp_xy)
    rows = []
    for i in np.flatnonzero(inside(kp_view, size)):
        row: dict[str, Any] = {"frame": frame, "keypoint": names[i], "group": str(groups[i]),
                               "marked": bool(marked[i])}  # fmt: skip
        for who, H in (("ref", H_ref), ("ours", H_ours)):
            px = None if H is None else float(np.linalg.norm(hg.project(H, kp_xy[i : i + 1])[0] - kp_view[i]))
            back = None if H is None else hg.backproject(H, kp_view[i : i + 1])[0][0]
            row[f"{who}_px"] = px
            row[f"{who}_m"] = None if back is None else float(np.linalg.norm(back - kp_xy[i]))
        rows.append(row)
    return rows


def table(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Rows → DataFrame with the schema inferred from every row: by default polars reads the first 100 rows
    only and silently drops keys that first appear later (for example after abstained frames)."""
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def stats(col: str, d: pl.DataFrame) -> dict[str, float] | None:
    if d.is_empty() or col not in d.columns:
        return None
    v = d[col].drop_nulls().to_numpy()
    return {"n": len(v), "p50": float(np.median(v)), "p90": float(np.quantile(v, 0.9))} if len(v) else None


def keypoint_summary(kps: pl.DataFrame) -> dict[str, Any]:
    """Keypoint errors of both cameras, over all keypoints in the views and per group."""

    def one(d: pl.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {"n": d.height}
        for who, label in (("ref", "reference"), ("ours", "ours")):
            out |= {f"{label}_px": stats(f"{who}_px", d), f"{label}_m": stats(f"{who}_m", d)}
        return out

    summary = {"all": one(kps)}
    if not kps.is_empty():
        summary |= {g: one(kps.filter(pl.col("group") == g)) for g in sorted(kps["group"].unique().to_list())}
    return summary


def player_summary(players: pl.DataFrame) -> dict[str, Any] | None:
    """Our positions against the reference camera's, over all player-frames and per half of the pitch."""
    if players.is_empty():
        return None

    def one(d: pl.DataFrame) -> dict[str, Any]:
        return {
            "n": d.height,
            "err_m": stats("err_m", d),
            "p90_radius_m": stats("p90_radius_m", d),
            "q_hat": stats("q", d),
            "coverage": float(d["hit"].mean()) if d.height else None,
        }

    return one(players) | {
        "near_half": one(players.filter(pl.col("y_ref") < 0)),
        "far_half": one(players.filter(pl.col("y_ref") >= 0)),
    }


def detections_of(nets: Any, rgb: np.ndarray) -> dict[str, Any]:
    (kps, ends) = nets.detect(rgb[None])[0]
    return {
        "kp_ids": kps.ids,
        "kp_uv": kps.uv,
        "kp_conf": kps.conf,
        "line_ids": ends.line_ids,
        "line_uv": ends.uv,
    }


def minimap_rgb(frame: TrackingFrame, height: int) -> np.ndarray:
    fig = Figure(
        figsize=(height * 1.33 / 100, height / 100), dpi=100, facecolor=SURFACE, layout="constrained"
    )
    FigureCanvasAgg(fig)
    draw_frame(fig.subplots(), frame)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return np.asarray(cv2.resize(img, (round(img.shape[1] * height / img.shape[0]), height)), dtype=np.uint8)


def compose(view: np.ndarray, minimap: np.ndarray, caption: str) -> np.ndarray:
    h = minimap.shape[0]
    left = cv2.resize(view, (round(view.shape[1] * h / view.shape[0]), h), interpolation=cv2.INTER_AREA)
    out = np.concatenate([left, minimap], axis=1)
    bar = np.full((34, out.shape[1], 3), 20, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(caption, font, 0.6, 1)
    scale = min(0.6, 0.6 * (out.shape[1] - 24) / max(text_w, 1))  # shrink a caption wider than the frame
    cv2.putText(bar, caption, (12, 23), font, scale, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([out, bar], axis=0)


def write_gif(
    frames_bgr: list[np.ndarray],
    path: Path,
    width: int,
    fps: float,
    max_bytes: int = 1_000_000,
    min_width: int = 160,
) -> int:
    """Animated GIF (palette, optimised); halves the width until the file is under ``max_bytes``.

    ValueError, and no file written, when it is still larger at the narrowest width not below ``min_width``:
    render fewer frames.
    """
    while True:
        imgs = []
        for f in frames_bgr:
            small = cv2.resize(
                f, (width, round(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA
            )
            imgs.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).quantize(colors=128))
        buf = io.BytesIO()
        duration = round(1000 / fps)
        imgs[0].save(
            buf, format="GIF", save_all=True, append_images=imgs[1:], duration=duration, loop=0, optimize=True
        )
        if buf.tell() <= max_bytes:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(buf.getvalue())
            return buf.tell()
        if width // 2 < min_width:
            raise ValueError(
                f"a GIF of {len(frames_bgr)} frames is {buf.tell() / 1e3:.0f} kB at {width} px wide, over "
                f"the {max_bytes / 1e3:.0f} kB cap: render fewer frames"
            )
        width //= 2


def ffmpeg_command(
    exe: str, path: Path, size: tuple[int, int], fps: float, crf: int, tune: str | None = None
) -> list[str]:
    """ffmpeg reading raw BGR frames of ``size`` (width, height) on stdin and writing web video: H.264,
    yuv420p, moov atom first (+faststart), no audio."""
    w, h = size
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}"]
    cmd += ["-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf)]
    if tune is not None:
        cmd += ["-tune", tune]
    return [*cmd, "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(path)]


class Mp4Writer:
    """Streams BGR frames of one size to ffmpeg; the file appears at ``path`` only once every frame is
    encoded (a partial file is removed on error)."""

    def __init__(
        self,
        path: Path,
        size: tuple[int, int],
        fps: float,
        crf: int = 28,
        tune: str | None = None,
        ffmpeg: str | None = None,
    ) -> None:
        if size[0] % 2 or size[1] % 2:
            raise ValueError(f"yuv420p needs an even frame size, got {size[0]} × {size[1]}")
        exe = ffmpeg or shutil.which("ffmpeg")
        if exe is None:
            raise FileNotFoundError("ffmpeg not found: install it or pass its path")
        self.path, self.size, self.n_frames = Path(path), size, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.part = self.path.with_name(self.path.name + ".part")
        self.proc = subprocess.Popen(
            ffmpeg_command(exe, self.part, size, fps, crf, tune),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape != (self.size[1], self.size[0], 3) or frame_bgr.dtype != np.uint8:
            raise ValueError(f"frame {frame_bgr.shape} {frame_bgr.dtype}, expected {self.size} uint8 BGR")
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        self.n_frames += 1

    def close(self) -> int:
        """Finish the file; its size in bytes. RuntimeError, and no file, when ffmpeg fails."""
        assert self.proc.stdin is not None
        assert self.proc.stderr is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0 or not self.n_frames:
            self.part.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed on {self.path} ({self.n_frames} frames): {err.strip()}")
        self.part.replace(self.path)
        return self.path.stat().st_size

    def abort(self) -> None:
        self.proc.kill()
        self.proc.wait()
        self.part.unlink(missing_ok=True)

    def __enter__(self) -> Mp4Writer:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is not None:
            self.abort()


def write_mp4(frames_bgr: list[np.ndarray], path: Path, fps: float, **kwargs: Any) -> int:
    """All frames to one H.264 file (see :class:`Mp4Writer`); its size in bytes."""
    if not frames_bgr:
        raise ValueError("no frames to encode")
    h, w = frames_bgr[0].shape[:2]
    with Mp4Writer(path, (w, h), fps, **kwargs) as writer:
        for frame in frames_bgr:
            writer.write(frame)
        return writer.close()


def write_still(frame_bgr: np.ndarray, path: Path, width: int = 1280, quality: int = 82) -> int:
    small = cv2.resize(
        frame_bgr,
        (width, round(frame_bgr.shape[0] * width / frame_bgr.shape[1])),
        interpolation=cv2.INTER_AREA,
    )
    ok, data = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"cannot encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())
    return len(data)


@app.command()
def main(
    match: Annotated[
        str, typer.Option(help="SoccerTrack v2 match with raw/, mot/ and its MOT clip on disk.")
    ] = "117093",
    start: Annotated[int, typer.Option(help="First clip frame (0-based, 25 fps).")] = 0,
    frames: Annotated[int, typer.Option(help="Number of frames to render (network inference on each).")] = 10,
    step: Annotated[int, typer.Option(help="Clip frames between rendered frames.")] = 5,
    fps: Annotated[float, typer.Option(help="Frame rate of demo.mp4.")] = 5.0,
    smooth: Annotated[
        bool, typer.Option(help="PTZ smoother with the dev noise parameters (else the per-frame camera).")
    ] = True,
    device: Annotated[str, typer.Option(help="Torch device for PnLCalib.")] = "auto",
    calibration: Annotated[
        Path,
        typer.Option(help="Noise parameters and conformal models from the SN-GSR dev evaluation (JSON)."),
    ] = dc.DEFAULT_PATH,
    show_reference: Annotated[
        bool, typer.Option(help="Also draw the reference camera's lines (cyan).")
    ] = True,
    hfov: Annotated[tuple[float, float], typer.Option(help="Horizontal field of view range, degrees.")] = (
        25.0,
        45.0,
    ),
    aim_y: Annotated[
        float, typer.Option(help="Metres added to the aim point's y (negative: nearer).")
    ] = -6.0,
    out: Annotated[Path | None, typer.Option(help="Output folder (default runs/<date>_<sha>_demo).")] = None,
    still: Annotated[
        Path | None, typer.Option(help="Write the middle composite frame as a JPEG here.")
    ] = None,
    gif: Annotated[Path | None, typer.Option(help="Write an animated GIF (at most 1 MB) here.")] = None,
    gif_width: Annotated[int, typer.Option(help="GIF width before automatic shrinking.")] = 800,
) -> None:
    t0 = time.perf_counter()
    weights = data_dir() / "weights" / "pnlcalib"
    missing = missing_inputs(match, weights, calibration)
    if missing:
        listing = "\n  ".join(map(str, missing))
        hints = [DOWNLOAD_HINT.format(match=match)] if set(missing) - {calibration} else []
        if calibration in missing:
            hints.append("The default calibration file is in the repository; scripts/fit_demo_calibration.py "
                         "rebuilds it from SN-GSR dev runs.")  # fmt: skip
        typer.echo("\n".join([f"missing inputs, nothing was run:\n  {listing}", *hints]), err=True)
        raise typer.Exit(code=2)
    camera = "ptz" if smooth else "none"
    cal = dc.load(calibration).get(camera)
    if cal is None:
        typer.echo(f"{calibration} has no calibration for the {camera!r} camera", err=True)
        raise typer.Exit(code=2)
    digest = cal.provenance.get("smoother_config_digest")
    if smooth and digest is not None and digest != SmootherConfig().digest():
        typer.echo(f"warning: noise parameters fitted with smoother config {digest}, running with "
                   f"{SmootherConfig().digest()}", err=True)  # fmt: skip

    from predfoot.keypoints.pnlcalib import PnLCalibNetworks

    out = out or REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{git_sha()}_demo"
    out.mkdir(parents=True, exist_ok=True)
    pano = vc.PanoramaCamera(st.load_fisheye(match), *st.load_rectify_maps(match))
    centre = pano.calib.C
    mot = st.load_mot(match)
    names, kp_xy, kp_uv = st.load_keypoints(match)
    idx = [start + k * step for k in range(frames)]
    boxes = {f: mot.filter(pl.col("frame") == f) for f in idx}
    feet_video = {f: st.foot_points(b) for f, b in boxes.items()}
    feet_model = [hg.backproject(pano.homography(), pano.video_to_rectified(feet_video[f]))[0] for f in idx]
    follow = vc.FollowConfig(hfov_deg=hfov, aim_offset_m=(0.0, aim_y))
    phis = vc.follow_action(feet_model, centre, fps=CLIP_FPS / step, config=follow)
    views = [vc.VirtualView(phi) for phi in phis]

    cap = cv2.VideoCapture(str(st.clip_path(match)))
    rendered = []
    for f, view in zip(idx, views, strict=True):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {f} of {st.clip_path(match)}")
        rendered.append(vc.render(img, vc.remap_tables(view, pano)))
    cap.release()

    nets = PnLCalibNetworks(weights / "SV_kp", weights / "SV_lines", device=device)
    t_net = time.perf_counter()
    dets = [detections_of(nets, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in rendered]
    t_net = time.perf_counter() - t_net
    solver = SolverConfig()
    raw: list[CalibResult] = []
    for f, det in zip(idx, dets, strict=True):
        xy, uv, w, lines = observations_from_detections(det)
        raw.append(
            solve(
                xy, uv, np.random.default_rng(int(match) * 100_003 + f), weights=w, config=solver, lines=lines
            )[0]
        )
    # Per-frame camera: raw Laplace covariance, scored with the quantiles of the per-frame dev run.
    results, sources = raw, ["frame"] * len(raw)
    smooth_info: dict[str, Any] = {"used": False}
    if smooth:
        assert cal.noise is not None  # checked when the calibration file is loaded
        sm, flags, _ = smooth_results(np.array(idx), raw, duplicate_mask(dets), SmootherConfig(), cal.noise)
        results, sources = sm, dc.cov_sources(flags)
        smooth_info = {
            "used": True,
            "params": cal.noise.to_dict(),
            "flags": {name: np.asarray(getattr(flags, name), dtype=bool).tolist() for name in FLAG_NAMES},
        }

    player_rows: list[dict[str, Any]] = []
    kp_rows: list[dict[str, Any]] = []
    composites = []
    for k, (f, view, img, res) in enumerate(zip(idx, views, rendered, results, strict=True)):
        H_ref = view.homography(centre)
        H_ours = res.H if res.ok and res.H is not None and res.cov_theta is not None else None
        feet = vc.video_to_view(feet_video[f], view, pano)
        ids = boxes[f]["track_id"].to_numpy()
        in_view = inside(feet, view.size)
        ref_xy = hg.backproject(H_ref, feet)[0]
        locs: list[Localization] = []
        canvas = img.copy()
        if show_reference:
            draw_pitch_lines(canvas, H_ref, REFERENCE_BGR, 2)
        if H_ours is not None and res.cov_theta is not None:
            draw_pitch_lines(canvas, H_ours, OURS_BGR, 2)
            pred, cov, valid = position_covariance(H_ours, res.cov_theta, feet[in_view])
            mpp = np.sqrt(np.abs(np.linalg.det(hg.jacobian_backproject_uv(H_ours, feet[in_view]))))
            model = cal.model_for(sources[k])
            qhat = model.quantile_for({"mpp": mpp, "n_inliers": np.full(len(mpp), float(res.n_inliers))})
            for j, i in enumerate(np.flatnonzero(in_view)):
                if not valid[j]:
                    continue
                cov_cal = cov[j] * qhat[j] / CHI2_2_90
                radius = float(np.sqrt(qhat[j] * np.linalg.eigvalsh(cov[j])[-1]))
                err = pred[j] - ref_xy[i]
                locs.append(Localization(int(ids[i]), xy=pred[j], cov_xy=cov_cal, p90_radius_m=radius))
                player_rows.append({
                    "frame": f, "track_id": int(ids[i]), "x": pred[j, 0], "y": pred[j, 1],
                    "x_ref": ref_xy[i, 0], "y_ref": ref_xy[i, 1], "err_m": float(np.linalg.norm(err)),
                    "p90_radius_m": radius,
                    "hit": bool(mahalanobis2(err[None], cov_cal[None])[0] <= CHI2_2_90), "mpp": float(mpp[j]),
                    "q": float(qhat[j]), "cov_source": sources[k],
                })  # fmt: skip
                colour = COLOURS_BGR[int(np.searchsorted(RADIUS_BINS_M, radius))]
                cv2.circle(canvas, tuple(np.round(feet[i]).astype(int)), 9, colour, -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(np.round(feet[i]).astype(int)), 9, (20, 20, 20), 2, cv2.LINE_AA)
        else:
            for i in np.flatnonzero(in_view):
                cv2.circle(canvas, tuple(np.round(feet[i]).astype(int)), 9, ABSTAIN_BGR, 2, cv2.LINE_AA)
        kp_view = vc.video_to_view(kp_uv, view, pano)
        kp_rows += keypoint_rows(f, names, kp_xy, kp_view, view.size, H_ref, H_ours)
        corners = None
        if H_ours is not None:
            w_, h_ = view.size
            img_corners = np.array([[0, 0], [w_ - 1, 0], [w_ - 1, h_ - 1], [0, h_ - 1]], dtype=np.float64)
            c_xy, c_ok = hg.backproject(H_ours, img_corners)
            corners = np.clip(c_xy, [-60, -45], [60, 39]) if c_ok.all() else None
        tf = TrackingFrame(frame=f, players=tuple(locs), timestamp=f"{f / CLIP_FPS:5.1f} s", corners=corners,
                           abstain=not res.ok)  # fmt: skip
        hfov_now = np.degrees(2 * np.arctan(view.size[0] / 2 / np.exp(view.phi[3])))
        caption = (
            f"{CREDIT} · frame {f} · hfov {hfov_now:.0f} deg · magenta: predfoot camera"
            + (" · cyan: reference camera (released calibration)" if show_reference else "")
            + f" · feet by 90 % radius: <{RADIUS_BINS_M[0]:g} m, <{RADIUS_BINS_M[1]:g} m, wider"
            + "; grey = abstained"
        )
        composites.append(compose(canvas, minimap_rgb(tf, 720), caption))

    writer = cv2.VideoWriter(str(out / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (composites[0].shape[1], composites[0].shape[0]))  # fmt: skip
    for c in composites:
        writer.write(c)
    writer.release()
    players, kps = table(player_rows), table(kp_rows)
    players.write_parquet(out / "players.parquet")
    kps.write_parquet(out / "keypoints.parquet")

    metrics = {
        "match": match, "frames": idx, "step": step, "commit": git_sha(), "follow": follow.__dict__,
        "calibrated": [r.ok for r in results], "raw_calibrated": [r.ok for r in raw],
        "n_inliers": [r.n_inliers for r in results], "rms_px": [r.rms_px for r in raw],
        "hfov_deg": [float(np.degrees(2 * np.arctan(v.size[0] / 2 / np.exp(v.phi[3])))) for v in views],
        "smoothing": smooth_info, "cov_sources": sources,
        "calibration": {
            "file": str(calibration), "camera": camera, "provenance": cal.provenance,
            "models": {s: json.loads(m.to_json()) for s, m in cal.models.items()},
        },
        "players_vs_reference_camera": player_summary(players),
        "keypoints_in_views": keypoint_summary(kps),
        "notes": ["the reference camera was fitted by the provider on the annotated keypoints (in-sample)",
                  "foot points are the ground-truth MOT boxes, not detections"],
        "network_seconds": t_net, "total_seconds": time.perf_counter() - t0,
    }  # fmt: skip
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    if still is not None:
        typer.echo(f"still: {still} ({write_still(composites[len(composites) // 2], still) / 1e3:.0f} kB)")
    gif_error = None
    if gif is not None:
        try:
            typer.echo(f"gif: {gif} ({write_gif(composites, gif, gif_width, fps) / 1e3:.0f} kB)")
        except ValueError as err:
            gif_error = str(err)
    summary_keys = ("calibrated", "players_vs_reference_camera", "keypoints_in_views", "network_seconds")
    typer.echo(json.dumps({k: metrics[k] for k in summary_keys}, indent=2))
    typer.echo(f"wrote {out}")
    if gif_error is not None:
        typer.echo(f"gif not written: {gif_error}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
