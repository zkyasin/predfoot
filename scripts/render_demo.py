"""Public demo on SoccerTrack v2 pixels (CC BY 4.0, Scott et al.): a virtual broadcast camera that follows the
play inside the fixed panorama, predfoot's camera and positions on it, and the minimap with 90 % regions.

    uv run python scripts/render_demo.py --match 117092 --start 0 --frames 10 --step 5 --fps 5 \\
        --still assets/demo.jpg --gif runs/demo.gif

Per rendered frame: the virtual view (PTZ camera at the panorama's centre, known pitch homography: the
reference camera) is rendered from the MOT clip frame with one remap; PnLCalib keypoints and lines →
robust calibration → optionally the PTZ smoother (noise parameters reused from the dev run, not refitted) →
delta-method positions of the MOT foot points with conformal quantiles fitted on the SN-GSR dev players
table (Mondrian bins per covariance source, as in scripts/eval_dev.py). That transfers a calibration from
SN-GSR broadcast frames to SoccerTrack virtual views: a domain shift, reported as such.

Two references are scored on the rendered frames: the reference camera (the released calibration, whose own
error is metres on the far half, see reports/soccertrack_v2.md) and the annotated keypoints mapped into each
view (true correspondences up to annotation noise).

Writes ``runs/<date>_<sha>_demo/``: demo.mp4 (virtual view | minimap), metrics.json, players.parquet,
keypoints.parquet. Network inference runs once per rendered frame.
"""

from __future__ import annotations

import io
import json
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
from predfoot.calib.temporal import NoiseParams, SmootherConfig, duplicate_mask, smooth_results
from predfoot.contracts import CalibResult, Localization
from predfoot.demo import virtual_camera as vc
from predfoot.geometry import homography as hg
from predfoot.geometry.pitch import Arc, sn_line_elements
from predfoot.io import soccertrack as st
from predfoot.io.skillcorner import TrackingFrame
from predfoot.paths import REPO_ROOT, data_dir
from predfoot.uncertainty.conformal import MondrianConformal
from predfoot.uncertainty.delta import CHI2_2_90, mahalanobis2, position_covariance
from predfoot.viz.minimap import SURFACE, draw_frame

DEV_RUN = REPO_ROOT.parent / "temporal" / "runs" / "20261031-153059_c3861c7_ptz"
ALPHA = 0.1
#: Foot-point colours (BGR) by calibrated 90 % radius: confident, wider, wide; grey when the frame abstains.
RADIUS_BINS_M = (1.0, 2.5)
COLOURS_BGR = ((80, 200, 60), (40, 170, 245), (60, 60, 230))
ABSTAIN_BGR = (150, 150, 150)
OURS_BGR = (255, 255, 255)
REFERENCE_BGR = (255, 160, 40)
CREDIT = "Pixels: SoccerTrack v2 (CC BY 4.0, Scott et al.), virtual view of the panorama"
FLAG_NAMES = ("smoothed", "repaired", "filled", "projected", "passthrough", "duplicate", "rejected")

app = typer.Typer(add_completion=False)


def git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "nogit"


def fit_conformal(table: Path) -> dict[str, MondrianConformal]:
    """Mondrian models per covariance source, fitted on every dev player-frame (reference: labels)."""
    d = pl.read_parquet(table).filter(pl.col("ex_lab").is_not_nan())
    e = d.select("ex_lab", "ey_lab").to_numpy()
    cov = np.stack([d["c00"], d["c01"], d["c01"], d["c11"]], axis=1).reshape(-1, 2, 2)
    score, groups, source = mahalanobis2(e, cov), d["clip"].to_numpy(), d["cov_source"].to_numpy()
    feats = {"mpp": d["mpp"].to_numpy(), "n_inliers": d["n_inliers"].to_numpy().astype(float)}
    models = {}
    for src in np.unique(source):
        sel = source == src
        models[str(src)] = MondrianConformal.fit(
            score[sel], groups[sel], {k: v[sel] for k, v in feats.items()}, {"mpp": 3, "n_inliers": 2}, ALPHA
        )
    return models


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
    cv2.putText(bar, caption, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([out, bar], axis=0)


def write_gif(
    frames_bgr: list[np.ndarray], path: Path, width: int, fps: float, max_bytes: int = 1_000_000
) -> int:
    """Animated GIF (palette, optimised); halves the width until the file is under ``max_bytes``."""
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
        if buf.tell() <= max_bytes or width <= 160:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(buf.getvalue())
            return buf.tell()
        width //= 2


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


def dev_noise_params(dev_run: Path) -> NoiseParams:
    """Median of the per-fold noise parameters fitted (without labels) on the SN-GSR dev clips."""
    folds = json.loads((dev_run / "smoothing.json").read_text())["folds"].values()
    keys = ("kappa2", "q_pan", "q_tilt", "q_logf")
    return NoiseParams(**{k: float(np.median([f["params"][k] for f in folds])) for k in keys})


@app.command()
def main(
    match: Annotated[
        str, typer.Option(help="SoccerTrack v2 match with raw/, mot/ and its MOT clip on disk.")
    ] = "117092",
    start: Annotated[int, typer.Option(help="First clip frame (0-based, 25 fps).")] = 0,
    frames: Annotated[int, typer.Option(help="Number of frames to render (network inference on each).")] = 10,
    step: Annotated[int, typer.Option(help="Clip frames between rendered frames.")] = 5,
    fps: Annotated[float, typer.Option(help="Frame rate of demo.mp4.")] = 5.0,
    smooth: Annotated[bool, typer.Option(help="PTZ smoother with the dev noise parameters.")] = True,
    device: Annotated[str, typer.Option(help="Torch device for PnLCalib.")] = "auto",
    dev_run: Annotated[Path, typer.Option(help="Dev run with players.parquet and smoothing.json.")] = DEV_RUN,
    show_reference: Annotated[bool, typer.Option(help="Also draw the reference camera's lines.")] = False,
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
    gif: Annotated[Path | None, typer.Option(help="Write an animated GIF (< 1 MB) here.")] = None,
    gif_width: Annotated[int, typer.Option(help="GIF width before automatic shrinking.")] = 800,
) -> None:
    from predfoot.keypoints.pnlcalib import PnLCalibNetworks

    t0 = time.perf_counter()
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
    clip_fps = 25.0
    follow = vc.FollowConfig(hfov_deg=hfov, aim_offset_m=(0.0, aim_y))
    phis = vc.follow_action(feet_model, centre, fps=clip_fps / step, config=follow)
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

    weights = data_dir() / "weights" / "pnlcalib"
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
    results, sources, smooth_info = raw, ["frame"] * len(raw), {"used": False}
    if smooth:
        params = dev_noise_params(dev_run)
        sm, flags, _ = smooth_results(np.array(idx), raw, duplicate_mask(dets), SmootherConfig(), params)
        results = sm
        sources = ["smoother" if (flags.smoothed[k] or flags.repaired[k] or flags.filled[k])
                   else "projected" if flags.projected[k] else "frame" for k in range(len(idx))]  # fmt: skip
        smooth_info = {
            "used": True,
            "params": params.to_dict(),
            "params_from": str(dev_run / "smoothing.json") + " (median of the six fold fits, not refitted)",
            "flags": {name: np.asarray(getattr(flags, name), dtype=bool).tolist() for name in FLAG_NAMES},
        }  # fmt: skip
    models = fit_conformal(dev_run / "players.parquet")

    player_rows: list[dict[str, Any]] = []
    kp_rows: list[dict[str, Any]] = []
    composites = []
    for k, (f, view, img, res) in enumerate(zip(idx, views, rendered, results, strict=True)):
        H_ref = view.homography(centre)
        feet = vc.video_to_view(feet_video[f], view, pano)
        ids = boxes[f]["track_id"].to_numpy()
        in_view = np.isfinite(feet).all(axis=1) & (feet[:, 0] >= 0) & (feet[:, 0] < view.size[0])
        in_view &= (feet[:, 1] >= 0) & (feet[:, 1] < view.size[1])
        ref_xy = hg.backproject(H_ref, feet)[0]
        locs: list[Localization] = []
        canvas = img.copy()
        if show_reference:
            draw_pitch_lines(canvas, H_ref, REFERENCE_BGR, 2)
        if res.ok and res.H is not None and res.cov_theta is not None:
            draw_pitch_lines(canvas, res.H, OURS_BGR, 2)
            pred, cov, valid = position_covariance(res.H, res.cov_theta, feet[in_view])
            mpp = np.sqrt(np.abs(np.linalg.det(hg.jacobian_backproject_uv(res.H, feet[in_view]))))
            model = models.get(sources[k], models["frame"])
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
        seen = np.isfinite(kp_view).all(axis=1) & (kp_view[:, 0] >= 0) & (kp_view[:, 0] < view.size[0])
        seen &= (kp_view[:, 1] >= 0) & (kp_view[:, 1] < view.size[1])
        for i in np.flatnonzero(seen):
            row: dict[str, Any] = {"frame": f, "keypoint": names[i], "ref_px": float(np.linalg.norm(
                hg.project(H_ref, kp_xy[i : i + 1])[0] - kp_view[i])), "ref_m": float(np.linalg.norm(
                hg.backproject(H_ref, kp_view[i : i + 1])[0][0] - kp_xy[i]))}  # fmt: skip
            if res.ok and res.H is not None:
                row["ours_px"] = float(np.linalg.norm(hg.project(res.H, kp_xy[i : i + 1])[0] - kp_view[i]))
                row["ours_m"] = float(
                    np.linalg.norm(hg.backproject(res.H, kp_view[i : i + 1])[0][0] - kp_xy[i])
                )
            kp_rows.append(row)
        corners = None
        if res.ok and res.H is not None:
            w_, h_ = view.size
            img_corners = np.array([[0, 0], [w_ - 1, 0], [w_ - 1, h_ - 1], [0, h_ - 1]], dtype=np.float64)
            c_xy, c_ok = hg.backproject(res.H, img_corners)
            corners = np.clip(c_xy, [-60, -45], [60, 39]) if c_ok.all() else None
        tf = TrackingFrame(frame=f, players=tuple(locs), timestamp=f"{f / clip_fps:5.1f} s", corners=corners,
                           abstain=not res.ok)  # fmt: skip
        hfov = np.degrees(2 * np.arctan(view.size[0] / 2 / np.exp(view.phi[3])))
        caption = (
            f"{CREDIT} · frame {f} · hfov {hfov:.0f} deg · predfoot camera (white lines), feet coloured by"
            f" 90 % radius: <{RADIUS_BINS_M[0]:g} m, <{RADIUS_BINS_M[1]:g} m, wider; grey = abstained"
        )
        composites.append(compose(canvas, minimap_rgb(tf, 720), caption))

    writer = cv2.VideoWriter(str(out / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (composites[0].shape[1], composites[0].shape[0]))  # fmt: skip
    for c in composites:
        writer.write(c)
    writer.release()
    players = pl.DataFrame(player_rows) if player_rows else pl.DataFrame()
    kps = pl.DataFrame(kp_rows) if kp_rows else pl.DataFrame()
    players.write_parquet(out / "players.parquet")
    kps.write_parquet(out / "keypoints.parquet")

    def stats(col: str, d: pl.DataFrame) -> dict[str, float] | None:
        if d.is_empty() or col not in d.columns:
            return None
        v = d[col].drop_nulls().to_numpy()
        return (
            {"n": len(v), "p50": float(np.median(v)), "p90": float(np.quantile(v, 0.9))} if len(v) else None
        )

    metrics = {
        "match": match, "frames": idx, "step": step, "commit": git_sha(), "follow": follow.__dict__,
        "calibrated": [r.ok for r in results], "raw_calibrated": [r.ok for r in raw],
        "n_inliers": [r.n_inliers for r in results], "rms_px": [r.rms_px for r in raw],
        "hfov_deg": [float(np.degrees(2 * np.arctan(960 / np.exp(v.phi[3])))) for v in views],
        "smoothing": smooth_info, "cov_sources": sources,
        "conformal": {s: json.loads(m.to_json()) for s, m in models.items()},
        "players_vs_reference_camera": {
            "err_m": stats("err_m", players), "p90_radius_m": stats("p90_radius_m", players),
            "coverage": float(players["hit"].mean()) if not players.is_empty() else None,
        },
        "keypoints_in_views": {"reference_px": stats("ref_px", kps), "reference_m": stats("ref_m", kps),
                               "ours_px": stats("ours_px", kps), "ours_m": stats("ours_m", kps)},
        "network_seconds": t_net, "total_seconds": time.perf_counter() - t0,
    }  # fmt: skip
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    if still is not None:
        typer.echo(f"still: {still} ({write_still(composites[len(composites) // 2], still) / 1e3:.0f} kB)")
    if gif is not None:
        typer.echo(f"gif: {gif} ({write_gif(composites, gif, gif_width, fps) / 1e3:.0f} kB)")
    summary_keys = ("calibrated", "players_vs_reference_camera", "keypoints_in_views", "network_seconds")
    typer.echo(json.dumps({k: metrics[k] for k in summary_keys}, indent=2))
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
