"""Feasibility of SoccerTrack v2 as a quasi-ground-truth benchmark (ticket T5.D1): the numbers behind
``reports/soccertrack_v2.md``.

    uv run python scripts/soccertrack_feasibility.py --match 117092 --out runs/soccertrack_feasibility.json

Reads the released calibration of one match, its MOT boxes and the first frame of its MOT clip; no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import typer
from scipy.interpolate import RBFInterpolator

from predfoot.demo import virtual_camera as vc
from predfoot.geometry import homography as hg
from predfoot.io import soccertrack as st

app = typer.Typer(add_completion=False)


def q(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {
        "median": round(float(np.median(v)), 3),
        "p90": round(float(np.quantile(v, 0.9)), 3),
        "max": round(float(v.max()), 3),
        "rms": round(float(np.sqrt(np.mean(v**2))), 3),
    }


def backproject_model(pano: vc.PanoramaCamera, uv: np.ndarray) -> np.ndarray:
    """Video pixels → internal pitch points through the released model."""
    return hg.backproject(pano.homography(), pano.video_to_rectified(uv))[0]


def loo_homography(xy: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """Leave-one-out predictions of rectified points by a homography fitted on the others (DLT)."""
    out = np.zeros_like(rect)
    for i in range(len(xy)):
        keep = np.arange(len(xy)) != i
        out[i] = hg.project(hg.dlt(xy[keep], rect[keep]), xy[i : i + 1])[0]
    return out


def jacobian(pano: vc.PanoramaCamera, p: np.ndarray) -> np.ndarray:
    """Jacobian ∂(x, y)/∂(u, v) (2 × 2) of the video pixel → pitch map at one pixel, central differences."""
    cols = [
        (backproject_model(pano, (p + d)[None]) - backproject_model(pano, (p - d)[None]))[0] / 2
        for d in (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    ]
    return np.column_stack(cols)


def metres_per_pixel(pano: vc.PanoramaCamera, p: np.ndarray) -> dict[str, float]:
    """Pitch metres per video pixel at one pixel: √|det J| (areal scale, the geometric mean of the two axes),
    the largest singular value of J (the worst direction: depth, for a camera behind a touchline) and the
    length of ∂(x, y)/∂v (one vertical video pixel)."""
    J = jacobian(pano, p)
    return {
        "areal": round(float(np.sqrt(abs(np.linalg.det(J)))), 3),
        "largest_axis": round(float(np.linalg.svd(J, compute_uv=False)[0]), 3),
        "vertical_pixel": round(float(np.linalg.norm(J[:, 1])), 3),
    }


@app.command()
def main(
    match: Annotated[str, typer.Option(help="SoccerTrack v2 match id.")] = "117092",
    out: Annotated[Path, typer.Option(help="JSON with every number of the report.")] = Path(
        "runs/soccertrack_feasibility.json"
    ),
) -> None:
    res: dict[str, Any] = {"match": match}
    calib = st.load_fisheye(match)
    map_x, map_y = st.load_rectify_maps(match)
    pano = vc.PanoramaCamera(calib, map_x, map_y)
    w, h = pano.size

    cap = cv2.VideoCapture(str(st.clip_path(match)))
    res["clip"] = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()

    # remap tables = fisheye model?
    ax, ay = cv2.fisheye.initUndistortRectifyMap(
        calib.K, calib.D.reshape(4, 1), np.eye(3), calib.Knew, (w, h), cv2.CV_32FC1
    )
    inside = (ax >= 0) & (ax <= w - 1) & (ay >= 0) & (ay <= h - 1)
    diff = np.hypot(map_x - ax, map_y - ay)[inside]
    res["maps"] = {
        "shape": [h, w],
        "max_diff_px_vs_fisheye_model": round(float(diff.max()), 4),
        "rectified_pixels_inside_video": round(float(inside.mean()), 4),
        "K": calib.K.tolist(),
        "D": calib.D.tolist(),
        "Knew": calib.Knew.tolist(),
        "provider_rms_px": calib.rms_px,
        "centre_internal_m": np.round(calib.C, 2).tolist(),
    }

    names, xy, uv = st.load_keypoints(match)
    corner = st.internal_to_corner_xy(xy)
    groups, marked = st.keypoint_groups(xy)
    interpolated = ~marked  # touchline points placed by eye
    far = groups == "far_touchline"
    near = groups == "near_touchline"
    proj = pano.project_video(xy)
    e_px = np.linalg.norm(proj - uv, axis=1)
    e_m = np.linalg.norm(backproject_model(pano, uv) - xy, axis=1)
    res["model_vs_keypoints"] = {
        "n": len(xy),
        "n_landmarks": int((~interpolated).sum()),
        "video_px": q(e_px),
        "metres": q(e_m),
        "metres_median_by_group": {
            "far_touchline": round(float(np.median(e_m[far])), 2),
            "near_touchline": round(float(np.median(e_m[near])), 2),
            "landmarks": round(float(np.median(e_m[~interpolated])), 2),
            "interpolated_touchline_points": round(float(np.median(e_m[interpolated])), 2),
        },
        "centre_spot_video_px_offset": np.round(
            proj[names.index("(52.5,34)")] - uv[names.index("(52.5,34)")], 1
        ).tolist(),
        "centre_spot_metres": round(float(e_m[names.index("(52.5,34)")]), 2),
    }
    res["video_metres_per_pixel"] = {
        n: metres_per_pixel(pano, uv[names.index(n)]) for n in ("(52.5,68)", "(52.5,34)", "(52.5,0)", "(0,0)")
    }

    # homographies in the rectified image
    rect = pano.video_to_rectified(uv)
    H_ds = st.homography_to_internal(st.load_homography(match))
    e_ds = np.linalg.norm(hg.project(H_ds, xy) - rect, axis=1)
    e_model = np.linalg.norm(hg.project(pano.homography(), xy) - rect, axis=1)
    H_ls = hg.dlt(xy, rect)
    e_ls = np.linalg.norm(hg.project(H_ls, xy) - rect, axis=1)
    loo = loo_homography(xy, rect)
    e_loo_video = np.linalg.norm(pano.rectified_to_video(loo) - uv, axis=1)
    res["rectified_homographies_px"] = {
        "homography_npy": {**q(e_ds), "under_3px": int((e_ds < 3).sum())},
        "model_Knew_r1r2t": q(e_model),
        "least_squares_all_65": q(e_ls),
        "least_squares_loo_in_video_px": q(e_loo_video),
        "homography_npy_metres": q(np.linalg.norm(hg.backproject(H_ds, rect)[0] - xy, axis=1)),
    }
    windows = {}
    for lo, hi in ((0.0, 40.0), (30.0, 75.0), (65.0, 105.0)):
        sel = (corner[:, 0] >= lo) & (corner[:, 0] <= hi)
        local = loo_homography(xy[sel], rect[sel])
        e_local = np.linalg.norm(pano.rectified_to_video(local) - uv[sel], axis=1)
        windows[f"x_{lo:g}_{hi:g}"] = {
            "n": int(sel.sum()),
            "local_loo_video_px_median": round(float(np.median(e_local)), 1),
            "global_loo_video_px_median": round(float(np.median(e_loo_video[sel])), 1),
        }
    res["local_windows"] = windows

    # best case for a corrected model: a smooth residual field (thin-plate spline) on the video image
    resid = uv - proj
    loo_tps = np.zeros(len(xy))
    for i in range(len(xy)):
        keep = np.arange(len(xy)) != i
        f = RBFInterpolator(proj[keep], resid[keep], kernel="thin_plate_spline", smoothing=100.0)
        loo_tps[i] = np.linalg.norm(proj[i] + f(proj[i : i + 1])[0] - uv[i])
    res["tps_corrected_model_loo_video_px"] = q(loo_tps)

    # MOT boxes
    mot = st.load_mot(match)
    feet = st.foot_points(mot)
    on_pitch = backproject_model(pano, feet)
    margin = 3.0
    inside_pitch = (np.abs(on_pitch[:, 0]) <= 52.5 + margin) & (np.abs(on_pitch[:, 1]) <= 34 + margin)
    per_frame = mot.group_by("frame").len()["len"].to_numpy()
    res["mot"] = {
        "rows": mot.height,
        "frames": [int(mot["frame"].min()), int(mot["frame"].max())],
        "ids": int(mot["track_id"].n_unique()),
        "boxes_per_frame": [int(per_frame.min()), int(per_frame.max())],
        "box_height_px": q(mot["h"].to_numpy()),
        "feet_inside_pitch_3m": round(float(inside_pitch.mean()), 4),
        "feet_far_half": round(float((on_pitch[:, 1] > 0).mean()), 4),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    typer.echo(json.dumps(res, indent=2))


if __name__ == "__main__":
    app()
