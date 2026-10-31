"""Run the keypoint and line networks once per clip and cache their outputs (decoded detections).

Every later stage (calibration, uncertainty, conformal, health) reads this cache, so experiments never pay
for network inference twice. The cache key is the model family and the number of frames on disk.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from predfoot.paths import cache_dir, data_dir

MODEL_FAMILY = "pnlcalib-sv"


def _read_rgb(path: Path) -> NDArray[np.uint8]:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise OSError(f"cannot read {path}")
    return np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def cache_path(clip_id: str, n_frames: int) -> Path:
    return cache_dir() / "kp" / MODEL_FAMILY / f"{clip_id}_{n_frames}.npz"


def cached_detections(
    clip_dir: Path, clip_id: str, device: str = "auto", batch: int = 1
) -> tuple[dict[int, dict[str, Any]], float, str]:
    """Detections for every frame image of ``clip_dir/img1``, computed once and cached as .npz.

    Returns ({frame: {kp_ids, kp_uv, kp_conf, line_ids, line_uv}}, inference seconds, model version).
    Batch 1 is the fastest setting on a 24 GB Apple GPU: larger batches of two HRNet-W48 run out of memory.
    """
    frame_files = sorted((clip_dir / "img1").glob("*.jpg"))
    path = cache_path(clip_id, len(frame_files))
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["dets"].item(), float(z["seconds"]), str(z["version"])

    from predfoot.keypoints.pnlcalib import PnLCalibNetworks

    weights = data_dir() / "weights" / "pnlcalib"
    nets = PnLCalibNetworks(weights / "SV_kp", weights / "SV_lines", device=device)
    out: dict[int, dict[str, Any]] = {}
    seconds = 0.0
    for start in range(0, len(frame_files), batch):
        chunk = frame_files[start : start + batch]
        imgs = np.stack([_read_rgb(f) for f in chunk])
        t0 = time.perf_counter()
        dets = nets.detect(imgs)
        seconds += time.perf_counter() - t0
        for f, (kps, ends) in zip(chunk, dets, strict=True):
            out[int(f.stem)] = {
                "kp_ids": kps.ids,
                "kp_uv": kps.uv,
                "kp_conf": kps.conf,
                "line_ids": ends.line_ids,
                "line_uv": ends.uv,
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, dets=np.array(out, dtype=object), seconds=seconds, version=nets.version)
    tmp.replace(path)
    return out, seconds, nets.version
