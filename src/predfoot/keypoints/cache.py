"""Run the keypoint and line networks once per clip and cache their outputs (decoded detections).

Every later stage (calibration, uncertainty, conformal, health) reads this cache, so experiments never pay
for network inference twice. The cache key is the model family and the number of frames on disk.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from predfoot.geometry.pitch import pnlcalib_keypoints, pnlcalib_lines
from predfoot.paths import cache_dir, data_dir

MODEL_FAMILY = "pnlcalib-sv"
#: Format of detections stored as JSON (small committed fixtures; the cache itself is .npz). Its frames are
#: keyed by the 0-based index of the clip's frames, whereas the cache keys them by image number.
DETECTIONS_FORMAT = "predfoot-detections/1"


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

    Returns ({frame: {kp_ids, kp_uv, kp_conf, line_ids, line_uv}}, inference seconds, model version), keyed by
    image number (the stem of the file name: 1, 6, 11, … for SN-GSR frames downloaded one in five); the real
    stages want the 0-based index of the clip's frames, see :func:`by_clip_index`.
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
    save_detections(path, out, seconds, nets.version)
    return out, seconds, nets.version


def by_clip_index[T](dets: Mapping[int, T]) -> dict[int, T]:
    """Detections keyed by image number, as the cache holds them, re-keyed by the 0-based index of the clip's
    frames (the images in increasing order): the keys of ``predfoot-detections/1`` and of the real stages."""
    return {i: dets[f] for i, f in enumerate(sorted(dets))}


def save_detections(path: Path, dets: dict[int, dict[str, Any]], seconds: float, version: str) -> None:
    """Write a clip's detections atomically (also used for detections computed elsewhere, e.g. on Modal)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, dets=np.array(dets, dtype=object), seconds=seconds, version=version)
    tmp.replace(path)


def dumps_detections(dets: Mapping[int, Mapping[str, Any]], model_version: str, decimals: int = 3) -> str:
    """Detections as JSON (format ``predfoot-detections/1``), one frame per line so that diffs stay readable.

    ``dets`` maps the 0-based index of the clip's frames (not the image number of the cache: see
    :func:`by_clip_index`) to the cache layout (``kp_ids``, ``kp_uv``, ``kp_conf``, ``line_ids``,
    ``line_uv``); pixels and confidences are rounded to ``decimals``.
    """

    def rounded(values: Any) -> Any:
        return np.round(np.asarray(values, dtype=np.float64), decimals).tolist()

    frames = [
        json.dumps(
            {
                "frame": int(f),
                "kp_ids": [int(k) for k in d["kp_ids"]],
                "kp_uv": rounded(np.reshape(d["kp_uv"], (-1, 2))),
                "kp_conf": rounded(d["kp_conf"]),
                "line_ids": [int(k) for k in d["line_ids"]],
                "line_uv": rounded(np.reshape(d["line_uv"], (-1, 2, 2))),
            },
            allow_nan=False,
        )
        for f, d in sorted(dets.items())
    ]
    head = f'{{"format": {json.dumps(DETECTIONS_FORMAT)}, "model_version": {json.dumps(model_version)},'
    return head + '\n "frames": [\n  ' + ",\n  ".join(frames) + "\n]}\n"


def _array(values: Any, shape: tuple[int, ...], what: str, dtype: type = np.float64) -> NDArray[Any]:
    try:
        arr = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{what}: {err}") from err
    if arr.size == 0:
        arr = arr.reshape((0, *shape[1:]))
    if arr.ndim != len(shape) or any(s not in (-1, n) for s, n in zip(shape, arr.shape, strict=True)):
        raise ValueError(f"{what} has shape {arr.shape}, expected {shape} (-1: any)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} must be finite")
    return arr


def load_detections(path: Path) -> tuple[dict[int, dict[str, Any]], str]:
    """Read a ``predfoot-detections/1`` file: ({index of the clip's frame: cache-layout dict}, model version).

    Raises ``ValueError`` on anything malformed: shapes, lengths, non-finite values, duplicate frames.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format") != DETECTIONS_FORMAT:
        raise ValueError(f"{path}: not a {DETECTIONS_FORMAT} file")
    frames = raw.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{path}: 'frames' must be a list")
    out: dict[int, dict[str, Any]] = {}
    for entry in frames:
        frame = entry.get("frame") if isinstance(entry, dict) else None
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 or frame in out:
            raise ValueError(f"{path}: frame {frame!r} is not a new non-negative integer")
        where = f"{path}: frame {frame}"
        ids = _array(entry.get("kp_ids"), (-1,), f"{where} kp_ids", np.int64)
        line_ids = _array(entry.get("line_ids"), (-1,), f"{where} line_ids", np.int64)
        unknown = sorted(set(ids.tolist()) - set(pnlcalib_keypoints())) + sorted(
            set(line_ids.tolist()) - set(pnlcalib_lines())
        )
        if unknown:
            raise ValueError(f"{where}: unknown keypoint or line ids {unknown}")
        det = {
            "kp_ids": ids,
            "kp_uv": _array(entry.get("kp_uv"), (len(ids), 2), f"{where} kp_uv"),
            "kp_conf": _array(entry.get("kp_conf"), (len(ids),), f"{where} kp_conf"),
            "line_ids": line_ids,
            "line_uv": _array(entry.get("line_uv"), (len(line_ids), 2, 2), f"{where} line_uv"),
        }
        out[frame] = det
    return out, str(raw.get("model_version", ""))
