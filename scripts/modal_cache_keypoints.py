"""Cache keypoint and line detections on Modal GPUs instead of the local machine.

Each container takes one SN-GSR clip: it reads every fifth frame straight from the Hugging Face zip over
HTTP range requests (nothing is uploaded from the laptop), runs both PnLCalib networks on a T4 in float32,
and returns the decoded detections; the local entrypoint writes them to the same cache as
``scripts/cache_keypoints.py``, so the two can share the work (this one walks the clips backwards).

Usage (from the repository, with the vision extra installed and a Modal token configured):
    uv run modal run scripts/modal_cache_keypoints.py --splits train,valid
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import modal

GPU = "T4"
EVERY = 5
WEIGHTS_URL = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/{name}"

app = modal.App("predfoot-keypoints")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "numpy>=2.0",
        "opencv-python-headless>=4.10",
        "polars>=1.10",
        "pyyaml>=6.0",
        "pydantic>=2.8",
    )
    .add_local_python_source("predfoot")
)
weights_volume = modal.Volume.from_name("predfoot-pnlcalib-weights", create_if_missing=True)


@app.function(image=image, volumes={"/weights": weights_volume}, timeout=1200)
def prepare_weights() -> list[str]:
    """Download the CC BY 4.0 weights into the volume once."""
    import urllib.request

    names = []
    for name in ("SV_kp", "SV_lines"):
        dest = Path("/weights") / name
        if not dest.exists():
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(WEIGHTS_URL.format(name=name), tmp)
            tmp.replace(dest)
        names.append(f"{name}: {dest.stat().st_size / 1e6:.0f} MB")
    weights_volume.commit()
    return names


@app.function(image=image, gpu=GPU, volumes={"/weights": weights_volume}, timeout=1800, max_containers=4)
def detect_clip(split: str, clip_id: str) -> tuple[str, dict[int, dict[str, Any]], float, str]:
    """Detections for frames 1, 6, 11, … of one clip: (clip_id, {frame: detections}, GPU seconds, version)."""
    import tempfile

    import cv2
    import numpy as np

    from predfoot.io.remote_zip import HttpRangeFetcher, RemoteZip
    from predfoot.io.sngsr import split_url
    from predfoot.keypoints.pnlcalib import PnLCalibNetworks

    nets = PnLCalibNetworks(Path("/weights/SV_kp"), Path("/weights/SV_lines"), device="cuda")
    archive = RemoteZip(HttpRangeFetcher(split_url(split)))
    names = [f"{clip_id}/img1/{i:06d}.jpg" for i in range(1, 751, EVERY)]
    dets: dict[int, dict[str, Any]] = {}
    seconds = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        targets = {name: Path(tmp) / Path(name).name for name in names}
        archive.extract(targets)
        files = sorted(targets.values())
        for start in range(0, len(files), 2):
            chunk = files[start : start + 2]
            imgs = []
            for f in chunk:
                img = cv2.imread(str(f))
                if img is None:
                    raise OSError(f"cannot decode {f.name} of {clip_id}")
                imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            t0 = time.perf_counter()
            out = nets.detect(np.stack(imgs))
            seconds += time.perf_counter() - t0
            for f, (kps, ends) in zip(chunk, out, strict=True):
                dets[int(f.stem)] = {
                    "kp_ids": kps.ids,
                    "kp_uv": kps.uv,
                    "kp_conf": kps.conf,
                    "line_ids": ends.line_ids,
                    "line_uv": ends.uv,
                }
    return clip_id, dets, seconds, nets.version


@app.local_entrypoint()
def main(splits: str = "train,valid", max_clips: int = 0) -> None:
    from predfoot.keypoints.cache import cache_path, save_detections
    from predfoot.paths import data_dir

    todo = []
    for split in splits.split(","):
        if split == "test":
            raise SystemExit("the test split stays locked until the preregistration is frozen")
        for clip_dir in sorted((data_dir() / "sn-gsr" / split).glob("SNGS-*"), reverse=True):
            if (clip_dir / "img1").is_dir() and not cache_path(clip_dir.name, 750 // EVERY).exists():
                todo.append((split, clip_dir.name))
    if max_clips:
        todo = todo[:max_clips]
    print(f"{len(todo)} clips to process on Modal ({GPU}): {', '.join(c for _, c in todo[:6])}…")
    print(prepare_weights.remote())
    t0 = time.perf_counter()
    gpu_seconds = 0.0
    for clip_id, dets, seconds, version in detect_clip.starmap(todo, order_outputs=False):
        path = cache_path(clip_id, len(dets))
        gpu_seconds += seconds
        if path.exists():
            print(f"{clip_id}: already cached locally, result dropped")
            continue
        save_detections(path, dets, seconds, version)
        elapsed = time.perf_counter() - t0
        print(f"{clip_id}: {len(dets)} frames, {seconds:.0f} s of inference ({elapsed:.0f} s elapsed)")
    wall = time.perf_counter() - t0
    print(f"Done: {len(todo)} clips, {gpu_seconds / 60:.1f} GPU-minutes of inference, {wall:.0f} s wall")
