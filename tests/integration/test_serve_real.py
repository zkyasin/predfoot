"""/v1/calibrate with the real PnLCalib networks on one SN-GSR dev frame (local only: weights and frames)."""

import numpy as np
import pytest

from predfoot.keypoints.cache import cache_path, cached_detections
from predfoot.paths import data_dir

pytestmark = [pytest.mark.vision, pytest.mark.data]

CLIP = "SNGS-056"


def test_calibrate_a_dev_frame_with_the_real_networks():
    pytest.importorskip("fastapi")
    pytest.importorskip("torch")
    clip_dir = data_dir() / "sn-gsr" / "valid" / CLIP
    frame = clip_dir / "img1" / "000001.jpg"
    if not frame.is_file() or not (data_dir() / "weights" / "pnlcalib" / "SV_kp").is_file():
        pytest.skip("needs the SN-GSR dev frames and the PnLCalib weights")
    from fastapi.testclient import TestClient

    from predfoot.serve.app import create_app

    client = TestClient(create_app())
    r = client.post("/v1/calibrate", files={"image": (frame.name, frame.read_bytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["image_size"] == [1920, 1080]
    assert body["n_inliers"] >= 8
    assert body["rms_px"] < 10.0
    assert client.get("/healthz").json()["models"]["pnlcalib"] == body["model_version"]
    if not cache_path(CLIP, 150).exists():
        return
    # Same preprocessing and decoding as the offline detection cache (possibly computed on another device).
    dets, _, version = cached_detections(clip_dir, CLIP)
    assert version == body["model_version"]
    cached = {int(k): uv for k, uv in zip(dets[1]["kp_ids"], dets[1]["kp_uv"], strict=True)}
    served = {kp["id"]: np.array([kp["u"], kp["v"]]) for kp in body["keypoints"]}
    common = sorted(set(cached) & set(served))
    assert len(common) >= 0.9 * len(cached)
    shifts = [np.linalg.norm(served[k] - cached[k]) for k in common]
    assert np.median(shifts) < 0.5
