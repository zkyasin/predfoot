"""ONNX Runtime against PyTorch on one SN-GSR dev frame (local only: weights, frames and exported models)."""

import pytest

from predfoot.keypoints.onnx_models import (
    MODEL_FILES,
    OnnxPnLCalib,
    onnx_dir,
    parity,
    read_frames,
    spread_frames,
)
from predfoot.paths import data_dir

pytestmark = [pytest.mark.vision, pytest.mark.data]


def test_onnx_runtime_matches_pytorch_on_a_dev_frame():
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    weights = data_dir() / "weights" / "pnlcalib"
    clip_dir = data_dir() / "sn-gsr" / "valid" / "SNGS-056"
    if not (weights / "SV_kp").is_file() or not (clip_dir / "img1").is_dir():
        pytest.skip("needs the PnLCalib weights and the SN-GSR dev frames")
    if not all((onnx_dir() / f).is_file() for f in MODEL_FILES.values()):
        pytest.skip("no exported models: run scripts/export_onnx.py")
    from predfoot.keypoints.pnlcalib import PnLCalibNetworks, preprocess

    nets = PnLCalibNetworks(weights / "SV_kp", weights / "SV_lines", device="cpu")
    rgb = read_frames(spread_frames(clip_dir, 1))
    x = preprocess(rgb)
    result = parity(nets.heatmaps_from_input(x), OnnxPnLCalib().heatmaps_from_input(x.numpy()), (1920, 1080))
    assert result.n_points >= 10
    assert result.ok, result
    assert result.n_flips == 0
