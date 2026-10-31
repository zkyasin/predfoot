"""ONNX export of small stand-ins for the PnLCalib networks: the same input, output names and heads (softmax
over the channels, sigmoid), without the weights. Needs torch and the 'export' extra."""

import numpy as np
import pytest

pytestmark = pytest.mark.vision


def test_export_network_writes_graphs_that_onnx_runtime_runs_like_pytorch(tmp_path):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    from predfoot.keypoints.onnx_models import (
        HEATMAP_TOLERANCE,
        INPUT_NAME,
        MODEL_FILES,
        OPSET,
        OUTPUT_NAME,
        OnnxPnLCalib,
        export_network,
        model_opset,
    )
    from predfoot.keypoints.pnlcalib import INPUT_SIZE, N_KEYPOINTS, N_LINES

    class Head(torch.nn.Module):
        """Stride-2 heatmaps of ``channels`` channels, background last, as the HRNet heads give them."""

        def __init__(self, channels: int, softmax: bool) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1)
            self.softmax = softmax

        def forward(self, x):
            y = self.conv(x)
            return torch.softmax(y, dim=1) if self.softmax else torch.sigmoid(y)

    torch.manual_seed(0)
    nets = {"kp": Head(N_KEYPOINTS + 1, softmax=True), "lines": Head(N_LINES + 1, softmax=False)}
    for name, net in nets.items():
        path = tmp_path / MODEL_FILES[name]
        export_network(net, path)
        model = onnx.load(str(path))
        assert model_opset(model) == OPSET
        [graph_input], [graph_output] = model.graph.input, model.graph.output
        assert (graph_input.name, graph_output.name) == (INPUT_NAME, OUTPUT_NAME)
        dims = [d.dim_value for d in graph_input.type.tensor_type.shape.dim]
        assert dims == [1, 3, INPUT_SIZE[1], INPUT_SIZE[0]]  # static: batch 1 at 960 × 540
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(MODEL_FILES.values())  # no temporary left

    x = torch.rand(2, 3, INPUT_SIZE[1], INPUT_SIZE[0])
    hk, hl = OnnxPnLCalib(tmp_path).heatmaps_from_input(x.numpy())
    with torch.inference_mode():
        ref_k, ref_l = nets["kp"](x)[:, :-1].numpy(), nets["lines"](x)[:, :-1].numpy()
    assert hk.shape == (2, N_KEYPOINTS, INPUT_SIZE[1] // 2, INPUT_SIZE[0] // 2)
    assert hl.shape == (2, N_LINES, INPUT_SIZE[1] // 2, INPUT_SIZE[0] // 2)
    assert np.abs(hk - ref_k).max() < HEATMAP_TOLERANCE
    assert np.abs(hl - ref_l).max() < HEATMAP_TOLERANCE


def test_the_file_has_the_requested_opset_or_the_export_fails(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from predfoot.keypoints import onnx_models

    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1), torch.nn.Sigmoid())
    path = tmp_path / "net.onnx"
    onnx_models.export_network(net, path, opset=17)  # built at opset 18, then converted down
    assert onnx_models.model_opset(onnx.load(str(path))) == 17
    # a conversion that silently kept the model as it was must not pass for the requested opset
    monkeypatch.setattr(onnx_models, "model_opset", lambda model: 18)
    path.unlink()
    with pytest.raises(RuntimeError, match="opset 18, not the 17 requested"):
        onnx_models.export_network(net, path, opset=17)
    assert list(tmp_path.iterdir()) == []
