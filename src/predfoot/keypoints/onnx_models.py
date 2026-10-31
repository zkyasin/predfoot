"""ONNX export of the PnLCalib networks, ONNX Runtime inference, and the parity check against PyTorch.

The exported graphs take the input of :func:`predfoot.keypoints.pnlcalib.preprocess` (float32 RGB in [0, 1],
1 × 3 × 540 × 960) and return the raw heatmaps, background channel included. The files derive from the
CC BY 4.0 weights and stay in ``PREDFOOT_DATA_DIR/models/onnx/``, never in git (ADR-0002).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from predfoot.keypoints.pnlcalib import INPUT_SIZE, decode_detections
from predfoot.paths import data_dir

OPSET = 18
INPUT_NAME = "frames"
OUTPUT_NAME = "heatmaps"
MODEL_FILES = {"kp": "pnlcalib-sv-kp.onnx", "lines": "pnlcalib-sv-lines.onnx"}
#: Acceptance of ticket T5.4: decoded keypoints and line ends within 0.5 px of PyTorch's.
PARITY_TOLERANCE_PX = 0.5

Heatmaps = tuple[NDArray[np.float32], NDArray[np.float32]]


def onnx_dir() -> Path:
    """Where the exported models live."""
    return data_dir() / "models" / "onnx"


def export_network(net: Any, path: Path, opset: int = OPSET) -> None:
    """Export one network for batch-1 inputs at 960 × 540 with torch's default exporter (torch.export), check
    the file with the ONNX checker, then move it into place."""
    import onnx
    import torch

    x = torch.zeros(1, 3, INPUT_SIZE[1], INPUT_SIZE[0])
    tmp = path.with_name(f".{path.name}.tmp")
    torch.onnx.export(
        net.eval(),
        (x,),
        str(tmp),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(str(tmp))
    tmp.replace(path)


class OnnxPnLCalib:
    """Both exported networks in ONNX Runtime sessions, with the outputs of
    :meth:`PnLCalibNetworks.heatmaps_from_input`."""

    def __init__(
        self, model_dir: Path | None = None, providers: Sequence[Any] = ("CPUExecutionProvider",)
    ) -> None:
        import onnxruntime as ort

        root = model_dir or onnx_dir()
        self.sessions = {
            name: ort.InferenceSession(str(root / file), providers=list(providers))
            for name, file in MODEL_FILES.items()
        }

    @property
    def providers(self) -> list[str]:
        """Execution providers in use, in priority order."""
        return list(self.sessions["kp"].get_providers())

    def heatmaps_from_input(self, x: Any) -> Heatmaps:
        """Heatmaps, background channel dropped, for a batch (B, 3, 540, 960) float32 (array or CPU tensor),
        run one frame at a time (the graphs have a fixed batch of 1)."""
        batch = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        hk, hl = [], []
        for i in range(len(batch)):
            frame = {INPUT_NAME: batch[i : i + 1]}
            hk.append(self.sessions["kp"].run([OUTPUT_NAME], frame)[0][:, :-1])
            hl.append(self.sessions["lines"].run([OUTPUT_NAME], frame)[0][:, :-1])
        return np.concatenate(hk), np.concatenate(hl)


def per_frame(run: Callable[[Any], Heatmaps], x: Any) -> Heatmaps:
    """``run`` on one frame of ``x`` at a time (bounded memory), outputs concatenated."""
    outs = [run(x[i : i + 1]) for i in range(len(x))]
    return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])


@dataclass(frozen=True)
class Parity:
    """How far a runtime's heatmaps and decoded detections are from a reference runtime's."""

    n_frames: int
    max_abs_heatmap: float  # both networks, every channel and cell
    max_keypoint_px: float  # keypoints both runtimes detect, pixels of the original frame
    max_line_end_px: float  # ends of the lines both runtimes detect
    n_points: int  # keypoints and line ends compared
    n_flips: int  # detections above the threshold in one runtime only

    @property
    def ok(self) -> bool:
        return max(self.max_keypoint_px, self.max_line_end_px) <= PARITY_TOLERANCE_PX

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"ok": self.ok, "tolerance_px": PARITY_TOLERANCE_PX}


def parity(reference: Heatmaps, other: Heatmaps, frame_size: tuple[int, int]) -> Parity:
    """Compare heatmaps of the same frames, then the detections decoded from them in frames of frame_size."""
    (hk_r, hl_r), (hk_o, hl_o) = reference, other
    max_abs = max(float(np.abs(hk_r - hk_o).max()), float(np.abs(hl_r - hl_o).max()))
    kp_px = line_px = 0.0
    n_points = n_flips = 0
    decoded = zip(
        decode_detections(hk_r, hl_r, frame_size), decode_detections(hk_o, hl_o, frame_size), strict=True
    )
    for (kr, lr), (ko, lo) in decoded:
        common, ir, io = np.intersect1d(kr.ids, ko.ids, return_indices=True)
        n_flips += len(kr.ids) + len(ko.ids) - 2 * len(common)
        if len(common):
            kp_px = max(kp_px, float(np.linalg.norm(kr.uv[ir] - ko.uv[io], axis=1).max()))
            n_points += len(common)
        common, jr, jo = np.intersect1d(lr.line_ids, lo.line_ids, return_indices=True)
        n_flips += len(lr.line_ids) + len(lo.line_ids) - 2 * len(common)
        if len(common):
            ends_r, ends_o = lr.uv[jr], lo.uv[jo]  # (m, 2, 2): the two ends come in either order
            same = np.linalg.norm(ends_r - ends_o, axis=2).max(axis=1)
            swapped = np.linalg.norm(ends_r - ends_o[:, ::-1], axis=2).max(axis=1)
            line_px = max(line_px, float(np.minimum(same, swapped).max()))
            n_points += 2 * len(common)
    return Parity(len(hk_r), max_abs, kp_px, line_px, n_points, n_flips)


def spread_frames(clip_dir: Path, n: int) -> list[Path]:
    """``n`` frame images of ``clip_dir/img1``, spread evenly over the clip."""
    files = sorted((clip_dir / "img1").glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no frame images in {clip_dir / 'img1'}")
    idx = np.unique(np.linspace(0, len(files) - 1, num=min(n, len(files))).round().astype(int))
    return [files[i] for i in idx]


def read_frames(paths: Sequence[Path]) -> NDArray[np.uint8]:
    """Frames (N, H, W, 3) uint8 RGB, read with OpenCV like the detection cache."""
    import cv2

    frames = []
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise OSError(f"cannot read {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames).astype(np.uint8)
