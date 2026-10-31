"""PnLCalib keypoint and line networks (CC BY 4.0 weights) behind predfoot's own preprocessing and decoding.

Only the MIT HRNet model definition is reused (``hrnet_mit.py``); everything else is written here.
Interface facts from the public release: inputs are RGB frames scaled to [0, 1] and resized to 960 × 540,
without mean/std normalisation; outputs are heatmaps at stride 2. The keypoint network has 57 channels plus a
background channel under a softmax; the line network has 23 channels plus background under a sigmoid, each
line channel peaking at the two visible ends of its line.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from predfoot.contracts import KeypointObs
from predfoot.keypoints.decode import cells_to_pixels, decode_peaks, decode_two_peaks

INPUT_SIZE = (960, 540)  # (width, height)
STRIDE = 2
N_KEYPOINTS = 57
N_LINES = 23
#: Detection thresholds published with the model (tuned by its authors on SoccerNet calibration data).
KEYPOINT_THRESHOLD = 0.3434
LINE_THRESHOLD = 0.7867


def hrnet_w48_config(num_outputs: int) -> dict[str, Any]:
    """Standard HRNet-W48 configuration (HRNet paper and Microsoft reference implementation)."""

    def stage(modules: int, branches: int, blocks: str, channels: list[int]) -> dict[str, Any]:
        return {
            "NUM_MODULES": modules,
            "NUM_BRANCHES": branches,
            "BLOCK": blocks,
            "NUM_BLOCKS": [4] * branches,
            "NUM_CHANNELS": channels,
            "FUSE_METHOD": "SUM",
        }

    return {
        "MODEL": {
            "IMAGE_SIZE": list(INPUT_SIZE),
            "NUM_JOINTS": num_outputs,
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "STAGE1": stage(1, 1, "BOTTLENECK", [64]),
                "STAGE2": stage(1, 2, "BASIC", [48, 96]),
                "STAGE3": stage(4, 3, "BASIC", [48, 96, 192]),
                "STAGE4": stage(3, 4, "BASIC", [48, 96, 192, 384]),
            },
        }
    }


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class LineEnds:
    """Visible ends of the detected pitch lines in one frame."""

    line_ids: NDArray[np.int64]  # (M,) 1-based PnLCalib line ids
    uv: NDArray[np.float64]  # (M, 2, 2) the two ends, pixels
    conf: NDArray[np.float64]  # (M, 2)


class PnLCalibNetworks:
    """Load both networks once; call :meth:`detect` on batches of frames."""

    def __init__(self, weights_kp: Path, weights_lines: Path, device: str = "auto") -> None:
        import torch

        from predfoot.keypoints.hrnet_mit import get_cls_net

        self.device = pick_device(device)
        self.kp = get_cls_net(hrnet_w48_config(N_KEYPOINTS + 1), activation="softmax")
        self.lines = get_cls_net(hrnet_w48_config(N_LINES + 1), activation="sigmoid")
        for net, path in ((self.kp, weights_kp), (self.lines, weights_lines)):
            state = torch.load(path, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
            net.eval().to(self.device)
        self.version = (
            f"pnlcalib-sv@{sha256_file(Path(weights_kp))[:12]}+{sha256_file(Path(weights_lines))[:12]}"
        )

    def heatmaps(self, frames_rgb: NDArray[np.uint8]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Raw heatmaps for frames (B, H, W, 3) uint8 RGB: keypoints (B, 57, h, w), lines (B, 23, h, w)."""
        import torch
        from torchvision.transforms.functional import resize

        x = torch.from_numpy(np.ascontiguousarray(frames_rgb)).permute(0, 3, 1, 2).float() / 255.0
        if (x.shape[-1], x.shape[-2]) != INPUT_SIZE:
            x = resize(x, [INPUT_SIZE[1], INPUT_SIZE[0]], antialias=True)
        with torch.inference_mode():
            x = x.to(self.device)
            hk = self.kp(x)[:, :-1].float().cpu().numpy()
            hl = self.lines(x)[:, :-1].float().cpu().numpy()
        return hk, hl

    def detect(
        self,
        frames_rgb: NDArray[np.uint8],
        kp_threshold: float = KEYPOINT_THRESHOLD,
        line_threshold: float = LINE_THRESHOLD,
    ) -> list[tuple[KeypointObs, LineEnds]]:
        """Keypoints and line ends, in pixels of the original frames."""
        frame_size = (frames_rgb.shape[2], frames_rgb.shape[1])
        hk, hl = self.heatmaps(frames_rgb)
        out = []
        for b in range(len(frames_rgb)):
            cells, score = decode_peaks(hk[b])
            keep = score > kp_threshold
            kps = KeypointObs(
                ids=np.flatnonzero(keep).astype(np.int64) + 1,
                uv=cells_to_pixels(cells[keep], STRIDE, INPUT_SIZE, frame_size),
                conf=score[keep],
            )
            ends, escore = decode_two_peaks(hl[b])
            lkeep = (escore > line_threshold).all(axis=1)
            lines = LineEnds(
                line_ids=np.flatnonzero(lkeep).astype(np.int64) + 1,
                uv=cells_to_pixels(ends[lkeep], STRIDE, INPUT_SIZE, frame_size),
                conf=escore[lkeep],
            )
            out.append((kps, lines))
        return out
