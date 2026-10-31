"""``/v1/calibrate``: PnLCalib detections of one frame → robust calibration → image footprint on the pitch.

The networks are loaded on the first request (or at start-up with ``preload``), once, and run one frame at a
time. Everything after the networks is the offline pipeline: :func:`observations_from_detections`, then the
RANSAC + soft-L1 solver with its Laplace covariance.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from predfoot.calib.observations import observations_from_detections
from predfoot.calib.solver import SolverConfig
from predfoot.calib.solver import calibrate as solve
from predfoot.contracts import CalibResult, FloatArray, KeypointObs
from predfoot.io.skillcorner import image_corners_projection
from predfoot.keypoints.pnlcalib import LineEnds
from predfoot.paths import data_dir

log = logging.getLogger(__name__)

#: Frame size the solver's pixel settings were tuned on (SN-GSR broadcast frames).
REFERENCE_SIZE = (1920, 1080)
WEIGHT_FILES = ("SV_kp", "SV_lines")


class Detector(Protocol):
    """What /v1/calibrate needs from a keypoint model: :class:`PnLCalibNetworks`, or a stand-in in tests."""

    version: str

    def detect(self, frames_rgb: NDArray[np.uint8]) -> list[tuple[KeypointObs, LineEnds]]: ...


class ModelUnavailable(RuntimeError):
    """The keypoint networks cannot be loaded here (torch missing, weights not downloaded)."""


def load_pnlcalib(weights_dir: Path | None = None, device: str = "auto") -> Detector:
    """PnLCalib networks from ``weights_dir`` (default ``PREDFOOT_DATA_DIR/weights/pnlcalib``).

    Raises :class:`ModelUnavailable` with a message for the API client (no server paths in it).
    """
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as err:
        raise ModelUnavailable(
            "the keypoint networks need torch, which is not installed on this server "
            "(install the 'vision' extra: uv sync --extra vision)"
        ) from err
    root = weights_dir or data_dir() / "weights" / "pnlcalib"
    paths = [root / name for name in WEIGHT_FILES]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        log.warning("PnLCalib weights not found: %s", ", ".join(map(str, missing)))
        raise ModelUnavailable(
            "the PnLCalib weights are not installed on this server "
            "(uv run python scripts/download_data.py pnlcalib-weights)"
        )
    from predfoot.keypoints.pnlcalib import PnLCalibNetworks

    return PnLCalibNetworks(paths[0], paths[1], device=device)


class LazyDetector:
    """Loads a detector on first use, exactly once even under concurrent requests, and runs it on one frame at
    a time: one GPU context, and two HRNet-W48 at 960 × 540 should not be held in memory twice.

    A failed load is not remembered: the next request tries again (for instance after the weights arrive).
    """

    def __init__(self, load: Callable[[], Detector]) -> None:
        self._load = load
        self._detector: Detector | None = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def version(self) -> str | None:
        """Version of the loaded detector, None before the first load."""
        detector = self._detector
        return None if detector is None else detector.version

    def get(self) -> Detector:
        detector = self._detector
        if detector is None:
            with self._load_lock:
                if self._detector is None:
                    self._detector = self._load()
                detector = self._detector
        return detector

    def detect(self, rgb: NDArray[np.uint8]) -> tuple[KeypointObs, LineEnds]:
        """Detections of one frame (H, W, 3) uint8 RGB."""
        detector = self.get()
        with self._run_lock:
            return detector.detect(rgb[None])[0]


def solver_config_for(image_size: tuple[int, int], base: SolverConfig | None = None) -> SolverConfig:
    """Solver settings for frames of ``image_size`` (width, height).

    The pixel settings were tuned on 1920 × 1080 frames. The networks see every frame at 960 × 540, so their
    errors, in pixels of the original frame, scale with its size, and so do these settings.
    """
    cfg = base or SolverConfig()
    s = math.sqrt(image_size[0] * image_size[1] / (REFERENCE_SIZE[0] * REFERENCE_SIZE[1]))
    return dataclasses.replace(
        cfg,
        ransac_threshold_px=cfg.ransac_threshold_px * s,
        loss_scale_px=cfg.loss_scale_px * s,
        sigma0_px=cfg.sigma0_px * s,
    )


def calibrate_detections(
    kps: KeypointObs, ends: LineEnds, image_size: tuple[int, int], seed: int = 0
) -> tuple[CalibResult, FloatArray | None]:
    """Calibration of one frame from its detections, and the SkillCorner image corners (None if unknown).

    RANSAC draws from a fixed seed, so the same image always gives the same answer.
    """
    det = {
        "kp_ids": kps.ids,
        "kp_uv": kps.uv,
        "kp_conf": kps.conf,
        "line_ids": ends.line_ids,
        "line_uv": ends.uv,
    }
    xy, uv, w, lines = observations_from_detections(det)
    result, _ = solve(
        xy, uv, np.random.default_rng(seed), weights=w, config=solver_config_for(image_size), lines=lines
    )
    corners = None
    if result.ok and result.H is not None:
        corners = image_corners_projection(result.H, image_size)
    return result, corners
