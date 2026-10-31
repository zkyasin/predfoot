"""SoccerTrack v2 (CC BY 4.0, atomscott/soccertrack-v2): calibration files and MOT boxes of one match.

Image spaces, checked on match 117092 (``reports/soccertrack_v2.md``):

- **video**: the half-match panorama videos and the 4-minute MOT clips share one pixel space (117092:
  3840 × 1906; most matches 4096 × 1080). It is a stitch of two wide-angle cameras. ``keypoints.json`` (65
  hand-annotated pitch points) and the MOT boxes are in it.
- **rectified**: the same scene undistorted with a fisheye model, same size, a pinhole image with intrinsics
  ``Knew``. ``mapx.npy`` and ``mapy.npy`` are OpenCV fixed-point remap tables (``CV_16SC2`` integer
  coordinates plus a ``CV_16UC1`` interpolation index) from rectified pixels to video pixels, equal to
  ``cv2.fisheye.initUndistortRectifyMap(K, D, I, Knew, size)``.
- ``homography.npy`` maps corner-origin pitch metres to rectified pixels.

Pitch frames: the raw files put the origin at a corner, x ∈ [0, 105] to the right and y ∈ [0, 68] towards the
near (camera) touchline. In the internal frame (ADR-0001) a point is (x − 52.5, 34 − y): the centre spot is
the origin and y points to the far touchline. The extrinsics of ``camera_intrinsics.npz`` use the corner frame
completed by z = x × y, which points down; the internal frame has z up.
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from predfoot.contracts import FloatArray
from predfoot.geometry import homography as hg
from predfoot.io.conventions import internal_to_sn_xy, sn_to_internal_xy
from predfoot.paths import data_dir

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
#: Corner-origin pitch point of the centre spot.
CENTRE_CORNER: FloatArray = np.array([PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2])
#: Homogeneous map from internal pitch points to corner-origin points: p_corner = A p_internal.
INTERNAL_TO_CORNER: FloatArray = np.array(
    [[1.0, 0.0, PITCH_LENGTH_M / 2], [0.0, -1.0, PITCH_WIDTH_M / 2], [0.0, 0.0, 1.0]]
)
#: The same change of frame in 3D, X_corner = A3 X_internal + (52.5, 34, 0): a proper rotation (det = +1).
_A3: FloatArray = np.diag([1.0, -1.0, -1.0])
_B3: FloatArray = np.array([PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2, 0.0])

MOT_COLUMNS = ("frame", "track_id", "x", "y", "w", "h")


def root_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "soccertrack-v2"


def raw_path(match: str, name: str, root: Path | None = None) -> Path:
    """``raw/<match>/<match>_<name>`` under the SoccerTrack v2 data folder."""
    return root_dir(root) / "raw" / match / f"{match}_{name}"


def mot_path(match: str, root: Path | None = None) -> Path:
    return root_dir(root) / "mot" / f"{match}.txt"


def clip_path(match: str, root: Path | None = None) -> Path:
    return root_dir(root) / "mot" / "clips" / f"{match}.mp4"


# --- pitch frames ------------------------------------------------------------------------------------------


def corner_to_internal_xy(xy: ArrayLike) -> FloatArray:
    """Corner-origin SoccerTrack pitch points (..., 2) → internal frame."""
    return sn_to_internal_xy(np.asarray(xy, dtype=np.float64) - CENTRE_CORNER)


def internal_to_corner_xy(xy: ArrayLike) -> FloatArray:
    """Inverse of :func:`corner_to_internal_xy`."""
    return internal_to_sn_xy(xy) + CENTRE_CORNER


def homography_to_internal(H_corner: ArrayLike) -> FloatArray:
    """A homography from corner-origin pitch points to pixels, rewritten for internal pitch points.

    The result is normalised so that H[2, 2] = 1 (the depth of the centre spot, see ``CalibResult``).
    """
    return hg.normalise(np.asarray(H_corner, dtype=np.float64) @ INTERNAL_TO_CORNER)


def extrinsics_to_internal(rvec: ArrayLike, tvec: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """OpenCV extrinsics of the corner frame (X_cam = R X_corner + t) → (R, C) of the internal frame.

    Returns the world → camera rotation R (X_cam = R (X − C)) and the projection centre C (metres, z up).
    """
    r_corner, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    rot = r_corner @ _A3
    t_int = t + r_corner @ _B3
    return np.asarray(rot, dtype=np.float64), np.asarray(-rot.T @ t_int, dtype=np.float64)


def internal_to_extrinsics(rot: ArrayLike, centre: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Inverse of :func:`extrinsics_to_internal`: internal (R, C) → corner-frame (rvec, tvec)."""
    r_int = np.asarray(rot, dtype=np.float64)
    c_int = np.asarray(centre, dtype=np.float64).reshape(3)
    r_corner = r_int @ _A3
    t_corner = -r_int @ c_int - r_corner @ _B3
    rvec, _ = cv2.Rodrigues(r_corner)
    return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(t_corner, dtype=np.float64)


# --- calibration files -------------------------------------------------------------------------------------


class _ArrayOnlyUnpickler(pickle.Unpickler):
    """Unpickler for numpy object arrays of plain numbers: any other global is refused."""

    ALLOWED = frozenset(
        {
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
        }
    )

    def find_class(self, module: str, name: str) -> type:
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


def load_npz_safely(path: Path) -> dict[str, FloatArray]:
    """Arrays of an .npz as float64, without ever unpickling anything but numbers.

    ``camera_intrinsics.npz`` stores ``rvecs`` and ``tvecs`` as object arrays (lists of floats), which
    ``np.load(allow_pickle=False)`` refuses and ``allow_pickle=True`` would load by running arbitrary code.
    """
    out: dict[str, FloatArray] = {}
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            raw = zf.read(member)
            buffer = io.BytesIO(raw)
            version = np.lib.format.read_magic(buffer)
            header = (
                np.lib.format.read_array_header_1_0(buffer)
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0(buffer)
            )
            shape, _, dtype = header
            if dtype.hasobject:
                array = np.asarray(_ArrayOnlyUnpickler(buffer).load(), dtype=object)
                values = np.array(array.tolist(), dtype=np.float64).reshape(shape)
            else:
                values = np.asarray(np.load(io.BytesIO(raw), allow_pickle=False), dtype=np.float64)
            out[member.removesuffix(".npy")] = values
    return out


@dataclass(frozen=True)
class FisheyeCalibration:
    """The released camera of a match: OpenCV fisheye model of the video and pose of the pitch.

    ``K`` and ``D`` (k1…k4) describe the video, ``Knew`` the rectified image; ``R`` and ``C`` are the pose in
    the internal pitch frame (X_cam = R (X − C), metres, z up). ``rms_px`` is the reprojection RMS the
    provider reports for its fit on the 65 keypoints.
    """

    K: FloatArray
    D: FloatArray  # (4,)
    Knew: FloatArray
    R: FloatArray
    C: FloatArray
    rms_px: float


def load_fisheye(match: str, root: Path | None = None) -> FisheyeCalibration:
    z = load_npz_safely(raw_path(match, "camera_intrinsics.npz", root))
    rot, centre = extrinsics_to_internal(z["rvecs"].reshape(3), z["tvecs"].reshape(3))
    return FisheyeCalibration(
        K=z["K"], D=z["D"].reshape(4), Knew=z["Knew"], R=rot, C=centre, rms_px=float(z["rms"])
    )


def load_homography(match: str, root: Path | None = None) -> FloatArray:
    """``homography.npy`` as released: corner-origin pitch metres → rectified pixels."""
    return np.asarray(np.load(raw_path(match, "homography.npy", root), allow_pickle=False), dtype=np.float64)


def parse_keypoints(text: str) -> tuple[list[str], FloatArray, FloatArray]:
    """``keypoints.json`` → (names, internal pitch points (n, 2), video pixels (n, 2))."""
    data: dict[str, list[float]] = json.loads(text)
    names = list(data)
    corner = np.array([[float(v) for v in name.strip("() ").split(",")] for name in names], dtype=np.float64)
    uv = np.array([data[name] for name in names], dtype=np.float64).reshape(-1, 2)
    return names, corner_to_internal_xy(corner.reshape(-1, 2)), uv


def load_keypoints(match: str, root: Path | None = None) -> tuple[list[str], FloatArray, FloatArray]:
    return parse_keypoints(raw_path(match, "keypoints.json", root).read_text())


def decode_maps(map1: NDArray[np.int16], map2: NDArray[np.uint16]) -> tuple[FloatArray, FloatArray]:
    """OpenCV fixed-point remap tables → float32 maps (x, y), exact to the 1/32 px grid of the tables."""
    mx, my = cv2.convertMaps(np.ascontiguousarray(map1), np.ascontiguousarray(map2), cv2.CV_32FC1)
    return np.asarray(mx, dtype=np.float32), np.asarray(my, dtype=np.float32)


def load_rectify_maps(match: str, root: Path | None = None) -> tuple[FloatArray, FloatArray]:
    """Float maps from rectified pixels to video pixels, each of shape (height, width)."""
    map1 = np.load(raw_path(match, "mapx.npy", root), allow_pickle=False)
    map2 = np.load(raw_path(match, "mapy.npy", root), allow_pickle=False)
    if map1.dtype != np.int16 or map1.ndim != 3 or map2.dtype != np.uint16:
        raise ValueError(f"unexpected remap tables: {map1.dtype}{map1.shape}, {map2.dtype}{map2.shape}")
    return decode_maps(map1, map2)


# --- MOT boxes ---------------------------------------------------------------------------------------------


def parse_mot(text: str) -> pl.DataFrame:
    """MOT ground truth (``frame,id,x,y,w,h,-1,-1,-1,-1``, top-left boxes in video pixels, 0-based frames)."""
    rows = [line.split(",")[:6] for line in text.splitlines() if line.strip()]
    if not rows:
        return pl.DataFrame(schema=dict.fromkeys(MOT_COLUMNS, pl.Float64))
    arr = np.array(rows, dtype=np.float64)
    df = pl.DataFrame({name: arr[:, i] for i, name in enumerate(MOT_COLUMNS)})
    return df.with_columns(pl.col("frame").cast(pl.Int64), pl.col("track_id").cast(pl.Int64)).sort(
        "frame", "track_id"
    )


def load_mot(match: str, root: Path | None = None) -> pl.DataFrame:
    return parse_mot(mot_path(match, root).read_text())


def foot_points(boxes: pl.DataFrame) -> FloatArray:
    """Bottom centre of each box, (n, 2) video pixels: the ground contact point (``FootObs`` convention)."""
    x, y, w, h = (boxes[c].to_numpy().astype(np.float64) for c in ("x", "y", "w", "h"))
    return np.column_stack([x + w / 2, y + h])
