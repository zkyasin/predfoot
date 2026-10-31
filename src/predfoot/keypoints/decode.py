"""Heatmap decoding: peak per channel with sub-pixel refinement (own implementation, numpy only)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _refine_1d(left: FloatArray, centre: FloatArray, right: FloatArray) -> FloatArray:
    """Offset in [−0.5, 0.5] of the vertex of the parabola through three equally spaced samples."""
    denom = left - 2 * centre + right
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(denom < 0, 0.5 * (left - right) / denom, 0.0)
    return np.clip(off, -0.5, 0.5)


def decode_peaks(heatmaps: NDArray[np.floating], border: int = 1) -> tuple[FloatArray, FloatArray]:
    """Strongest peak of each channel.

    ``heatmaps`` has shape (C, H, W). Peaks closer than ``border`` cells to the edge are ignored, as border
    responses are the least reliable. Returns positions (C, 2) as (x, y) in heatmap cells, refined to
    sub-cell precision with a separable quadratic fit, and the peak scores (C,).
    """
    hm = np.asarray(heatmaps, dtype=np.float64)
    c, h, w = hm.shape
    inner = hm[:, border : h - border, border : w - border].reshape(c, -1)
    flat = inner.argmax(axis=1)
    iy, ix = np.divmod(flat, w - 2 * border)
    iy, ix = iy + border, ix + border
    ch = np.arange(c)
    score = hm[ch, iy, ix]
    has_x = (ix > 0) & (ix < w - 1)
    has_y = (iy > 0) & (iy < h - 1)
    dx = np.where(
        has_x,
        _refine_1d(hm[ch, iy, np.maximum(ix - 1, 0)], score, hm[ch, iy, np.minimum(ix + 1, w - 1)]),
        0.0,
    )
    dy = np.where(
        has_y,
        _refine_1d(hm[ch, np.maximum(iy - 1, 0), ix], score, hm[ch, np.minimum(iy + 1, h - 1), ix]),
        0.0,
    )
    return np.stack([ix + dx, iy + dy], axis=1), score


def decode_two_peaks(
    heatmaps: NDArray[np.floating], min_separation: int = 3, border: int = 1
) -> tuple[FloatArray, FloatArray]:
    """The two strongest, separated peaks of each channel (the two visible ends of a line).

    Returns positions (C, 2, 2) in heatmap cells and scores (C, 2); the second peak is searched outside a
    window of ``min_separation`` cells around the first.
    """
    hm = np.asarray(heatmaps, dtype=np.float64).copy()
    first_xy, first_s = decode_peaks(hm, border)
    for k in range(hm.shape[0]):
        x, y = round(first_xy[k, 0]), round(first_xy[k, 1])
        hm[
            k,
            max(0, y - min_separation) : y + min_separation + 1,
            max(0, x - min_separation) : x + min_separation + 1,
        ] = -np.inf
    second_xy, second_s = decode_peaks(hm, border)
    return np.stack([first_xy, second_xy], axis=1), np.stack([first_s, second_s], axis=1)


def cells_to_pixels(
    cells: FloatArray, stride: float, input_size: tuple[int, int], frame_size: tuple[int, int]
) -> FloatArray:
    """Map heatmap cells to pixels of the original frame.

    Cell i of a heatmap with ``stride`` sits at input pixel stride·i (the convention the network was trained
    with), and the input (width, height) is a resize of the frame (width, height).
    """
    sx = frame_size[0] / input_size[0]
    sy = frame_size[1] / input_size[1]
    out = np.asarray(cells, dtype=np.float64) * stride
    return out * np.array([sx, sy])
