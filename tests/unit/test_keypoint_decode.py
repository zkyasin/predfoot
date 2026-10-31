import os
from pathlib import Path

import numpy as np
import pytest

from predfoot.keypoints.decode import cells_to_pixels, decode_peaks, decode_two_peaks


def gaussian_map(h, w, centres, sigma=2.0):
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros((h, w))
    for cx, cy in centres:
        out = np.maximum(out, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    return out


def test_sub_pixel_peaks_are_recovered(rng):
    centres = rng.uniform([5, 5], [115, 60], size=(20, 2))
    hm = np.stack([gaussian_map(68, 120, [c]) for c in centres])
    xy, score = decode_peaks(hm)
    assert np.abs(xy - centres).max() < 0.1
    assert (score > 0.9).all()


def test_border_peaks_are_ignored():
    hm = gaussian_map(30, 40, [(0, 15)], sigma=0.3)[None]  # a sharp response on the outermost column
    hm[0, 20, 30] = 0.5
    xy, _ = decode_peaks(hm, border=1)
    assert tuple(np.round(xy[0]).astype(int)) == (30, 20)


def test_two_ends_of_a_line():
    hm = gaussian_map(60, 100, [(10.3, 12.7), (80.6, 40.2)])[None]
    xy, score = decode_two_peaks(hm)
    found = sorted(map(tuple, np.round(xy[0], 1)))
    assert found == [(10.3, 12.7), (80.6, 40.2)]
    assert (score > 0.95).all()


def test_cells_map_to_frame_pixels():
    # Cell (240, 135) of a stride-2 heatmap on a 960 × 540 input is the centre of a 1920 × 1080 frame.
    px = cells_to_pixels(np.array([[240.0, 135.0]]), 2, (960, 540), (1920, 1080))
    np.testing.assert_allclose(px, [[960.0, 540.0]])


WEIGHTS = Path(os.environ.get("PREDFOOT_DATA_DIR", "data")) / "weights" / "pnlcalib"


@pytest.mark.vision
@pytest.mark.data
@pytest.mark.skipif(not (WEIGHTS / "SV_kp").exists(), reason="PnLCalib weights not downloaded")
def test_networks_load_and_run_on_a_blank_frame():
    from predfoot.keypoints.pnlcalib import PnLCalibNetworks

    nets = PnLCalibNetworks(WEIGHTS / "SV_kp", WEIGHTS / "SV_lines", device="cpu")
    hk, hl = nets.heatmaps(np.zeros((1, 540, 960, 3), dtype=np.uint8))
    assert hk.shape == (1, 57, 270, 480)
    assert hl.shape == (1, 23, 270, 480)
    assert np.isfinite(hk).all()
    assert np.isfinite(hl).all()
