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


def test_detections_are_decoded_from_heatmaps_of_any_runtime():
    # numpy heatmaps stand in for an ONNX Runtime output: decoding needs neither torch nor the networks
    from predfoot.keypoints.pnlcalib import N_KEYPOINTS, N_LINES, decode_detections

    hk = np.zeros((1, N_KEYPOINTS, 270, 480))
    hk[0, 4] = gaussian_map(270, 480, [(240.0, 135.0)])  # keypoint id 5 at the frame centre
    hk[0, 9] = 0.2 * gaussian_map(270, 480, [(100.0, 50.0)])  # below the detection threshold
    hl = np.zeros((1, N_LINES, 270, 480))
    hl[0, 2] = gaussian_map(270, 480, [(50.0, 60.0), (300.0, 200.0)])  # line id 3, both ends visible
    [(kps, lines)] = decode_detections(hk, hl, (1920, 1080))
    np.testing.assert_array_equal(kps.ids, [5])
    np.testing.assert_allclose(kps.uv, [[960.0, 540.0]], atol=1e-6)
    np.testing.assert_array_equal(lines.line_ids, [3])
    ends = sorted(map(tuple, np.round(lines.uv[0], 3)))
    assert ends == [(200.0, 240.0), (1200.0, 800.0)]


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


def test_parity_measures_decoded_shifts_in_frame_pixels():
    # the ONNX parity check on synthetic heatmaps: no runtime needed
    from predfoot.keypoints.onnx_models import parity
    from predfoot.keypoints.pnlcalib import N_KEYPOINTS, N_LINES

    def heatmaps(kp_centre, line_ends):
        hk = np.zeros((1, N_KEYPOINTS, 270, 480), dtype=np.float32)
        hk[0, 0] = gaussian_map(270, 480, [kp_centre])
        hl = np.zeros((1, N_LINES, 270, 480), dtype=np.float32)
        hl[0, 4] = gaussian_map(270, 480, line_ends)
        return hk, hl

    reference = heatmaps((100.0, 50.0), [(20.0, 30.0), (200.0, 150.0)])
    same = parity(reference, reference, (1920, 1080))
    assert same.ok
    assert (same.max_abs_heatmap, same.max_keypoint_px, same.max_line_end_px, same.n_flips) == (0, 0, 0, 0)
    assert same.n_points == 3  # one keypoint, two line ends
    # a keypoint 0.25 cell away is 1 px away in a 1920 × 1080 frame (stride 2, input 960 × 540): too far
    moved = parity(reference, heatmaps((100.25, 50.0), [(200.0, 150.0), (20.0, 30.0)]), (1920, 1080))
    assert moved.max_keypoint_px == pytest.approx(1.0, abs=0.05)
    assert moved.max_line_end_px == pytest.approx(0.0, abs=1e-6)  # the order of the two ends does not matter
    assert not moved.ok
