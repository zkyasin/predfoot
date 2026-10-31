import numpy as np

from predfoot.calib.observations import LINE_WEIGHT, observations_from_detections
from predfoot.geometry.pitch import pnlcalib_keypoints


def test_elevated_keypoints_are_dropped_and_lines_become_observations():
    kps = pnlcalib_keypoints()
    det = {
        "kp_ids": np.array([5, 12, 25, 51]),  # 12 is the top of a goal post
        "kp_uv": np.array([[100.0, 200.0], [150.0, 250.0], [300.0, 400.0], [900.0, 500.0]]),
        "kp_conf": np.array([0.9, 0.8, 0.7, 0.6]),
        "line_ids": np.array([2, 7]),  # left penalty-area line (planar) and left crossbar (not planar)
        "line_uv": np.array([[[100.0, 210.0], [300.0, 390.0]], [[10.0, 10.0], [20.0, 20.0]]]),
    }
    xy, uv, w, lines = observations_from_detections(det)
    assert uv.shape == (3, 2)
    assert len(xy) == 3
    np.testing.assert_allclose(xy[0], kps[5][:2])
    np.testing.assert_allclose(w, [0.9, 0.7, 0.6])
    assert len(lines) == 1
    assert lines[0].weight == LINE_WEIGHT
    assert lines[0].uv.shape == (2, 2)
    xy2, _, _, lines2 = observations_from_detections(det, use_lines=False)
    assert len(xy2) == 3
    assert lines2 == []
