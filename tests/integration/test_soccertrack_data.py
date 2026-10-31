"""Orientation of the SoccerTrack v2 conversion, checked on the downloaded matches.

The unit tests build synthetic files on our reading of the dataset (corner y = 0 is the far touchline, the
corner frame completed by z = x × y points down). If a match used the other orientation, the conversion would
mirror the pitch and every synthetic test would still pass; these checks would not.
"""

import cv2
import numpy as np
import pytest

from predfoot.io import soccertrack as st

pytestmark = pytest.mark.data


@pytest.mark.parametrize("match", ["117092", "117093"])
def test_released_pose_in_the_internal_frame(match):
    if not st.raw_path(match, "camera_intrinsics.npz").is_file():
        pytest.skip(f"SoccerTrack v2 match {match} not downloaded (scripts/download_data.py soccertrack-v2)")
    calib = st.load_fisheye(match)
    names, xy, uv = st.load_keypoints(match)
    cam = (np.column_stack([xy, np.zeros(len(xy))]) - calib.C) @ calib.R.T
    assert (cam[:, 2] > 0).all()  # every keypoint in front of the camera
    proj, _ = cv2.fisheye.projectPoints(
        cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), calib.K, calib.D.reshape(4, 1)
    )
    rms = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - uv) ** 2, axis=1)))
    assert rms == pytest.approx(calib.rms_px, abs=1e-3)  # the provider's own fit, reproduced
    assert calib.C[2] > 0  # above the pitch
    assert calib.C[1] < -34  # behind the near touchline (internal y points to the far touchline)
    for x in ("0", "52.5", "105"):  # the near touchline (corner y = 68) is lower in the video
        assert uv[names.index(f"({x},68)"), 1] > uv[names.index(f"({x},0)"), 1]
