import io
import json
import pickle
import zipfile

import cv2
import numpy as np
import pytest

from predfoot.geometry import homography as hg
from predfoot.geometry import ptz
from predfoot.io import soccertrack as st


def test_far_touchline_has_positive_y_and_corners_map_to_the_pitch_rectangle():
    corners = np.array([[0.0, 0.0], [105.0, 0.0], [0.0, 68.0], [105.0, 68.0], [52.5, 34.0]])
    out = st.corner_to_internal_xy(corners)
    np.testing.assert_allclose(out, [[-52.5, 34], [52.5, 34], [-52.5, -34], [52.5, -34], [0, 0]])
    np.testing.assert_allclose(st.internal_to_corner_xy(out), corners)


def test_homography_to_internal_projects_the_same_pixels(rng):
    H = np.array([[0.9, -18.0, 1800.0], [-0.5, -7.0, 700.0], [-8.0e-4, -9.0e-3, 1.0]])  # synthetic
    corner = rng.uniform([0, 0], [105, 68], size=(50, 2))
    H_int = st.homography_to_internal(H)
    assert H_int[2, 2] == pytest.approx(1.0)
    np.testing.assert_allclose(
        hg.project(H_int, st.corner_to_internal_xy(corner)), hg.project(H, corner), rtol=1e-10
    )


def test_extrinsics_round_trip_and_project_identically(rng):
    rot = ptz.rotation(0.1, 0.3, 0.02)
    centre = np.array([1.0, -38.0, 12.0])
    rvec, tvec = st.internal_to_extrinsics(rot, centre)
    rot2, centre2 = st.extrinsics_to_internal(rvec, tvec)
    np.testing.assert_allclose(rot2, rot, atol=1e-12)
    np.testing.assert_allclose(centre2, centre, atol=1e-10)
    corner = rng.uniform([0, 0], [105, 68], size=(20, 2))
    K = np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]])
    via_cv, _ = cv2.projectPoints(np.c_[corner, np.zeros(len(corner))], rvec, tvec, K, None)
    cam = (np.c_[st.corner_to_internal_xy(corner), np.zeros(len(corner))] - centre) @ rot.T
    via_int = (cam @ K.T)[:, :2] / cam[:, 2:3]
    np.testing.assert_allclose(via_cv.reshape(-1, 2), via_int, atol=1e-8)


def _write_intrinsics(path, rvec, tvec):
    np.savez(
        path,
        K=np.eye(3) * 2.0,
        D=np.zeros((4, 1)),
        Knew=np.eye(3),
        rvecs=np.array([[[v] for v in rvec]], dtype=object),
        tvecs=np.array([[[v] for v in tvec]], dtype=object),
        rms=np.array(12.5),
    )


def test_npz_with_object_arrays_loads_without_pickle_code(tmp_path):
    path = tmp_path / "117092_camera_intrinsics.npz"
    _write_intrinsics(path, [-1.1, 0.05, -0.1], [-50.0, -20.0, 70.0])
    with pytest.raises(ValueError, match="allow_pickle"):
        np.load(path)["rvecs"]
    z = st.load_npz_safely(path)
    np.testing.assert_allclose(z["rvecs"].reshape(3), [-1.1, 0.05, -0.1])
    assert z["rms"] == pytest.approx(12.5)


class _Evil:
    def __reduce__(self):
        return (print, ("pwned",))


def test_npz_loader_refuses_other_pickled_globals(tmp_path):
    payload = pickle.dumps(np.array([_Evil()], dtype=object), protocol=4)
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer, {"descr": "|O", "fortran_order": False, "shape": (1,)})
    path = tmp_path / "evil.npz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("rvecs.npy", buffer.getvalue() + payload)  # the header writer adds the magic
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        st.load_npz_safely(path)


def test_load_fisheye_converts_the_pose(tmp_path):
    rot, centre = ptz.rotation(0.05, 0.25, 0.0), np.array([1.0, -38.0, 12.0])
    rvec, tvec = st.internal_to_extrinsics(rot, centre)
    folder = tmp_path / "soccertrack-v2" / "raw" / "1"
    folder.mkdir(parents=True)
    _write_intrinsics(folder / "1_camera_intrinsics.npz", rvec, tvec)
    cam = st.load_fisheye("1", root=tmp_path)
    np.testing.assert_allclose(cam.C, centre, atol=1e-9)
    np.testing.assert_allclose(cam.R, rot, atol=1e-12)
    assert cam.D.shape == (4,)


def test_fixed_point_maps_decode_to_the_float_maps(tmp_path, rng):
    h, w = 40, 64
    mx = (np.tile(np.arange(w, dtype=np.float32), (h, 1)) * 1.3 + rng.uniform(0, 1, (h, w))).astype(
        np.float32
    )
    my = (np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)) * 0.7 + 3.2).astype(np.float32)
    map1, map2 = cv2.convertMaps(mx, my, cv2.CV_16SC2)
    folder = tmp_path / "soccertrack-v2" / "raw" / "7"
    folder.mkdir(parents=True)
    np.save(folder / "7_mapx.npy", map1)
    np.save(folder / "7_mapy.npy", map2)
    fx, fy = st.load_rectify_maps("7", root=tmp_path)
    assert fx.shape == (h, w)
    assert np.abs(fx - mx).max() <= 1 / 32
    assert np.abs(fy - my).max() <= 1 / 32


def test_keypoints_and_mot_parsing():
    names, xy, uv = st.parse_keypoints(json.dumps({"(0,68)": [250.0, 1100.0], "(52.5,34)": [1900.0, 640.0]}))
    assert names == ["(0,68)", "(52.5,34)"]
    np.testing.assert_allclose(xy, [[-52.5, -34.0], [0.0, 0.0]])
    np.testing.assert_allclose(uv, [[250, 1100], [1900, 640]])
    mot = st.parse_mot("1,2,10,20,4,8,-1,-1,-1,-1\n0,1,950.25,740.5,20.5,35.0,-1,-1,-1,-1\n")
    assert mot["frame"].to_list() == [0, 1]
    np.testing.assert_allclose(st.foot_points(mot), [[950.25 + 10.25, 740.5 + 35.0], [12.0, 28.0]])
    assert st.parse_mot("").height == 0


def test_keypoint_groups_and_marked_landmarks():
    corner = [[0, 0], [16.5, 0], [52.5, 0], [105, 68], [77.5, 68], [52.5, 34], [16.5, 13.84], [0, 30.34]]
    groups, marked = st.keypoint_groups(st.corner_to_internal_xy(np.array(corner, dtype=np.float64)))
    assert groups.tolist() == ["far_touchline"] * 3 + ["near_touchline"] * 2 + ["interior"] * 3
    # touchline points other than the corners and the halfway line were placed by eye
    assert marked.tolist() == [True, False, True, True, False, True, True, True]
