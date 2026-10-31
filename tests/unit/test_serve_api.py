"""HTTP API through FastAPI's test client, with a synthetic camera and a stand-in detector (no torch)."""

import json

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from predfoot import __version__
from predfoot.contracts import KeypointObs
from predfoot.geometry import homography as hg
from predfoot.io.skillcorner import CORNER_KEYS, image_corners_projection
from predfoot.serve.app import MAX_JSON_BYTES, create_app
from predfoot.serve.schemas import MAX_FEET
from predfoot.uncertainty.delta import CHI2_2_90, region_quantile
from tests.unit.serve_fakes import SyntheticDetector, broadcast_camera, laplace_covariance

PLAYERS = np.array([[5.0, -10.0], [20.0, 5.0], [30.0, 15.0], [45.0, -30.0]])
CAM = broadcast_camera()


def localize_body(**overrides):
    uv = hg.project(CAM.H, PLAYERS)
    body = {
        "homography": CAM.H.tolist(),
        "cov_theta": laplace_covariance(CAM).tolist(),
        "feet": [{"track_id": i, "u": float(u), "v": float(v)} for i, (u, v) in enumerate(uv)],
    }
    return body | overrides


def frame_png(width=1920, height=1080):
    ok, buf = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def client():
    return TestClient(create_app(detector=SyntheticDetector(CAM)))


def test_healthz_reports_versions(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__, "models": {}}  # networks load lazily


def test_localize_places_feet_on_the_pitch(client):
    r = client.post("/v1/localize", json=localize_body())
    assert r.status_code == 200
    body = r.json()
    assert body["quantile"] == pytest.approx(CHI2_2_90)
    assert body["quantile_source"] == "laplace_chi2"
    got = np.array([[p["x"], p["y"]] for p in body["positions"]])
    assert np.abs(got - PLAYERS).max() < 0.01  # within 1 cm of the geometry
    for i, p in enumerate(body["positions"]):
        assert p["track_id"] == i
        cov = np.array(p["pos_cov_m2"])
        np.testing.assert_allclose(cov, cov.T, atol=1e-15)
        assert np.linalg.eigvalsh(cov)[0] >= 0
        # the radius is the circle around the 90 % ellipse of the calibrated covariance
        assert p["p90_radius_m"] == pytest.approx(np.sqrt(CHI2_2_90 * np.linalg.eigvalsh(cov)[-1]))


def test_localize_scales_regions_with_the_quantile(client):
    base = client.post("/v1/localize", json=localize_body()).json()
    wide = client.post("/v1/localize", json=localize_body(conformal_quantile=4 * CHI2_2_90)).json()
    assert wide["quantile_source"] == "conformal"
    for b, w in zip(base["positions"], wide["positions"], strict=True):
        assert w["x"] == b["x"]
        assert w["p90_radius_m"] == pytest.approx(2 * b["p90_radius_m"])
        np.testing.assert_allclose(np.array(w["pos_cov_m2"]), 4 * np.array(b["pos_cov_m2"]))
    with_dof = client.post("/v1/localize", json=localize_body(dof=12.0)).json()
    assert with_dof["quantile_source"] == "laplace_f"
    assert with_dof["quantile"] == pytest.approx(region_quantile(2, 0.9, 12.0))


def test_localize_adds_foot_pixel_noise(client):
    body = localize_body()
    body["feet"][0]["cov_uv"] = [[4.0, 0.0], [0.0, 9.0]]
    got = client.post("/v1/localize", json=body).json()["positions"]
    base = client.post("/v1/localize", json=localize_body()).json()["positions"]
    assert got[0]["p90_radius_m"] > base[0]["p90_radius_m"]
    assert got[1]["p90_radius_m"] == pytest.approx(base[1]["p90_radius_m"])  # others stay exact


def test_feet_above_the_horizon_get_null_positions(client):
    g = np.linalg.inv(CAM.H)[2]
    v_horizon = -(g[0] * 960.0 + g[2]) / g[1]
    body = localize_body()
    body["feet"].append({"track_id": 99, "u": 960.0, "v": float(v_horizon - 40.0)})
    positions = client.post("/v1/localize", json=body).json()["positions"]
    assert positions[-1] == {"track_id": 99, "x": None, "y": None, "pos_cov_m2": None, "p90_radius_m": None}
    assert all(p["x"] is not None for p in positions[:-1])


def _bad_bodies():
    h = CAM.H.copy()
    cov = laplace_covariance(CAM)
    asym = cov.copy()
    asym[0, 1] += 10 * np.sqrt(cov[0, 0] * cov[1, 1])
    not_psd = cov - 2 * np.linalg.eigvalsh(cov)[-1] * np.eye(8)
    feet = localize_body()["feet"]
    return {
        "H[2][2] not 1": localize_body(homography=(2 * h).tolist()),
        "H not 3x3": localize_body(homography=h[:2].tolist()),
        "singular H": localize_body(homography=[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]),
        "cov_theta not 8x8": localize_body(cov_theta=cov[:7].tolist()),
        "cov_theta not symmetric": localize_body(cov_theta=asym.tolist()),
        "cov_theta not PSD": localize_body(cov_theta=not_psd.tolist()),
        "negative track id": localize_body(feet=[{"track_id": -1, "u": 1.0, "v": 2.0}]),
        "missing pixel": localize_body(feet=[{"track_id": 1, "u": 1.0}]),
        "cov_uv not PSD": localize_body(
            feet=[{"track_id": 1, "u": 1.0, "v": 2.0, "cov_uv": [[1.0, 0.0], [0.0, -1.0]]}]
        ),
        "unknown field": localize_body(conformal_q=5.0),
        "zero quantile": localize_body(conformal_quantile=0.0),
        "negative dof": localize_body(dof=-3.0),
        "too many feet": localize_body(feet=feet[:1] * (MAX_FEET + 1)),
        "no homography": {k: v for k, v in localize_body().items() if k != "homography"},
    }


@pytest.mark.parametrize("case", list(_bad_bodies()))
def test_invalid_localize_requests_are_rejected(client, case):
    r = client.post("/v1/localize", json=_bad_bodies()[case])
    assert r.status_code == 422, r.text


def test_non_finite_numbers_are_rejected(client):
    body = localize_body()
    body["feet"][0]["u"] = float("nan")
    text = json.dumps(body)  # Python writes the non-standard NaN token, which json.loads accepts
    assert "NaN" in text
    r = client.post("/v1/localize", content=text, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_oversized_json_bodies_are_refused(client):
    r = client.post(
        "/v1/localize", content=b" " * (MAX_JSON_BYTES + 1), headers={"content-type": "application/json"}
    )
    assert r.status_code == 413


def test_metrics_count_requests_by_route(client):
    client.get("/healthz")
    client.post("/v1/localize", json=localize_body(conformal_quantile=-1.0))
    client.get("/no/such/page")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert 'predfoot_http_requests_total{method="GET",route="/healthz",status="200"} 1.0' in text
    assert 'predfoot_http_requests_total{method="POST",route="/v1/localize",status="422"} 1.0' in text
    assert 'predfoot_http_requests_total{method="GET",route="other",status="404"} 1.0' in text
    assert 'predfoot_http_request_duration_seconds_count{route="/healthz"} 1.0' in text


def test_calibrate_returns_a_camera_that_localize_accepts():
    detector = SyntheticDetector(CAM)
    client = TestClient(create_app(detector=detector))
    r = client.post("/v1/calibrate", files={"image": ("frame.png", frame_png(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["image_size"] == [1920, 1080]
    assert body["n_inliers"] == len(body["keypoints"]) == 25
    assert body["lines"] == []
    assert body["model_version"] == "synthetic@test"
    assert set(body["timings_ms"]) == {"decode", "networks", "calibration"}
    cov = np.array(body["cov_theta"])
    assert cov.shape == (8, 8)
    np.testing.assert_allclose(cov, cov.T)
    assert body["homography"][2][2] == pytest.approx(1.0)
    corners = np.array([body["image_corners_projection"][k] for k in CORNER_KEYS]).reshape(4, 2)
    assert np.abs(corners - image_corners_projection(CAM.H, (1920, 1080))).max() < 0.2
    assert client.get("/healthz").json()["models"] == {"pnlcalib": "synthetic@test"}
    # the calibration feeds /v1/localize as it is
    loc = client.post(
        "/v1/localize",
        json=localize_body(homography=body["homography"], cov_theta=body["cov_theta"], dof=body["dof"]),
    )
    assert loc.status_code == 200, loc.text
    got = np.array([[p["x"], p["y"]] for p in loc.json()["positions"]])
    assert np.abs(got - PLAYERS).max() < 0.1
    assert detector.calls == 1


class ThreeKeypoints(SyntheticDetector):
    """A frame where the networks find only three keypoints: not enough for a homography."""

    def detect(self, frames_rgb):
        return [
            (KeypointObs(ids=k.ids[:3], uv=k.uv[:3], conf=k.conf[:3]), ends)
            for k, ends in super().detect(frames_rgb)
        ]


def test_calibrate_without_enough_keypoints_returns_nulls():
    client = TestClient(create_app(detector=ThreeKeypoints(CAM)))
    body = client.post("/v1/calibrate", files={"image": ("f.png", frame_png(), "image/png")}).json()
    assert body["status"] == "few_points"
    assert body["homography"] is None
    assert body["cov_theta"] is None
    assert body["dof"] is None
    assert set(body["image_corners_projection"].values()) == {None}


def _multipart(payload: bytes, content_type="image/png", boundary="predfoot-test"):
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="f.png"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    return head + payload + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def test_oversized_uploads_are_refused_before_being_read():
    detector = SyntheticDetector(CAM)
    client = TestClient(create_app(detector=detector, max_upload_bytes=10_000))
    body, content_type = _multipart(b"\x89PNG" + b"0" * 200_000)

    # announced by Content-Length: answered without reading the body
    r = client.post("/v1/calibrate", content=body, headers={"content-type": content_type})
    assert r.status_code == 413

    # streamed without a length: cut off once the limit is passed
    def chunks():
        for start in range(0, len(body), 4096):
            yield body[start : start + 4096]

    r = client.post("/v1/calibrate", content=chunks(), headers={"content-type": content_type})
    assert r.status_code == 413

    # a body within the multipart allowance whose image is just above the limit
    r = client.post("/v1/calibrate", files={"image": ("f.png", b"0" * 10_001, "image/png")})
    assert r.status_code == 413
    assert detector.calls == 0


@pytest.mark.parametrize(
    ("payload", "content_type"),
    [
        (b"just text", "text/plain"),
        (b"not a png", "image/png"),
        ("png", "image/jpeg"),
        ("png", "application/octet-stream"),
    ],
)
def test_non_images_are_refused(payload, content_type):
    detector = SyntheticDetector(CAM)
    client = TestClient(create_app(detector=detector))
    data = frame_png(64, 36) if payload == "png" else payload
    r = client.post("/v1/calibrate", files={"image": ("f", data, content_type)})
    assert r.status_code == 415
    assert detector.calls == 0


def test_calibrate_needs_an_image_field(client):
    r = client.post("/v1/calibrate", files={"picture": ("f.png", frame_png(64, 36), "image/png")})
    assert r.status_code == 422


def test_calibrate_answers_503_without_the_networks(tmp_path):
    client = TestClient(create_app(weights_dir=tmp_path / "no-weights"))
    r = client.post("/v1/calibrate", files={"image": ("f.png", frame_png(64, 36), "image/png")})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "weights" in detail or "torch" in detail
    assert str(tmp_path) not in detail
    assert client.get("/healthz").json()["models"] == {}
