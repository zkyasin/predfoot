"""The HTTP API. ``create_app()`` builds it; ``predfoot serve`` runs it with uvicorn.

- ``GET /healthz``: liveness, package version, loaded model versions.
- ``POST /v1/localize``: camera (H, Σθ) + foot pixels → pitch positions with calibrated 90 % regions.
- ``POST /v1/calibrate``: one broadcast frame (multipart upload) → PnLCalib keypoints and lines → calibration
  and image corners. Needs torch (extra ``vision``) and the PnLCalib weights, else 503.
- ``GET /metrics``: request counts and latencies in the Prometheus text format.

Uploaded images are only held in memory for the request (Starlette spools parts above 1 MiB to an anonymous
temporary file that has no name on disk); predfoot never writes them anywhere.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from predfoot import __version__
from predfoot.io.skillcorner import CORNER_KEYS
from predfoot.serve.calibrate import (
    Detector,
    LazyDetector,
    ModelUnavailable,
    calibrate_detections,
    load_pnlcalib,
)
from predfoot.serve.images import MAX_UPLOAD_BYTES, ImageRejected, decode_rgb, inspect_image
from predfoot.serve.localize import localize_feet, region_threshold
from predfoot.serve.middleware import (
    PROMETHEUS_CONTENT_TYPE,
    BodySizeLimitMiddleware,
    MetricsMiddleware,
    RequestMetrics,
)
from predfoot.serve.schemas import (
    CalibrateResponse,
    Health,
    Keypoint,
    LineDetection,
    LocalizeRequest,
    LocalizeResponse,
    Position,
)

log = logging.getLogger(__name__)

#: Body limit of the JSON endpoints: 1000 feet with covariances take about 150 kB.
MAX_JSON_BYTES = 1024 * 1024
#: Room for the multipart boundaries and part headers around the image of /v1/calibrate.
MULTIPART_SLACK_BYTES = 64 * 1024

_ERRORS: dict[int | str, dict[str, Any]] = {
    413: {"description": "Upload or image too large."},
    415: {"description": "Not a JPEG or PNG image, or a media type that does not match its content."},
    503: {"description": "Keypoint networks unavailable on this server (torch or weights missing)."},
}


def create_app(
    *,
    detector: Detector | None = None,
    weights_dir: Path | None = None,
    device: str = "auto",
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    preload: bool = False,
) -> FastAPI:
    """Build the API.

    ``detector`` replaces the PnLCalib networks (tests); otherwise they are loaded from ``weights_dir``
    (default ``PREDFOOT_DATA_DIR/weights/pnlcalib``) on ``device`` at the first /v1/calibrate request, or at
    start-up with ``preload``.
    """
    networks = LazyDetector(
        (lambda: detector) if detector is not None else (lambda: load_pnlcalib(weights_dir, device))
    )
    metrics = RequestMetrics()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if preload:
            try:
                await run_in_threadpool(networks.get)
            except ModelUnavailable as err:
                log.warning("keypoint networks not loaded at start-up: %s", err)
        yield

    app = FastAPI(
        title="predfoot",
        version=__version__,
        summary="Broadcast camera calibration and player pitch positions with calibrated 90 % regions.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's own handler echoes the offending input, and JSON has no NaN: a request holding NaN would
        # get a 500 instead of a 422.
        return JSONResponse(status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))})

    @app.get("/healthz")
    def healthz() -> Health:
        version = networks.version
        models = {} if version is None else {"pnlcalib": version}
        return Health(status="ok", version=__version__, models=models)

    @app.post("/v1/localize")
    def localize(request: LocalizeRequest) -> LocalizeResponse:
        """Pitch positions of foot pixels with their calibrated covariances and 90 % radii (null above the
        horizon)."""
        q, source = region_threshold(request.conformal_quantile, request.dof)
        feet = request.feet
        uv = np.array([[f.u, f.v] for f in feet], dtype=np.float64).reshape(-1, 2)
        cov_uv = None
        if any(f.cov_uv is not None for f in feet):
            cov_uv = np.array([np.zeros((2, 2)) if f.cov_uv is None else f.cov_uv for f in feet])
        out = localize_feet(request.homography, request.cov_theta, uv, cov_uv, q)
        positions = [
            Position(
                track_id=f.track_id,
                x=float(out.xy[i, 0]),
                y=float(out.xy[i, 1]),
                pos_cov_m2=out.cov[i].tolist(),
                p90_radius_m=float(out.radius[i]),
            )
            if out.valid[i]
            else Position(track_id=f.track_id, x=None, y=None, pos_cov_m2=None, p90_radius_m=None)
            for i, f in enumerate(feet)
        ]
        return LocalizeResponse(quantile=q, quantile_source=source, positions=positions)

    @app.post("/v1/calibrate", responses=_ERRORS)
    def calibrate(
        image: Annotated[UploadFile, File(description="Broadcast frame, JPEG or PNG, at most 20 MiB.")],
    ) -> CalibrateResponse:
        """Calibrate one broadcast frame: PnLCalib keypoints and lines, then the robust solver of the offline
        pipeline. The result feeds /v1/localize."""
        start = time.perf_counter()
        if image.size is not None and image.size > max_upload_bytes:
            raise HTTPException(413, f"upload larger than {max_upload_bytes} bytes")
        data = image.file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            raise HTTPException(413, f"upload larger than {max_upload_bytes} bytes")
        try:
            _, size = inspect_image(data, image.content_type)
            rgb = decode_rgb(data)
        except ImageRejected as err:
            raise HTTPException(err.status_code, str(err)) from err
        decoded = time.perf_counter()
        try:
            networks.get()  # loads the networks on the first request
        except ModelUnavailable as err:
            raise HTTPException(503, str(err)) from err
        loaded = time.perf_counter()
        kps, ends = networks.detect(rgb)
        detected = time.perf_counter()
        result, corners = calibrate_detections(kps, ends, size)
        done = time.perf_counter()
        ok = result.ok and result.H is not None and result.cov_theta is not None
        corner_values = [None] * len(CORNER_KEYS) if corners is None else corners.reshape(-1).tolist()
        return CalibrateResponse(
            status=result.status.value,
            homography=result.H.tolist() if ok and result.H is not None else None,
            cov_theta=result.cov_theta.tolist() if ok and result.cov_theta is not None else None,
            dof=result.dof if ok else None,
            rms_px=result.rms_px if ok else None,
            n_inliers=result.n_inliers,
            image_size=size,
            image_corners_projection=dict(zip(CORNER_KEYS, corner_values, strict=True)),
            keypoints=[
                Keypoint(id=int(k), u=float(p[0]), v=float(p[1]), conf=float(c))
                for k, p, c in zip(kps.ids, kps.uv, kps.conf, strict=True)
            ],
            lines=[
                LineDetection(id=int(k), ends=e.tolist(), conf=c.tolist())
                for k, e, c in zip(ends.line_ids, ends.uv, ends.conf, strict=True)
            ],
            model_version=networks.version or "",
            timings_ms={
                "decode": 1e3 * (decoded - start),
                "load": 1e3 * (loaded - decoded),
                "networks": 1e3 * (detected - loaded),
                "calibration": 1e3 * (done - detected),
            },
        )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type=PROMETHEUS_CONTENT_TYPE)

    routes = frozenset(path for route in app.routes if isinstance(path := getattr(route, "path", None), str))
    app.add_middleware(
        BodySizeLimitMiddleware,
        limits={"/v1/calibrate": max_upload_bytes + MULTIPART_SLACK_BYTES},
        default=MAX_JSON_BYTES,
    )
    # Added last, so outermost: it also counts the 413 answers of the size limit.
    app.add_middleware(MetricsMiddleware, metrics=metrics, routes=routes)
    return app


def _json_safe(value: Any) -> Any:
    """``value`` with its non-finite floats written as strings, since JSON has no NaN or infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value
