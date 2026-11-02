# HTTP API and inference latency

```bash
uv sync --extra serve --extra vision      # vision (torch) is only needed by /v1/calibrate
uv run python scripts/fit_conformal.py runs/<eval_dev run>/players.parquet --out conformal.json
uv run predfoot serve --port 8000 --conformal conformal.json   # OpenAPI docs at http://127.0.0.1:8000/docs
```

| Endpoint | In → out |
|---|---|
| `GET /healthz` | Status, package version, versions of the loaded models: networks (loaded on first use, or with `--preload`) and conformal model |
| `POST /v1/calibrate` | Multipart `image` (JPEG or PNG, at most 20 MiB) → PnLCalib keypoints and lines → `status`, `homography`, `cov_theta`, `dof`, `rms_px`, `n_inliers`, `image_corners_projection`. 503 when the networks cannot load (torch or weights missing), or with `Retry-After` when 8 calibrations are already running or queued |
| `POST /v1/localize` | `homography` (pitch → image, H[2][2] = 1), `cov_theta` (8 × 8), `n_inliers`, optional `dof`, `feet` [{`track_id`, `u`, `v`, optional `cov_uv`}], optional `conformal_quantile` → per foot `x`, `y`, `pos_cov_m2`, `p90_radius_m`, `quantile`, all null above the horizon |
| `GET /metrics` | Request counters and latency histograms (Prometheus text format) |

```bash
curl -s -F "image=@frame.jpg;type=image/jpeg" localhost:8000/v1/calibrate > calib.json
jq '{homography, cov_theta, dof, n_inliers, feet: [{track_id: 7, u: 960, v: 800}]}' calib.json \
  | curl -s -H "content-type: application/json" -d @- localhost:8000/v1/localize
```

`/v1/localize` uses the definitions of the pipeline: the calibrated covariance is (q / 4.605) · Σp and the
radius √(q · λmax(Σp)). With `--conformal`, each foot gets the q̂ of its Mondrian bin (metres per pixel at
the feet × RANSAC inliers, hence `n_inliers`), exactly as the real pipeline stages do with the same model
(`scripts/run_real_clip.py`); this is the construction that covered 90.4 % of the held-out dev errors
(leave one game out, per-frame camera, `reports/dev_per_frame.md`). A `conformal_quantile` in the request
overrides the model with one threshold for all feet. Without either, q is the uncalibrated Laplace threshold,
which covers only 17.9 % of the errors on the dev games. The model is calibrated on Σp from Σθ alone: a foot
sent with `cov_uv` gets a wider region than the one calibrated. predfoot never saves uploaded images
(Starlette spools uploads above 1 MiB to an anonymous temporary file that goes away with the request).

The CPU Docker image serves `/healthz`, `/v1/localize` and `/metrics`; `/v1/calibrate` needs torch
([`docker.md`](docker.md)).

## ONNX export and latency

The keypoint networks also export to ONNX (opset 18, parity with PyTorch within 0.5 px on decoded keypoints)
and run on ONNX Runtime. [`reports/latency_m4.md`](../reports/latency_m4.md) times both networks per frame on
an M4 Pro: 0.46 s with ONNX Runtime CoreML, 0.56 s with PyTorch MPS, about 1.1-1.9 s on the CPU (a 3-frame
smoke run on a busy machine, which inflated the CPU frames).

```bash
uv sync --extra vision --extra export
uv run python scripts/export_onnx.py      # → $PREDFOOT_DATA_DIR/models/onnx/, never git
uv run python scripts/bench_latency.py --frames 3 --out reports/latency_m4.md
```
