# predfoot

**Calibrated position uncertainty for broadcast football tracking.**

Broadcast tracking data gives each player a position, but not how far off that position might be. predfoot
estimates the camera from the pitch lines of each broadcast frame, then gives every visible player:

- a pitch position with a **90 % uncertainty ellipse** whose coverage is checked on matches held out from
  every tuning step;
- a **calibration health score** that abstains before positions go wrong;
- an **error budget for physical KPIs** (distance, high-speed running, sprints, PSV-99).

Outputs follow the SkillCorner open-data JSONL schema, with the uncertainty as extra fields.

> Status: early development. The pipeline runs end to end on real broadcast frames; development results are
> in [`reports/`](reports/). Test-set numbers will appear here only after the preregistered run
> ([`docs/preregistration.md`](docs/preregistration.md), draft).

## Layout

| Path | Contents |
|---|---|
| `src/predfoot/` | The package: `geometry`, `keypoints`, `calib`, `uncertainty`, `health`, `detect`, `io`, `kpi`, `eval`, `viz`, `serve` |
| `tests/` | `unit`, `property` (hypothesis), `integration`, synthetic `fixtures` only |
| `scripts/` | Data download and verification, experiments, checks |
| `configs/` | YAML configs, validated with pydantic |
| `docs/adr/` | Architecture decision records |
| `notebooks/tutorials/` | Numbered tutorials |
| `reports/` | Generated reports with aggregate numbers |
| `assets/` | Images for the documentation |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline and [CLAUDE.md](CLAUDE.md) for the working rules.

## Real pipeline on a development clip

```bash
uv sync --extra vision
uv run python scripts/download_data.py sn-gsr-labels --splits train valid
uv run python scripts/build_index.py
uv run python scripts/download_data.py sn-gsr-frames --clips SNGS-056 --every 5
uv run python scripts/download_data.py pnlcalib-weights
uv run python scripts/cache_keypoints.py --splits valid      # network inference, once per clip
uv run python scripts/run_real_clip.py SNGS-056 --out runs/real/SNGS-056
```

SoccerNet-derived outputs stay local: they are for evaluation only and are never published.

## Tracer bullet

The pipeline already runs end to end with stub stages (homographies and foot points read from a small clip
JSON, placeholder uncertainty); each real module will replace one stub.

```bash
uv run predfoot run tests/fixtures/synthetic_clip.json --out out/
uv run predfoot validate out/tracking.jsonl --match out/match.json
```

`out/` then holds `tracking.jsonl` (SkillCorner open-data schema at 10 fps, plus `p90_radius_m`, `pos_cov_m2`,
`calib_health` and `abstain`), `match.json` (loadable with kloppy), `meta.json`, `metrics.json` and
`minimap.png`. A killed run resumes from its finished blocks and produces the same bytes. The synthetic clip
comes from `scripts/make_synthetic_clip.py`; the frame schema is `src/predfoot/io/schema/frame-1.0.json`.

```bash
docker build -f docker/Dockerfile.cpu -t predfoot:cpu .   # CPU image, no torch
```

## Serving

```bash
uv sync --extra serve --extra vision      # vision (torch) is only needed by /v1/calibrate
uv run predfoot serve --port 8000         # OpenAPI docs at http://127.0.0.1:8000/docs
```

| Endpoint | In → out |
|---|---|
| `GET /healthz` | Status, package version, versions of the loaded models (the networks load on first use, or with `--preload`) |
| `POST /v1/calibrate` | Multipart `image` (JPEG or PNG, at most 20 MiB) → PnLCalib keypoints and lines → `status`, `homography`, `cov_theta`, `dof`, `rms_px`, `n_inliers`, `image_corners_projection`. 503 when torch or the weights are missing |
| `POST /v1/localize` | `homography` (pitch → image, H[2][2] = 1), `cov_theta` (8 × 8), optional `dof`, `feet` [{`track_id`, `u`, `v`, optional `cov_uv`}], optional `conformal_quantile` → per foot `x`, `y`, `pos_cov_m2`, `p90_radius_m`, all null above the horizon |
| `GET /metrics` | Request counters and latency histograms (Prometheus text format) |

```bash
curl -s -F "image=@frame.jpg;type=image/jpeg" localhost:8000/v1/calibrate > calib.json
jq '{homography, cov_theta, dof, feet: [{track_id: 7, u: 960, v: 800}]}' calib.json \
  | curl -s -H "content-type: application/json" -d @- localhost:8000/v1/localize
```

`/v1/localize` uses the definitions of the pipeline: the calibrated covariance is (q / 4.605) · Σp and the
radius √(q · λmax(Σp)). Without `conformal_quantile`, q is the uncalibrated Laplace threshold, which covers
only about 18 % of the errors on the dev games: pass the conformal quantile of a calibration run. predfoot
never saves uploaded images (Starlette spools uploads above 1 MiB to an anonymous temporary file that goes
away with the request).

The keypoint networks also export to ONNX (opset 18, parity with PyTorch within 0.5 px on decoded keypoints)
and run on ONNX Runtime. [`reports/latency_m4.md`](reports/latency_m4.md) times both networks per frame on an
M4 Pro: 0.46 s with ONNX Runtime CoreML, 0.56 s with PyTorch MPS, 1.8-1.9 s on the CPU (3-frame smoke run).

```bash
uv sync --extra vision --extra export
uv run python scripts/export_onnx.py      # → $PREDFOOT_DATA_DIR/models/onnx/, never git
uv run python scripts/bench_latency.py --frames 3 --out reports/latency_m4.md
```

## Development

```bash
uv sync                      # core + dev tools
uv sync --extra vision       # + torch, for the keypoint model
uv sync --extra serve        # + FastAPI, for the HTTP API and its tests (CI installs it)
uv run pre-commit install
uv run pytest
```

Datasets are downloaded at runtime and never committed. Set `PREDFOOT_DATA_DIR` to share them between
checkouts.

## Licence and credits

Apache-2.0. Third-party data and weights are listed in [ATTRIBUTION.md](ATTRIBUTION.md).
predfoot is an independent project, not affiliated with SkillCorner.
