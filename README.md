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
`calib_health` and `abstain`), `match.json` (loadable with kloppy), `meta.json` (versions, hashes, the stages
that ran and their conformal model), `metrics.json`, `minimap.png` and `qa_report.html`. A killed run resumes
from its finished blocks and produces the same bytes (only `metrics.json`, which holds timings, differs). The
synthetic clip comes from `scripts/make_synthetic_clip.py`; the frame schema is
`src/predfoot/io/schema/frame-1.0.json`.

### Real stages from detections

With a detections file (`predfoot-detections/1`: the PnLCalib keypoints and line ends of the clip's frames),
`predfoot run` runs the real stages instead of the stubs: robust calibration of each frame, delta-method
positions, and 90 % regions scaled by a Mondrian conformal model when one is given. No torch is needed, so the
CPU images run them too. Frames are keyed by their 0-based index in the clip file (`frame` in
`predfoot-clip/1`), not by image number as in the detection cache (1, 6, 11, … for SN-GSR frames downloaded
one in five): re-key cached detections with `predfoot.keypoints.cache.by_clip_index` before writing them with
`dumps_detections`, as `scripts/run_real_clip.py` does when it writes `input_clip.json`, `detections.json` and
`conformal.json` next to its outputs, so that `predfoot run --fps 5 --detections ...` reruns it from files. A
file that names frames beyond the clip is refused before anything runs, and a file without frame 0 gets a
warning. The golden fixture is such an input:

```bash
uv run predfoot run tests/fixtures/golden/clip.json --out out-golden/ \
    --detections tests/fixtures/golden/detections.json --conformal tests/fixtures/golden/conformal.json
uv run python -m predfoot.io.compare out-golden/ tests/fixtures/golden/expected/
```

### QA report

`qa_report.html` is one self-contained file that opens offline: inline CSS and SVG, the minimap embedded as
a PNG, no script and no external reference (a Content-Security-Policy forbids loading anything), every string
from the run files escaped. It shows the run summary (frames, published share, abstentions, provenance and
stages), the validation of the outputs plus consistency checks between `metrics.json`, `match.json` and
`tracking.jsonl`, a timeline of calibration health, abstentions and median p90 radius per frame (when a
column groups several frames, its shading is as high as the share of them that abstain), the distribution of
the p90 radii, the worst frames, the abstained spans and short definitions of the fields.
`predfoot qa` rebuilds it from the outputs of a finished run, byte for byte:

```bash
uv run predfoot qa out/                       # → out/qa_report.html
uv run predfoot qa out/ --output report.html
```

## Docker

`docker/Dockerfile.cpu` builds two CPU images without torch, both running as an unprivileged user
(uid 10001):

| Target | Contents | Default command |
|---|---|---|
| `cli` (default) | The command-line tools | `predfoot --help` |
| `serve` | The tools and the HTTP API (extra `serve`); `EXPOSE 8000`, a `HEALTHCHECK` on `/healthz` | `predfoot serve`, on `PREDFOOT_HOST=0.0.0.0` and `PREDFOOT_PORT=8000` |

```bash
docker build -f docker/Dockerfile.cpu --build-arg GIT_SHA="$(git rev-parse HEAD)" -t predfoot:cli .
mkdir -p out && chmod 0777 out
docker run --rm -v "$PWD/tests/fixtures:/in:ro" -v "$PWD/out:/out" \
    predfoot:cli run /in/synthetic_clip.json --out /out
docker run --rm -v "$PWD/out:/out:ro" predfoot:cli validate /out/tracking.jsonl --match /out/match.json

docker build -f docker/Dockerfile.cpu --target serve -t predfoot:serve .
docker run --rm -p 8000:8000 predfoot:serve
```

In the `serve` image, `/healthz`, `/v1/localize` and `/metrics` work; `/v1/calibrate` answers 503 because the
keypoint networks need torch (extra `vision`), which the CPU images leave out. `predfoot serve` reads its
interface and port from `PREDFOOT_HOST` and `PREDFOOT_PORT` when the command line does not set them, so
`docker run -p 8000:8000 predfoot:serve serve --conformal /models/conformal.json` still listens on every
interface; the `HEALTHCHECK` probes `PREDFOOT_PORT`, without proxies.

Two scripts smoke-test the images, locally or in CI: `docker/smoke-cli.sh IMAGE` (non-root user,
`predfoot run`, `validate` and `qa`, then the real stages on the golden fixture, whose outputs must match
`tests/fixtures/golden/expected/`; it runs on both targets) and `docker/smoke-serve.sh IMAGE` (a healthy
`HEALTHCHECK`, `/healthz`, `POST /v1/localize` with 200 and finite values, `POST /v1/calibrate` with 503,
`/metrics`). The `docker` workflow builds both targets on every pull request that touches what the images
contain and runs them; nothing is pushed there.

### Release

Bump the version in `pyproject.toml` and in `src/predfoot/__init__.py` (a unit test checks that they agree),
merge, then push a tag `v<version>`. The `release` workflow checks that the tag matches the package version,
runs the smoke scripts of the `docker` workflow on each image (and `predfoot version`), the HTTP checks
included for the `serve` image, and only then pushes them to the GitHub Container Registry with
`GITHUB_TOKEN`, with OCI labels (source, revision, version, licence):

| Image | Tags |
|---|---|
| `ghcr.io/zkyasin/predfoot` (cli) | `<version>`, `sha-<commit>` |
| `ghcr.io/zkyasin/predfoot` (serve) | `<version>-serve`, `sha-<commit>-serve` |

## Serving

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
(leave one game out). A `conformal_quantile` in the request overrides the model with one threshold for all
feet. Without either, q is the uncalibrated Laplace threshold, which covers
only about 18 % of the errors on the dev games. The model is calibrated on Σp from Σθ alone: a foot sent
with `cov_uv` gets a wider region than the one calibrated. predfoot never saves uploaded images (Starlette
spools uploads above 1 MiB to an anonymous temporary file that goes away with the request).

The keypoint networks also export to ONNX (opset 18, parity with PyTorch within 0.5 px on decoded keypoints)
and run on ONNX Runtime. [`reports/latency_m4.md`](reports/latency_m4.md) times both networks per frame on an
M4 Pro: 0.46 s with ONNX Runtime CoreML, 0.56 s with PyTorch MPS, about 1.1-1.9 s on the CPU (a 3-frame
smoke run on a busy machine, which inflated the CPU frames).

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

The golden test (`tests/integration/test_golden_real.py`) runs the real stages (calibration, conformal
regions, localization, export) on synthetic network detections of a known camera, committed in
`tests/fixtures/golden/`, and compares the outputs with `tests/fixtures/golden/expected/`: exact on structure,
ids and flags, within one unit of the last written decimal on numbers (macOS and Linux round a few values
differently). After an intended change of the outputs, regenerate and review the diff:

```bash
uv run python scripts/make_golden.py --update
```

Datasets are downloaded at runtime and never committed. Set `PREDFOOT_DATA_DIR` to share them between
checkouts.

## Licence and credits

Apache-2.0. Third-party data and weights are listed in [ATTRIBUTION.md](ATTRIBUTION.md).
predfoot is an independent project, not affiliated with SkillCorner.
