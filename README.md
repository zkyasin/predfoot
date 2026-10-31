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

## Development

```bash
uv sync                      # core + dev tools
uv sync --extra vision       # + torch, for the keypoint model
uv run pre-commit install
uv run pytest
```

Datasets are downloaded at runtime and never committed. Set `PREDFOOT_DATA_DIR` to share them between
checkouts.

## Licence and credits

Apache-2.0. Third-party data and weights are listed in [ATTRIBUTION.md](ATTRIBUTION.md).
predfoot is an independent project, not affiliated with SkillCorner.
