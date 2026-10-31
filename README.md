# predfoot

**Calibrated position uncertainty for broadcast football tracking.**

Broadcast tracking data gives each player a position, but not how far off that position might be. predfoot
estimates the camera from the pitch lines of each broadcast frame, then gives every visible player:

- a pitch position with a **90 % uncertainty ellipse** whose coverage is checked on matches held out from
  every tuning step;
- a **calibration health score** that abstains before positions go wrong;
- an **error budget for physical KPIs** (distance, high-speed running, sprints, PSV-99).

Outputs follow the SkillCorner open-data JSONL schema, with the uncertainty as extra fields.

> Status: early development (milestone M0). Numbers will appear here only once measured.

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
