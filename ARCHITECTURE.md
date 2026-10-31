# Architecture

## Pipeline

```mermaid
flowchart LR
  V["Frames"] --> KP["keypoints/<br/>HRNet heatmaps → keypoints + line points"]
  KP --> CAL["calib/<br/>DLT + RANSAC → robust LM → Laplace Σθ"]
  DET["detect/<br/>ground-truth or detector boxes"] -- "feet u, Σu" --> UNC
  CAL -- "H, Σθ" --> UNC["uncertainty/<br/>delta method → Σp"]
  CAL --> HL["health/<br/>features + critic → abstain"]
  UNC --> CF["uncertainty/conformal<br/>Mondrian quantiles"]
  HL --> CF
  CF --> IO["io/<br/>SkillCorner JSONL + extensions"]
  IO --> KPI["kpi/<br/>physical KPI error budget"]
  IO --> VZ["viz/<br/>minimap, QA report"]
  IO --> SV["cli, serve/"]
```

The keypoint network runs **once per frame**; its outputs are cached as Parquet, keyed by the hash of the
weights and of the decoding config. Every later stage reads the cache, so experiments run in minutes.

## Modules

| Module | Input → output |
|---|---|
| `geometry` | Pitch template (FIFA dimensions), homography helpers, projections, Jacobians |
| `keypoints` | Frame → `KeypointObs`, `LinePointObs` (pixels); ONNX export and ONNX Runtime inference |
| `calib` | Observations → `CalibResult` (H, Σθ, residuals, status) |
| `uncertainty` | `CalibResult` + `FootObs` → `Localization` (position, Σp); conformal calibration |
| `health` | Calibration features (+ frame) → failure probability, abstention |
| `detect` | Frame → `FootObs` (ground-truth boxes for evaluation, a detector for demos) |
| `io` | Dataset loaders, coordinate conventions, SkillCorner JSONL writer and reader, export time base, JSON Schema and validator |
| `kpi` | Tracking → physical KPIs; error injection; KPI error budget |
| `eval` | Metrics, coverage, risk–coverage, cluster bootstrap |
| `viz` | Minimaps, overlays, per-run QA report |
| `serve` | HTTP API (FastAPI): `/v1/calibrate`, `/v1/localize`, `/healthz`, `/metrics` |
| `pipeline` | Orchestration of the stages per frame, export blocks with resume; stub stages until the real modules land |

Shared contracts live in `predfoot/contracts.py`; conventions in [ADR-0001](docs/adr/0001-conventions.md).

## Outputs of a run

`tracking.jsonl` (one SkillCorner-style record per frame at 10 fps, plus `p90_radius_m`, `pos_cov_m2`,
`calib_health`, `abstain`; schema `src/predfoot/io/schema/frame-1.0.json`), `match.json`, `meta.json`
(versions, hashes, conformal quantiles), `metrics.json`, `minimap.png` and, later, `qa_report.html`.

Runs are processed in blocks of export frames written atomically to `OUT/.parts/`. A re-run with the same
input, configuration and stages reuses the finished blocks, so a killed run resumes to byte-identical files
(only `metrics.json`, which holds timings, differs).
