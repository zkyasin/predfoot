"""Inputs and expected outputs of the golden test of the real pipeline stages.

    uv run python scripts/make_golden.py             # rewrite the inputs: clip, detections, conformal model
    uv run python scripts/make_golden.py --update    # also rerun the real stages and rewrite expected/

Everything is synthetic; nothing comes from SoccerNet. The clip is the first 20 native frames (25 fps) of the
tracer-bullet clip (``scripts/make_synthetic_clip.py``), without its homographies: the camera has to be
recovered from network-style keypoint and line detections generated from the true camera (1 px noise,
outlying keypoints on frames 2, 10 and 15, three keypoints and no line on frame 7, no detection at all on
frame 8). The Mondrian conformal model is made up: its numbers were not fitted on any data, they only make
the players of the clip fall into different bins, one of which falls back to the pooled quantile.

``tests/integration/test_golden_real.py`` runs calibration, conformal regions, localization and export on
these inputs (:func:`run_golden`) and compares the outputs with ``expected/``: exact on structure, ids,
flags and texts, within one unit of the last written decimal on numbers (macOS and Linux CI may round a few
values differently). Only ``--update`` rewrites ``expected/``: use it when a change of the outputs is
intended, and review the diff.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from predfoot.geometry import homography as hg
from predfoot.keypoints.cache import dumps_detections, load_detections
from predfoot.paths import REPO_ROOT
from predfoot.pipeline.clip import ClipFrame, load_clip
from predfoot.pipeline.real import real_stages
from predfoot.pipeline.run import RunConfig, RunResult, Stages, run_clip
from predfoot.pipeline.synthetic import dumps_clip, make_broadcast_clip, synthetic_detections
from predfoot.uncertainty.conformal import MondrianConformal

GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"
CLIP_ID = "synthetic-golden-001"
N_FRAMES = 20
SEED = 20261101
#: native frame → outlying keypoints
OUTLIERS = {2: 1, 10: 2, 15: 1}
SPARSE_FRAMES = (7,)  # three keypoints and no line: too few to calibrate
MISSING_FRAMES = (8,)  # no detection at all
MODEL_VERSION = "synthetic-detections-1"
STAGES_VERSION = "golden-1"
BLOCK_SIZE = 3  # several resumable blocks, as in a real run
CONFORMAL = MondrianConformal(
    alpha=0.1,
    edges={"mpp": [0.035, 0.062], "n_inliers": [26.5]},
    quantiles={"0-0": 4.2, "0-1": 3.8, "1-0": 5.6, "1-1": 5.0, "2-1": 7.4},  # "2-0": pooled
    counts={},
    pooled=6.5,
)
#: A player closer than this (relative) to an mpp edge could change bin on another platform.
MIN_EDGE_MARGIN = 0.02
EXPECTED_FILES = ("tracking.jsonl", "match.json", "meta.json", "metrics.json")
#: meta.json fields that change with every commit or release, left out of the comparison.
VOLATILE_META = ("git_sha", "predfoot_version")


def build_inputs() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """(clip as a JSON-ready dict, detections by native frame)."""
    source, _ = make_broadcast_clip()
    frames = source["frames"][:N_FRAMES]
    detections = {}
    for f in frames:
        k = f["frame"]
        if k in MISSING_FRAMES:
            continue
        sparse = k in SPARSE_FRAMES
        detections[k] = synthetic_detections(
            np.array(f["homography"]),
            np.random.default_rng([SEED, k]),
            n_outliers=OUTLIERS.get(k, 0),
            max_keypoints=3 if sparse else None,
            with_lines=not sparse,
        )
    clip = {k: v for k, v in source.items() if k != "frames"} | {"clip_id": CLIP_ID}
    clip["frames"] = [f | {"homography": None} for f in frames]  # the camera is not an input
    return clip, detections


def write_inputs(fixture_dir: Path) -> None:
    clip, detections = build_inputs()
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "clip.json").write_text(dumps_clip(clip), encoding="utf-8")
    detections_json = dumps_detections(detections, MODEL_VERSION)
    (fixture_dir / "detections.json").write_text(detections_json, encoding="utf-8")
    (fixture_dir / "conformal.json").write_text(CONFORMAL.to_json() + "\n", encoding="utf-8")


def golden_stages(fixture_dir: Path) -> tuple[Stages, MondrianConformal]:
    detections, model_version = load_detections(fixture_dir / "detections.json")
    conformal = MondrianConformal.from_json((fixture_dir / "conformal.json").read_text(encoding="utf-8"))
    return real_stages(detections, conformal, f"{STAGES_VERSION}+{model_version}"), conformal


def run_golden(fixture_dir: Path, out_dir: Path) -> RunResult:
    """The real stages on the golden inputs, through ``run_clip`` as ``predfoot run`` uses it."""
    stages, _ = golden_stages(fixture_dir)
    return run_clip(fixture_dir / "clip.json", out_dir, RunConfig(block_size=BLOCK_SIZE), stages)


def normalised_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if k not in VOLATILE_META}


def normalised_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """The deterministic part of metrics.json (no timings, no block counts)."""
    return {k: metrics[k] for k in ("clip_id", "frames", "players")}


def player_bins(fixture_dir: Path) -> list[tuple[int, int, str, float]]:
    """(native frame, track, Mondrian bin, smallest relative distance to an mpp edge) of every foot of a
    calibrated frame, as the real stages compute them."""
    stages, conformal = golden_stages(fixture_dir)
    clip = load_clip(fixture_dir / "clip.json")
    edges = np.asarray(conformal.edges["mpp"])
    out = []
    for frame in clip.frames:
        calib = stages.calibrate(ClipFrame(frame=frame.frame))
        if not calib.ok or calib.H is None or not frame.feet:
            continue
        uv = np.array([[p.u, p.v] for p in frame.feet])
        _, valid = hg.backproject(calib.H, uv)
        mpp = np.sqrt(np.abs(np.linalg.det(hg.jacobian_backproject_uv(calib.H, uv))))
        # the bins exactly as MondrianConformal.quantile_for finds them
        keys = conformal._keys({"mpp": mpp, "n_inliers": np.full(len(uv), float(calib.n_inliers))})
        for p, m, key, ok in zip(frame.feet, mpp, keys, valid, strict=True):
            if ok:
                out.append((frame.frame, p.track_id, str(key), float(np.min(np.abs(m - edges) / edges))))
    return out


def write_expected(fixture_dir: Path) -> RunResult:
    expected = fixture_dir / "expected"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        result = run_golden(fixture_dir, out)
        if expected.exists():
            shutil.rmtree(expected)
        expected.mkdir(parents=True)
        for name in ("tracking.jsonl", "match.json"):
            shutil.copyfile(out / name, expected / name)
        for name, normalise in (("meta.json", normalised_meta), ("metrics.json", normalised_metrics)):
            obj = normalise(json.loads((out / name).read_text(encoding="utf-8")))
            (expected / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, default=GOLDEN_DIR, help="fixture folder")
    parser.add_argument(
        "--update",
        action="store_true",
        help="also rerun the real stages and rewrite expected/ (then review the diff)",
    )
    args = parser.parse_args(argv)
    write_inputs(args.dir)
    bins = player_bins(args.dir)
    margin = min(m for *_, m in bins)
    if margin < MIN_EDGE_MARGIN:
        raise SystemExit(f"a player sits {margin:.1%} from an mpp edge of the conformal model: move it")
    used = sorted({key for _, _, key, _ in bins})
    print(f"inputs written to {args.dir}; bins used {used}, closest player {margin:.1%} from an mpp edge")
    if args.update:
        result = write_expected(args.dir)
        print(
            f"expected/ rewritten: {result.n_export} export frames from {result.n_native} native frames, "
            f"{result.n_abstained} abstained"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
