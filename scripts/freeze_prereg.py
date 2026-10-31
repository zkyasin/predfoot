"""Freeze the preregistration (ticket T3.4): fit the frozen artifacts on dev only and write the lock.

    uv run python scripts/freeze_prereg.py --calib-from runs/<per-frame dev run>/calib.npz

Refuses a dirty working tree, a preregistration that still carries a draft marker ("DRAFT", "pending owner
approval"), an existing lock, an existing artifact folder and an environment without the packages every stage
needs (torch included: the lock pins the versions stage A runs the networks with). Then, from the per-frame
calibrations of every dev clip (recomputed from the detection cache, or read from ``--calib-from`` after
checking that the solver settings match and that recalibrating every clip from its detections reproduces them
exactly) and the dev labels, it fits (:mod:`predfoot.eval.frozen`, the code of ``scripts/eval_dev.py`` and
``scripts/eval_health.py``):

- the PTZ smoother's noise parameters, label-free, on all six dev games;
- the Mondrian conformal regions for each camera (per-frame, PTZ-smoothed), reference (raw labels, trusted
  line oracle) and covariance source, calibrated on all six dev games;
- the failure detectors (gradient boosting, logistic regression, single features) on the per-frame
  calibrations of all six dev games, as plain JSON.

It writes them to ``artifacts/prereg/v<version>/``, then ``docs/preregistration.lock.json``: SHA-256 of the
preregistration and of every artifact, code commit, date, configuration digests, the detection model version
of the dev caches and the SHA-256 of the network weights, input hashes, the environment, and the test clip
list (count and digest of the (clip id, game) pairs of ``index/clips.parquet``: the split's structure, which
``reports/data_audit.md`` already reports; no test label or frame is read). It prints the line to quote in
README.md. Commit the artifacts and the lock together, in one commit that holds nothing else;
``scripts/run_test_once.py`` refuses to run on anything else.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import typer

from predfoot.calib.solver import SolverConfig
from predfoot.eval import frozen as fz
from predfoot.eval import positions as ps
from predfoot.eval import prereg as pr
from predfoot.io.manifest import sha256_file
from predfoot.io.sngsr import DEV_SPLITS
from predfoot.paths import REPO_ROOT, data_dir, index_dir

app = typer.Typer(add_completion=False)
WEIGHTS = ("SV_kp", "SV_lines")


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(fz.to_jsonable(obj), sort_keys=True).encode()).hexdigest()[:12]


def same_results(a: Sequence[Any], b: Sequence[Any]) -> bool:
    """Two lists of CalibResult are identical (arrays compared exactly, NaN equal to NaN)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if (x.status, x.n_inliers, x.s2, x.rms_px, x.cond, x.dof) != (
            y.status,
            y.n_inliers,
            y.s2,
            y.rms_px,
            y.cond,
            y.dof,
        ):
            return False
        for u, v in ((x.H, y.H), (x.cov_theta, y.cov_theta)):
            if (u is None) != (v is None) or (u is not None and not np.array_equal(u, v, equal_nan=True)):
                return False
    return True


def load_dev_clips(
    clip_ids: Sequence[str] | None = None, calib_from: Path | None = None, verify: int | None = None
) -> list[ps.ClipCalib]:
    """Labelled per-frame calibrations of the dev clips (all of them, or ``clip_ids``; dev splits only).

    With ``calib_from``, saved calibrations are reused after two checks: the solver settings they were made
    with equal today's, and recalibrating the first ``verify`` clips (every clip when None) from their
    detections gives exactly the saved results, so that the frozen models describe the camera that the test
    run will compute.
    """
    dev = pl.read_parquet(index_dir() / "clips.parquet").filter(pl.col("split").is_in(list(DEV_SPLITS)))
    dev = dev.sort("clip_id")
    if clip_ids is not None:
        unknown = sorted(set(clip_ids) - set(dev["clip_id"].to_list()))
        if unknown:
            raise typer.BadParameter(f"not dev clips: {unknown}")
        dev = dev.filter(pl.col("clip_id").is_in(list(clip_ids)))
    persons = pl.read_parquet(index_dir() / "persons.parquet")
    lines = pl.read_parquet(index_dir() / "lines.parquet")
    saved: dict[str, list[Any]] = {}
    if calib_from is not None:
        saved, meta = ps.load_calib(calib_from)
        if meta.get("solver") != asdict(SolverConfig()):
            raise typer.BadParameter(f"{calib_from} was made with other solver settings")
    n_verify = dev.height if verify is None else verify
    ccs = []
    for i, c in enumerate(dev.iter_rows(named=True)):
        cid = c["clip_id"]
        if calib_from is not None and cid not in saved:
            raise typer.BadParameter(f"{cid} is not in {calib_from}")
        cc = ps.calibrate_clip(
            cid,
            c["split"],
            int(c["game_id"]),
            persons.filter(pl.col("clip_id") == cid),
            lines.filter(pl.col("clip_id") == cid),
            saved.get(cid),
        )
        if cc is None:
            raise typer.BadParameter(f"{cid} has no detection cache")
        if calib_from is not None and i < n_verify:
            dets = ps.clip_detections(cid, c["split"])
            assert dets is not None
            fresh = ps.calibrate_detections(cid, c["split"], int(c["game_id"]), dets)
            if not same_results(fresh.results, cc.results):
                raise typer.BadParameter(f"{calib_from}: recalibrating {cid} does not reproduce it")
        ccs.append(cc)
    return ccs


def detection_model(ccs: Sequence[ps.ClipCalib]) -> str:
    """The one model version of the dev detection caches the artifacts are fitted on."""
    versions = {ps.clip_model_version(cc.clip_id, cc.split) for cc in ccs}
    if len(versions) != 1 or None in versions:
        raise typer.BadParameter(
            f"the dev detection caches hold several model versions: {sorted(map(str, versions))}"
        )
    return str(versions.pop())


def freeze(
    root: Path,
    art_dir: Path,
    ccs: Sequence[ps.ClipCalib],
    *,
    code_commit: str,
    inputs: dict[str, Any],
    test_clips: Sequence[tuple[str, int]],
    detection_model: str,
    lock_path: Path = pr.LOCK_PATH,
    prereg_path: Path = pr.PREREG_PATH,
) -> pr.Lock:
    """Fit and write every artifact under ``root / art_dir``, then the lock at ``root / lock_path``.

    ``test_clips``: the (clip id, game) pairs the test run evaluates, pinned by count and digest.
    """
    tables = fz.dev_tables(ccs)
    clips = sorted(cc.clip_id for cc in ccs)
    provenance = {
        "dev_clips": len(clips),
        "dev_clip_ids_sha256": hashlib.sha256(",".join(clips).encode()).hexdigest(),
        "dev_games": sorted({cc.game for cc in ccs}),
        "code_commit": code_commit,
    }
    fz.write_artifacts(tables, root / art_dir, provenance)
    per_game = Counter(g for _, g in test_clips)
    lock = pr.make_lock(
        root,
        art_dir,
        code_commit=code_commit,
        frozen_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        config_digests={
            "solver": digest(asdict(SolverConfig())),
            "smoother_config": tables.cfg.digest(),
            "noise_params": tables.params.digest(),
            "analysis": digest(fz.analysis_config()),
            "detection_model": detection_model,
        },
        inputs=inputs,
        test_split={
            "split": "test",
            "games": sorted(per_game),
            "clips": len(test_clips),
            "clips_per_game": {str(g): n for g, n in sorted(per_game.items())},
            "clip_ids_sha256": fz.clip_list_digest(test_clips),
        },
        environment=pr.environment(),
        prereg_path=prereg_path,
    )
    (root / lock_path).parent.mkdir(parents=True, exist_ok=True)
    (root / lock_path).write_text(lock.to_json(), encoding="utf-8")
    pr.verify_hashes(root, lock)
    return lock


def input_hashes(calib_from: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        f"index/{name}.parquet": sha256_file(index_dir() / f"{name}.parquet")
        for name in ("clips", "persons", "lines")
    }
    weights = data_dir() / "weights" / "pnlcalib"
    missing = [name for name in WEIGHTS if not (weights / name).is_file()]
    if missing:
        raise typer.BadParameter(f"network weights missing under {weights}: {missing}")
    out["weights"] = {name: sha256_file(weights / name) for name in WEIGHTS}
    if calib_from is not None:
        _, meta = ps.load_calib(calib_from)
        out["calib_from"] = {
            "name": calib_from.name,
            "sha256": sha256_file(calib_from),
            "git_sha": meta.get("git_sha"),
        }
    return out


@app.command()
def main(
    calib_from: Annotated[
        Path | None, typer.Option(help="calib.npz of a per-frame eval_dev run on every dev clip.")
    ] = None,
) -> None:
    root = REPO_ROOT
    try:
        pr.require_clean_tree(root)
        text = (root / pr.PREREG_PATH).read_text(encoding="utf-8")
        pr.require_final_text(text)
        version = pr.prereg_version(text)
        missing = [pkg for pkg, v in pr.environment().items() if v == "missing"]
        if missing:
            raise pr.PreregError(
                f"packages missing: {missing}; the lock pins the environment of every stage, torch included"
                " (uv sync --extra serve --extra vision)"
            )
    except pr.PreregError as err:
        typer.echo(f"Refused: {err}", err=True)
        raise typer.Exit(2) from err
    art_dir = pr.ARTIFACTS_ROOT / f"v{version}"
    for path in (root / pr.LOCK_PATH, root / art_dir):
        if path.exists():
            typer.echo(
                f"Refused: {path.relative_to(root)} already exists (the preregistration is frozen)", err=True
            )
            raise typer.Exit(2)
    test_clips = fz.locked_clip_list(index_dir() / "clips.parquet")
    if len(test_clips) != fz.TEST_CLIPS:
        raise typer.BadParameter(f"{len(test_clips)} test clips in the index, preregistered {fz.TEST_CLIPS}")
    commit = pr.head_commit(root)
    inputs = input_hashes(calib_from)
    ccs = load_dev_clips(calib_from=calib_from)
    games = sorted({cc.game for cc in ccs})
    if games != list(fz.DEV_GAMES):
        raise typer.BadParameter(f"dev games {games}, expected {list(fz.DEV_GAMES)}")
    lock = freeze(
        root,
        art_dir,
        ccs,
        code_commit=commit,
        inputs=inputs,
        test_clips=test_clips,
        detection_model=detection_model(ccs),
    )
    typer.echo(f"Frozen {len(lock.artifacts)} artifacts from {len(ccs)} dev clips into {art_dir}:")
    for path, sha in lock.artifacts.items():
        typer.echo(f"  {path}  {sha[:16]}")
    typer.echo(f"Lock: {pr.LOCK_PATH} (code commit {commit[:12]}).")
    typer.echo(
        "Commit the artifacts and the lock together, in one commit that holds nothing else, then quote this"
        " line in README.md:\n"
    )
    typer.echo(lock.readme_line(pr.lock_sha256(root)))


if __name__ == "__main__":
    app()
