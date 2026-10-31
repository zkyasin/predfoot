"""Freeze the preregistration (ticket T3.4): fit the frozen artifacts on dev only and write the lock.

    uv run python scripts/freeze_prereg.py --calib-from runs/<per-frame dev run>/calib.npz

Refuses a dirty working tree, a preregistration that still carries a draft marker ("DRAFT", "pending owner
approval"), an existing lock and an existing artifact folder. Then, from the per-frame calibrations of every
dev clip (recomputed from the detection cache, or read from ``--calib-from`` after checking that the solver
settings match and that recalibrating the first clip reproduces them exactly) and the dev labels, it fits
(:mod:`predfoot.eval.frozen`, the code of ``scripts/eval_dev.py`` and ``scripts/eval_health.py``):

- the PTZ smoother's noise parameters, label-free, on all six dev games;
- the Mondrian conformal regions for each camera (per-frame, PTZ-smoothed), reference (raw labels, trusted
  line oracle) and covariance source, calibrated on all six dev games;
- the failure detectors (gradient boosting, logistic regression, single features) on the per-frame
  calibrations of all six dev games, as plain JSON.

It writes them to ``artifacts/prereg/v<version>/``, then ``docs/preregistration.lock.json`` (SHA-256 of the
preregistration and of every artifact, code commit, date, configuration digests, input hashes), and prints the
line to quote in README.md. Commit the artifacts and the lock together; ``scripts/run_test_once.py`` refuses
to run on anything else. Never reads a test file.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
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
from predfoot.paths import REPO_ROOT, index_dir

app = typer.Typer(add_completion=False)


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
    clip_ids: Sequence[str] | None = None, calib_from: Path | None = None, verify: int = 1
) -> list[ps.ClipCalib]:
    """Labelled per-frame calibrations of the dev clips (all of them, or ``clip_ids``; dev splits only).

    With ``calib_from``, saved calibrations are reused after two checks: the solver settings they were made
    with equal today's, and recalibrating the first ``verify`` clips from their detections gives exactly the
    saved results (so the frozen models describe the camera that the test run will compute).
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
        if calib_from is not None and i < verify:
            dets = ps.clip_detections(cid, c["split"])
            assert dets is not None
            fresh = ps.calibrate_detections(cid, c["split"], int(c["game_id"]), dets)
            if not same_results(fresh.results, cc.results):
                raise typer.BadParameter(f"{calib_from}: recalibrating {cid} does not reproduce it")
        ccs.append(cc)
    return ccs


def environment() -> dict[str, str]:
    env = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "scikit-learn", "polars", "opencv-python-headless"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg] = "missing"
    return env


def freeze(
    root: Path,
    art_dir: Path,
    ccs: Sequence[ps.ClipCalib],
    *,
    code_commit: str,
    inputs: dict[str, Any],
    lock_path: Path = pr.LOCK_PATH,
    prereg_path: Path = pr.PREREG_PATH,
) -> pr.Lock:
    """Fit and write every artifact under ``root / art_dir``, then the lock at ``root / lock_path``."""
    tables = fz.dev_tables(ccs)
    clips = sorted(cc.clip_id for cc in ccs)
    provenance = {
        "dev_clips": len(clips),
        "dev_clip_ids_sha256": hashlib.sha256(",".join(clips).encode()).hexdigest(),
        "dev_games": sorted({cc.game for cc in ccs}),
        "code_commit": code_commit,
    }
    fz.write_artifacts(tables, root / art_dir, provenance)
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
        },
        inputs=inputs,
        test_split={"split": "test", "games": list(fz.TEST_GAMES), "clips": fz.TEST_CLIPS},
        environment=environment(),
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
    if calib_from is not None:
        out["calib_from"] = {"name": calib_from.name, "sha256": sha256_file(calib_from)}
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
    commit = pr.head_commit(root)
    ccs = load_dev_clips(calib_from=calib_from)
    games = sorted({cc.game for cc in ccs})
    if games != list(fz.DEV_GAMES):
        raise typer.BadParameter(f"dev games {games}, expected {list(fz.DEV_GAMES)}")
    lock = freeze(root, art_dir, ccs, code_commit=commit, inputs=input_hashes(calib_from))
    typer.echo(f"Frozen {len(lock.artifacts)} artifacts from {len(ccs)} dev clips into {art_dir}:")
    for path, sha in lock.artifacts.items():
        typer.echo(f"  {path}  {sha[:16]}")
    typer.echo(f"Lock: {pr.LOCK_PATH} (code commit {commit[:12]}).")
    typer.echo("Commit the artifacts and the lock together, then quote this line in README.md:\n")
    typer.echo(lock.readme_line())


if __name__ == "__main__":
    app()
