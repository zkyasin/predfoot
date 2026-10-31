"""The single preregistered evaluation on the locked SN-GSR test games (ticket T6.1).

    uv run python scripts/run_test_once.py              # stages A, B, C in order (A and B resume)
    uv run python scripts/run_test_once.py --stage b    # one stage
    uv run python scripts/run_test_once.py --dry-run --calib-from runs/<per-frame dev run>/calib.npz

Before every stage the lock must match (:func:`predfoot.eval.prereg.verify_lock`): the preregistration and the
frozen artifacts have their locked SHA-256, nothing under src/, scripts/ or configs/ changed since the lock
commit, and the working tree is clean.

- A, label-free preparation: test frames at 5 fps (``scripts/download_data.py sn-gsr-frames --allow-locked``),
  keypoint and line networks into a separate detection cache (``cache/kp-test/``). Reads only the clip list
  (clip id, game) of ``index/clips.parquet``. Resumable: downloads and per-clip caches are idempotent.
- B, camera, label-free: per-frame calibration of every cached test frame (seeded like dev, saved per clip,
  resumable), then PTZ smoothing with the frozen noise parameters.
- C, evaluation, the only stage that reads test labels (``index/test/``), exactly once: the markers
  (``PREDFOOT_DATA_DIR/test_once/`` and ``runs/test_once/``) are written before any label is read, so a second
  C is refused even after a crash. Every endpoint for both cameras and both references, the success criteria
  evaluated as written, into ``reports/test_run.md`` (aggregate numbers) and ``runs/test_once/`` (local).

``--dry-run`` exercises the same chain on dev clips standing in for test clips (``--dry-clips``, default three
clips of game 4), in a scratch folder: a scratch freeze on a few clips of the five other dev games
(``--dry-train-per-game`` each) with its own scratch lock, the stand-ins' cached detections copied into a
scratch detection cache, the keypoint networks on at most ``--smoke-frames`` (≤ 10) frames compared with that
cache, stages B and C as above (C reads the dev labels), then a second C, which must be refused. It never
reads, writes or checks the real lock, markers or test files.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import typer

from predfoot.calib import temporal as tm
from predfoot.eval import frozen as fz
from predfoot.eval import positions as ps
from predfoot.eval import prereg as pr
from predfoot.io.manifest import sha256_file
from predfoot.io.sngsr import DEV_SPLITS
from predfoot.keypoints.cache import cache_path, cached_detections
from predfoot.paths import REPO_ROOT, cache_dir, data_dir, index_dir

app = typer.Typer(add_completion=False)
MAX_SMOKE_FRAMES = 10
DRY_CLIPS = ("SNGS-073", "SNGS-074", "SNGS-077")  # game 4, clips with failures in the per-frame dev run


def _load_freeze_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "freeze_prereg", Path(__file__).with_name("freeze_prereg.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Context:
    """Where a run reads and writes; the real run and the dry run differ only here."""

    dry: bool
    root: Path  # repository root (real) or scratch folder holding the scratch lock (dry)
    run_dir: Path
    markers: list[Path]
    cache_root: Path
    report: Path
    persons_path: Path
    lines_path: Path
    lock_path: Path = pr.LOCK_PATH
    clips: list[tuple[str, str, int]] = field(default_factory=list)  # (clip id, split, game)
    lock: pr.Lock | None = None
    log_lines: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        line = f"[{datetime.now(UTC):%H:%M:%S}] {msg}"
        self.log_lines.append(line)
        typer.echo(line)


def real_context() -> Context:
    run_dir = REPO_ROOT / "runs" / "test_once"
    return Context(
        dry=False,
        root=REPO_ROOT,
        run_dir=run_dir,
        markers=[data_dir() / "test_once" / "STAGE_C_STARTED.json", run_dir / "STAGE_C_STARTED.json"],
        cache_root=cache_dir() / "kp-test",
        report=REPO_ROOT / "reports" / "test_run.md",
        persons_path=index_dir() / "test" / "persons.parquet",
        lines_path=index_dir() / "test" / "lines.parquet",
    )


def guard(ctx: Context) -> pr.Lock:
    """The lock checks run before every stage; the dry run checks its scratch lock's hashes only."""
    if ctx.dry:
        lock = pr.read_lock(ctx.root, ctx.lock_path)
        pr.verify_hashes(ctx.root, lock)
    else:
        lock = pr.verify_lock(ctx.root, ctx.lock_path)
    ctx.lock = lock
    return lock


def artifacts(ctx: Context) -> fz.Artifacts:
    assert ctx.lock is not None
    return fz.load_artifacts(ctx.root / ctx.lock.artifacts_dir)


def test_clips() -> list[tuple[str, str, int]]:
    """The locked split's clip list (clip id, game) from the clip index: no label is read."""
    clips = pl.read_parquet(index_dir() / "clips.parquet").filter(pl.col("split") == "test").sort("clip_id")
    games = sorted(int(g) for g in clips["game_id"].unique())
    if games != list(fz.TEST_GAMES):
        raise pr.PreregError(f"test games {games}, preregistered {list(fz.TEST_GAMES)}")
    return [(c, "test", int(g)) for c, g in clips.select("clip_id", "game_id").iter_rows()]


def ensure_clips(ctx: Context) -> None:
    if not ctx.clips:
        ctx.clips = test_clips()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(fz.dump(payload), encoding="utf-8")
    tmp.replace(path)


def clip_dir(split: str, clip_id: str) -> Path:
    return data_dir() / "sn-gsr" / split / clip_id


# --- stage A ---------------------------------------------------------------------------------------------


def smoke_inference(ctx: Context, clip_id: str, split: str, n_frames: int) -> dict[str, Any]:
    """Run the networks on the first ``n_frames`` (≤ 10) frames of a clip and compare with the cache."""
    n_frames = min(n_frames, MAX_SMOKE_FRAMES)
    try:
        from predfoot.keypoints.cache import _read_rgb
        from predfoot.keypoints.pnlcalib import PnLCalibNetworks
    except ImportError as err:  # torch missing: the smoke check is skipped, not failed
        return {"skipped": f"torch unavailable ({err})"}
    files = sorted((clip_dir(split, clip_id) / "img1").glob("*.jpg"))[:n_frames]
    cached, _, _ = cached_detections(clip_dir(split, clip_id), clip_id, root=ctx.cache_root)
    weights = data_dir() / "weights" / "pnlcalib"
    nets = PnLCalibNetworks(weights / "SV_kp", weights / "SV_lines")
    kp_diff, line_diff, same_ids, t0 = 0.0, 0.0, 0, time.perf_counter()
    for f in files:
        (kps, ends) = nets.detect(_read_rgb(f)[None])[0]
        ref = cached[int(f.stem)]
        if np.array_equal(kps.ids, ref["kp_ids"]) and np.array_equal(ends.line_ids, ref["line_ids"]):
            same_ids += 1
            if len(kps.ids):
                kp_diff = max(kp_diff, float(np.abs(kps.uv - ref["kp_uv"]).max()))
            if len(ends.line_ids):
                line_diff = max(line_diff, float(np.abs(ends.uv - ref["line_uv"]).max()))
    return {
        "clip": clip_id,
        "frames": len(files),
        "frames_with_same_ids": same_ids,
        "max_kp_px_diff": kp_diff,
        "max_line_px_diff": line_diff,
        "seconds": time.perf_counter() - t0,
        "model_version": nets.version,
    }


def stage_a(ctx: Context, smoke_frames: int = 0) -> None:
    guard(ctx)
    ensure_clips(ctx)
    done = ctx.run_dir / "stage_a.json"
    if done.exists():
        ctx.log("stage A already done, skipped")
        return
    ctx.log(f"stage A: {len(ctx.clips)} clips, detections into {ctx.cache_root}")
    if not ctx.dry:
        cmd = ["uv", "run", "python", str(REPO_ROOT / "scripts" / "download_data.py"), "sn-gsr-frames",
               "--every", "5", "--allow-locked", "--clips", *[c for c, _, _ in ctx.clips]]  # fmt: skip
        if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
            raise RuntimeError("frame download failed; rerun stage A (it resumes)")
    smoke: dict[str, Any] | None = None
    cache = {}
    for clip_id, split, _ in ctx.clips:
        n = ps.clip_frame_count(clip_id, split)
        target = cache_path(clip_id, n, root=ctx.cache_root)
        if ctx.dry and not target.exists():  # stand-ins: the dev cache is copied, never written
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_path(clip_id, n), target)
        dets, seconds, version = cached_detections(clip_dir(split, clip_id), clip_id, root=ctx.cache_root)
        cache[clip_id] = {"frames": len(dets), "sha256": sha256_file(target), "model_version": version}
        ctx.log(
            f"  {clip_id}: {len(dets)} frames cached (inference time recorded in the cache: {seconds:.0f} s)"
        )
    if smoke_frames:
        clip_id, split, _ = ctx.clips[0]
        smoke = smoke_inference(ctx, clip_id, split, smoke_frames)
        ctx.log(f"  smoke check of the networks: {smoke}")
    write_json(done, {"clips": ctx.clips, "cache": cache, "smoke": smoke, "finished_utc": _now()})


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- stage B ---------------------------------------------------------------------------------------------


def _clip_dets(ctx: Context, clip_id: str, split: str) -> dict[int, dict[str, Any]]:
    dets = ps.clip_detections(clip_id, split, ctx.cache_root)
    if dets is None:
        raise RuntimeError(f"{clip_id} has no detection cache under {ctx.cache_root}: run stage A")
    return dets


def stage_b(ctx: Context) -> None:
    guard(ctx)
    ensure_clips(ctx)
    if not (ctx.run_dir / "stage_a.json").exists():
        raise RuntimeError("stage A has not finished")
    done = ctx.run_dir / "stage_b.json"
    if done.exists():
        ctx.log("stage B already done, skipped")
        return
    art = artifacts(ctx)
    ctx.log(f"stage B: per-frame calibration of {len(ctx.clips)} clips, then smoothing (frozen noise)")
    ccs = []
    for clip_id, split, game in ctx.clips:
        part = ctx.run_dir / "stage_b" / f"{clip_id}.npz"
        dets = _clip_dets(ctx, clip_id, split)
        if part.exists():
            saved, _ = ps.load_calib(part)
            cc = ps.calibrate_detections(clip_id, split, game, dets, saved[clip_id])
        else:
            t0 = time.perf_counter()
            cc = ps.calibrate_detections(clip_id, split, game, dets)
            ps.save_calib(part, [cc], {"clip": clip_id})
            rate = np.mean([r.ok for r in cc.results])
            seconds = time.perf_counter() - t0
            ctx.log(f"  {clip_id}: {len(cc.frames)} frames, {rate:.0%} calibrated ({seconds:.0f} s)")
        ccs.append(cc)
    meta = {
        "stage": "B",
        "smoother_config_digest": art.smoother_config_digest,
        "params": art.params.to_dict(),
    }
    ps.save_calib(ctx.run_dir / "calib_perframe.npz", ccs, meta)
    cfg = tm.SmootherConfig()
    info = ps.SmoothingInfo()
    outputs = list(ps.smooth_outputs(ccs, ps.Smooth.ptz, ps.FitScope.all, cfg, info, params=art.params))
    fz.save_outputs(ctx.run_dir / "calib_ptz.npz", outputs, meta)
    write_json(
        done,
        {
            "clips": len(ccs),
            "frames": int(sum(len(cc.frames) for cc in ccs)),
            "perframe_ok": float(np.mean([r.ok for cc in ccs for r in cc.results])),
            "ptz_ok": float(np.mean([r.ok for o in outputs for r in o.results])),
            "params_digest": art.params.digest(),
            "sha256": {
                name: sha256_file(ctx.run_dir / name) for name in ("calib_perframe.npz", "calib_ptz.npz")
            },
            "finished_utc": _now(),
        },
    )
    ctx.log(f"stage B done: {len(ccs)} clips, noise parameters {art.params.digest()} (frozen)")


# --- stage C ---------------------------------------------------------------------------------------------


def stage_c(ctx: Context) -> dict[str, Any]:
    lock = guard(ctx)  # nothing below runs without a matching lock
    if ctx.report.exists():  # a committed result, e.g. from another clone with its own data folder
        raise pr.PreregError(f"{ctx.report} exists: the test evaluation already ran, a second one is refused")
    ensure_clips(ctx)
    b = ctx.run_dir / "stage_b.json"
    if not b.exists():
        raise RuntimeError("stage B has not finished")
    stage_b_info = json.loads(b.read_text())
    for name, sha in stage_b_info["sha256"].items():
        if sha256_file(ctx.run_dir / name) != sha:
            raise RuntimeError(f"{name} changed since stage B")
    art = artifacts(ctx)
    payload = {
        "started_utc": _now(),
        "lock_sha256": sha256_file(ctx.root / ctx.lock_path),
        "code_commit": lock.code_commit,
        "head": pr.head_commit(REPO_ROOT),
        "dry_run": ctx.dry,
    }
    pr.claim_once(ctx.markers, payload)  # written before any label is read
    ctx.log(f"stage C started, markers written: {', '.join(str(m) for m in ctx.markers)}")
    try:
        return _evaluate(ctx, art, lock, payload)
    finally:
        (ctx.run_dir / "stage_c.log").write_text("\n".join(ctx.log_lines) + "\n", encoding="utf-8")


def _evaluate(ctx: Context, art: fz.Artifacts, lock: pr.Lock, payload: dict[str, Any]) -> dict[str, Any]:
    ids = [c for c, _, _ in ctx.clips]
    persons = pl.read_parquet(ctx.persons_path).filter(pl.col("clip_id").is_in(ids))
    lines = pl.read_parquet(ctx.lines_path).filter(pl.col("clip_id").is_in(ids))
    ctx.log(f"labels read: {persons.height} person rows, {lines.height} line points")
    perframe, _ = ps.load_calib(ctx.run_dir / "calib_perframe.npz")
    smoothed, _ = fz.load_outputs(ctx.run_dir / "calib_ptz.npz")
    ccs, outputs = [], []
    for clip_id, split, game in ctx.clips:
        dets = _clip_dets(ctx, clip_id, split)
        cc = ps.calibrate_detections(clip_id, split, game, dets, perframe[clip_id])
        ps.attach_labels(
            cc, dets, persons.filter(pl.col("clip_id") == clip_id), lines.filter(pl.col("clip_id") == clip_id)
        )
        ccs.append(cc)
        res, flags, diag = smoothed[clip_id]
        outputs.append(ps.Output(cc, None, res, flags, diag))
    tables = {"perframe": fz.perframe_tables(ccs), "ptz": fz.output_tables(outputs)}
    for name, t in tables.items():
        if t.players.height:
            t.players.write_parquet(ctx.run_dir / f"players_{name}.parquet")
        t.frames.write_parquet(ctx.run_dir / f"frames_{name}.parquet")
    res = fz.evaluate(tables, art)
    res["provenance"] = {**payload, "lock_code_commit": lock.code_commit, "artifacts": lock.artifacts}
    write_json(ctx.run_dir / "results.json", res)
    title = "Dry run of the test-run tooling on dev clips" if ctx.dry else "Preregistered test run"
    header = [
        f"# {title}",
        "",
        f"`uv run python scripts/run_test_once.py{' --dry-run' if ctx.dry else ''}` · lock code commit"
        f" `{lock.code_commit[:12]}` · preregistration SHA-256 `{lock.preregistration['sha256'][:16]}…` ·"
        f" stage C started {payload['started_utc']}.",
    ]
    if ctx.dry:
        header += [
            "",
            "**DRY RUN: dev clips stand in for the test games and the artifacts come from a scratch freeze"
            " on a few dev clips. These numbers say nothing about the test games.**",
        ]
    else:
        header += [
            "",
            "Run once, as preregistered in `docs/preregistration.md`; every result is published, failed"
            " criteria included. Local outputs in `runs/test_once/` (not in git).",
        ]
    ctx.report.parent.mkdir(parents=True, exist_ok=True)
    ctx.report.write_text(fz.render_report(res, header), encoding="utf-8")
    write_json(ctx.run_dir / "stage_c.json", {"finished_utc": _now(), "report": str(ctx.report)})
    ctx.log(f"stage C done: report {ctx.report}")
    return res


# --- dry run ---------------------------------------------------------------------------------------------


def dry_context(scratch: Path, clips: list[str], train_per_game: int, calib_from: Path | None) -> Context:
    """Scratch freeze (own artifacts and lock under ``scratch``) and a context that never touches the real
    lock, markers, cache or test files."""
    dev = pl.read_parquet(index_dir() / "clips.parquet").filter(pl.col("split").is_in(list(DEV_SPLITS)))
    dev = dev.sort("clip_id")
    unknown = sorted(set(clips) - set(dev["clip_id"].to_list()))
    if unknown:
        raise typer.BadParameter(f"--dry-clips must be dev clips: {unknown}")
    stand_in = dev.filter(pl.col("clip_id").is_in(clips))
    held_games = set(stand_in["game_id"].to_list())
    train = (
        dev.filter(~pl.col("game_id").is_in(list(held_games)))
        .group_by("game_id", maintain_order=True)
        .head(train_per_game)["clip_id"]
        .to_list()
    )
    freeze = _load_freeze_module()
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "docs").mkdir(exist_ok=True)
    shutil.copy2(REPO_ROOT / pr.PREREG_PATH, scratch / pr.PREREG_PATH)
    lock_file = scratch / pr.LOCK_PATH
    if not lock_file.exists():
        train_games = sorted(set(dev.filter(pl.col("clip_id").is_in(train))["game_id"].to_list()))
        typer.echo(
            f"scratch freeze on {len(train)} dev clips of games {train_games}"
            f" (stand-ins: {', '.join(clips)}, game {sorted(held_games)})"
        )
        t0 = time.perf_counter()
        ccs = freeze.load_dev_clips(train, calib_from=calib_from, verify=1)
        lock = freeze.freeze(
            scratch,
            pr.ARTIFACTS_ROOT / "dry",
            ccs,
            code_commit=pr.head_commit(REPO_ROOT),
            inputs={"dry_run": True, "train_clips": train, "stand_in_clips": clips},
        )
        typer.echo(f"scratch freeze: {len(lock.artifacts)} artifacts in {time.perf_counter() - t0:.0f} s")
    run_dir = scratch / "run"
    return Context(
        dry=True,
        root=scratch,
        run_dir=run_dir,
        markers=[scratch / "marker_data" / "STAGE_C_STARTED.json", run_dir / "STAGE_C_STARTED.json"],
        cache_root=scratch / "cache-kp",
        report=scratch / "test_run.md",
        persons_path=index_dir() / "persons.parquet",
        lines_path=index_dir() / "lines.parquet",
        clips=[(r["clip_id"], r["split"], int(r["game_id"])) for r in stand_in.iter_rows(named=True)],
    )


def compare_with_dev(ctx: Context, calib_from: Path) -> None:
    """Stage B recalibrated the stand-ins from their detections: they must equal the dev calibrations."""
    freeze = _load_freeze_module()
    saved, _ = ps.load_calib(calib_from)
    mine, _ = ps.load_calib(ctx.run_dir / "calib_perframe.npz")
    same = {c: freeze.same_results(mine[c], saved[c]) for c in mine}
    ctx.log(f"stage B per-frame calibrations equal to {calib_from.name}: {same}")


@app.command()
def main(
    stage: Annotated[str, typer.Option(help="a, b, c or all (A and B resume, C runs once).")] = "all",
    dry_run: Annotated[
        bool, typer.Option(help="Exercise the chain on dev clips in a scratch folder.")
    ] = False,
    dry_clips: Annotated[
        str, typer.Option(help="Comma-separated dev clips standing in for test clips.")
    ] = ",".join(DRY_CLIPS),
    dry_train_per_game: Annotated[
        int, typer.Option(help="Dev clips per other game in the scratch freeze.")
    ] = 3,
    scratch: Annotated[
        Path | None, typer.Option(help="Dry-run folder (default runs/test_once_dry/<time>).")
    ] = None,
    calib_from: Annotated[
        Path | None,
        typer.Option(help="Dry run: per-frame dev calib.npz for the scratch freeze and the check."),
    ] = None,
    smoke_frames: Annotated[int, typer.Option(help="Dry run: frames of real network inference (≤ 10).")] = 10,
) -> None:
    if stage not in ("a", "b", "c", "all"):
        raise typer.BadParameter("--stage is a, b, c or all")
    if smoke_frames > MAX_SMOKE_FRAMES:
        raise typer.BadParameter(f"--smoke-frames is at most {MAX_SMOKE_FRAMES}")
    try:
        if dry_run:
            folder = scratch or REPO_ROOT / "runs" / "test_once_dry" / f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
            clips = [c.strip() for c in dry_clips.split(",") if c.strip()]
            ctx = dry_context(folder, clips, dry_train_per_game, calib_from)
        else:
            ctx = real_context()
            smoke_frames = 0
        if stage in ("a", "all"):
            stage_a(ctx, smoke_frames)
        if stage in ("b", "all"):
            stage_b(ctx)
            if dry_run and calib_from is not None:
                compare_with_dev(ctx, calib_from)
        if stage in ("c", "all"):
            res = stage_c(ctx)
            crit = res["criteria"]
            typer.echo(
                "criteria: " + ", ".join(f"{k} {'met' if v['pass'] else 'not met'}" for k, v in crit.items())
            )
            if dry_run:  # the report is set aside so that the marker alone has to refuse
                aside = ctx.report.with_name(ctx.report.name + ".aside")
                ctx.report.rename(aside)
                try:
                    stage_c(ctx)
                except pr.PreregError as err:
                    typer.echo(f"second stage C refused as expected: {err}")
                else:
                    raise RuntimeError("a second stage C was not refused")
                finally:
                    aside.rename(ctx.report)
    except pr.PreregError as err:
        typer.echo(f"Refused: {err}", err=True)
        raise typer.Exit(2) from err
    except Exception:
        traceback.print_exc()
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
