"""The single preregistered evaluation on the locked SN-GSR test games (ticket T6.1).

    uv run python scripts/run_test_once.py              # stages A, B, C in order (A and B resume)
    uv run python scripts/run_test_once.py --stage b    # one stage
    uv run python scripts/run_test_once.py --dry-run --calib-from runs/<per-frame dev run>/calib.npz

Before every stage the lock must match (:func:`predfoot.eval.prereg.verify_lock` and
:func:`~predfoot.eval.prereg.verify_environment`): the preregistration and the frozen artifacts have their
locked SHA-256, the lock is anchored in its freeze commit, nothing under the protected paths changed since the
lock commit, the working tree is clean, and the interpreter and packages are those of the freeze. The clips
must be the pinned list (count and digest in the lock).

Every output of a freeze lives under ``runs/test_once/lock-<lock SHA-256>/`` and its detections under
``cache/kp-test/lock-<lock SHA-256>/``, and every record (``stage_a.json``, ``stage_b.json``, each clip's
part file, both saved camera files) carries the lock's SHA-256: a re-freeze starts from nothing, and an output
of another lock is never reused or evaluated.

- A, label-free preparation: test frames at 5 fps (``scripts/download_data.py sn-gsr-frames --allow-locked``,
  run with this interpreter), the network weights checked against the lock, keypoint and line networks into
  the lock's detection cache; every cache must carry the detection model version the artifacts were fitted
  on. Reads only the clip list (clip id, game) of ``index/clips.parquet``. Resumable.
- B, camera, label-free: per-frame calibration of every cached test frame (seeded like dev, saved per clip,
  resumable), then PTZ smoothing with the frozen noise parameters. The detection caches must still have the
  SHA-256 recorded by stage A.
- C, evaluation, the only stage that reads test labels (``index/test/``), exactly once. Every label-free input
  is checked and loaded first (stage records, caches, saved cameras, clip lists, frame counts), and both label
  tables must exist with the columns stage C reads (their parquet schemas, no label); then the markers
  (``PREDFOOT_DATA_DIR/test_once/`` and ``runs/test_once/``) are written before any label is read, so a second
  C is refused even after a crash. Every endpoint for both cameras and both references, the success criteria
  evaluated as written, into ``reports/test_run.md`` (aggregate numbers) and the lock's run folder (local). A
  crash or an interruption after the markers (an exception, Ctrl-C, SIGTERM or SIGHUP) still writes
  ``reports/test_run.md``, naming the failure, with every result computed before it; a hard kill (SIGKILL) or
  a power loss leaves only the markers.

``--dry-run`` exercises the same chain on dev clips standing in for test clips (``--dry-clips``, default three
clips of game 4), in a scratch folder: a scratch freeze on a few clips of the five other dev games
(``--dry-train-per-game`` each) with its own scratch lock that pins the stand-ins, the stand-ins' cached
detections copied into a scratch detection cache, the keypoint networks on at most ``--smoke-frames`` (≤ 10)
frames compared with that cache, stages B and C as above (C reads the dev labels), then a second C, which must
be refused. It never reads, writes or checks the real lock, markers, test clip list or test files.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import typer

from predfoot.calib import temporal as tm
from predfoot.contracts import CalibResult
from predfoot.eval import frozen as fz
from predfoot.eval import positions as ps
from predfoot.eval import prereg as pr
from predfoot.io.manifest import sha256_file
from predfoot.io.sngsr import DEV_SPLITS
from predfoot.keypoints.cache import cache_path, cached_detections
from predfoot.paths import REPO_ROOT, cache_dir, data_dir, index_dir

app = typer.Typer(add_completion=False)
MAX_SMOKE_FRAMES = 10
SMOKE_MAX_PX = 0.01  # the networks must reproduce the cached detections (exact on the dev runs)
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
    run_dir: Path  # holds a marker; each lock's outputs go to run_dir / lock-<sha>/ (work_dir)
    markers: list[Path]
    cache_root: Path  # each lock's detections go to cache_root / lock-<sha>/ (det_root)
    report: Path
    persons_path: Path
    lines_path: Path
    lock_path: Path = pr.LOCK_PATH
    clips: list[tuple[str, str, int]] = field(default_factory=list)  # (clip id, split, game)
    lock: pr.Lock | None = None
    lock_sha: str = ""
    log_lines: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        line = f"[{datetime.now(UTC):%H:%M:%S}] {msg}"
        self.log_lines.append(line)
        with contextlib.suppress(OSError):  # a closed terminal or pipe must not stop a stage
            typer.echo(line)

    @property
    def work_dir(self) -> Path:
        assert self.lock_sha, "guard() reads the lock first"
        return self.run_dir / f"lock-{self.lock_sha[:16]}"

    @property
    def det_root(self) -> Path:
        assert self.lock_sha, "guard() reads the lock first"
        return self.cache_root / f"lock-{self.lock_sha[:16]}"


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
    """The lock checks run before every stage; the dry run checks its scratch lock's hashes only (it has no
    git history of its own). Both check the environment."""
    if ctx.dry:
        lock = pr.read_lock(ctx.root, ctx.lock_path)
        pr.verify_hashes(ctx.root, lock)
    else:
        lock = pr.verify_lock(ctx.root, ctx.lock_path)
    pr.verify_environment(lock)
    ctx.lock, ctx.lock_sha = lock, pr.lock_sha256(ctx.root, ctx.lock_path)
    return lock


def artifacts(ctx: Context) -> fz.Artifacts:
    assert ctx.lock is not None
    return fz.load_artifacts(ctx.root / ctx.lock.artifacts_dir)


def test_clips() -> list[tuple[str, str, int]]:
    """The locked split's clip list (clip id, game) from the clip index: no label is read."""
    try:
        return [(c, "test", g) for c, g in fz.locked_clip_list(index_dir() / "clips.parquet")]
    except ValueError as err:
        raise pr.PreregError(str(err)) from err


def ensure_clips(ctx: Context) -> None:
    if ctx.clips:
        return
    if ctx.dry:  # the dry run never falls back to the locked split
        raise pr.PreregError("the dry run has no stand-in clips")
    ctx.clips = test_clips()


def check_clips(ctx: Context) -> None:
    """The clips are exactly the list pinned by the lock (count and digest of the (clip id, game) pairs)."""
    assert ctx.lock is not None
    ensure_clips(ctx)
    pinned = ctx.lock.test_split
    pairs = [(c, g) for c, _, g in ctx.clips]
    got = fz.clip_list_digest(pairs)
    if len(pairs) != pinned.get("clips") or got != pinned.get("clip_ids_sha256"):
        raise pr.PreregError(
            f"the clip list ({len(pairs)} clips, digest {got[:12]}) is not the frozen one"
            f" ({pinned.get('clips')} clips, digest {str(pinned.get('clip_ids_sha256'))[:12]})"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(fz.dump(payload), encoding="utf-8")
    tmp.replace(path)


def read_stage(ctx: Context, name: str) -> dict[str, Any]:
    """A stage record of this lock's run folder; refuses a record of another lock or of other clips."""
    path = ctx.work_dir / name
    if not path.exists():
        raise RuntimeError(f"{name} is missing: run the earlier stage first")
    info: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if info.get("lock_sha256") != ctx.lock_sha:
        raise pr.PreregError(f"{path} was written under another lock")
    if [tuple(c) for c in info.get("clips", [])] != ctx.clips:
        raise pr.PreregError(f"{path} holds another clip list")
    return info


def clip_dir(split: str, clip_id: str) -> Path:
    return data_dir() / "sn-gsr" / split / clip_id


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- stage A ---------------------------------------------------------------------------------------------


def verify_weights(lock: pr.Lock) -> None:
    """The network weights on disk are those whose SHA-256 the freeze recorded."""
    frozen = lock.inputs.get("weights") or {}
    if not frozen:
        raise pr.PreregError("the lock records no network weights")
    folder = data_dir() / "weights" / "pnlcalib"
    for name, sha in sorted(frozen.items()):
        path = folder / name
        if not path.is_file() or sha256_file(path) != sha:
            raise pr.PreregError(f"weights/pnlcalib/{name} is not the frozen file")


def smoke_inference(ctx: Context, clip_id: str, split: str, n_frames: int) -> dict[str, Any]:
    """Run the networks on the first ``n_frames`` (≤ 10) frames of a clip and compare with the cache."""
    n_frames = max(0, min(n_frames, MAX_SMOKE_FRAMES))
    try:
        from predfoot.keypoints.cache import _read_rgb
        from predfoot.keypoints.pnlcalib import PnLCalibNetworks
    except ImportError as err:  # torch missing: the smoke check is skipped, not failed
        return {"skipped": f"torch unavailable ({err})"}
    files = sorted((clip_dir(split, clip_id) / "img1").glob("*.jpg"))[:n_frames]
    cached, _, _ = cached_detections(clip_dir(split, clip_id), clip_id, root=ctx.det_root)
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


def check_smoke(smoke: dict[str, Any], expected_model: str) -> None:
    """The networks reproduce the cache: same ids on every frame, same pixels, same model version."""
    if "skipped" in smoke:
        return
    if (
        smoke["frames_with_same_ids"] != smoke["frames"]
        or max(smoke["max_kp_px_diff"], smoke["max_line_px_diff"]) > SMOKE_MAX_PX
        or smoke["model_version"] != expected_model
    ):
        raise RuntimeError(f"the networks do not reproduce the cached detections: {smoke}")


def stage_a(ctx: Context, smoke_frames: int = 0) -> None:
    lock = guard(ctx)
    check_clips(ctx)
    done = ctx.work_dir / "stage_a.json"
    if done.exists():
        read_stage(ctx, "stage_a.json")
        ctx.log("stage A already done for this lock, skipped")
        return
    expected_model = str(lock.config_digests.get("detection_model") or "")
    if not expected_model:
        raise pr.PreregError("the lock records no detection model version")
    ctx.log(f"stage A: {len(ctx.clips)} clips, detections into {ctx.det_root}")
    if not ctx.dry:
        verify_weights(lock)
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "download_data.py"), "sn-gsr-frames",
               "--every", "5", "--allow-locked", "--clips", *[c for c, _, _ in ctx.clips]]  # fmt: skip
        if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
            raise RuntimeError("frame download failed; rerun stage A (it resumes)")
    smoke: dict[str, Any] | None = None
    cache = {}
    for clip_id, split, _ in ctx.clips:
        n = ps.clip_frame_count(clip_id, split)
        target = cache_path(clip_id, n, root=ctx.det_root)
        if ctx.dry and not target.exists():  # stand-ins: the dev cache is copied, never written
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_path(clip_id, n), target)
        dets, seconds, version = cached_detections(clip_dir(split, clip_id), clip_id, root=ctx.det_root)
        if version != expected_model:
            raise pr.PreregError(
                f"{clip_id}: detections of model {version}, the artifacts were fitted on {expected_model}"
            )
        cache[clip_id] = {
            "frames": len(dets),
            "image_files": n,
            "sha256": sha256_file(target),
            "model_version": version,
        }
        ctx.log(
            f"  {clip_id}: {len(dets)} frames cached (inference time recorded in the cache: {seconds:.0f} s)"
        )
    if smoke_frames:
        clip_id, split, _ = ctx.clips[0]
        smoke = smoke_inference(ctx, clip_id, split, smoke_frames)
        ctx.log(f"  smoke check of the networks: {smoke}")
        check_smoke(smoke, expected_model)
    write_json(
        done,
        {
            "lock_sha256": ctx.lock_sha,
            "code_commit": lock.code_commit,
            "clips": ctx.clips,
            "cache": cache,
            "smoke": smoke,
            "finished_utc": _now(),
        },
    )


# --- stage B ---------------------------------------------------------------------------------------------


def _clip_dets(ctx: Context, clip_id: str, split: str) -> dict[int, dict[str, Any]]:
    dets = ps.clip_detections(clip_id, split, ctx.det_root)
    if dets is None:
        raise RuntimeError(f"{clip_id} has no detection cache under {ctx.det_root}: run stage A")
    return dets


def verify_caches(ctx: Context, stage_a_info: dict[str, Any]) -> None:
    """Every clip's detection cache is still the file stage A recorded (same frames on disk, same SHA-256)."""
    for clip_id, split, _ in ctx.clips:
        rec = stage_a_info["cache"].get(clip_id)
        path = cache_path(clip_id, ps.clip_frame_count(clip_id, split), root=ctx.det_root)
        if rec is None or not path.exists() or sha256_file(path) != rec["sha256"]:
            raise pr.PreregError(f"the detection cache of {clip_id} changed since stage A ({path})")


def check_stage_b(ctx: Context, info: dict[str, Any], art: fz.Artifacts) -> None:
    """Stage B smoothed with the frozen parameters and its two camera files are unchanged."""
    if (
        info.get("params_digest") != art.params.digest()
        or info.get("smoother_config_digest") != art.smoother_config_digest
    ):
        raise pr.PreregError("stage B used other noise parameters or another smoother configuration")
    for name, sha in info["sha256"].items():
        if sha256_file(ctx.work_dir / name) != sha:
            raise pr.PreregError(f"{name} changed since stage B")


def _load_part(ctx: Context, part: Path, clip_id: str) -> list[CalibResult] | None:
    """A clip's saved per-frame calibrations, if readable and made under this lock; else None (recompute)."""
    if not part.exists():
        return None
    try:
        saved, meta = ps.load_calib(part)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as err:
        ctx.log(f"  {clip_id}: unreadable part file ({type(err).__name__}), recomputed")
        return None
    if meta.get("lock_sha256") != ctx.lock_sha or clip_id not in saved:
        ctx.log(f"  {clip_id}: part file of another lock, recomputed")
        return None
    return saved[clip_id]


def stage_b(ctx: Context) -> None:
    guard(ctx)
    check_clips(ctx)
    a = read_stage(ctx, "stage_a.json")
    verify_caches(ctx, a)
    art = artifacts(ctx)
    done = ctx.work_dir / "stage_b.json"
    if done.exists():
        check_stage_b(ctx, read_stage(ctx, "stage_b.json"), art)
        ctx.log("stage B already done for this lock, skipped")
        return
    ctx.log(f"stage B: per-frame calibration of {len(ctx.clips)} clips, then smoothing (frozen noise)")
    ccs = []
    for clip_id, split, game in ctx.clips:
        part = ctx.work_dir / "stage_b" / f"{clip_id}.npz"
        dets = _clip_dets(ctx, clip_id, split)
        saved = _load_part(ctx, part, clip_id)
        if saved is not None:
            cc = ps.calibrate_detections(clip_id, split, game, dets, saved)
        else:
            t0 = time.perf_counter()
            cc = ps.calibrate_detections(clip_id, split, game, dets)
            ps.save_calib(part, [cc], {"clip": clip_id, "lock_sha256": ctx.lock_sha})
            rate = np.mean([r.ok for r in cc.results])
            seconds = time.perf_counter() - t0
            ctx.log(f"  {clip_id}: {len(cc.frames)} frames, {rate:.0%} calibrated ({seconds:.0f} s)")
        ccs.append(cc)
    meta = {
        "stage": "B",
        "lock_sha256": ctx.lock_sha,
        "smoother_config_digest": art.smoother_config_digest,
        "params": art.params.to_dict(),
        "params_digest": art.params.digest(),
    }
    ps.save_calib(ctx.work_dir / "calib_perframe.npz", ccs, meta)
    cfg = tm.SmootherConfig()
    info = ps.SmoothingInfo()
    outputs = list(ps.smooth_outputs(ccs, ps.Smooth.ptz, ps.FitScope.all, cfg, info, params=art.params))
    fz.save_outputs(ctx.work_dir / "calib_ptz.npz", outputs, meta)
    write_json(
        done,
        {
            "lock_sha256": ctx.lock_sha,
            "clips": ctx.clips,
            "frames": int(sum(len(cc.frames) for cc in ccs)),
            "perframe_ok": float(np.mean([r.ok for cc in ccs for r in cc.results])),
            "ptz_ok": float(np.mean([r.ok for o in outputs for r in o.results])),
            "params_digest": art.params.digest(),
            "smoother_config_digest": art.smoother_config_digest,
            "sha256": {
                name: sha256_file(ctx.work_dir / name) for name in ("calib_perframe.npz", "calib_ptz.npz")
            },
            "finished_utc": _now(),
        },
    )
    ctx.log(f"stage B done: {len(ccs)} clips, noise parameters {art.params.digest()} (frozen)")


# --- stage C ---------------------------------------------------------------------------------------------


@dataclass
class Prepared:
    """Every label-free input of stage C, checked and loaded before the claim."""

    art: fz.Artifacts
    dets: dict[str, dict[int, dict[str, Any]]]
    ccs: list[ps.ClipCalib]  # per-frame calibrations of stage B, no label attached yet
    smoothed: dict[str, fz.SavedOutput]


#: Columns of the two label tables that stage C reads (:func:`predfoot.eval.positions.attach_labels`).
LABEL_COLUMNS = {
    "persons": ("clip_id", "frame", "track_id", "foot_u", "foot_v", "x", "y", "x_raw", "y_raw", "truncated"),
    "lines": ("clip_id", "frame", "line_class", "point_idx", "u", "v", "is_planar"),
}


def check_label_tables(ctx: Context) -> None:
    """Both label tables exist and have the columns stage C reads. Only their parquet schemas are read, no
    label: a missing, empty or damaged table is refused before the claim, not after it."""
    for name, path in (("persons", ctx.persons_path), ("lines", ctx.lines_path)):
        if not path.is_file() or path.stat().st_size == 0:
            raise pr.PreregError(f"the {name} label table {path} is missing or empty")
        try:
            schema = pl.read_parquet_schema(path)
        except Exception as err:  # a damaged file: polars raises its own error types
            raise pr.PreregError(f"the {name} label table {path} is unreadable ({err})") from err
        missing = [c for c in LABEL_COLUMNS[name] if c not in schema]
        if missing:
            raise pr.PreregError(f"the {name} label table {path} lacks the columns {missing}")


def prepare_c(ctx: Context) -> Prepared:
    """Check and load everything stage C needs except the labels, so that nothing label-free can fail after
    the claim: the clip list, the two label tables (their schemas, no label), both stage records, the
    detection caches, the frozen artifacts, the two camera files (lock, parameters, clips) and every clip's
    frame count."""
    check_clips(ctx)
    check_label_tables(ctx)
    verify_caches(ctx, read_stage(ctx, "stage_a.json"))
    art = artifacts(ctx)
    check_stage_b(ctx, read_stage(ctx, "stage_b.json"), art)
    perframe, meta_p = ps.load_calib(ctx.work_dir / "calib_perframe.npz")
    smoothed, meta_s = fz.load_outputs(ctx.work_dir / "calib_ptz.npz")
    ids = [c for c, _, _ in ctx.clips]
    for name, meta in (("calib_perframe.npz", meta_p), ("calib_ptz.npz", meta_s)):
        if (
            meta.get("lock_sha256") != ctx.lock_sha
            or meta.get("params_digest") != art.params.digest()
            or meta.get("clips") != ids
        ):
            raise pr.PreregError(f"{name} was not made by stage B of this lock for these clips")
    dets: dict[str, dict[int, dict[str, Any]]] = {}
    ccs = []
    for clip_id, split, game in ctx.clips:
        dets[clip_id] = _clip_dets(ctx, clip_id, split)
        try:
            cc = ps.calibrate_detections(clip_id, split, game, dets[clip_id], perframe[clip_id])
        except ValueError as err:
            raise pr.PreregError(str(err)) from err
        res, flags, _ = smoothed[clip_id]
        if len(res) != len(cc.frames) or len(flags.smoothed) != len(cc.frames):
            raise pr.PreregError(f"{clip_id}: {len(res)} smoothed cameras for {len(cc.frames)} cached frames")
        ccs.append(cc)
    return Prepared(art, dets, ccs, smoothed)


def report_header(ctx: Context, lock: pr.Lock, started_utc: str) -> list[str]:
    title = "Dry run of the test-run tooling on dev clips" if ctx.dry else "Preregistered test run"
    header = [
        f"# {title}",
        "",
        f"`uv run python scripts/run_test_once.py{' --dry-run' if ctx.dry else ''}` · lock code commit"
        f" `{lock.code_commit[:12]}` · preregistration SHA-256 `{lock.preregistration['sha256'][:16]}…` ·"
        f" lock SHA-256 `{ctx.lock_sha[:16]}…` · stage C started {started_utc}.",
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
    return header


class Interrupted(KeyboardInterrupt):
    """A signal (SIGINT, SIGTERM, SIGHUP) while stage C holds its claim, raised like Ctrl-C so that the
    failure report is still written."""


@dataclass
class SignalGuard:
    """Raises :class:`Interrupted` on the first signal; once ``held`` is set (by that first signal, or while
    the failure report is written), later signals are ignored."""

    held: bool = False

    def handler(self, signum: int, _frame: Any) -> None:
        if not self.held:
            self.held = True
            raise Interrupted(signal.Signals(signum).name)


@contextlib.contextmanager
def signals_raise() -> Iterator[SignalGuard]:
    """While stage C holds its claim, SIGINT, SIGTERM and SIGHUP raise :class:`Interrupted`: by default
    SIGTERM and SIGHUP (a closed terminal) end Python without running ``finally`` blocks, so no report would
    be written. A signal already ignored, as under ``nohup``, stays ignored; outside the main thread nothing
    changes. The previous handlers are restored afterwards."""
    guard = SignalGuard()
    previous: dict[int, Any] = {}
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            old = signal.getsignal(sig)
            if old == signal.SIG_IGN:
                continue
            signal.signal(sig, guard.handler)
        except ValueError:  # not the main thread
            continue
        previous[sig] = old
    try:
        yield guard
    finally:
        for sig, old in previous.items():
            signal.signal(sig, signal.SIG_DFL if old is None else old)


def stage_c(ctx: Context) -> dict[str, Any]:
    lock = guard(ctx)  # nothing below runs without a matching lock
    if ctx.report.exists():  # a committed result, e.g. from another clone with its own data folder
        raise pr.PreregError(f"{ctx.report} exists: the test evaluation already ran, a second one is refused")
    prepared = prepare_c(ctx)  # label-free: a failure here leaves the evaluation unclaimed
    existing = [m for m in ctx.markers if m.exists()]
    if existing:  # claim_once refuses too; refused here, a second stage C never writes a failure report
        raise pr.PreregError(
            f"the test evaluation already started ({existing[0]}): a second evaluation is refused"
        )
    payload = {
        "started_utc": _now(),
        "lock_sha256": ctx.lock_sha,
        "code_commit": lock.code_commit,
        "freeze_commit": None if ctx.dry else pr.freeze_commit(ctx.root, ctx.lock_path),
        "head": pr.head_commit(REPO_ROOT),
        "dry_run": ctx.dry,
    }
    with signals_raise() as signals:
        try:
            pr.claim_once(ctx.markers, payload)  # written before any label is read
            ctx.log(f"stage C started, markers written: {', '.join(str(m) for m in ctx.markers)}")
            return _evaluate(ctx, prepared, lock, payload)
        except BaseException as err:  # Ctrl-C, SIGTERM and SIGHUP included
            signals.held = True  # a second signal must not stop the failure report
            if any(m.exists() for m in ctx.markers):  # the claim is spent, even if half written
                _failure_report(ctx, lock, payload, err)
            raise
        finally:
            with contextlib.suppress(OSError):
                (ctx.work_dir / "stage_c.log").write_text("\n".join(ctx.log_lines) + "\n", encoding="utf-8")


def _failure_report(ctx: Context, lock: pr.Lock, payload: dict[str, Any], err: BaseException) -> None:
    """The claim is spent: a crash or an interruption after it still leaves the report, which names the
    failure (a deviation) and carries every result computed before it. The traceback goes to the log."""
    ctx.log_lines.append("".join(traceback.format_exception(err)).rstrip())
    if ctx.report.exists():
        return
    what = f"{type(err).__name__}: {err}" if str(err) else type(err).__name__
    md = [
        *report_header(ctx, lock, payload["started_utc"]),
        "",
        "## Stage C failed",
        "",
        f"Stage C stopped after its markers were written: `{what}`. As preregistered, the evaluation is not"
        " run a second time; this failure is a deviation from the preregistration. The traceback is in"
        " `stage_c.log`, in the lock's run folder (local).",
        "",
        *_computed_results(ctx),
    ]
    ctx.report.parent.mkdir(parents=True, exist_ok=True)
    ctx.report.write_text("\n".join(md) + "\n", encoding="utf-8")
    ctx.log(f"stage C failed after the claim ({what}); failure report {ctx.report}")


def _computed_results(ctx: Context) -> list[str]:
    """Every result computed before a failure: ``results.json`` is written before the report is rendered."""
    try:
        res = json.loads((ctx.work_dir / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["No result was computed before the failure."]
    return [
        "Every endpoint was computed before the failure, but the report could not be rendered. The success"
        " criteria and every result, as saved in `results.json` (aggregate numbers):",
        "",
        "```json",
        fz.dump(res).rstrip(),
        "```",
    ]


def _evaluate(ctx: Context, prepared: Prepared, lock: pr.Lock, payload: dict[str, Any]) -> dict[str, Any]:
    ids = [c for c, _, _ in ctx.clips]
    persons = pl.read_parquet(ctx.persons_path).filter(pl.col("clip_id").is_in(ids))
    lines = pl.read_parquet(ctx.lines_path).filter(pl.col("clip_id").is_in(ids))
    ctx.log(f"labels read: {persons.height} person rows, {lines.height} line points")
    outputs = []
    for cc in prepared.ccs:
        ps.attach_labels(
            cc,
            prepared.dets[cc.clip_id],
            persons.filter(pl.col("clip_id") == cc.clip_id),
            lines.filter(pl.col("clip_id") == cc.clip_id),
        )
        res, flags, diag = prepared.smoothed[cc.clip_id]
        outputs.append(ps.Output(cc, None, res, flags, diag))
    tables = {"perframe": fz.perframe_tables(prepared.ccs), "ptz": fz.output_tables(outputs)}
    for name, t in tables.items():
        if t.players.height:
            t.players.write_parquet(ctx.work_dir / f"players_{name}.parquet")
        t.frames.write_parquet(ctx.work_dir / f"frames_{name}.parquet")
    ctx.log(
        f"labels attached ({tables['perframe'].frames.height} labelled frames); computing every endpoint"
        f" ({fz.N_BOOT} bootstrap draws per interval)"
    )
    res = fz.evaluate(tables, prepared.art)
    res["provenance"] = {**payload, "lock_code_commit": lock.code_commit, "artifacts": lock.artifacts}
    write_json(ctx.work_dir / "results.json", res)
    ctx.log("every endpoint computed and saved (results.json); writing the report")
    ctx.report.parent.mkdir(parents=True, exist_ok=True)
    ctx.report.write_text(
        fz.render_report(res, report_header(ctx, lock, payload["started_utc"])), encoding="utf-8"
    )
    write_json(ctx.work_dir / "stage_c.json", {"finished_utc": _now(), "report": str(ctx.report)})
    ctx.log(f"stage C done: report {ctx.report}")
    return res


# --- dry run ---------------------------------------------------------------------------------------------


def dry_context(scratch: Path, clips: list[str], train_per_game: int, calib_from: Path | None) -> Context:
    """Scratch freeze (own artifacts and lock under ``scratch``, pinning the stand-ins) and a context that
    never touches the real lock, markers, cache, test clip list or test files."""
    if not clips:
        raise typer.BadParameter("--dry-clips needs at least one dev clip")
    dev = pl.read_parquet(index_dir() / "clips.parquet").filter(pl.col("split").is_in(list(DEV_SPLITS)))
    dev = dev.sort("clip_id")
    unknown = sorted(set(clips) - set(dev["clip_id"].to_list()))
    if unknown:
        raise typer.BadParameter(f"--dry-clips must be dev clips: {unknown}")
    stand_in = dev.filter(pl.col("clip_id").is_in(clips))
    stand_in_rows = [(r["clip_id"], r["split"], int(r["game_id"])) for r in stand_in.iter_rows(named=True)]
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
            test_clips=[(c, g) for c, _, g in stand_in_rows],
            detection_model=freeze.detection_model(ccs),
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
        clips=stand_in_rows,
    )


def compare_with_dev(ctx: Context, calib_from: Path) -> None:
    """Stage B recalibrated the stand-ins from their detections: they must equal the dev calibrations."""
    freeze = _load_freeze_module()
    saved, _ = ps.load_calib(calib_from)
    mine, _ = ps.load_calib(ctx.work_dir / "calib_perframe.npz")
    same = {c: freeze.same_results(mine[c], saved[c]) for c in mine}
    ctx.log(f"stage B per-frame calibrations equal to {calib_from.name}: {same}")
    if not all(same.values()):
        raise RuntimeError(f"stage B does not reproduce the dev calibrations of {calib_from}: {same}")


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
    if not 0 <= smoke_frames <= MAX_SMOKE_FRAMES:
        raise typer.BadParameter(f"--smoke-frames is between 0 and {MAX_SMOKE_FRAMES}")
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
