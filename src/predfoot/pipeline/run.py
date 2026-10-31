"""``predfoot run``: the end-to-end tracer bullet, processed in blocks that survive a crash.

Native frames → stages (``CalibResult``, ``FootObs``, ``Localization``) → export frames at 10 fps →
``tracking.jsonl``, ``match.json``, ``meta.json``, ``metrics.json``, ``minimap.png`` and ``qa_report.html`` in
the output folder.
The stages are the stubs of :mod:`predfoot.pipeline.stub` until the real modules replace them.

Resume: export frames are processed in blocks of ``block_size``, each written atomically to ``OUT/.parts/``.
A re-run reuses the complete blocks of the same input and configuration (checked by a fingerprint) and
recomputes the others. Every stage is deterministic, so killing a run at any point and running it again gives
byte-identical outputs; only ``metrics.json`` differs (timings, reused blocks). One run per output folder at a
time: there is no lock.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from predfoot import __version__
from predfoot.contracts import CalibResult, FloatArray, FootObs, Localization
from predfoot.io.files import (
    atomic_write_bytes,
    atomic_write_chunks,
    sha256_file,
    sha256_json,
    write_json,
)
from predfoot.io.skillcorner import (
    EXPORT_FPS,
    TrackingFrame,
    dumps_record,
    frame_to_record,
    image_corners_projection,
    make_match_json,
    read_tracking,
)
from predfoot.io.timebase import ExportSlot, export_schedule, format_timestamp, interpolate_localization
from predfoot.paths import REPO_ROOT
from predfoot.pipeline import stub
from predfoot.pipeline.clip import ClipFrame, ClipInput, load_clip
from predfoot.viz.minimap import render_minimap, select_panels
from predfoot.viz.qa_report import write_qa_report

#: Bump when the stub stages change their outputs, so stale blocks are not reused.
STUB_VERSION = 1
SCHEMA_VERSION = "1.0"
PARTS_DIR = ".parts"


@dataclass(frozen=True)
class RunConfig:
    """Run settings. Everything except ``block_size`` changes the outputs and enters the config hash."""

    export_fps: float = EXPORT_FPS
    block_size: int = 100
    placeholder_radius_m: float = stub.PLACEHOLDER_RADIUS_M
    n_panels: int = 6

    def __post_init__(self) -> None:
        if self.export_fps <= 0:
            raise ValueError("export_fps must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be at least 1")

    def output_settings(self) -> dict[str, Any]:
        return {
            "export_fps": float(self.export_fps),  # 10 and 10.0 are one setting, with one config hash
            "placeholder_radius_m": self.placeholder_radius_m,
            "n_panels": self.n_panels,
            "schema_version": SCHEMA_VERSION,
        }


class ClipMismatch(ValueError):
    """The stages cannot run on this clip, for example detections that name frames the clip does not have."""


@dataclass(frozen=True)
class Stages:
    """The pipeline's replaceable parts, typed by the shared contracts.

    ``version`` identifies what the stages compute (module versions, weight hashes): it enters the config
    hash, so blocks computed by other stages are never reused.
    """

    version: str
    calibrate: Callable[[ClipFrame], CalibResult]
    detect: Callable[[ClipFrame], list[FootObs]]
    localize: Callable[[CalibResult, Sequence[FootObs]], list[Localization]]
    health: Callable[[CalibResult], tuple[float | None, bool]]
    #: What each stage does (calibration, detection, localization, health), for meta.json. None: the stubs.
    description: Mapping[str, str] | None = None
    #: Provenance of the conformal model behind the regions, for meta.json; None without one.
    conformal: Mapping[str, Any] | None = None
    #: Raises :class:`ClipMismatch` when the stages were not built for the clip; ``run_clip`` calls it before
    #: computing anything. None: the stages run on any clip.
    check_clip: Callable[[ClipInput], None] | None = None


def stub_stages(config: RunConfig) -> Stages:
    return Stages(
        version=f"stub-{STUB_VERSION}",
        calibrate=stub.calibrate,
        detect=stub.detect,
        localize=functools.partial(stub.localize, radius_m=config.placeholder_radius_m),
        health=stub.health,
    )


@dataclass(frozen=True)
class NativeResult:
    """Everything the export needs from one native frame."""

    abstain: bool
    calib_health: float | None
    corners: FloatArray | None
    localizations: dict[int, Localization]


def process_native_frame(frame: ClipFrame, clip: ClipInput, stages: Stages) -> NativeResult:
    calib = stages.calibrate(frame)
    localizations = stages.localize(calib, stages.detect(frame))
    calib_health, abstain = stages.health(calib)
    corners = None
    if not abstain and calib.ok and calib.H is not None:
        corners = image_corners_projection(calib.H, clip.image_size, pitch_width=clip.pitch.width)
    if abstain:  # an abstained frame publishes no position, whatever the stages returned
        localizations = [Localization(p.track_id, xy=None, is_detected=p.is_detected) for p in localizations]
    return NativeResult(abstain, calib_health, corners, {loc.track_id: loc for loc in localizations})


def export_frame(slot: ExportSlot, r0: NativeResult, r1: NativeResult, export_fps: float) -> TrackingFrame:
    """Export frame between native results ``r0`` (weight 1 - w) and ``r1`` (weight w).

    A track is exported when both native frames see it. The frame abstains if either native frame does; the
    corners come from the nearer one; the health is the lower of the two.
    """
    w = slot.w
    abstain = r0.abstain or (w > 0 and r1.abstain)
    if w == 0:
        players = list(r0.localizations.values())
    else:
        players = [
            interpolate_localization(loc, r1.localizations[tid], w)
            for tid, loc in r0.localizations.items()
            if tid in r1.localizations
        ]
    if abstain:
        players = [Localization(p.track_id, xy=None, is_detected=p.is_detected) for p in players]
    healths = [r.calib_health for r in ((r0,) if w == 0 else (r0, r1)) if r.calib_health is not None]
    return TrackingFrame(
        frame=slot.export_frame,
        timestamp=format_timestamp(slot.export_frame / export_fps),
        period=1,
        players=tuple(players),
        corners=None if abstain else (r0.corners if w < 0.5 else r1.corners),
        calib_health=min(healths) if healths else None,
        abstain=abstain,
    )


def _compute_block(clip: ClipInput, slots: Sequence[ExportSlot], stages: Stages, export_fps: float) -> bytes:
    """JSONL bytes of one block of export frames."""
    needed = sorted({s.i0 for s in slots} | {s.i1 for s in slots})
    results = {i: process_native_frame(clip.frames[i], clip, stages) for i in needed}
    lines = [
        dumps_record(frame_to_record(export_frame(s, results[s.i0], results[s.i1], export_fps))) + "\n"
        for s in slots
    ]
    return "".join(lines).encode("utf-8")


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def git_sha() -> str | None:
    """Commit of the running code: ``PREDFOOT_GIT_SHA`` (Docker builds), else git in a source checkout."""
    if sha := os.environ.get("PREDFOOT_GIT_SHA"):
        return sha
    if not (REPO_ROOT / "pyproject.toml").is_file() or not (REPO_ROOT / "src" / "predfoot").is_dir():
        return None  # an installed package: a git repo above it would describe something else
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None


@dataclass(frozen=True)
class RunResult:
    out_dir: Path
    n_native: int
    n_export: int
    n_abstained: int
    blocks_computed: int
    blocks_reused: int


def _prepare_parts(parts: Path, fingerprint: dict[str, Any]) -> None:
    """Keep ``parts`` only if it was written for the same input, configuration and block size."""
    marker = parts / "fingerprint.json"
    if parts.exists():
        try:
            same = json.loads(marker.read_text(encoding="utf-8")) == fingerprint
        except (OSError, ValueError):
            same = False
        if not same:
            shutil.rmtree(parts)
    if not parts.exists():
        parts.mkdir(parents=True)
        write_json(marker, fingerprint)


def run_clip(
    input_path: Path, out_dir: Path, config: RunConfig | None = None, stages: Stages | None = None
) -> RunResult:
    """Run the pipeline on a clip file and write every output to ``out_dir`` (resuming if possible)."""
    started = time.perf_counter()
    config = config or RunConfig()
    stages = stages or stub_stages(config)
    input_path, out_dir = Path(input_path), Path(out_dir)
    clip = load_clip(input_path)
    if stages.check_clip is not None:
        stages.check_clip(clip)
    input_sha = sha256_file(input_path)
    settings = config.output_settings() | {"stages": stages.version}
    config_hash = sha256_json(settings)
    sha = git_sha()
    parts = out_dir / PARTS_DIR
    _prepare_parts(
        parts,
        {
            "input_sha256": input_sha,
            "config_hash": config_hash,
            "block_size": config.block_size,
            "predfoot_version": __version__,
            "git_sha": sha,
        },
    )

    slots = export_schedule(len(clip.frames), clip.fps, config.export_fps)
    blocks = [slots[i : i + config.block_size] for i in range(0, len(slots), config.block_size)]
    block_paths = [parts / f"block-{b:06d}.jsonl" for b in range(len(blocks))]
    computed = reused = 0
    for block, path in zip(blocks, block_paths, strict=True):
        if path.exists() and _count_lines(path) == len(block):
            reused += 1
            continue
        atomic_write_bytes(path, _compute_block(clip, block, stages, config.export_fps))
        computed += 1
    pipeline_s = time.perf_counter() - started

    tracking_path = out_dir / "tracking.jsonl"
    atomic_write_chunks(tracking_path, (p.read_bytes() for p in block_paths))
    summary = _summarise(tracking_path)  # streamed: a full match does not fit in memory as objects
    write_json(
        out_dir / "match.json",
        make_match_json(
            summary.track_ids,
            team_of=clip.teams,
            pitch_length=clip.pitch.length,
            pitch_width=clip.pitch.width,
            n_frames=summary.n_frames,
            fps=config.export_fps,
        ),
    )
    panels = set(select_panels(summary.abstain, config.n_panels))
    render_minimap(
        [f for i, f in enumerate(read_tracking(tracking_path)) if i in panels],
        out_dir / "minimap.png",
        n_panels=len(panels),
        pitch_length=clip.pitch.length,
        pitch_width=clip.pitch.width,
        team_of=clip.teams,
        title=clip.clip_id,
    )
    write_json(out_dir / "meta.json", _meta(clip, input_path, input_sha, settings, config_hash, sha, stages))
    n_abstained = sum(summary.abstain)
    write_json(
        out_dir / "metrics.json",
        {
            "clip_id": clip.clip_id,
            "frames": {
                "native": len(clip.frames),
                "exported": summary.n_frames,
                "abstained": n_abstained,
                "abstention_rate": round(n_abstained / summary.n_frames, 4),
            },
            "players": {
                "tracks": len(summary.track_ids),
                "positions": summary.positions,
                "unlocated_in_live_frames": summary.unlocated,
            },
            "blocks": {
                "size": config.block_size,
                "total": len(blocks),
                "computed": computed,
                "reused": reused,
            },
            "timing_s": {
                "pipeline": round(pipeline_s, 4),
                "total": round(time.perf_counter() - started, 4),
            },
        },
    )
    write_qa_report(out_dir)  # reads the files above; holds no timings, so it is byte-stable too
    shutil.rmtree(parts)
    return RunResult(out_dir, len(clip.frames), summary.n_frames, n_abstained, computed, reused)


@dataclass(frozen=True)
class _Summary:
    n_frames: int
    abstain: list[bool]
    track_ids: list[int]
    positions: int
    unlocated: int


def _summarise(tracking_path: Path) -> _Summary:
    abstain: list[bool] = []
    track_ids: set[int] = set()
    positions = unlocated = 0
    for frame in read_tracking(tracking_path):
        abstain.append(frame.abstain)
        for player in frame.players:
            track_ids.add(player.track_id)
            if player.xy is not None:
                positions += 1
            elif not frame.abstain:
                unlocated += 1
    return _Summary(len(abstain), abstain, sorted(track_ids), positions, unlocated)


def _meta(
    clip: ClipInput,
    input_path: Path,
    input_sha: str,
    settings: dict[str, Any],
    config_hash: str,
    sha: str | None,
    stages: Stages | None = None,
) -> dict[str, Any]:
    """Provenance of a run. No timestamps, so that a resumed run writes the same bytes."""
    described = stages.description if stages is not None else None
    description = dict(described) if described is not None else _stub_description(settings)
    return {
        "schema_version": SCHEMA_VERSION,
        "predfoot_version": __version__,
        "git_sha": sha,
        "config_hash": config_hash,
        "config": settings,
        "input": {"file": input_path.name, "sha256": input_sha, "format": clip.format},
        "clip_id": clip.clip_id,
        "native_fps": clip.fps,
        "export_fps": settings["export_fps"],
        "image_size": list(clip.image_size),
        "pitch": {"length": clip.pitch.length, "width": clip.pitch.width},
        "time_resampling": "linear interpolation between the two native frames around each export instant",
        "stages": description,
        "conformal": None if stages is None or stages.conformal is None else dict(stages.conformal),
    }


def _stub_description(settings: dict[str, Any]) -> dict[str, str]:
    return {
        "calibration": "stub: homography read from the input",
        "detection": "stub: foot points read from the input",
        "localization": (
            "inverse homography; placeholder uncertainty: constant "
            f"{settings['placeholder_radius_m']:g} m radius, isotropic covariance"
        ),
        "health": "none: calib_health is null; a frame abstains when its calibration failed",
    }
