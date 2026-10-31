"""Per-frame latency of the PnLCalib networks on this machine (ticket T5.5).

Backends: PyTorch CPU, PyTorch MPS (Apple GPU, when available), ONNX Runtime CPU and ONNX Runtime with the
CoreML execution provider (when available).

Usage:
    uv sync --extra vision --extra export
    uv run python scripts/export_onnx.py                 # the ONNX models, once
    uv run python scripts/bench_latency.py --frames 3 --out reports/latency_m4.md

Method: frames of a dev clip are decoded and preprocessed once (resized to 960 × 540, RGB in [0, 1]). Each
backend then runs both networks on one frame at a time (batch 1). The clock covers the copy to the device,
both forward passes and the heatmaps back in host memory; MPS is synchronised before the clock starts and
stops. The first frame is a warm-up and is excluded; loading is timed apart. JPEG decoding, resizing and
heatmap decoding, which surround the networks in the pipeline, are timed separately on the CPU.
"""

from __future__ import annotations

import dataclasses
import gc
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import typer

from predfoot.keypoints.onnx_models import (
    MODEL_FILES,
    Heatmaps,
    OnnxPnLCalib,
    Parity,
    onnx_dir,
    parity,
    read_frames,
    spread_frames,
)
from predfoot.keypoints.pnlcalib import PnLCalibNetworks, decode_detections, preprocess
from predfoot.paths import REPO_ROOT, data_dir

app = typer.Typer(add_completion=False)
DEFAULT_CLIP = Path("sn-gsr") / "valid" / "SNGS-056"
COREML_OPTIONS = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
MATCH_FRAMES_10FPS = 90 * 60 * 10


@dataclasses.dataclass(frozen=True)
class Result:
    backend: str
    load_s: float
    warmup_ms: float
    times_ms: list[float]
    max_shift_px: float | None = None  # decoded keypoints and line ends against PyTorch CPU
    flips: int | None = None

    def against(self, p: Parity) -> Result:
        return dataclasses.replace(
            self, max_shift_px=max(p.max_keypoint_px, p.max_line_end_px), flips=p.n_flips
        )


def timed_runs(
    run: Callable[[Any], Heatmaps], frames: list[Any], sync: Callable[[], None]
) -> tuple[float, list[float], Heatmaps]:
    """Warm-up time and per-frame times (ms), one frame at a time, and the heatmaps of frames[1:]."""
    times, outputs = [], []
    for x in frames:
        sync()
        start = time.perf_counter()
        out = run(x)
        sync()
        times.append(1e3 * (time.perf_counter() - start))
        outputs.append(out)
    timed = outputs[1:]
    return times[0], times[1:], (np.concatenate([o[0] for o in timed]), np.concatenate([o[1] for o in timed]))


@contextmanager
def stderr_to(path: Path) -> Iterator[None]:
    """Send the process-level stderr, C++ logs included, to ``path``."""
    sys.stderr.flush()
    saved = os.dup(2)
    with path.open("wb") as fh:
        os.dup2(fh.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def coreml_partition(log: str) -> str:
    """How much of each graph the CoreML provider took, from ONNX Runtime's verbose log."""
    nodes = re.findall(r"number of nodes in the graph: (\d+) number of nodes supported by CoreML: (\d+)", log)
    parts = re.findall(r"number of partitions supported by CoreML: (\d+)", log)
    if len(nodes) < len(MODEL_FILES) or len(parts) < len(MODEL_FILES):
        return "how ONNX Runtime split the graphs between CoreML and the CPU was not logged"
    graphs = [
        f"{taken} of {total} nodes of the {name} graph in {n_parts} partition(s)"
        for name, (total, taken), n_parts in zip(("keypoint", "line"), nodes, parts, strict=False)
    ]
    return "CoreML ran " + " and ".join(graphs)


def machine() -> str:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
    return f"{chip.stdout.strip() or platform.processor()}, macOS {platform.mac_ver()[0]}"


def git_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    return f"{sha}+dirty" if git("status", "--porcelain") else sha


def report(results: list[Result], ctx: dict[str, Any]) -> str:
    lines = [
        "# Keypoint network latency on an Apple M4 Pro",
        "",
        f"`uv run python scripts/bench_latency.py --frames {ctx['n']}` · commit `{ctx['commit']}` · "
        f"{ctx['date']} · model `{ctx['model']}`",
        "",
        f"Machine: {ctx['machine']}; torch {ctx['torch']}, onnxruntime {ctx['onnxruntime']}. "
        f"Load average at start (1, 5, 15 min): {ctx['loadavg']}; other jobs may have shared the machine.",
        "",
        "| Backend | Load (s) | Warm-up (ms) | ms / frame, median (min–max) | Frames / s | "
        "h per match at 10 fps | Max shift vs PyTorch CPU (px) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        med = statistics.median(r.times_ms)
        shift = "reference" if r.max_shift_px is None else f"{r.max_shift_px:.4f}"
        if r.flips:
            shift += f" ({r.flips} detections in one backend only)"
        lines.append(
            f"| {r.backend} | {r.load_s:.1f} | {r.warmup_ms:.0f} | {med:.0f} ({min(r.times_ms):.0f}–"
            f"{max(r.times_ms):.0f}) | {1e3 / med:.1f} | {MATCH_FRAMES_10FPS * med / 3.6e6:.1f} | {shift} |"
        )
    lines += [
        "",
        f"Around the networks, on the CPU: JPEG decoding and resizing {ctx['pre_ms']:.0f} ms per frame, "
        f"heatmap decoding {ctx['post_ms']:.0f} ms per frame (medians).",
        "",
        "**Method.** Both networks (keypoints and lines, HRNet-W48) run on one 960 × 540 frame at a time "
        "(batch 1, float32). The clock covers the copy to the device, both forward passes and the heatmaps "
        "back in host memory; MPS is synchronised (`torch.mps.synchronize()`) before the clock starts and "
        "stops. The first frame warms each backend up and is excluded; loading (weights and their hashes, "
        f"session creation, CoreML compilation) is timed apart. Frames: {ctx['n']} timed frames of dev clip "
        f"{ctx['clip']} after one warm-up frame, the same preprocessed inputs for every backend. PyTorch "
        f"CPU uses {ctx['torch_threads']} threads, ONNX Runtime its defaults. {ctx['coreml']}. Max shift: "
        "the largest distance between the keypoints and line ends decoded from this backend's heatmaps and "
        "from PyTorch CPU's, in pixels of the 1920 × 1080 frame (the ONNX export itself is checked within "
        "0.5 px by `scripts/export_onnx.py`).",
        "",
        f"**Caveats.** {ctx['n']} timed frames per backend make a smoke run, not a stable estimate: no "
        "confidence interval and no thermal steady state. Hours per match extrapolate the median to 54,000 "
        "frames (90 minutes at 10 fps), networks only. No T4 / CUDA row yet.",
        "",
    ]
    return "\n".join(lines)


@app.command()
def main(
    frames: int = typer.Option(3, min=1, help="Timed frames per backend (after one warm-up frame)."),
    clip_dir: Path | None = typer.Option(None, help="Clip folder with img1/ (default: dev clip SNGS-056)."),  # noqa: B008
    out: Path | None = typer.Option(None, help="Markdown report to write (default: print it)."),  # noqa: B008
) -> None:
    import torch

    clip = clip_dir or data_dir() / DEFAULT_CLIP
    if "test" in clip.parts:
        raise typer.BadParameter("the test split stays locked", param_hint="--clip-dir")
    loadavg = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    loadavg = ", ".join(loadavg.strip(" {}\n").split())
    paths = spread_frames(clip, frames + 1)
    if len(paths) < 2:
        raise typer.BadParameter("the clip needs at least two frames")
    rgb = read_frames(paths)
    frame_size = (rgb.shape[2], rgb.shape[1])
    x = preprocess(rgb)
    inputs = [x[i : i + 1] for i in range(len(x))]
    weights = data_dir() / "weights" / "pnlcalib"

    def bench_torch(device: str, sync: Callable[[], None]) -> tuple[Result, Heatmaps, str]:
        start = time.perf_counter()
        nets = PnLCalibNetworks(weights / "SV_kp", weights / "SV_lines", device=device)
        load_s = time.perf_counter() - start
        warm, times, heatmaps = timed_runs(nets.heatmaps_from_input, inputs, sync)
        typer.echo(f"PyTorch {device}: {statistics.median(times):.0f} ms per frame")
        return Result(f"PyTorch {device.upper()}", load_s, warm, times), heatmaps, nets.version

    cpu, reference, model_version = bench_torch("cpu", lambda: None)
    results = [cpu]
    pre_ms, post_ms = [], []
    for i, path in enumerate(paths[1:]):
        start = time.perf_counter()
        preprocess(read_frames([path]))
        pre_ms.append(1e3 * (time.perf_counter() - start))
        start = time.perf_counter()
        decode_detections(reference[0][i : i + 1], reference[1][i : i + 1], frame_size)
        post_ms.append(1e3 * (time.perf_counter() - start))
    gc.collect()

    if torch.backends.mps.is_available():
        mps, heatmaps, _ = bench_torch("mps", torch.mps.synchronize)
        results.append(mps.against(parity(reference, heatmaps, frame_size)))
        gc.collect()
        torch.mps.empty_cache()

    coreml_note = "The CoreML execution provider was not available"
    ort_version = "not used"
    if all((onnx_dir() / f).is_file() for f in MODEL_FILES.values()):
        import onnxruntime as ort

        ort_version = version("onnxruntime")
        arrays = [b.numpy() for b in inputs]
        backends: list[tuple[str, list[Any]]] = [("ONNX Runtime CPU", ["CPUExecutionProvider"])]
        if "CoreMLExecutionProvider" in ort.get_available_providers():
            coreml = [("CoreMLExecutionProvider", COREML_OPTIONS), "CPUExecutionProvider"]
            backends.append(("ONNX Runtime CoreML", coreml))
        for name, providers in backends:
            start = time.perf_counter()
            if name.endswith("CoreML"):
                with tempfile.TemporaryDirectory() as tmp:
                    log = Path(tmp) / "ort.log"
                    ort.set_default_logger_severity(1)  # logs how each graph is split between CoreML and CPU
                    with stderr_to(log):
                        session = OnnxPnLCalib(providers=providers)
                    ort.set_default_logger_severity(2)
                    coreml_note = f"CoreML options {COREML_OPTIONS}: " + coreml_partition(
                        log.read_text(errors="replace")
                    )
            else:
                session = OnnxPnLCalib(providers=providers)
            load_s = time.perf_counter() - start
            warm, times, heatmaps = timed_runs(session.heatmaps_from_input, arrays, lambda: None)
            results.append(Result(name, load_s, warm, times).against(parity(reference, heatmaps, frame_size)))
            typer.echo(
                f"{name} ({', '.join(session.providers)}): {statistics.median(times):.0f} ms per frame"
            )
            del session
            gc.collect()
    else:
        typer.echo("no exported ONNX models: run scripts/export_onnx.py for the ONNX Runtime rows", err=True)

    text = report(
        results,
        {
            "n": len(paths) - 1,
            "commit": git_commit(),
            "date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "model": model_version,
            "machine": machine(),
            "torch": version("torch"),
            "onnxruntime": ort_version,
            "loadavg": loadavg,
            "clip": clip.name,
            "torch_threads": torch.get_num_threads(),
            "pre_ms": statistics.median(pre_ms),
            "post_ms": statistics.median(post_ms),
            "coreml": coreml_note,
        },
    )
    if out is None:
        typer.echo(text)
    else:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"→ {out}")


if __name__ == "__main__":
    app()
