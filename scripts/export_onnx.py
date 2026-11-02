"""Export the PnLCalib keypoint and line networks to ONNX and check parity with PyTorch.

Usage:
    uv sync --extra vision --extra export
    uv run python scripts/export_onnx.py --frames 3

Writes PREDFOOT_DATA_DIR/models/onnx/pnlcalib-sv-{kp,lines}.onnx (derived from the CC BY 4.0 weights, so
never in git) and export.json (versions, opset, hashes, parity). The models are exported to a staging folder
and replace the installed ones only once they pass the parity check; a failed export leaves the installed
models as they were and writes its parity record to export-failed.json. Parity: PyTorch and ONNX Runtime run
on the CPU, batch 1, on frames of a dev clip; the export fails (exit code 1) when the heatmaps differ by more
than 1e-4, when a detection is found by one runtime only or none is found, or when a decoded keypoint or line
end moves by more than 0.5 px between the two.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import typer

from predfoot.io.files import sha256_file, write_json
from predfoot.keypoints.onnx_models import (
    INPUT_NAME,
    MODEL_FILES,
    OPSET,
    OUTPUT_NAME,
    OnnxPnLCalib,
    export_network,
    onnx_dir,
    parity,
    per_frame,
    read_frames,
    spread_frames,
)
from predfoot.keypoints.pnlcalib import INPUT_SIZE, PnLCalibNetworks, preprocess
from predfoot.paths import REPO_ROOT, data_dir

app = typer.Typer(add_completion=False)
DEFAULT_CLIP = Path("sn-gsr") / "valid" / "SNGS-056"


def git_commit() -> str:
    """Short commit of the code that ran, with "+dirty" when the working tree has uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    return f"{sha}+dirty" if git("status", "--porcelain") else sha


@app.command()
def main(
    frames: int = typer.Option(3, min=1, help="Dev frames for the parity check."),
    clip_dir: Path | None = typer.Option(None, help="Clip folder with img1/ (default: dev clip SNGS-056)."),  # noqa: B008
    out_dir: Path | None = typer.Option(None, help="Output folder (default: PREDFOOT_DATA_DIR/models/onnx)."),  # noqa: B008
    opset: int = typer.Option(OPSET, min=17, help="ONNX opset."),
) -> None:
    clip = clip_dir or data_dir() / DEFAULT_CLIP
    if "test" in clip.parts:
        raise typer.BadParameter("the test split stays locked", param_hint="--clip-dir")
    out = out_dir or onnx_dir()
    out.mkdir(parents=True, exist_ok=True)
    staging = out / ".staging"  # same file system as out: each file then moves into place at once
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        export_and_check(clip, frames, opset, staging, out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def export_and_check(clip: Path, frames: int, opset: int, staging: Path, out: Path) -> None:
    """Export both networks into ``staging``, compare them with PyTorch, and install them in ``out`` only if
    they pass."""
    weights = data_dir() / "weights" / "pnlcalib"
    nets = PnLCalibNetworks(weights / "SV_kp", weights / "SV_lines", device="cpu")
    for name, net in (("kp", nets.kp), ("lines", nets.lines)):
        start = time.perf_counter()
        export_network(net, staging / MODEL_FILES[name], opset)  # checks the opset of the file
        size_mb = (staging / MODEL_FILES[name]).stat().st_size / 1e6
        typer.echo(f"{MODEL_FILES[name]}: {size_mb:.0f} MB, exported in {time.perf_counter() - start:.0f} s")

    paths = spread_frames(clip, frames)
    rgb = read_frames(paths)
    x = preprocess(rgb)
    reference = per_frame(nets.heatmaps_from_input, x)
    other = per_frame(OnnxPnLCalib(staging).heatmaps_from_input, x.numpy())
    result = parity(reference, other, (rgb.shape[2], rgb.shape[1]))
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model_version": nets.version,
        "opset": opset,
        "input": {"name": INPUT_NAME, "shape": [1, 3, INPUT_SIZE[1], INPUT_SIZE[0]], "dtype": "float32"},
        "output": {"name": OUTPUT_NAME, "note": "raw heatmaps at stride 2, background channel last"},
        "files": {f: sha256_file(staging / f) for f in MODEL_FILES.values()},
        "versions": {p: version(p) for p in ("torch", "onnx", "onnxruntime", "onnxscript")},
        "parity": result.as_dict() | {"frames": [f"{p.parent.parent.name}/{p.name}" for p in paths]},
    }
    typer.echo(
        f"Parity on {result.n_frames} frames ({result.n_points} points, PyTorch CPU vs ONNX Runtime CPU): "
        f"max |Δheatmap| {result.max_abs_heatmap:.2e}, keypoints {result.max_keypoint_px:.4f} px, "
        f"line ends {result.max_line_end_px:.4f} px, detections in one runtime only {result.n_flips}"
    )
    if not result.ok:
        write_json(out / "export-failed.json", record)
        typer.echo(f"FAILED: {'; '.join(result.problems())}. The installed models are unchanged.", err=True)
        raise typer.Exit(code=1)
    for f in MODEL_FILES.values():
        (staging / f).replace(out / f)
    write_json(out / "export.json", record)
    (out / "export-failed.json").unlink(missing_ok=True)
    typer.echo(f"OK → {out}")


if __name__ == "__main__":
    app()
