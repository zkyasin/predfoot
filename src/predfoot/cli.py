"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from predfoot import __version__

app = typer.Typer(
    help="predfoot: calibrated position uncertainty for broadcast football tracking.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """predfoot command-line tools."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def run(
    clip: Annotated[
        Path,
        typer.Argument(
            metavar="INPUT", exists=True, dir_okay=False, readable=True, help="Clip JSON (predfoot-clip/1)."
        ),
    ],
    out: Annotated[Path, typer.Option(file_okay=False, help="Output folder (created if needed).")],
    fps: Annotated[float, typer.Option(help="Export frame rate.")] = 10.0,
    block_size: Annotated[int, typer.Option(help="Export frames per block: the resume granularity.")] = 100,
) -> None:
    """Run the pipeline on a clip: tracking.jsonl, match.json, meta.json, metrics.json, minimap.png.

    Re-running after a crash reuses the finished blocks and gives byte-identical outputs.
    """
    from pydantic import ValidationError

    from predfoot.pipeline.run import RunConfig, run_clip

    if fps <= 0:
        raise typer.BadParameter("must be positive", param_hint="--fps")
    if block_size < 1:
        raise typer.BadParameter("must be at least 1", param_hint="--block-size")
    try:
        result = run_clip(clip, out, RunConfig(export_fps=fps, block_size=block_size))
    except ValidationError as err:
        typer.echo(f"invalid clip file {clip}:\n{err}", err=True)
        raise typer.Exit(code=2) from err
    typer.echo(
        f"{result.n_export} frames at {fps:g} fps from {result.n_native} native frames, "
        f"{result.n_abstained} abstained; blocks computed {result.blocks_computed}, "
        f"reused {result.blocks_reused} → {out}"
    )


@app.command()
def validate(
    tracking: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="JSONL tracking file to check.")
    ],
    match: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="match.json: every player_id must be listed in it; also gives the pitch width.",
        ),
    ] = None,
    fps: Annotated[float, typer.Option(help="Frame rate the timestamps must follow.")] = 10.0,
    pitch_width: Annotated[
        float | None,
        typer.Option(help="Pitch width in metres for corner clipping (default: match.json, else 68)."),
    ] = None,
    skillcorner: Annotated[
        bool,
        typer.Option(
            "--skillcorner", help="SkillCorner open-data file: predfoot's extra fields are optional."
        ),
    ] = False,
    max_issues: Annotated[int, typer.Option(help="Report at most this many problems.")] = 20,
) -> None:
    """Check a tracking file against the frame schema and the semantic rules."""
    from predfoot.io.validate import validate_tracking

    match_obj = json.loads(match.read_text(encoding="utf-8")) if match else None
    report = validate_tracking(
        tracking,
        match=match_obj,
        pitch_width=pitch_width,
        fps=fps,
        extensions=not skillcorner,
        max_issues=max_issues,
    )
    if report.ok:
        typer.echo(f"OK: {report.n_records} records in {tracking}")
        return
    for issue in report.issues:
        typer.echo(str(issue), err=True)
    more = "+" if report.truncated else ""
    typer.echo(f"FAILED: {len(report.issues)}{more} problem(s) in {report.n_records} records", err=True)
    raise typer.Exit(code=1)


@app.command()
def serve(
    host: Annotated[
        str, typer.Option(help="Interface to listen on (0.0.0.0 for every interface).")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="TCP port.")] = 8000,
    device: Annotated[
        str, typer.Option(help="Torch device of /v1/calibrate: auto, cpu, mps or cuda.")
    ] = "auto",
    preload: Annotated[
        bool,
        typer.Option(help="Load the keypoint networks at start-up rather than on the first calibration."),
    ] = False,
) -> None:
    """Serve the HTTP API: /healthz, /v1/localize, /v1/calibrate and /metrics (OpenAPI docs at /docs).

    Needs the 'serve' extra; /v1/calibrate also needs the 'vision' extra and the PnLCalib weights.
    """
    try:
        import uvicorn

        from predfoot.serve.app import create_app
    except ImportError as err:
        typer.echo(f"predfoot serve needs the 'serve' extra (uv sync --extra serve): {err}", err=True)
        raise typer.Exit(code=2) from err
    uvicorn.run(create_app(device=device, preload=preload), host=host, port=port)
