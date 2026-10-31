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
