"""Command-line interface."""

from __future__ import annotations

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
