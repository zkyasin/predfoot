"""Check every file listed in PREDFOOT_DATA_DIR/MANIFEST.sha256: present, same size, same SHA-256.

Exits with status 1 on any mismatch.

    uv run python scripts/verify_data.py
    uv run python scripts/verify_data.py --prefix sn-gsr/valid/
"""

from __future__ import annotations

from typing import Annotated

import typer

from predfoot.io.manifest import verify_manifest
from predfoot.paths import data_dir

app = typer.Typer(add_completion=False)


@app.command()
def main(
    prefix: Annotated[str, typer.Option(help="Only check paths starting with this prefix.")] = "",
    jobs: Annotated[int, typer.Option(help="Files hashed in parallel.")] = 8,
) -> None:
    """Verify the manifest of PREDFOOT_DATA_DIR."""
    root = data_dir()
    try:
        checked, problems = verify_manifest(root, jobs=jobs, prefix=prefix)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(1) from exc
    for problem in problems:
        typer.echo(f"{problem.kind.upper():<8} {problem.path}: {problem.detail}", err=True)
    typer.echo(f"{root}: {checked} files checked, {len(problems)} problem(s)")
    raise typer.Exit(1 if problems or not checked else 0)


if __name__ == "__main__":
    app()
