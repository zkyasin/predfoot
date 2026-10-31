"""Build the SN-GSR Parquet index (clips, persons, lines) from the labels under PREDFOOT_DATA_DIR.

    uv run python scripts/build_index.py

The locked test split's per-frame tables go to index/test/ (see predfoot.io.sngsr.build_index).
"""

from __future__ import annotations

import time
from typing import Annotated

import polars as pl
import typer

from predfoot.io.sngsr import build_index
from predfoot.paths import data_dir

app = typer.Typer(add_completion=False)


@app.command()
def main(workers: Annotated[int, typer.Option(help="Label files parsed in parallel.")] = 6) -> None:
    """Parse every downloaded Labels-GameState.json into index/*.parquet."""
    start = time.monotonic()
    written = build_index(data_dir(), workers=workers)
    for name, path in written.items():
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        typer.echo(f"{name:<14} {rows:>10,} rows  {path.stat().st_size / 1e6:7.1f} MB  {path}")
    typer.echo(f"done in {time.monotonic() - start:.0f} s")


if __name__ == "__main__":
    app()
