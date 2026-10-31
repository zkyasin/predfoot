"""Fit the Mondrian conformal model that ``predfoot serve --conformal`` loads, from an eval_dev run.

Usage:
    uv run python scripts/fit_conformal.py runs/<eval_dev run>/players.parquet --out conformal.json

Scores, features and bins are those of scripts/eval_dev.py and scripts/run_real_clip.py: squared Mahalanobis
distances of the errors against the labels under Σp (from Σθ only, exact foot pixels), clips as groups, bins
of metres per pixel at the feet (3) × RANSAC inliers (2), α = 0.1. ``--exclude-game`` leaves a game out, to
serve frames of that game with quantiles fitted on the other games only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

from predfoot.uncertainty.conformal import MondrianConformal
from predfoot.uncertainty.delta import mahalanobis2

app = typer.Typer(add_completion=False)
BINS = {"mpp": 3, "n_inliers": 2}


@app.command()
def main(
    players: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="players.parquet of an eval_dev run.")
    ],
    out: Annotated[Path, typer.Option(help="JSON file to write.")],
    exclude_game: Annotated[int | None, typer.Option(help="Game left out of the fit.")] = None,
) -> None:
    d = pl.read_parquet(players).filter(pl.col("ex_lab").is_not_nan())
    if exclude_game is not None:
        d = d.filter(pl.col("game") != exclude_game)
    if d.is_empty():
        raise typer.BadParameter("no labelled player-frames to fit on", param_hint="PLAYERS")
    e = d.select("ex_lab", "ey_lab").to_numpy()
    cov = np.stack([d["c00"], d["c01"], d["c01"], d["c11"]], axis=1).reshape(-1, 2, 2)
    feats = {"mpp": d["mpp"].to_numpy(), "n_inliers": d["n_inliers"].to_numpy().astype(float)}
    model = MondrianConformal.fit(mahalanobis2(e, cov), d["clip"].to_numpy(), feats, BINS)
    text = model.to_json()
    out.write_text(text, encoding="utf-8")
    tag = hashlib.sha256(text.encode()).hexdigest()[:12]
    typer.echo(
        f"mondrian@{tag}: {len(model.quantiles)} of {len(model.counts)} bins fitted, pooled q̂ "
        f"{model.pooled:.1f}, from {d.height} player-frames of {d['clip'].n_unique()} clips → {out}"
    )


if __name__ == "__main__":
    app()
