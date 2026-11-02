"""Position-error percentiles of both cameras, for the accuracy chart of the showcase page.

Reads the player tables of the preregistered test run (``runs/test_once/lock-<id>/``, git-ignored) and prints
JSON: for the per-frame and the PTZ-smoothed camera, the error in metres at the percentiles 0.5, 1.5, ...,
99.5, on the same player-frames and against the same reference (SN-GSR raw labels) as the position-error
table of reports/test_run.md. It also prints p50, p90 and p99 so they can be checked against that table.
These are aggregates of the frozen run: they change none of its results.

Usage:
    uv run python scripts/showcase_percentiles.py runs/test_once/lock-191bb06dbb75aba8
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import typer

from predfoot.eval.frozen import KEYS

#: One dot per percent of the player-frames, each at the middle of its percent.
LEVELS = (np.arange(100) + 0.5) / 100


def errors(run_dir: Path) -> dict[str, np.ndarray]:
    """Error distances (m) of both cameras on the player-frames they share, as in frozen.error_endpoint."""
    ptz = pl.read_parquet(run_dir / "players_ptz.parquet").filter(~pl.col("filled"))
    perframe = pl.read_parquet(run_dir / "players_perframe.parquet")
    cols = ["ex_lab", "ey_lab"]
    joined = ptz.select(*KEYS, *cols).join(perframe.select(*KEYS, *cols), on=KEYS, suffix="_pf")
    out = {
        "perframe": np.hypot(joined["ex_lab_pf"].to_numpy(), joined["ey_lab_pf"].to_numpy()),
        "ptz": np.hypot(joined["ex_lab"].to_numpy(), joined["ey_lab"].to_numpy()),
    }
    return {name: e[np.isfinite(e)] for name, e in out.items()}


def main(run_dir: Path) -> None:
    """Print the percentiles as JSON."""
    errs = errors(run_dir)
    report = {
        "n": {name: int(e.size) for name, e in errs.items()},
        "check_p50_p90_p99": {
            name: [round(float(np.quantile(e, p)), 2) for p in (0.5, 0.9, 0.99)] for name, e in errs.items()
        },
        "percentiles_m": {
            name: [round(float(v), 3) for v in np.quantile(e, LEVELS)] for name, e in errs.items()
        },
    }
    typer.echo(json.dumps(report))


if __name__ == "__main__":
    typer.run(main)
