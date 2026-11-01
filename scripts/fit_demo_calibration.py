"""Rebuild ``src/predfoot/demo/dev_calibration.json``, the uncertainty calibration of the SoccerTrack demo,
from two dev runs of ``scripts/eval_dev.py`` on SN-GSR, one with ``--smooth ptz`` and one with
``--smooth none``.

    uv run python scripts/fit_demo_calibration.py \\
        --ptz-run runs/<date>_<sha>_ptz --none-run runs/<date>_<sha>_none

The file holds aggregate numbers only (bin edges, quantiles, bin counts and four noise parameters); see
:mod:`predfoot.demo.calibration`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from predfoot.demo import calibration as dc

app = typer.Typer(add_completion=False)


@app.command()
def main(
    ptz_run: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Dev run with --smooth ptz (smoothing.json).")
    ],
    none_run: Annotated[Path, typer.Option(exists=True, file_okay=False, help="Dev run with --smooth none.")],
    out: Annotated[Path, typer.Option(help="JSON file to write.")] = dc.DEFAULT_PATH,
) -> None:
    cameras = {"ptz": dc.from_dev_run(ptz_run, "ptz"), "none": dc.from_dev_run(none_run, "none")}
    out.write_text(dc.to_json(cameras), encoding="utf-8")
    for name, cal in cameras.items():
        typer.echo(f"{name}: {cal.provenance['player_frames']} player-frames")
        for src, m in cal.models.items():
            typer.echo(f"  {src}: {len(m.quantiles)} of {len(m.counts)} bins fitted, pooled q {m.pooled:.1f}")
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
