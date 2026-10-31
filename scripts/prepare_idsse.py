"""Download the IDSSE tracking data and parse each match once into Parquet (ticket T0.8).

    uv run python scripts/prepare_idsse.py                 # all 7 matches
    uv run python scripts/prepare_idsse.py J03WPY J03WOH   # some matches
    uv run python scripts/prepare_idsse.py --force         # parse again

Raw XML goes to ``PREDFOOT_DATA_DIR/idsse/raw/`` and tables to ``PREDFOOT_DATA_DIR/idsse/`` (see
:mod:`predfoot.io.idsse`). IDSSE: Bassek, Rein, Weber & Memmert (2025), CC BY 4.0,
doi:10.6084/m9.figshare.28196177.
"""

from __future__ import annotations

import argparse
import time

import polars as pl

from predfoot.io import idsse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("matches", nargs="*", default=sorted(idsse.IDSSE_MATCHES), help="IDSSE match ids")
    parser.add_argument("--force", action="store_true", help="parse again even if the tables exist")
    args = parser.parse_args()

    for match_id in args.matches:
        t0 = time.perf_counter()
        paths = idsse.prepare_match(match_id, force=args.force)
        players = pl.read_parquet(paths.players)
        summary = players.group_by("period").agg(
            pl.len().alias("rows"), pl.col("player_id").n_unique().alias("players")
        )
        periods = ", ".join(
            f"P{r['period']}: {r['players']} players, {r['rows']} rows"
            for r in summary.sort("period").iter_rows(named=True)
        )
        print(f"{match_id}: {periods} ({time.perf_counter() - t0:.1f} s) -> {paths.players}")


if __name__ == "__main__":
    main()
