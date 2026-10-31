"""Tracking data as per-period arrays, and per-player KPIs under simulated position errors.

Shared by the KPI error-budget scripts (``scripts/kpi_sensitivity.py``, ``scripts/kpi_spectral.py``,
``scripts/kpi_card.py``): errors are added to clean positions at the native rate, then the standard pipeline
of :mod:`predfoot.kpi.speed` and the KPIs of :mod:`predfoot.kpi.metrics` run on each player.
"""

from __future__ import annotations

import zlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from predfoot.kpi.config import KpiConfig
from predfoot.kpi.metrics import compute_kpis
from predfoot.kpi.speed import FloatArray, Kinematics, drop_glitches, kmh_to_ms, pipeline


@dataclass(frozen=True)
class PeriodTracks:
    """Positions of every player of one period on the period's frame grid, NaN where a player is absent."""

    period: int
    frame0: int  # native frame number of row 0
    player_ids: tuple[str, ...]
    xy: FloatArray  # (n_frames, n_players, 2), metres

    def span(self, i: int) -> tuple[int, int] | None:
        """First and one-past-last row where player ``i`` is present, or None if never."""
        valid = np.flatnonzero(np.isfinite(self.xy[:, i, 0]))
        return (int(valid[0]), int(valid[-1]) + 1) if valid.size else None


def period_tracks(players: pl.DataFrame, cfg: KpiConfig) -> tuple[list[PeriodTracks], int, int]:
    """One :class:`PeriodTracks` per period of a ``frame, period, player_id, x, y`` table.

    Tracking glitches of the source feed are cut out first (:func:`~predfoot.kpi.speed.drop_glitches`). Also
    returns the number of samples dropped and the number of valid samples before cleaning.
    """
    out = []
    dropped = total = 0
    max_speed = float(kmh_to_ms(cfg.glitch_speed_kmh))
    for period in sorted(players["period"].unique().to_list()):
        df = players.filter(pl.col("period") == period)
        ids = sorted(df["player_id"].unique().to_list())
        frames = df["frame"].to_numpy()
        f0 = int(frames.min())
        n = int(frames.max()) - f0 + 1
        cube = np.full((n, len(ids), 2), np.nan)
        col = df["player_id"].replace_strict({pid: i for i, pid in enumerate(ids)}).to_numpy()
        row = frames - f0
        cube[row, col, 0] = df["x"].to_numpy()
        cube[row, col, 1] = df["y"].to_numpy()
        for i in range(len(ids)):
            cube[:, i], n_dropped = drop_glitches(cube[:, i], cfg.native_hz, max_speed)
            dropped += n_dropped
        total += df.height
        out.append(PeriodTracks(int(period), f0, tuple(ids), cube))
    return out, dropped, total


def seeded_rng(seed: int, *keys: str | int) -> np.random.Generator:
    """Generator seeded from a study seed and stable keys (match, error model, period...)."""
    entropy = [seed] + [k if isinstance(k, int) else zlib.crc32(k.encode()) for k in keys]
    return np.random.default_rng(entropy)


def player_kpis(
    periods: Sequence[PeriodTracks],
    cfg: KpiConfig,
    *,
    resample_to: int | None,
    errors: Sequence[FloatArray] | None = None,
    players: Collection[str] | None = None,
) -> dict[str, dict[str, float]]:
    """KPIs of every player (or only ``players``), each period's errors added to its positions first.

    ``errors[p]`` has the shape of ``periods[p].xy``. Each player's track runs from their first to their last
    valid sample of the period; gaps inside stay NaN and are never bridged.
    """
    parts: dict[str, list[Kinematics]] = {}
    for p, tracks in enumerate(periods):
        xy_all = tracks.xy if errors is None else tracks.xy + errors[p]
        for i, pid in enumerate(tracks.player_ids):
            if players is not None and pid not in players:
                continue
            span = tracks.span(i)
            if span is None:
                continue
            parts.setdefault(pid, []).append(
                pipeline(xy_all[span[0] : span[1], i], cfg, resample_to=resample_to)
            )
    return {pid: compute_kpis(kins, cfg) for pid, kins in parts.items()}
