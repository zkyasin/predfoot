"""Dev evaluation (tickets T1.5 and T2.3): calibrate every cached dev frame, predict positions with their
uncertainty, and measure leave-one-game-out (LOGO) conformal coverage against two references.

References:
- ``labels``: the SN-GSR raw ground truth (``x_raw``, ``y_raw``), produced by the annotators' camera;
- ``oracle``: positions from a homography fitted to the annotated lines of the same frame, kept only where
  it fits those lines within 3 px (see reports/spike_s0.md for why both are needed).

For each held-out game, conformal quantiles are fitted on the five other dev games only. Two regions are
compared at the same nominal level: our ellipses {e : eᵀ Σp⁻¹ e ≤ q̂} (pooled and Mondrian) and a constant
circle whose radius is the conformal quantile of the error norms. Bootstrap intervals resample clips.

Usage:
    uv run python scripts/eval_dev.py --out reports/m2_logo_dev.md
Needs the index (scripts/build_index.py) and the detection cache (scripts/cache_keypoints.py).
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import typer

from predfoot.calib.lines import line_oracle, line_reprojection_px
from predfoot.calib.observations import observations_from_detections
from predfoot.calib.solver import SolverConfig, calibrate
from predfoot.geometry import homography as hg
from predfoot.geometry.pitch import pnlcalib_keypoints
from predfoot.health.features import calibration_features
from predfoot.keypoints.cache import cache_path, cached_detections
from predfoot.paths import data_dir, index_dir
from predfoot.uncertainty.conformal import MondrianConformal, grouped_quantile
from predfoot.uncertainty.delta import circumscribed_radius, mahalanobis2, position_covariance

app = typer.Typer(add_completion=False)
KP = pnlcalib_keypoints()
ORACLE_MAX_PX = 3.0
ALPHA = 0.1


def predict_clip(
    clip_id: str, split: str, game: int, persons: pl.DataFrame, lines: pl.DataFrame
) -> tuple[list[dict], list[dict]]:
    """Per-frame rows (health features and failure labels) and per-player rows (errors, covariances)."""
    clip_dir = data_dir() / "sn-gsr" / split / clip_id
    n = len(list((clip_dir / "img1").glob("*.jpg")))
    if not cache_path(clip_id, n).exists():
        return [], []
    dets, _, _ = cached_detections(clip_dir, clip_id)
    cfg = SolverConfig()
    out: list[dict] = []
    frame_rows: list[dict] = []
    prev_h = None
    p_by_frame = persons.partition_by("frame", as_dict=True)
    l_by_frame = lines.partition_by("frame", as_dict=True)
    for frame, det in sorted(dets.items()):
        p = p_by_frame.get((frame,))
        if p is None:
            continue
        p = p.filter(~pl.col("truncated") & pl.col("x_raw").is_not_null())
        frame_lines = {}
        lf = l_by_frame.get((frame,))
        if lf is not None:
            for (cls,), g in lf.filter(pl.col("is_planar")).partition_by("line_class", as_dict=True).items():
                if g.height >= 2:
                    frame_lines[cls] = g.sort("point_idx").select("u", "v").to_numpy()
        h_or = line_oracle(frame_lines)
        oracle_px = line_reprojection_px(h_or, frame_lines)
        trusted = h_or is not None and oracle_px <= ORACLE_MAX_PX
        xy, uv, w, obs = observations_from_detections(det)
        rng = np.random.default_rng(frame * 7919 + int(clip_id.split("-")[1]))
        res, _ = calibrate(xy, uv, rng, weights=w, config=cfg, lines=obs)
        evidence = np.vstack([uv.reshape(-1, 2), *[o.uv for o in obs]]) if obs else uv
        n_kp = int(sum(KP[int(k)][2] for k in det["kp_ids"]))
        feats = calibration_features(res, evidence, n_kp, len(obs), det["kp_conf"], prev_h)
        prev_h = res.H if res.ok else None
        line_px = line_reprojection_px(res.H, frame_lines) if res.ok else float("nan")
        feet = p.select("foot_u", "foot_v").to_numpy()
        gt = p.select("x_raw", "y_raw").to_numpy()
        tracks = p["track_id"].to_numpy()
        base = {
            "clip": clip_id,
            "game": game,
            "frame": frame,
            "ok": res.ok,
            "n_corr": len(xy) + len(obs),
            "n_inliers": res.n_inliers,
        }
        frame_row = {**base, **feats, "line_px": line_px, "oracle_px": oracle_px, "trusted": trusted}
        if not res.ok or p.height == 0:
            frame_rows.append(frame_row | {"err_p90_lab": np.nan, "err_p90_or": np.nan})
            continue
        pred, cov, valid = position_covariance(res.H, res.cov_theta, feet)
        ref_or = hg.backproject(h_or, feet)[0] if trusted else np.full_like(gt, np.nan)
        mpp = np.sqrt(np.abs(np.linalg.det(hg.jacobian_backproject_uv(res.H, feet))))
        e_lab = np.linalg.norm(pred - gt, axis=1)[valid]
        e_or = np.linalg.norm(pred - ref_or, axis=1)[valid]
        frame_rows.append(
            frame_row
            | {
                "err_p90_lab": float(np.quantile(e_lab, 0.9)) if len(e_lab) else np.nan,
                "err_p90_or": float(np.quantile(e_or, 0.9)) if trusted and len(e_or) else np.nan,
            }
        )
        for i in np.flatnonzero(valid):
            out.append(
                {
                    **base,
                    "track_id": int(tracks[i]),
                    "ex_lab": pred[i, 0] - gt[i, 0],
                    "ey_lab": pred[i, 1] - gt[i, 1],
                    "ex_or": pred[i, 0] - ref_or[i, 0],
                    "ey_or": pred[i, 1] - ref_or[i, 1],
                    "c00": cov[i, 0, 0],
                    "c01": cov[i, 0, 1],
                    "c11": cov[i, 1, 1],
                    "mpp": mpp[i],
                    "rms_px": res.rms_px,
                    "dof": res.dof,
                }
            )
    return frame_rows, out


def logo(df: pl.DataFrame, ref: str, rng: np.random.Generator, n_boot: int = 300) -> list[dict]:
    """Leave-one-game-out coverage and size for our ellipses and the constant-radius baseline."""
    d = df.filter(pl.col(f"ex_{ref}").is_not_nan())
    e = d.select(f"ex_{ref}", f"ey_{ref}").to_numpy()
    cov = np.stack([d["c00"], d["c01"], d["c01"], d["c11"]], axis=1).reshape(-1, 2, 2)
    score = mahalanobis2(e, cov)
    norm = np.linalg.norm(e, axis=1)
    groups = d["clip"].to_numpy()
    games = d["game"].to_numpy()
    feats = {"mpp": d["mpp"].to_numpy(), "n_inliers": d["n_inliers"].to_numpy().astype(float)}
    rows = []
    for g in np.unique(games):
        cal, tst = games != g, games == g
        q_pool = grouped_quantile(score[cal], groups[cal], ALPHA)
        mond = MondrianConformal.fit(
            score[cal],
            groups[cal],
            {k: v[cal] for k, v in feats.items()},
            {"mpp": 3, "n_inliers": 2},
            alpha=ALPHA,
        )
        q_mond = mond.quantile_for({k: v[tst] for k, v in feats.items()})
        r_const = grouped_quantile(norm[cal], groups[cal], ALPHA)
        hit_pool = score[tst] <= q_pool
        hit_mond = score[tst] <= q_mond
        hit_const = norm[tst] <= r_const
        rad_pool = circumscribed_radius(cov[tst], q_pool)
        rad_mond = np.sqrt(q_mond * np.linalg.eigvalsh(cov[tst])[:, -1])
        area_mond = np.pi * q_mond * np.sqrt(np.clip(np.linalg.det(cov[tst]), 0, None))
        clips = np.unique(groups[tst])

        def boot(
            hit: np.ndarray, clip_ids: np.ndarray = groups[tst], uniq: np.ndarray = clips
        ) -> tuple[float, float]:
            idx = {c: np.flatnonzero(clip_ids == c) for c in uniq}
            stats = []
            for _ in range(n_boot):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                stats.append(np.mean(np.concatenate([hit[idx[c]] for c in pick])))
            lo, hi = np.quantile(stats, [0.025, 0.975])
            return float(lo), float(hi)

        rows.append(
            {
                "game": int(g),
                "clips": len(clips),
                "n": int(tst.sum()),
                "cov_pool": float(hit_pool.mean()),
                "cov_mond": float(hit_mond.mean()),
                "cov_mond_ci": boot(hit_mond),
                "cov_const": float(hit_const.mean()),
                "r_pool": float(np.median(rad_pool)),
                "r_mond": float(np.median(rad_mond)),
                "a_mond": float(np.median(area_mond)),
                "r_const": float(r_const),
                "a_const": float(np.pi * r_const**2),
            }
        )
    return rows


@app.command()
def main(out: Path = Path("reports/m2_logo_dev.md"), max_clips: int = 0) -> None:
    clips = pl.read_parquet(index_dir() / "clips.parquet").filter(pl.col("split").is_in(["train", "valid"]))
    persons = pl.read_parquet(index_dir() / "persons.parquet")
    lines = pl.read_parquet(index_dir() / "lines.parquet")
    rows: list[dict] = []
    frame_rows: list[dict] = []
    for i, c in enumerate(clips.sort("clip_id").iter_rows(named=True)):
        if max_clips and i >= max_clips:
            break
        cp = persons.filter(pl.col("clip_id") == c["clip_id"])
        cl = lines.filter(pl.col("clip_id") == c["clip_id"])
        f_rows, p_rows = predict_clip(c["clip_id"], c["split"], int(c["game_id"]), cp, cl)
        frame_rows.extend(f_rows)
        rows.extend(p_rows)
    if not rows:
        typer.echo("No cached clips found: run scripts/cache_keypoints.py first.")
        raise typer.Exit(1)
    players = pl.DataFrame(rows, infer_schema_length=None)
    frames = pl.DataFrame(frame_rows, infer_schema_length=None)
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    run_dir = Path("runs") / f"{datetime.now(UTC):%Y%m%d-%H%M}_{sha}"
    run_dir.mkdir(parents=True, exist_ok=True)
    players.write_parquet(run_dir / "players.parquet")
    frames.write_parquet(run_dir / "frames.parquet")

    n_clips, n_games = players["clip"].n_unique(), players["game"].n_unique()
    col = lambda name: players[name].cast(pl.Float64).fill_null(np.nan).to_numpy()  # noqa: E731
    err_lab = np.hypot(col("ex_lab"), col("ey_lab"))
    err_or = np.hypot(col("ex_or"), col("ey_or"))
    rng = np.random.default_rng(0)

    def q3(x: np.ndarray) -> str:
        v = np.asarray(x, dtype=np.float64)
        return " | ".join(f"{np.nanquantile(v, p):.2f}" for p in (0.5, 0.9, 0.99))

    line_px = frames["line_px"].to_numpy().astype(float)
    md = [
        "# Dev evaluation: calibration, positions and leave-one-game-out coverage",
        "",
        f"`uv run python scripts/eval_dev.py` · commit `{sha}` · per-player table in `{run_dir}`"
        " (not in git).",
        "",
        f"Dev clips with cached detections: {n_clips} from {n_games} games. Frames: {frames.height}"
        f" (one in five), calibrated {frames['ok'].mean():.1%}. Player-frames: {players.height} (ground-truth"
        " boxes; truncated boxes and frames without a labelled camera excluded).",
        "",
        "## Position error (metres)",
        "",
        "| Reference | p50 | p90 | p99 |",
        "|---|---|---|---|",
        f"| labels (`x_raw`, `y_raw`) | {q3(err_lab)} |",
        f"| trusted line oracle ({np.isfinite(err_or).mean():.0%} of player-frames) | {q3(err_or)} |",
        "",
        "Calibration (mean distance of annotated line points to the projected template):"
        f" p50 {np.nanquantile(line_px, 0.5):.1f} px, p90 {np.nanquantile(line_px, 0.9):.1f} px.",
    ]
    header = (
        "| Held-out game | Clips | Player-frames | Ellipses, pooled | Ellipses, Mondrian [95 % CI] |"
        " Constant circle | Median radius pooled / Mondrian / constant (m) |"
        " Median area Mondrian / constant |"
    )
    for ref, title in (("lab", "labels"), ("or", "trusted line oracle")):
        res = logo(players, ref, rng)
        md += [
            "",
            f"## LOGO coverage at 90 %, reference: {title}",
            "",
            "Each row holds one dev game out; its quantiles come from the five others. Radius: circumscribed",
            "circle of the ellipse (median over player-frames). Area in m².",
            "",
            header,
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in res:
            lo, hi = r["cov_mond_ci"]
            md.append(
                f"| {r['game']} | {r['clips']} | {r['n']} | {r['cov_pool']:.1%} | {r['cov_mond']:.1%}"
                f" [{lo:.1%}, {hi:.1%}] | {r['cov_const']:.1%} | {r['r_pool']:.2f} / {r['r_mond']:.2f} /"
                f" {r['r_const']:.2f} | {r['a_mond']:.2f} / {r['a_const']:.2f} |"
            )
    md += [
        "",
        "Notes. Exploratory dev numbers, not the preregistered test. Frames are subsampled at 5 fps and",
        "remain correlated within a clip, hence clip-level bootstrap intervals. Mondrian bins: metres per",
        "pixel at the feet (3 bins) × number of RANSAC inliers (2 bins); small bins use the pooled",
        "quantile.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    typer.echo("\n".join(md))


if __name__ == "__main__":
    app()
