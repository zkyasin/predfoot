"""Run ``predfoot run`` with the real stages on a cached SN-GSR dev clip.

Foot points come from the ground-truth boxes (players and goalkeepers, truncated boxes dropped); the camera
comes from our calibration of the cached detections; the conformal quantiles, when given, are fitted on the
dev games other than the clip's own (scores against the SN-GSR labels). Frames are the cached ones (one in
five), so the clip runs at 5 fps. Internal check only: outputs derived from SoccerNet data are not published.

Usage:
    uv run python scripts/run_real_clip.py SNGS-056 --out runs/real/SNGS-056 \
        --conformal-from runs/<eval_dev run>/players.parquet
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import typer

from predfoot.keypoints.cache import cache_path, cached_detections
from predfoot.paths import data_dir, index_dir
from predfoot.pipeline.real import real_stages
from predfoot.pipeline.run import RunConfig, run_clip
from predfoot.uncertainty.conformal import MondrianConformal
from predfoot.uncertainty.delta import mahalanobis2

app = typer.Typer(add_completion=False)


def fit_conformal(players_path: Path, exclude_game: int) -> MondrianConformal:
    d = pl.read_parquet(players_path).filter((pl.col("game") != exclude_game) & pl.col("ex_lab").is_not_nan())
    e = d.select("ex_lab", "ey_lab").to_numpy()
    cov = np.stack([d["c00"], d["c01"], d["c01"], d["c11"]], axis=1).reshape(-1, 2, 2)
    feats = {"mpp": d["mpp"].to_numpy(), "n_inliers": d["n_inliers"].to_numpy().astype(float)}
    return MondrianConformal.fit(
        mahalanobis2(e, cov), d["clip"].to_numpy(), feats, {"mpp": 3, "n_inliers": 2}
    )


@app.command()
def main(
    clip_id: str,
    out: Path = typer.Option(..., help="Output folder."),  # noqa: B008
    conformal_from: Path | None = typer.Option(None, help="players.parquet of an eval_dev run."),  # noqa: B008
) -> None:
    clip = (
        pl.read_parquet(index_dir() / "clips.parquet").filter(pl.col("clip_id") == clip_id).row(0, named=True)
    )
    if clip["split"] == "test":
        raise typer.BadParameter("test clips stay locked")
    clip_dir = data_dir() / "sn-gsr" / clip["split"] / clip_id
    n = len(list((clip_dir / "img1").glob("*.jpg")))
    if not cache_path(clip_id, n).exists():
        raise typer.BadParameter(f"no cached detections for {clip_id}: run scripts/cache_keypoints.py")
    dets, _, model_version = cached_detections(clip_dir, clip_id)
    source_frames = sorted(dets)
    persons = pl.read_parquet(index_dir() / "persons.parquet").filter(
        (pl.col("clip_id") == clip_id) & ~pl.col("truncated") & pl.col("role").is_in(["player", "goalkeeper"])
    )
    by_frame = persons.partition_by("frame", as_dict=True)
    frames, teams = [], {}
    for i, f in enumerate(source_frames):
        p = by_frame.get((f,))
        feet = []
        if p is not None:
            for row in p.iter_rows(named=True):
                feet.append({"track_id": int(row["track_id"]), "u": row["foot_u"], "v": row["foot_v"]})
                if row["team"] in ("left", "right"):
                    teams[int(row["track_id"])] = "home" if row["team"] == "left" else "away"
        frames.append({"frame": i, "homography": None, "feet": feet})
    out.mkdir(parents=True, exist_ok=True)
    input_path = out / "input_clip.json"
    clip_json = {
        "format": "predfoot-clip/1",
        "clip_id": clip_id,
        "fps": 25.0 / 5,
        "image_size": [int(clip["width"]), int(clip["height"])],
        "teams": {str(k): v for k, v in sorted(teams.items())},
        "frames": frames,
    }
    input_path.write_text(json.dumps(clip_json))
    conformal = fit_conformal(conformal_from, int(clip["game_id"])) if conformal_from else None
    tag = hashlib.sha256(conformal.to_json().encode()).hexdigest()[:12] if conformal else "laplace"
    stages = real_stages(
        dict(enumerate(dets[f] for f in source_frames)), conformal, f"real-{model_version}-{tag}"
    )
    result = run_clip(input_path, out, RunConfig(export_fps=5.0), stages)
    typer.echo(f"{clip_id}: {result}")


if __name__ == "__main__":
    app()
