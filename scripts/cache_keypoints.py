"""Cache keypoint and line detections for every downloaded clip of the given splits (ticket T1.6).

Usage:
    uv run python scripts/cache_keypoints.py --splits train valid
Resumable: clips already cached are skipped.
"""

from __future__ import annotations

import time

import typer

from predfoot.keypoints.cache import cache_path, cached_detections
from predfoot.paths import data_dir

app = typer.Typer(add_completion=False)


@app.command()
def main(
    splits: list[str] = typer.Option(["train", "valid"]),  # noqa: B008
    device: str = "auto",
    reverse: bool = typer.Option(
        False, help="Walk the clips backwards (to share the work with a second run)."
    ),
) -> None:
    if "test" in splits:
        raise typer.BadParameter("the test split stays locked until the preregistration is frozen")
    clip_dirs = sorted(
        d for s in splits for d in (data_dir() / "sn-gsr" / s).glob("SNGS-*") if (d / "img1").is_dir()
    )
    if reverse:
        clip_dirs = clip_dirs[::-1]
    t_start, done = time.perf_counter(), 0
    for i, clip_dir in enumerate(clip_dirs, start=1):
        n = len(list((clip_dir / "img1").glob("*.jpg")))
        if cache_path(clip_dir.name, n).exists():
            continue
        _, seconds, _ = cached_detections(clip_dir, clip_dir.name, device=device)
        done += 1
        elapsed = time.perf_counter() - t_start
        progress = f"[{i}/{len(clip_dirs)}] {clip_dir.name}"
        typer.echo(f"{progress}: {n} frames, {seconds:.0f} s, total {elapsed / 60:.1f} min")
    typer.echo(f"Done: {done} clips cached, {len(clip_dirs)} available.")


if __name__ == "__main__":
    app()
