"""Calibration failure detection on dev: can a frame's own features flag a bad camera?

Reads ``frames.parquet`` written by scripts/eval_dev.py. Only frames whose calibration succeeded are scored
(the others already abstain). Failure label, chosen before looking at any score: the calibration leaves the
annotated pitch lines more than 10 px away on average (about twice the typical error). It uses the line
annotations only, never the position labels, whose own camera is off by several pixels in some clips, and
only frames whose annotated lines are self-consistent (a homography fits them within 3 px), so that a
"failure" is never an annotation that no camera could explain.

Detectors (:mod:`predfoot.health.detector`): each feature alone (its sign set on the training games),
logistic regression and gradient boosting, all evaluated leave-one-game-out. Metrics: AUROC, average
precision, AURC and the share of failures removed when abstaining on 5 % or 10 % of frames.

Usage:
    uv run python scripts/eval_health.py runs/<run>/frames.parquet --out reports/m3_health_dev.md
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import typer
from sklearn.metrics import average_precision_score, roc_auc_score

from predfoot.eval.selective import aurc, removed_at
from predfoot.health.detector import FAIL_LINE_PX, failure_table, logo_scores
from predfoot.health.features import FEATURES

app = typer.Typer(add_completion=False)


@app.command()
def main(frames_path: Path, out: Path = Path("reports/m3_health_dev.md")) -> None:
    df = failure_table(pl.read_parquet(frames_path))
    y = df["fail"].to_numpy().astype(bool)
    line_px = df["line_px"].to_numpy().astype(float)
    scores = logo_scores(df)
    rows = []
    for name, s in scores.items():
        rows.append(
            {
                "name": name,
                "auroc": roc_auc_score(y, s),
                "ap": average_precision_score(y, s),
                "aurc": aurc(s, y.astype(float)),
                "aurc_px": aurc(s, np.minimum(line_px, 50.0)),
                "rem5": removed_at(s, y, 0.05),
                "rem10": removed_at(s, y, 0.10),
            }
        )
    rows.sort(key=lambda r: -r["auroc"])
    best_single = next(r for r in rows if r["name"] in FEATURES)
    per_game = []
    for g in np.unique(df["game"].to_numpy()):
        sel = df["game"].to_numpy() == g
        if len(np.unique(y[sel])) == 2:
            auc = roc_auc_score(y[sel], scores["logistic"][sel])
            per_game.append((int(g), int(sel.sum()), float(y[sel].mean()), auc))
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    lr = next(r for r in rows if r["name"] == "logistic")
    md = [
        "# Dev failure detection: flagging a wrong camera from the frame's own features",
        "",
        f"`uv run python scripts/eval_health.py {frames_path}` · commit `{sha}`",
        "",
        f"Calibrated dev frames with self-consistent line annotations: {len(df)}, from"
        f" {df['clip'].n_unique()} clips and {df['game'].n_unique()} games. Failures (mean distance to the"
        f" annotated lines > {FAIL_LINE_PX:.0f} px): {y.mean():.1%}.",
        "All scores are out-of-game: each game is scored by detectors trained on the other games.",
        "",
        f"**Abstaining on the 10 % riskiest frames removes {lr['rem10']:.0%} of the failures (logistic"
        f" regression); the best single feature, `{best_single['name']}`, removes"
        f" {best_single['rem10']:.0%}.**",
        "",
        "| Detector | AUROC | Average precision | AURC (failure rate) | AURC (line px, capped at 50) |"
        " Failures removed at 5 % / 10 % abstention |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['name']} | {r['auroc']:.3f} | {r['ap']:.3f} | {r['aurc']:.3f} | {r['aurc_px']:.2f} |"
            f" {r['rem5']:.0%} / {r['rem10']:.0%} |"
            for r in rows
        ],
        "",
        "Per held-out game (logistic regression):",
        "",
        "| Game | Frames | Failure rate | AUROC |",
        "|---|---|---|---|",
        *[f"| {g} | {n} | {rate:.1%} | {auc:.3f} |" for g, n, rate, auc in per_game],
        "",
        "Notes. Exploratory dev numbers. Frames are one in five and correlated within a clip, so the",
        "effective sample is closer to the number of clips than to the number of frames. The critic CNN",
        "has to beat the best row of this table to be worth its cost.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    typer.echo("\n".join(md))


if __name__ == "__main__":
    app()
