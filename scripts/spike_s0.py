"""Spike S0 (ticket T0.5): does the calibration pipeline work on real broadcast frames, and does its predicted
uncertainty rank the real position errors?

Per frame: PnLCalib keypoints (+ intersections of detected lines) → robust calibration → positions of the
ground-truth boxes. Positions are compared with two references:
- the SN-GSR labels (``bbox_pitch_raw``), produced by the annotators' camera;
- the line oracle: a homography fitted to the annotated pitch lines of the same frame.
The report also measures how well the annotators' camera explains the annotated lines, and inference speed.
Network outputs are cached per clip, so re-running only redoes the geometry.

Usage:
    uv run python scripts/spike_s0.py CLIP_DIR [CLIP_DIR ...] --out reports/spike_s0.md
A clip directory holds ``Labels-GameState.json`` and ``img1/*.jpg`` (frames may be subsampled).
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import typer
from scipy.stats import spearmanr

from predfoot.calib.lines import line_oracle, line_reprojection_px
from predfoot.calib.observations import observations_from_detections
from predfoot.calib.solver import SolverConfig, calibrate
from predfoot.geometry import homography as hg
from predfoot.io.conventions import sn_normalised_to_pixels, sn_to_internal_xy
from predfoot.keypoints.cache import cached_detections
from predfoot.uncertainty.delta import position_covariance

app = typer.Typer(add_completion=False)
ORACLE_MAX_PX = 3.0  # the line oracle is a reference only where it fits its own lines within 3 px


@dataclass
class FrameLabels:
    feet: np.ndarray  # (n, 2) pixels
    gt: np.ndarray  # (n, 2) metres, internal frame (bbox_pitch_raw)
    gt_smooth: np.ndarray  # (n, 2) metres (bbox_pitch)
    lines: dict[str, np.ndarray] = field(default_factory=dict)  # class -> (m, 2) pixels


def load_labels(path: Path) -> tuple[dict, dict[int, FrameLabels]]:
    """Minimal reader for the spike (the full index lives in predfoot.io.sngsr)."""
    raw = json.loads(path.read_text())
    by_image = {im["image_id"]: im for im in raw["images"]}
    persons: dict[int, list] = defaultdict(list)
    lines: dict[int, dict] = {}
    for a in raw["annotations"]:
        im = by_image[a["image_id"]]
        frame, w, h = int(Path(im["file_name"]).stem), im["width"], im["height"]
        if a.get("category_id") in (1, 2, 3) and a.get("bbox_pitch_raw") and a.get("bbox_pitch"):
            b = a["bbox_image"]
            if b["y"] + b["h"] >= h - 2 or b["x"] <= 1 or b["x"] + b["w"] >= w - 2:
                continue  # truncated box: the feet may be outside the frame
            r, s = a["bbox_pitch_raw"], a["bbox_pitch"]
            persons[frame].append(
                (b["x"] + b["w"] / 2, b["y"] + b["h"], r["x_bottom_middle"], r["y_bottom_middle"],
                 s["x_bottom_middle"], s["y_bottom_middle"])
            )  # fmt: skip
        elif a.get("category_id") == 5 and isinstance(a.get("lines"), dict):
            lines[frame] = {
                n: sn_normalised_to_pixels([[p["x"], p["y"]] for p in pts], w, h)
                for n, pts in a["lines"].items()
                if len(pts) >= 2
            }
    frames = {}
    for frame in set(persons) | set(lines):
        arr = np.array(persons.get(frame, []), dtype=np.float64).reshape(-1, 6)
        frames[frame] = FrameLabels(
            arr[:, :2], sn_to_internal_xy(arr[:, 2:4]), sn_to_internal_xy(arr[:, 4:6]), lines.get(frame, {})
        )
    return raw["info"], frames


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "unknown"


def summarise(rows: np.ndarray) -> dict[str, float]:
    err, sigma = rows[:, 0], rows[:, 1]
    lo, hi = np.quantile(sigma, [1 / 3, 2 / 3])
    return {
        "p50": float(np.quantile(err, 0.5)),
        "p90": float(np.quantile(err, 0.9)),
        "rho": float(spearmanr(sigma, err).statistic),
        "p90_low": float(np.quantile(err[sigma <= lo], 0.9)),
        "p90_high": float(np.quantile(err[sigma >= hi], 0.9)),
    }


@app.command()
def main(
    clip_dirs: list[Path], out: Path = Path("reports/spike_s0.md"), device: str = "auto", batch: int = 1
) -> None:
    cfg = SolverConfig()
    variants = {"keypoints": False, "keypoints + lines": True}
    rows: dict[tuple[str, str], list] = defaultdict(list)  # (variant, reference) -> [err, sigma, clip]
    frames: list[dict] = []
    seconds_total, n_inferred, version = 0.0, 0, ""
    for clip_dir in clip_dirs:
        info, labels = load_labels(clip_dir / "Labels-GameState.json")
        dets, seconds, version = cached_detections(clip_dir, info["name"], device=device, batch=batch)
        seconds_total, n_inferred = seconds_total + seconds, n_inferred + len(dets)
        for frame, det in sorted(dets.items()):
            lab = labels.get(frame)
            if lab is None:
                continue
            h_oracle = line_oracle(lab.lines)
            if line_reprojection_px(h_oracle, lab.lines) > ORACLE_MAX_PX:
                h_oracle = None  # the annotated lines do not pin the camera down: no trusted reference
            h_label = hg.dlt(lab.gt, lab.feet) if len(lab.feet) >= 8 else None
            rec = {
                "clip": info["name"], "game": info["game_id"], "frame": frame,
                "oracle_px": line_reprojection_px(h_oracle, lab.lines),
                "label_px": line_reprojection_px(h_label, lab.lines),
            }  # fmt: skip
            refs = {"labels": lab.gt}
            if h_oracle is not None and len(lab.feet):
                refs["line oracle"] = hg.backproject(h_oracle, lab.feet)[0]
            for name, use_lines in variants.items():
                xy, uv, w, obs = observations_from_detections(det, use_lines)
                rng = np.random.default_rng(int(frame) * 7919 + sum(map(ord, info["name"])))
                result, _ = calibrate(xy, uv, rng, weights=w, config=cfg, lines=obs)
                rec[f"{name} ok"] = result.ok
                rec[f"{name} px"] = line_reprojection_px(result.H, lab.lines) if result.ok else np.nan
                rec[f"{name} n"] = len(xy)
                if not result.ok or not len(lab.feet):
                    continue
                pred, cov, valid = position_covariance(result.H, result.cov_theta, lab.feet)
                sigma = np.sqrt(np.linalg.eigvalsh(cov)[:, -1])
                for ref_name, ref in refs.items():
                    err = np.linalg.norm(pred - ref, axis=1)
                    rows[(name, ref_name)].extend(
                        [e, s] for e, s in zip(err[valid], sigma[valid], strict=True)
                    )
            frames.append(rec)
            rows[("floor", "labels")].extend([e, 0.0] for e in np.linalg.norm(lab.gt_smooth - lab.gt, axis=1))
            if "line oracle" in refs:
                rows[("oracle", "labels")].extend(
                    [e, 0.0] for e in np.linalg.norm(refs["line oracle"] - lab.gt, axis=1)
                )

    def nanq(values: list, p: float) -> float:
        return float(np.nanquantile(np.array(values, dtype=np.float64), p))

    def p50_p90(values: list, digits: int = 2) -> str:
        return f"{nanq(values, 0.5):.{digits}f} | {nanq(values, 0.9):.{digits}f}"

    def yes(ok: bool) -> str:
        return "yes" if ok else "no"

    fps = n_inferred / seconds_total if seconds_total > 0 else float("nan")
    clips = sorted({f["clip"] for f in frames})
    best = "keypoints + lines"
    s_lab = summarise(np.array(rows[(best, "labels")]))
    s_orc = summarise(np.array(rows[(best, "line oracle")]))
    calibrated = float(np.mean([f[f"{best} ok"] for f in frames]))
    floor_lab = nanq([r[0] for r in rows[("oracle", "labels")]], 0.9)
    ranks = s_orc["rho"] >= 0.3 and s_orc["p90_high"] >= 2 * s_orc["p90_low"]
    gate_rows = [
        (f"Throughput ≥ 2 frames/s (both networks, M4 Pro, batch {batch})", fps >= 2, f"{fps:.1f} frames/s"),
        ("≥ 80 % of frames calibrated", calibrated >= 0.8, f"{calibrated:.0%}"),
        (
            "Error p90 ≥ 2 × floor (labels as reference)",
            s_lab["p90"] >= 2 * floor_lab,
            f"{s_lab['p90']:.2f} m vs floor {floor_lab:.2f} m (line oracle vs labels)",
        ),
        (
            "Spearman(σ̂, error) ≥ 0.3 and tercile ratio ≥ 2 (line-oracle reference)",
            ranks,
            f"ρ = {s_orc['rho']:.2f}; p90 top/bottom σ̂ tercile "
            f"{s_orc['p90_high']:.2f} / {s_orc['p90_low']:.2f} m",
        ),
    ]
    md = [
        "# Spike S0: calibration on real broadcast frames",
        "",
        f"`uv run python scripts/spike_s0.py <{len(clip_dirs)} clips>` · commit `{git_sha()}`"
        f" · model `{version}`",
        "",
        f"Clips: {', '.join(clips)}. Frames: {len(frames)}. Ground-truth boxes, truncated boxes excluded.",
        "",
        "## Gate G0",
        "",
        "| Criterion | Pass | Measured |",
        "|---|---|---|",
        *[f"| {name} | {yes(ok)} | {measured} |" for name, ok, measured in gate_rows],
        "",
        "## Position errors (metres)",
        "",
        "| Calibration | Reference | p50 | p90 | Spearman(σ̂, error) |",
        "|---|---|---|---|---|",
    ]
    for (variant, ref), vals in sorted(rows.items()):
        arr = np.array(vals)
        if variant in variants:
            s = summarise(arr)
            md.append(f"| {variant} | {ref} | {s['p50']:.2f} | {s['p90']:.2f} | {s['rho']:.2f} |")
        else:
            label = "raw vs smoothed labels" if variant == "floor" else "line oracle"
            md.append(f"| {label} | {ref} | {p50_p90(list(arr[:, 0]))} | — |")
    camera_rows = [
        ("keypoints", "keypoints px"),
        ("keypoints + lines", f"{best} px"),
        ("line oracle (fitted to these lines)", "oracle_px"),
        ("annotators' camera (homography implied by the position labels)", "label_px"),
    ]
    md += [
        "",
        "## Calibration quality (pixels: mean distance of annotated line points to the projected template)",
        "",
        "| Camera | p50 | p90 |",
        "|---|---|---|",
        *[f"| {name} | {p50_p90([f[key] for f in frames], 1)} |" for name, key in camera_rows],
        "",
        "## Per clip (keypoints + lines, labels as reference)",
        "",
        "| Clip | Game | Frames | Calibrated | Line px p50 | Labels' camera px p50 |",
        "|---|---|---|---|---|---|",
    ]
    for c in clips:
        fr = [f for f in frames if f["clip"] == c]
        ok_share = np.mean([f[f"{best} ok"] for f in fr])
        ours, theirs = nanq([f[f"{best} px"] for f in fr], 0.5), nanq([f["label_px"] for f in fr], 0.5)
        md.append(f"| {c} | {fr[0]['game']} | {len(fr)} | {ok_share:.0%} | {ours:.1f} | {theirs:.1f} |")
    md += [
        "",
        "Notes. The position labels fit a homography exactly (they are the annotators' camera applied to the",
        "box bottoms), so their line reprojection error measures how far that camera is from the annotated",
        "lines. σ̂ = √λmax(Σp) comes from the Laplace covariance before any conformal calibration; only its",
        "ranking matters here. Frames within a clip are strongly correlated: treat these numbers as a",
        "feasibility check, not results.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    typer.echo("\n".join(md))


if __name__ == "__main__":
    app()
