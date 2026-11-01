"""Frozen artifacts and endpoints of the preregistered test run (docs/preregistration.md).

Freeze (dev only, :func:`dev_tables` then :func:`write_artifacts`), from the per-frame calibrations of every
dev clip and their labels:

- smoother noise parameters, fitted without labels on every dev clip (the ``--fit-scope all`` fit of
  ``scripts/eval_dev.py``);
- conformal regions for each camera (per-frame, PTZ-smoothed with those parameters) and each reference (raw
  labels, trusted line oracle): :func:`predfoot.eval.positions.fit_frozen_regions`, the construction of one
  leave-one-game-out fold with every dev game on the calibration side;
- failure detectors on the per-frame calibrations (:class:`predfoot.health.detector.FrozenDetector`).

Evaluation (test games, once, :func:`evaluate`): every primary and secondary endpoint for both cameras and
both references, with the success criteria as written, then :func:`render_report` (aggregate numbers only).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from predfoot.calib import temporal as tm
from predfoot.calib.solver import SolverConfig
from predfoot.contracts import CalibResult
from predfoot.eval import positions as ps
from predfoot.eval.selective import removed_at
from predfoot.health.detector import FAIL_LINE_PX, FrozenDetector, failure_table
from predfoot.health.features import FEATURES
from predfoot.kpi.proxy import players_kpi_proxy, restrict_shared
from predfoot.uncertainty.delta import region_quantile

ARTIFACT_FORMAT = "predfoot-prereg-artifact/1"
CAMERAS = ("perframe", "ptz")
REFERENCES = ("lab", "or")
CAMERA_TITLES = {"perframe": "per-frame camera", "ptz": "PTZ-smoothed camera"}
REF_TITLES = {"lab": "SN-GSR raw labels", "or": "trusted line oracle", "sm": "smoothed labels"}
#: Method under test and primary reference (P1, P2); everything else is secondary.
PRIMARY = ("ptz", "lab")
#: Weighting of P1's coverage: "player-frames" pools the headline player-frames (the dev headline); "clips"
#: weighs every clip equally. The other one is reported as a secondary analysis.
P1_WEIGHTING = "player-frames"
P1_COVERAGE_KEYS = {"player-frames": "coverage", "clips": "coverage_clips"}
P1_WEIGHTING_TITLES = {"player-frames": "player-frames pooled", "clips": "clips weighted equally"}
N_BOOT = 2000
BOOT_SEED = 0
TEST_GAMES = (7, 8, 11)
TEST_CLIPS = 49
DEV_GAMES = (2, 3, 4, 5, 6, 9)
KEYS = ["clip", "frame", "track_id"]
OUTPUT_TYPES = ("smoothed", "repaired", "filled", "bridged", "passthrough", "duplicate")


def analysis_config() -> dict[str, Any]:
    """Every constant of the analysis plan, recorded with the artifacts (and digested in the lock)."""
    return {
        "alpha": ps.ALPHA,
        "n_boot": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "unit": "clip",
        "oracle_max_px": ps.ORACLE_MAX_PX,
        "raw_gap_max_m": ps.RAW_GAP_MAX_M,
        "fail_line_px": FAIL_LINE_PX,
        "mondrian_bins": dict(ps.MONDRIAN_BINS),
        "pooled_sources": sorted(ps.POOLED_SOURCES),
        "min_source_clips": ps.MIN_SOURCE_CLIPS,
        "frames_every": 5,
        "cameras": list(CAMERAS),
        "references": list(REFERENCES),
        "primary": {"camera": PRIMARY[0], "reference": PRIMARY[1]},
        "p1_weighting": P1_WEIGHTING,
        "dev_games": list(DEV_GAMES),
        "test_games": list(TEST_GAMES),
        "test_clips": TEST_CLIPS,
        "solver": asdict(SolverConfig()),
    }


def clip_list_digest(clips: Iterable[tuple[str, int]]) -> str:
    """SHA-256 of a clip list (clip id, game), order-free: the lock pins the test clips with it."""
    text = "".join(f"{clip},{int(game)}\n" for clip, game in sorted(clips))
    return hashlib.sha256(text.encode()).hexdigest()


def locked_clip_list(clips_parquet: Path) -> list[tuple[str, int]]:
    """(clip id, game) of every clip of the locked split from the clip index (``index/clips.parquet``, the
    split's structure only: no label is read); refuses games other than the preregistered ones."""
    clips = pl.read_parquet(clips_parquet, columns=["clip_id", "split", "game_id"])
    rows = clips.filter(pl.col("split") == "test").sort("clip_id").select("clip_id", "game_id").iter_rows()
    out = [(str(c), int(g)) for c, g in rows]
    games = sorted({g for _, g in out})
    if games != list(TEST_GAMES):
        raise ValueError(f"test games {games}, preregistered {list(TEST_GAMES)}")
    return out


def to_jsonable(obj: Any) -> Any:
    """Plain JSON: numpy scalars as Python numbers, tuples as lists, NaN and infinities as null."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


def _mean(values: pl.Series) -> float:
    return float(np.mean(values.to_numpy()))


def dump(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), indent=1, sort_keys=True, allow_nan=False) + "\n"


# --- dev tables and freeze --------------------------------------------------------------------------------


@dataclass
class CameraTables:
    """Player rows (all output frames) and frame rows of one camera over a set of labelled clips."""

    players: pl.DataFrame
    frames: pl.DataFrame


def _concat(tables: Sequence[pl.DataFrame]) -> pl.DataFrame:
    kept = [t for t in tables if t.height]
    return pl.concat(kept, how="vertical") if kept else pl.DataFrame()


def perframe_tables(ccs: Sequence[ps.ClipCalib]) -> CameraTables:
    players = _concat([ps.player_rows(cc, cc.results, None, None) for cc in ccs])
    rows = [r for cc in ccs for r in ps.frame_rows_out(cc, cc.results, None, {}, None)]
    return CameraTables(players, pl.DataFrame(rows, infer_schema_length=None))


def output_tables(outputs: Sequence[ps.Output]) -> CameraTables:
    players = _concat([ps.player_rows(o.cc, o.results, o.flags, None) for o in outputs])
    rows = [r for o in outputs for r in ps.frame_rows_out(o.cc, o.results, o.flags, o.diag, None)]
    return CameraTables(players, pl.DataFrame(rows, infer_schema_length=None))


@dataclass
class DevTables:
    perframe: CameraTables
    ptz: CameraTables
    smoothing: ps.SmoothingInfo
    params: tm.NoiseParams
    cfg: tm.SmootherConfig


def dev_tables(ccs: Sequence[ps.ClipCalib], cfg: tm.SmootherConfig | None = None) -> DevTables:
    """Both cameras on the labelled dev clips, the smoother's noise fitted on all of them without labels."""
    cfg = cfg or tm.SmootherConfig()
    info = ps.SmoothingInfo()
    outputs = list(ps.smooth_outputs(ccs, ps.Smooth.ptz, ps.FitScope.all, cfg, info))
    params = tm.NoiseParams.from_dict(info.fits["all"]["params"])
    return DevTables(perframe_tables(ccs), output_tables(outputs), info, params, cfg)


def camera_table(tables: DevTables, camera: str) -> CameraTables:
    return tables.perframe if camera == "perframe" else tables.ptz


def fit_camera_regions(players: pl.DataFrame, ref: str) -> ps.RegionModels:
    """Frozen regions of one camera and reference: headline rows calibrate, filled rows their own source."""
    return ps.fit_frozen_regions(players.filter(~pl.col("filled")), ref, players.filter(pl.col("filled")))


def write_artifacts(tables: DevTables, out_dir: Path, provenance: Mapping[str, Any]) -> dict[str, Path]:
    """Fit every frozen artifact from the dev tables and write them as JSON under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def write(name: str, payload: Mapping[str, Any]) -> None:
        path = out_dir / f"{name}.json"
        path.write_text(dump({"format": ARTIFACT_FORMAT, "kind": name, **payload}), encoding="utf-8")
        written[name] = path

    write("analysis", {"config": analysis_config(), "provenance": dict(provenance)})
    fit = tables.smoothing.fits["all"]
    write(
        "smoother",
        {
            "config": asdict(tables.cfg),
            "config_digest": tables.cfg.digest(),
            "params": tables.params.to_dict(),
            "params_digest": tables.params.digest(),
            "fit": {k: v for k, v in fit.items() if k not in ("params", "params_digest")},
        },
    )
    for camera in CAMERAS:
        players = camera_table(tables, camera).players
        for ref in REFERENCES:
            rm = fit_camera_regions(players, ref)
            head = players.filter(~pl.col("filled") & pl.col(f"ex_{ref}").is_not_nan())
            sources = head["cov_source"].value_counts().sort("cov_source").rows()
            write(
                f"regions_{camera}_{ref}",
                {
                    "camera": camera,
                    "reference": ref,
                    **rm.to_dict(),
                    "calibration": {
                        "player_frames": head.height,
                        "filled_player_frames": players.filter(
                            pl.col("filled") & pl.col(f"ex_{ref}").is_not_nan()
                        ).height,
                        "clips": head["clip"].n_unique(),
                        "games": sorted(int(g) for g in head["game"].unique()),
                        "sources": {str(k): int(v) for k, v in sources},
                    },
                },
            )
    det = FrozenDetector.fit(failure_table(tables.perframe.frames))
    write("detector", {"detector": det.to_dict()})
    return written


@dataclass
class Artifacts:
    params: tm.NoiseParams
    smoother_config_digest: str
    regions: dict[tuple[str, str], ps.RegionModels]
    detector: FrozenDetector
    analysis: dict[str, Any]


def load_artifacts(art_dir: Path) -> Artifacts:
    """Read the frozen artifacts; refuses a smoother configuration other than the current default."""

    def read(name: str) -> dict[str, Any]:
        d: dict[str, Any] = json.loads((art_dir / f"{name}.json").read_text(encoding="utf-8"))
        if d.get("format") != ARTIFACT_FORMAT or d.get("kind") != name:
            raise ValueError(f"{art_dir / name}.json is not a frozen {name} artifact")
        return d

    sm = read("smoother")
    if sm["config_digest"] != tm.SmootherConfig().digest():
        raise ValueError("the smoother configuration changed since the freeze")
    regions = {
        (c, r): ps.RegionModels.from_dict(read(f"regions_{c}_{r}")) for c in CAMERAS for r in REFERENCES
    }
    return Artifacts(
        params=tm.NoiseParams.from_dict(sm["params"]),
        smoother_config_digest=str(sm["config_digest"]),
        regions=regions,
        detector=FrozenDetector.from_dict(read("detector")["detector"]),
        analysis=read("analysis")["config"],
    )


# --- saved output cameras ---------------------------------------------------------------------------------


def save_outputs(path: Path, outputs: Sequence[ps.Output], meta: Mapping[str, Any]) -> None:
    """Output cameras (calibrations, frame flags, smoother diagnostics) of every clip, exactly."""
    arrays: dict[str, np.ndarray] = {
        "meta": np.array(json.dumps({**meta, "clips": [o.cc.clip_id for o in outputs]}))
    }
    for o in outputs:
        arrays |= ps._result_arrays(o.cc.clip_id, o.cc.frames, o.cc.duplicate, o.results)
        key = f"{o.cc.clip_id}__"
        flags = o.flags if o.flags is not None else tm.FrameFlags.empty(len(o.cc.frames))
        for name, values in flags.as_dict().items():
            arrays[key + "flag_" + name] = np.asarray(values)
        for name, values in o.diag.items():
            arrays[key + "diag_" + name] = np.asarray(values, dtype=np.float64)
    ps.savez_atomic(path, arrays)


SavedOutput = tuple[list[CalibResult], tm.FrameFlags, dict[str, np.ndarray]]


def load_outputs(path: Path) -> tuple[dict[str, SavedOutput], dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    meta: dict[str, Any] = json.loads(str(z["meta"]))
    out: dict[str, SavedOutput] = {}
    names = [f.name for f in tm.FrameFlags.__dataclass_fields__.values()]
    for clip_id in meta["clips"]:
        key = f"{clip_id}__"
        flags = tm.FrameFlags(**{n: z[key + "flag_" + n].copy() for n in names})
        diag = {k[len(key) + 5 :]: z[k].copy() for k in z.files if k.startswith(key + "diag_")}
        out[clip_id] = (ps._load_results(z, clip_id), flags, diag)
    return out, meta


# --- endpoints --------------------------------------------------------------------------------------------


def _boot_stat(
    values: np.ndarray, groups: np.ndarray, rng: np.random.Generator, n_boot: int, stat: Any
) -> tuple[float, float]:
    """95 % percentile interval of ``stat`` over resampled clips."""
    uniq = np.unique(groups)
    idx = {c: np.flatnonzero(groups == c) for c in uniq}
    draws = [
        stat(np.concatenate([values[idx[c]] for c in rng.choice(uniq, size=len(uniq), replace=True)]))
        for _ in range(n_boot)
    ]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def region_endpoint(
    players: pl.DataFrame,
    rm: ps.RegionModels,
    ref: str,
    rng: np.random.Generator,
    n_boot: int,
    raw_ok: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Coverage and size of the frozen regions of one camera against one reference, with clip-bootstrap
    intervals; filled frames reported apart.

    Headline rows: player-frames whose raw calibration succeeded and for which this camera publishes a
    position. ``raw_ok`` (the per-frame camera's rows: every raw-OK player-frame) gives the player-frames
    where this camera abstains although the raw calibration succeeded; they are counted, and a secondary
    coverage counts them as misses.
    """
    d = players.filter(~pl.col("filled") & pl.col(f"ex_{ref}").is_not_nan()) if players.height else players
    if d.height == 0:
        return {"n": 0}
    s = ps.score_table(d, ref)
    feats = {k: s[k] for k in ps.FEATURE_NAMES}
    q = rm.quantile_for(feats, s["source"])
    hit = s["score"] <= q
    area = np.pi * q * np.sqrt(np.clip(np.linalg.det(s["cov"]), 0, None))
    hit_c = s["norm"] <= rm.r_const
    a_const = float(np.pi * rm.r_const**2)
    groups, games = s["groups"], s["games"]
    raw_q = np.array([region_quantile(2, 1 - ps.ALPHA, v) for v in s["dof"]])
    cov_clips, cov_clips_ci = ps.clip_weighted_coverage(hit, groups, rng, n_boot)
    out: dict[str, Any] = {
        "n": d.height,
        "clips": len(np.unique(groups)),
        "coverage": float(hit.mean()),
        "coverage_ci": ps.clip_bootstrap(hit, groups, rng, n_boot),
        "coverage_clips": cov_clips,
        "coverage_clips_ci": cov_clips_ci,
        "coverage_const": float(hit_c.mean()),
        "coverage_const_ci": ps.clip_bootstrap(hit_c, groups, rng, n_boot),
        "coverage_raw_laplace": float(np.mean(s["score"] <= raw_q)),
        "median_area": float(np.median(area)),
        "const_area": a_const,
        "area_ratio": float(np.median(area)) / a_const,
        "area_ratio_ci": _boot_stat(area, groups, rng, n_boot, lambda a: float(np.median(a)) / a_const),
    }
    out["per_game"] = []
    for g in np.unique(games):
        sel = games == g
        out["per_game"].append(
            {
                "game": int(g),
                "clips": len(np.unique(groups[sel])),
                "n": int(sel.sum()),
                "coverage": float(hit[sel].mean()),
                "coverage_ci": ps.clip_bootstrap(hit[sel], groups[sel], rng, n_boot),
                "median_area": float(np.median(area[sel])),
            }
        )
    edges = np.quantile(feats["mpp"], [1 / 3, 2 / 3])
    b = np.searchsorted(edges, feats["mpp"])
    out["by_tercile"] = [
        {"coverage": float(hit[b == i].mean()), "coverage_const": float(hit_c[b == i].mean())}
        for i in range(3)
    ]
    out["tercile_edges"] = edges.tolist()
    per_frame = (
        pl.DataFrame({"clip": d["clip"], "frame": d["frame"], "hit": hit})
        .group_by("clip", "frame")
        .agg(pl.col("hit").all())
    )
    out["simultaneous_frames"] = per_frame.height
    out["simultaneous_coverage"] = _mean(per_frame["hit"])
    filled = players.filter(pl.col("filled") & pl.col(f"ex_{ref}").is_not_nan())
    if filled.height:
        fs = ps.score_table(filled, ref)
        fq = rm.quantile_for({k: fs[k] for k in ps.FEATURE_NAMES}, fs["source"])
        out["filled"] = {
            "n": filled.height,
            "coverage": float(np.mean(fs["score"] <= fq)),
            "coverage_const": float(np.mean(fs["norm"] <= rm.r_const)),
        }
    if raw_ok is not None and raw_ok.height:
        base = raw_ok.filter(pl.col(f"ex_{ref}").is_not_nan()).select(KEYS)
        missing = base.join(d.select(KEYS), on=KEYS, how="anti")
        n_ab = missing.height
        out["abstained"] = {
            "n": n_ab,
            "frames": missing.select("clip", "frame").unique().height,
            "share": n_ab / (d.height + n_ab),
            "coverage_as_misses": float(hit.sum()) / (d.height + n_ab),
        }
    return out


def detector_endpoint(
    frames: pl.DataFrame, det: FrozenDetector, rng: np.random.Generator, n_boot: int
) -> dict[str, Any]:
    """Failure detection on the per-frame calibrations: AUROC of every frozen detector, the preregistered
    comparison (boosting against the single feature chosen on dev) with a clip-bootstrap interval of the
    difference, and the best single feature on these frames (post hoc, stricter)."""
    df = failure_table(frames)
    y = df["fail"].to_numpy().astype(bool)
    out: dict[str, Any] = {"frames": len(df), "failures": int(y.sum()), "best_single_dev": det.best_single}
    if len(np.unique(y)) < 2:
        out["auroc"] = None
        return out
    scores = det.scores(df)
    out["detectors"] = {
        name: {
            "auroc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)),
            "removed_5": removed_at(s, y, 0.05),
            "removed_10": removed_at(s, y, 0.10),
        }
        for name, s in scores.items()
    }
    best = det.best_single
    out["auroc_boosting"] = out["detectors"]["boosting"]["auroc"]
    out["auroc_best_single"] = out["detectors"][best]["auroc"]
    out["auroc_diff"] = out["auroc_boosting"] - out["auroc_best_single"]
    clips = df["clip"].to_numpy()
    uniq = np.unique(clips)
    idx = {c: np.flatnonzero(clips == c) for c in uniq}
    diffs = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[rows])) == 2:
            diffs.append(
                roc_auc_score(y[rows], scores["boosting"][rows]) - roc_auc_score(y[rows], scores[best][rows])
            )
    out["auroc_diff_ci"] = tuple(float(v) for v in np.quantile(diffs, [0.025, 0.975])) if diffs else None
    out["auroc_diff_draws"] = len(diffs)
    singles = {f: out["detectors"][f]["auroc"] for f in FEATURES}
    out["best_single_test"] = max(singles, key=lambda f: singles[f])
    games = df["game"].to_numpy()
    out["per_game"] = [
        {
            "game": int(g),
            "frames": int((games == g).sum()),
            "failure_rate": float(y[games == g].mean()),
            "auroc_boosting": float(roc_auc_score(y[games == g], scores["boosting"][games == g]))
            if len(np.unique(y[games == g])) == 2
            else None,
            "auroc_best_single": float(roc_auc_score(y[games == g], scores[best][games == g]))
            if len(np.unique(y[games == g])) == 2
            else None,
        }
        for g in np.unique(games)
    ]
    return out


def residual_endpoint(frames_perframe: pl.DataFrame, frames_ptz: pl.DataFrame) -> dict[str, Any]:
    """Residual failures of the smoothed camera: frames with self-consistent lines whose camera is more than
    10 px from the annotated lines.

    Both cameras on the same frames, as on dev (``scripts/eval_dev.py``): raw and output cameras both
    calibrated. Frames that only the smoother fills, and raw-OK frames where it abstains, are counted apart;
    the share per output type covers every published frame of the smoothed camera.
    """
    if not frames_ptz.height:
        return {"frames": 0, "by_type": {}}
    trusted = frames_ptz.filter(pl.col("trusted"))
    both = trusted.filter(
        pl.col("ok") & pl.col("ok_out") & pl.col("line_px").is_not_nan() & pl.col("line_px_out").is_not_nan()
    )
    filled = trusted.filter(~pl.col("ok") & pl.col("ok_out") & pl.col("line_px_out").is_not_nan())
    t = trusted.filter(pl.col("ok_out") & pl.col("line_px_out").is_not_nan())
    lp, lo = both["line_px"].to_numpy().astype(float), both["line_px_out"].to_numpy().astype(float)
    out: dict[str, Any] = {
        "frames": both.height,
        "fail_share_perframe": float(np.mean(lp > FAIL_LINE_PX)) if both.height else None,
        "fail_share_ptz": float(np.mean(lo > FAIL_LINE_PX)) if both.height else None,
        "line_px_p90_perframe": float(np.quantile(lp, 0.9)) if both.height else None,
        "line_px_p90_ptz": float(np.quantile(lo, 0.9)) if both.height else None,
        "filled_frames": filled.height,
        "fail_share_filled": _mean(filled["line_px_out"] > FAIL_LINE_PX) if filled.height else None,
        "abstained_frames": trusted.filter(pl.col("ok") & ~pl.col("ok_out")).height,
        "published_rate_ptz": _mean(frames_ptz["ok_out"]),
        "published_rate_perframe": _mean(frames_perframe["ok"]) if frames_perframe.height else None,
        "by_type": {},
    }
    for name in OUTPUT_TYPES:
        part = t.filter(pl.col(name))
        if part.height:
            out["by_type"][name] = {
                "frames": part.height,
                "fail_share": _mean(part["line_px_out"] > FAIL_LINE_PX),
            }
    return out


def error_endpoint(players: Mapping[str, pl.DataFrame]) -> dict[str, Any]:
    """Error quantiles (p50, p90, p99, m) of both cameras on the same player-frames, per reference."""
    a = players["ptz"].filter(~pl.col("filled"))
    b = players["perframe"]
    if not a.height or not b.height:
        return {"n": 0}
    cols = [f"e{x}_{r}" for r in ("lab", "or", "sm") for x in "xy"]
    joined = a.select(*KEYS, *cols).join(b.select(*KEYS, *cols), on=KEYS, suffix="_pf")
    out: dict[str, Any] = {"n": joined.height}
    for r in ("lab", "or", "sm"):
        for cam, suffix in (("ptz", ""), ("perframe", "_pf")):
            e = np.hypot(joined[f"ex_{r}{suffix}"].to_numpy(), joined[f"ey_{r}{suffix}"].to_numpy())
            out[f"{cam}_{r}"] = (
                [float(np.nanquantile(e, p)) for p in (0.5, 0.9, 0.99)] if np.isfinite(e).any() else None
            )
    return out


def kpi_endpoint(players: Mapping[str, pl.DataFrame]) -> dict[str, Any]:
    """Model-free KPI proxy of both cameras on the same player-frames, against the smoothed labels."""
    if not players["ptz"].height or not players["perframe"].height:
        return {}
    a, b = restrict_shared(players["ptz"].filter(~pl.col("filled")), players["perframe"])
    return {"ptz": players_kpi_proxy(a), "perframe": players_kpi_proxy(b)}


def p1_criterion(prim: Mapping[str, Any]) -> dict[str, Any]:
    """P1 as preregistered: the coverage of the primary camera and reference, weighted as ``P1_WEIGHTING``
    says, is met when its 95 % interval contains the nominal level 1 − α."""
    if not prim.get("n"):
        return {"pass": None}
    key = P1_COVERAGE_KEYS[P1_WEIGHTING]
    lo, hi = prim[f"{key}_ci"]
    return {
        "weighting": P1_WEIGHTING,
        "coverage": prim[key],
        "ci": prim[f"{key}_ci"],
        "pass": lo <= 1 - ps.ALPHA <= hi,
    }


def evaluate(
    tables: Mapping[str, CameraTables], art: Artifacts, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> dict[str, Any]:
    """Every endpoint of the preregistration, the success criteria evaluated as written."""
    rng = np.random.default_rng(seed)
    players = {c: tables[c].players for c in CAMERAS}
    raw_ok = {"perframe": None, "ptz": players["perframe"]}  # every per-frame row is a raw-OK player-frame
    regions = {
        c: {r: region_endpoint(players[c], art.regions[c, r], r, rng, n_boot, raw_ok[c]) for r in REFERENCES}
        for c in CAMERAS
    }
    det = detector_endpoint(tables["perframe"].frames, art.detector, rng, n_boot)
    prim = regions[PRIMARY[0]][PRIMARY[1]]
    p1 = p1_criterion(prim)
    p2 = (
        {"ratio": prim.get("area_ratio"), "pass": prim["area_ratio"] < 1} if prim.get("n") else {"pass": None}
    )
    p3 = (
        {"diff": det["auroc_diff"], "pass": det["auroc_diff"] > 0}
        if det.get("auroc_diff") is not None
        else {"pass": None}
    )
    frames_pf = tables["perframe"].frames
    return {
        "clips": int(frames_pf["clip"].n_unique()) if frames_pf.height else 0,
        "games": sorted(int(g) for g in frames_pf["game"].unique()) if frames_pf.height else [],
        "labelled_frames": frames_pf.height,
        "n_boot": n_boot,
        "seed": seed,
        "regions": regions,
        "detector": det,
        "residual": residual_endpoint(frames_pf, tables["ptz"].frames),
        "errors": error_endpoint(players),
        "kpi": kpi_endpoint(players),
        "criteria": {"P1": p1, "P2": p2, "P3": p3},
    }


# --- report -------------------------------------------------------------------------------------------------


def _pct(v: float | None) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.1%}"


def _ci(ci: Sequence[float] | None) -> str:
    return "–" if ci is None else f"[{ci[0]:.1%}, {ci[1]:.1%}]"


def _verdict(p: bool | None) -> str:
    return "not evaluable" if p is None else ("**met**" if p else "**not met**")


def render_report(res: Mapping[str, Any], header: Sequence[str]) -> str:
    """Markdown report of :func:`evaluate` (aggregate numbers only)."""
    crit = res["criteria"]
    prim = res["regions"][PRIMARY[0]][PRIMARY[1]]
    det = res["detector"]
    md = [*header, ""]
    md += [
        f"Clips: {res['clips']} from game{'s' if len(res['games']) > 1 else ''}"
        f" {', '.join(map(str, res['games']))}; labelled frames (one in"
        " five):"
        f" {res['labelled_frames']}. Intervals: 95 % percentile bootstrap over clips ({res['n_boot']} draws,"
        f" seed {res['seed']}).",
        "",
        "## Success criteria (as preregistered)",
        "",
        "| # | Endpoint | Result | Criterion | Verdict |",
        "|---|---|---|---|---|",
        f"| P1 | Coverage of the 90 % Mondrian regions, {CAMERA_TITLES[PRIMARY[0]]},"
        f" {REF_TITLES[PRIMARY[1]]}, {P1_WEIGHTING_TITLES[P1_WEIGHTING]} |"
        f" {_pct(crit['P1'].get('coverage'))} {_ci(crit['P1'].get('ci'))} | 95 % CI contains 90 % |"
        f" {_verdict(crit['P1']['pass'])} |",
        f"| P2 | Median region area / constant circle at the same level, same camera and reference |"
        f" {prim.get('area_ratio', float('nan')):.2f} | below 1 | {_verdict(crit['P2']['pass'])} |",
        f"| P3 | AUROC of gradient boosting − AUROC of `{det.get('best_single_dev')}` (best single feature on"
        f" dev), per-frame calibrations | {det.get('auroc_diff', float('nan')):+.3f} | above 0 |"
        f" {_verdict(crit['P3']['pass'])} |",
        "",
        "## Coverage and size of the regions",
        "",
        "Headline rows: player-frames whose raw calibration succeeded (frames filled by the smoother are"
        " reported apart).",
        "",
        "| Camera | Reference | Player-frames | Clips | Mondrian coverage [95 % CI] |"
        " Mondrian coverage, clips weighted equally [95 % CI] | Constant circle [95 % CI] |"
        " Uncalibrated Laplace | Median area (m²) | Constant circle area (m²) | Area ratio [95 % CI] |"
        " All players of a frame covered |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in CAMERAS:
        for r in REFERENCES:
            e = res["regions"][c][r]
            if not e.get("n"):
                md.append(f"| {CAMERA_TITLES[c]} | {REF_TITLES[r]} | 0 |" + " – |" * 9)
                continue
            lo, hi = e["area_ratio_ci"]
            md.append(
                f"| {CAMERA_TITLES[c]} | {REF_TITLES[r]} | {e['n']} | {e['clips']} | {_pct(e['coverage'])}"
                f" {_ci(e['coverage_ci'])} | {_pct(e['coverage_clips'])} {_ci(e['coverage_clips_ci'])} |"
                f" {_pct(e['coverage_const'])} {_ci(e['coverage_const_ci'])} |"
                f" {_pct(e['coverage_raw_laplace'])} | {e['median_area']:.2f} | {e['const_area']:.2f} |"
                f" {e['area_ratio']:.2f} [{lo:.2f}, {hi:.2f}] | {_pct(e['simultaneous_coverage'])} |"
            )
    ab = prim.get("abstained")
    if ab:
        md += [
            "",
            f"Raw-OK player-frames where the {CAMERA_TITLES[PRIMARY[0]]} abstains (not in the headline rows):"
            f" {ab['n']} ({_pct(ab['share'])}, {ab['frames']} frames); P1 coverage counting them as misses:"
            f" {_pct(ab['coverage_as_misses'])}.",
        ]
    md += ["", "Per test game (Mondrian coverage [95 % CI], player-frames):", ""]
    md += ["| Camera | Reference | " + " | ".join(f"Game {g}" for g in res["games"]) + " |"]
    md += ["|---|---|" + "---|" * len(res["games"])]
    for c in CAMERAS:
        for r in REFERENCES:
            e = res["regions"][c][r]
            by_game = {pg["game"]: f"{_pct(pg['coverage'])} {_ci(pg['coverage_ci'])}, n = {pg['n']}"
                     for pg in e.get("per_game", [])}  # fmt: skip
            md.append(
                f"| {CAMERA_TITLES[c]} | {REF_TITLES[r]} | "
                + " | ".join(by_game.get(g, "–") for g in res["games"])
                + " |"
            )
    md += ["", "Coverage by distance tercile (metres per pixel at the feet), Mondrian / constant circle:", ""]
    md += ["| Camera | Reference | Near | Middle | Far |", "|---|---|---|---|---|"]
    for c in CAMERAS:
        for r in REFERENCES:
            e = res["regions"][c][r]
            if e.get("n"):
                cells = [f"{_pct(t['coverage'])} / {_pct(t['coverage_const'])}" for t in e["by_tercile"]]
                md.append(f"| {CAMERA_TITLES[c]} | {REF_TITLES[r]} | " + " | ".join(cells) + " |")
    filled = [(r, res["regions"]["ptz"][r].get("filled")) for r in REFERENCES]
    if any(f for _, f in filled):
        md += [
            "",
            "Frames filled by the smoother (raw calibration failed), coverage Mondrian / constant circle:",
        ]
        for r, f in filled:
            if f:
                md.append(
                    f"- {REF_TITLES[r]}: {_pct(f['coverage'])} / {_pct(f['coverage_const'])} (n = {f['n']})."
                )
    err = res["errors"]
    if err.get("n"):
        md += [
            "",
            f"## Position error (m) on the same {err['n']} player-frames",
            "",
            "| Camera | Labels p50 / p90 / p99 | Line oracle p50 / p90 / p99 |"
            " Smoothed labels p50 / p90 / p99 |",
            "|---|---|---|---|",
        ]
        for c in ("perframe", "ptz"):
            cells = [
                " / ".join(f"{v:.2f}" for v in q) if (q := err.get(f"{c}_{r}")) else "–"
                for r in ("lab", "or", "sm")
            ]
            md.append(f"| {CAMERA_TITLES[c]} | " + " | ".join(cells) + " |")
    md += ["", "## Failure detection (per-frame calibrations)", ""]
    md.append(
        f"Frames with self-consistent lines and a calibration: {det['frames']}; failures (mean distance to"
        " the"
        f" annotated lines > {FAIL_LINE_PX:.0f} px): {det['failures']}."
    )
    if det.get("detectors"):
        lo_hi = det.get("auroc_diff_ci")
        md += [
            f"Boosting − `{det['best_single_dev']}`: {det['auroc_diff']:+.3f}"
            + (f" [{lo_hi[0]:+.3f}, {lo_hi[1]:+.3f}]" if lo_hi else "")
            + f" ({det['auroc_diff_draws']} bootstrap draws with both classes). Best single feature on these"
            f" frames (post hoc, stricter): `{det['best_single_test']}`.",
            "",
            "| Detector | AUROC | Average precision | Failures removed at 5 % / 10 % abstention |",
            "|---|---|---|---|",
        ]
        rows = sorted(det["detectors"].items(), key=lambda kv: -kv[1]["auroc"])
        md += [
            f"| {n} | {d['auroc']:.3f} | {d['ap']:.3f} | {_pct(d['removed_5'])} / {_pct(d['removed_10'])} |"
            for n, d in rows
        ]
        md += [
            "",
            "| Game | Frames | Failure rate | AUROC boosting | AUROC best single (dev) |",
            "|---|---|---|---|---|",
        ]
        for pg in det["per_game"]:
            ab, bs = pg["auroc_boosting"], pg["auroc_best_single"]
            md.append(
                f"| {pg['game']} | {pg['frames']} | {_pct(pg['failure_rate'])} |"
                f" {'–' if ab is None else f'{ab:.3f}'} | {'–' if bs is None else f'{bs:.3f}'} |"
            )
    rs = res["residual"]
    md += ["", "## Residual failures of the smoothed camera", ""]
    if rs.get("frames"):
        md += [
            f"{rs['frames']} frames with self-consistent lines where both cameras are calibrated: above"
            f" {FAIL_LINE_PX:.0f} px, per-frame camera {_pct(rs['fail_share_perframe'])}, smoothed camera"
            f" {_pct(rs['fail_share_ptz'])} (line distance p90 {rs['line_px_p90_perframe']:.1f} and"
            f" {rs['line_px_p90_ptz']:.1f} px). Apart: {rs['filled_frames']} frames filled by the smoother"
            f" (above {FAIL_LINE_PX:.0f} px {_pct(rs['fail_share_filled'])}) and {rs['abstained_frames']}"
            " raw-OK frames where it abstains. Published-position rate: per-frame"
            f" {_pct(rs['published_rate_perframe'])}, smoothed {_pct(rs['published_rate_ptz'])}.",
            "",
        ]
    md += [
        "| Output type | Frames | Above 10 px |",
        "|---|---|---|",
        *[f"| {n} | {v['frames']} | {_pct(v['fail_share'])} |" for n, v in rs["by_type"].items()],
    ]
    kpi = res["kpi"]
    if kpi:
        md += [
            "",
            "## KPI proxy (5 fps, against the smoothed labels, same input player-frames; each camera's runs"
            " break at its own errors above 10 m)",
            "",
            "| Camera | Runs | Distance bias | HSR-distance bias | Speed error RMS / p90 (m/s) |",
            "|---|---|---|---|---|",
        ]
        for c in ("perframe", "ptz"):
            p = kpi.get(c)
            if p is None:
                md.append(f"| {CAMERA_TITLES[c]} | – | – | – | – |")
                continue
            md.append(
                f"| {CAMERA_TITLES[c]} | {p['runs']} | {p['distance_bias']:+.1%} | {p['hsr_bias']:+.1%} |"
                f" {p['speed_rms']:.2f} / {p['speed_p90']:.2f} |"
            )
        p0 = kpi.get("ptz") or kpi.get("perframe")
        if p0:
            md.append(
                f"\nNoise floor: the raw labels' own distance bias against the smoothed labels is"
                f" {p0['raw_distance_bias']:+.1%} (HSR distance {p0['raw_hsr_bias']:+.1%})."
            )
    return "\n".join(md) + "\n"
