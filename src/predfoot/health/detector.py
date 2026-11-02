"""Calibration failure detectors on the per-frame health features: the failure label, out-of-game scores and a
frozen detector stored as plain JSON.

Failure label, chosen before looking at any score (reports/dev_failure_detection.md): the calibration leaves
the annotated pitch lines more than 10 px away on average, on frames whose calibration succeeded and whose
annotated lines are self-consistent (a homography fits them within 3 px). It uses the line annotations only,
never the position labels.

Detectors: each feature alone (its sign set on the training games), logistic regression and gradient boosting.
``scripts/eval_health.py`` scores them leave-one-game-out (:func:`logo_scores`); ``scripts/freeze_prereg.py``
fits them once on every dev game (:meth:`FrozenDetector.fit`) and writes their parameters as JSON, which
:meth:`FrozenDetector.scores` evaluates with numpy alone: no pickle, no dependence on the scikit-learn version
at evaluation time.

Gradient boosting runs with ``early_stopping=False``. scikit-learn's default (``"auto"``) switches early
stopping on above 10,000 training rows, with a random validation split; the leave-one-game-out folds of dev
train on fewer rows (at most 8,975), so the default never triggered there, but a fit on all six dev games
(10,024 frames) would silently have used a different, non-deterministic construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from predfoot.health.features import FEATURES

FloatArray = NDArray[np.float64]
FAIL_LINE_PX = 10.0
DETECTOR_FORMAT = "predfoot-failure-detector/1"
BOOSTING_PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 300,
    "class_weight": "balanced",
    "early_stopping": False,
    "random_state": 0,
}
LOGISTIC_PARAMS: dict[str, Any] = {"class_weight": "balanced", "max_iter": 2000}


def failure_table(frames: pl.DataFrame) -> pl.DataFrame:
    """Frames that enter the failure evaluation, with the label ``fail``: calibration OK, line reprojection
    known, annotated lines self-consistent (``trusted``); fail when the mean line distance exceeds 10 px."""
    df = frames.filter(pl.col("ok") & pl.col("line_px").is_not_nan() & pl.col("trusted"))
    return df.with_columns((pl.col("line_px") > FAIL_LINE_PX).alias("fail"))


def impute(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Missing features replaced by the training medians (0 for a feature never observed in training)."""
    med = training_medians(train)
    return np.where(np.isfinite(train), train, med), np.where(np.isfinite(test), test, med)


def training_medians(train: np.ndarray) -> np.ndarray:
    med = np.nanmedian(train, axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def make_logistic() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(**LOGISTIC_PARAMS))


def make_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**BOOSTING_PARAMS)


def feature_sign(y: np.ndarray, x: np.ndarray) -> float:
    """+1 when larger values mean failure on these rows, −1 otherwise (+1 when ``y`` has one class)."""
    if len(np.unique(y)) == 2 and roc_auc_score(y, x) < 0.5:
        return -1.0
    return 1.0


def logo_scores(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Out-of-game risk scores for every detector, aligned with the rows of ``df``."""
    x = df.select(FEATURES).to_numpy().astype(float)
    y = df["fail"].to_numpy().astype(int)
    games = df["game"].to_numpy()
    scores = {name: np.full(len(df), np.nan) for name in [*FEATURES, "logistic", "boosting"]}
    for g in np.unique(games):
        tr, te = games != g, games == g
        x_tr, x_te = impute(x[tr], x[te])
        for j, name in enumerate(FEATURES):
            scores[name][te] = feature_sign(y[tr], x_tr[:, j]) * x_te[:, j]
        lr = make_logistic()
        lr.fit(x_tr, y[tr])
        scores["logistic"][te] = lr.predict_proba(x_te)[:, 1]
        gb = make_boosting()
        gb.fit(x_tr, y[tr])
        scores["boosting"][te] = gb.predict_proba(x_te)[:, 1]
    return scores


# --- frozen detector ---------------------------------------------------------------------------------------


def _export_trees(gb: HistGradientBoostingClassifier) -> list[dict[str, list[Any]]]:
    trees = []
    for predictors in gb._predictors:
        if len(predictors) != 1:
            raise ValueError("only binary gradient boosting can be frozen")
        nodes = predictors[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits cannot be frozen")
        trees.append(
            {
                "feature": nodes["feature_idx"].astype(int).tolist(),
                "threshold": nodes["num_threshold"].astype(float).tolist(),
                "missing_left": nodes["missing_go_to_left"].astype(bool).tolist(),
                "left": nodes["left"].astype(int).tolist(),
                "right": nodes["right"].astype(int).tolist(),
                "leaf": nodes["is_leaf"].astype(bool).tolist(),
                "value": nodes["value"].astype(float).tolist(),
            }
        )
    return trees


def _tree_values(tree: Mapping[str, list[Any]], x: np.ndarray) -> FloatArray:
    """Leaf value of every row, following scikit-learn's rule on raw features: NaN goes to the side the split
    learnt, otherwise left when the value is at most the threshold."""
    feature = np.asarray(tree["feature"], dtype=np.int64)
    threshold = np.asarray(tree["threshold"], dtype=np.float64)
    missing_left = np.asarray(tree["missing_left"], dtype=bool)
    left = np.asarray(tree["left"], dtype=np.int64)
    right = np.asarray(tree["right"], dtype=np.int64)
    leaf = np.asarray(tree["leaf"], dtype=bool)
    value = np.asarray(tree["value"], dtype=np.float64)
    node = np.zeros(len(x), dtype=np.int64)
    active = ~leaf[node]
    while active.any():
        idx = np.flatnonzero(active)
        nd = node[idx]
        v = x[idx, feature[nd]]
        go_left = np.where(np.isnan(v), missing_left[nd], v <= threshold[nd])
        node[idx] = np.where(go_left, left[nd], right[nd])
        active = ~leaf[node]
    return value[node]


@dataclass
class FrozenDetector:
    """Failure detectors fitted once on every training frame, as plain numbers.

    ``medians``: imputation values; ``signs``: orientation of each feature used alone; ``best_single``: the
    single feature with the highest out-of-game AUROC on the training games (the baseline to beat);
    ``logistic`` and ``boosting``: model parameters (standardisation and coefficients; trees and baseline).
    """

    features: list[str]
    medians: list[float]
    signs: dict[str, float]
    best_single: str
    logistic: dict[str, Any]
    boosting: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pl.DataFrame, provenance: Mapping[str, Any] | None = None) -> FrozenDetector:
        """Fit on every row of a :func:`failure_table` (``game`` for the out-of-game baseline choice)."""
        x = df.select(FEATURES).to_numpy().astype(float)
        y = df["fail"].to_numpy().astype(int)
        if len(np.unique(y)) != 2:
            raise ValueError("the training frames need both failures and successes")
        med = training_medians(x)
        xi = np.where(np.isfinite(x), x, med)
        signs = {name: feature_sign(y, xi[:, j]) for j, name in enumerate(FEATURES)}
        logo = logo_scores(df)
        logo_auroc = {name: float(roc_auc_score(y, s)) for name, s in logo.items()}
        best_single = max(FEATURES, key=lambda name: logo_auroc[name])
        lr = make_logistic()
        lr.fit(xi, y)
        scaler, logit = lr.named_steps["standardscaler"], lr.named_steps["logisticregression"]
        gb = make_boosting()
        gb.fit(xi, y)
        det = cls(
            features=list(FEATURES),
            medians=med.astype(float).tolist(),
            signs=signs,
            best_single=best_single,
            logistic={
                "mean": scaler.mean_.astype(float).tolist(),
                "scale": scaler.scale_.astype(float).tolist(),
                "coef": logit.coef_.astype(float).tolist(),
                "intercept": logit.intercept_.astype(float).tolist(),
                "params": dict(LOGISTIC_PARAMS),
            },
            boosting={
                "baseline": float(np.asarray(gb._baseline_prediction).ravel()[0]),
                "trees": _export_trees(gb),
                "params": dict(BOOSTING_PARAMS),
                "n_iter": int(gb.n_iter_),
            },
            provenance={
                **(provenance or {}),
                "training_frames": len(df),
                "training_failures": int(y.sum()),
                "training_games": sorted(int(g) for g in np.unique(df["game"].to_numpy())),
                "training_clips": int(df["clip"].n_unique()) if "clip" in df.columns else None,
                "dev_logo_auroc": logo_auroc,
            },
        )
        # The frozen parameters must reproduce the fitted models on their own training rows.
        mine = det.scores(df)
        for name, model in (("logistic", lr), ("boosting", gb)):
            ref = model.predict_proba(xi)[:, 1]
            gap = float(np.max(np.abs(mine[name] - ref))) if len(ref) else 0.0
            if gap > 1e-12:
                raise RuntimeError(f"frozen {name} differs from scikit-learn by {gap:.3g}")
        return det

    def _x(self, df: pl.DataFrame) -> np.ndarray:
        x = df.select(self.features).to_numpy().astype(float)
        med = np.asarray(self.medians, dtype=np.float64)
        return np.asarray(np.where(np.isfinite(x), x, med), dtype=np.float64)

    def scores(self, df: pl.DataFrame) -> dict[str, FloatArray]:
        """Risk scores of every detector for the rows of ``df`` (higher means more likely a failure)."""
        x = self._x(df)
        out = {name: self.signs[name] * x[:, j] for j, name in enumerate(self.features)}
        lg = self.logistic
        z = (x - np.asarray(lg["mean"])) / np.asarray(lg["scale"])
        decision = (z @ np.asarray(lg["coef"]).T + np.asarray(lg["intercept"])).ravel()
        out["logistic"] = np.asarray(expit(decision), dtype=np.float64)
        raw = np.zeros(len(x), dtype=np.float64)
        raw += self.boosting["baseline"]
        for tree in self.boosting["trees"]:
            raw += _tree_values(tree, x)
        out["boosting"] = np.asarray(expit(raw), dtype=np.float64)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DETECTOR_FORMAT,
            "features": self.features,
            "medians": self.medians,
            "signs": self.signs,
            "best_single": self.best_single,
            "failure_line_px": FAIL_LINE_PX,
            "logistic": self.logistic,
            "boosting": self.boosting,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FrozenDetector:
        if d.get("format") != DETECTOR_FORMAT:
            raise ValueError(f"not a {DETECTOR_FORMAT} file")
        if list(d["features"]) != list(FEATURES):
            raise ValueError("the detector was frozen with other health features")
        return cls(
            features=list(d["features"]),
            medians=[float(v) for v in d["medians"]],
            signs={k: float(v) for k, v in d["signs"].items()},
            best_single=str(d["best_single"]),
            logistic=dict(d["logistic"]),
            boosting=dict(d["boosting"]),
            provenance=dict(d.get("provenance", {})),
        )
