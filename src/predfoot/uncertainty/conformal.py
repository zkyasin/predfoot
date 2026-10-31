"""Split conformal calibration of the position uncertainty, grouped by clip, with Mondrian bins.

Scores are squared Mahalanobis distances d² = eᵀ Σp⁻¹ e of the position errors under the predicted
covariance. The calibrated 90 % region of a player is then {e : eᵀ Σp⁻¹ e ≤ q̂}, where q̂ is a conformal
quantile of the scores of matches never used for anything else.

Frames of a clip are strongly correlated, so each clip gets the same total weight in the quantile, and the
finite-sample correction uses the number of clips, not the number of player-frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Standard split-conformal quantile: the ⌈(n+1)(1−α)⌉-th smallest score (+inf if that exceeds n)."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= len(s) else float("inf")


def grouped_quantile(scores: ArrayLike, groups: ArrayLike, alpha: float) -> float:
    """Quantile of the scores where every group (clip) carries the same total weight.

    The level is raised to (1 − α)(1 + 1/G) for G groups, the conformal correction at the group level.
    """
    s = np.asarray(scores, dtype=np.float64)
    g = np.asarray(groups)
    if len(s) == 0:
        return float("inf")
    uniq, inverse, counts = np.unique(g, return_inverse=True, return_counts=True)
    level = (1 - alpha) * (1 + 1 / len(uniq))
    if level >= 1:
        return float(s.max())
    w = 1.0 / counts[inverse]
    order = np.argsort(s)
    cum = np.cumsum(w[order]) / w.sum()
    return float(s[order][np.searchsorted(cum, level, side="left")])


def group_split(groups: ArrayLike, held_out: ArrayLike) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Boolean masks (calibration, evaluation) that never share a group (match)."""
    g = np.asarray(groups)
    test = np.isin(g, np.asarray(held_out))
    return ~test, test


@dataclass
class MondrianConformal:
    """Per-bin conformal quantiles over a grid of features (for example zoom × distance × visible lines).

    Bins with fewer than ``min_count`` scores or ``min_groups`` clips fall back to the pooled quantile.
    """

    alpha: float = 0.1
    edges: dict[str, list[float]] = field(default_factory=dict)
    quantiles: dict[str, float] = field(default_factory=dict)  # bin key "i-j-k" -> q̂
    counts: dict[str, int] = field(default_factory=dict)
    pooled: float = float("inf")

    @classmethod
    def fit(
        cls,
        scores: ArrayLike,
        groups: ArrayLike,
        features: dict[str, ArrayLike],
        n_bins: dict[str, int],
        alpha: float = 0.1,
        min_count: int = 200,
        min_groups: int = 3,
    ) -> MondrianConformal:
        s = np.asarray(scores, dtype=np.float64)
        g = np.asarray(groups)
        edges = {
            name: np.quantile(
                np.asarray(features[name], dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1]
            ).tolist()
            for name, n in n_bins.items()
        }
        model = cls(alpha=alpha, edges=edges, pooled=grouped_quantile(s, g, alpha))
        keys = model._keys(features)
        for key in np.unique(keys):
            sel = keys == key
            model.counts[str(key)] = int(sel.sum())
            if sel.sum() >= min_count and len(np.unique(g[sel])) >= min_groups:
                model.quantiles[str(key)] = grouped_quantile(s[sel], g[sel], alpha)
        return model

    def _keys(self, features: dict[str, ArrayLike]) -> NDArray[np.str_]:
        idx = [
            np.searchsorted(np.asarray(e), np.asarray(features[n], dtype=np.float64))
            for n, e in self.edges.items()
        ]
        if not idx:
            return np.array(["all"] * len(np.asarray(next(iter(features.values())))))
        return np.array(["-".join(map(str, t)) for t in zip(*idx, strict=True)])

    def quantile_for(self, features: dict[str, ArrayLike]) -> FloatArray:
        """q̂ for each sample, from its bin or the pooled quantile."""
        return np.array([self.quantiles.get(str(k), self.pooled) for k in self._keys(features)])

    def to_json(self) -> str:
        return json.dumps(
            {"alpha": self.alpha, "edges": self.edges, "quantiles": self.quantiles, "counts": self.counts,
             "pooled": self.pooled},
            indent=2,
        )  # fmt: skip

    @classmethod
    def from_json(cls, text: str) -> MondrianConformal:
        d: dict[str, Any] = json.loads(text)
        return cls(
            alpha=d["alpha"],
            edges=d["edges"],
            quantiles=d["quantiles"],
            counts=d["counts"],
            pooled=d["pooled"],
        )


def frame_scores(scores: ArrayLike, frame_ids: ArrayLike) -> tuple[FloatArray, NDArray[Any]]:
    """Simultaneous variant: the maximum score of each frame, so q̂ covers all its players at once."""
    s = np.asarray(scores, dtype=np.float64)
    f = np.asarray(frame_ids)
    uniq, inverse = np.unique(f, return_inverse=True)
    out = np.full(len(uniq), -np.inf)
    np.maximum.at(out, inverse, s)
    return out, uniq
