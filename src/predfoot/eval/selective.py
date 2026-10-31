"""Selective prediction: how much error is removed by abstaining on the riskiest frames."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def risk_coverage(risk: ArrayLike, loss: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Keep frames from the least to the most risky: returns (coverage, mean loss of the kept frames)."""
    r = np.asarray(risk, dtype=np.float64)
    loss_arr = np.asarray(loss, dtype=np.float64)
    order = np.argsort(r, kind="stable")
    kept_loss = np.cumsum(loss_arr[order]) / np.arange(1, len(r) + 1)
    coverage = np.arange(1, len(r) + 1) / len(r)
    return coverage, kept_loss


def aurc(risk: ArrayLike, loss: ArrayLike) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, kept = risk_coverage(risk, loss)
    return float(np.mean(kept))


def removed_at(risk: ArrayLike, failure: ArrayLike, abstain_fraction: float) -> float:
    """Share of failures removed by abstaining on the ``abstain_fraction`` riskiest frames."""
    r = np.asarray(risk, dtype=np.float64)
    fail = np.asarray(failure, dtype=bool)
    if fail.sum() == 0:
        return float("nan")
    k = round(abstain_fraction * len(r))
    if k == 0:
        return 0.0
    dropped = np.argsort(r, kind="stable")[::-1][:k]
    return float(fail[dropped].sum() / fail.sum())
