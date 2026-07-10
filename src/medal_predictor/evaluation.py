"""Honest evaluation metrics for an imbalanced (~10.8% positive) target.

Reports precision/recall, calibration and bootstrap confidence intervals instead
of leaning on AUC + accuracy alone.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)


def precision_recall_summary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """PR-AUC, average precision, and the max-F1 threshold.

    On a 10.8%-positive target the PR curve is far more informative than ROC:
    it ignores the easy true-negatives that inflate accuracy and AUC-ROC.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # f1 over the thresholded points (drop the trailing recall=0 sentinel point).
    p, r = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    best = int(np.argmax(f1)) if f1.size else 0
    return {
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "max_f1": float(f1[best]) if f1.size else 0.0,
        "max_f1_threshold": float(thresholds[best]) if thresholds.size else 0.0,
    }


def calibration_report(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> dict[str, np.ndarray | float]:
    """Reliability-diagram bin data plus the Brier score.

    ``prob_pred`` is the mean predicted probability per bin; ``prob_true`` the
    observed positive rate. A well-calibrated model has the two roughly equal.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
) -> tuple[float, float]:
    """95% confidence interval for ``metric_fn`` via bootstrap resampling.

    Resamples rows with replacement ``n`` times and returns the (2.5th, 97.5th)
    percentiles. Resamples in which ``metric_fn`` is undefined (e.g. a single
    class present) are skipped. Seeded for reproducibility.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(0)
    size = len(y_true)
    stats: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        try:
            stats.append(float(metric_fn(y_true[idx], y_score[idx])))
        except ValueError:
            continue
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (float(lo), float(hi))
