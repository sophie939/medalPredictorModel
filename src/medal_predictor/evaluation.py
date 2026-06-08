"""Honest evaluation metrics for an imbalanced (~10.8% positive) target.

Reports precision/recall, calibration and bootstrap confidence intervals instead
of leaning on AUC + accuracy alone.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def precision_recall_summary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """PR-AUC, average precision, and the max-F1 threshold."""
    raise NotImplementedError


def calibration_report(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> dict[str, np.ndarray | float]:
    """Reliability-diagram bin data plus the Brier score."""
    raise NotImplementedError


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
) -> tuple[float, float]:
    """95% confidence interval for ``metric_fn`` via bootstrap resampling."""
    raise NotImplementedError
