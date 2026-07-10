"""Tests for ``medal_predictor.evaluation`` — imbalanced-aware metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from medal_predictor.evaluation import (
    bootstrap_metric,
    calibration_report,
    precision_recall_summary,
)


def _data(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y_true = (rng.random(n) < 0.11).astype(int)  # ~11% positive
    # Scores correlate with truth but are noisy.
    y_score = np.clip(0.1 + 0.5 * y_true + rng.normal(scale=0.3, size=n), 0, 1)
    return y_true, y_score


def test_precision_recall_summary_keys_and_ranges() -> None:
    s = precision_recall_summary(*_data())
    assert {"pr_auc", "average_precision", "max_f1", "max_f1_threshold"} == set(s)
    for k in ("pr_auc", "average_precision", "max_f1"):
        assert 0.0 <= s[k] <= 1.0
    # An informative score beats the ~0.11 positive-rate baseline.
    assert s["average_precision"] > 0.11


def test_calibration_report_shapes_and_brier() -> None:
    rep = calibration_report(*_data(), n_bins=5)
    assert rep["prob_true"].shape == rep["prob_pred"].shape
    assert len(rep["prob_true"]) <= 5
    assert 0.0 <= rep["brier"] <= 1.0


def test_bootstrap_ci_brackets_point_estimate_and_orders() -> None:
    y_true, y_score = _data()
    point = average_precision_score(y_true, y_score)
    lo, hi = bootstrap_metric(y_true, y_score, average_precision_score, n=300)
    assert lo < hi
    assert lo <= point <= hi


def test_bootstrap_ci_shrinks_with_more_samples() -> None:
    # Plan's check: a bigger sample gives a tighter interval.
    rng = np.random.default_rng(1)
    small_t = (rng.random(80) < 0.11).astype(int)
    small_s = np.clip(0.1 + 0.5 * small_t + rng.normal(scale=0.3, size=80), 0, 1)
    big_t = (rng.random(2000) < 0.11).astype(int)
    big_s = np.clip(0.1 + 0.5 * big_t + rng.normal(scale=0.3, size=2000), 0, 1)

    lo_s, hi_s = bootstrap_metric(small_t, small_s, average_precision_score, n=300)
    lo_b, hi_b = bootstrap_metric(big_t, big_s, average_precision_score, n=300)
    assert (hi_b - lo_b) < (hi_s - lo_s)
