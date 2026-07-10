"""Model training.

Builds and fits the prediction pipeline (scaler + gradient-boosted classifier),
with hyper-parameter search driven by the walk-forward splitter in ``cv.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from medal_predictor.cv import WalkForwardOlympicsSplit

# Defaults are the ``GradientBoostingClassifier`` keys (no ``clf__`` prefix); they
# are re-prefixed onto the pipeline step below.
_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "min_samples_split": 2,
}


def build_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """Return an unfitted StandardScaler + GradientBoostingClassifier pipeline."""
    p = {**_DEFAULT_PARAMS, **(params or {})}
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=0, **p)),
        ]
    )


def train(X: pd.DataFrame, y: pd.Series, params: dict[str, Any] | None = None) -> Pipeline:
    """Fit and return a StandardScaler + GradientBoostingClassifier pipeline."""
    return build_pipeline(params).fit(X, y)


def search_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    grid: dict[str, list[Any]],
    scoring: str = "average_precision",
) -> GridSearchCV:
    """Grid-search pipeline hyper-parameters using walk-forward CV.

    ``grid`` keys are bare estimator params (e.g. ``"max_depth"``); they are
    prefixed to ``"clf__..."`` for the pipeline. CV folds come from
    :class:`WalkForwardOlympicsSplit` so no future Olympic informs a past one.
    Defaults to ``average_precision`` (PR-AUC) — the right scorer for this
    imbalanced target, not accuracy.
    """
    splitter = WalkForwardOlympicsSplit()
    cv = list(splitter.split(years))
    prefixed = {f"clf__{k}": v for k, v in grid.items()}
    search = GridSearchCV(build_pipeline(), prefixed, scoring=scoring, cv=cv)
    search.fit(np.asarray(X), np.asarray(y))
    return search
