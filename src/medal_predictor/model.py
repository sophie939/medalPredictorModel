"""Model training.

Builds and fits the prediction pipeline (scaler + gradient-boosted classifier),
with hyper-parameter search driven by the walk-forward splitter in ``cv.py``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline


def train(X: pd.DataFrame, y: pd.Series, params: dict[str, Any] | None = None) -> Pipeline:
    """Fit and return a StandardScaler + GradientBoostingClassifier pipeline."""
    raise NotImplementedError
