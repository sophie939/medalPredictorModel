"""Tests for ``medal_predictor.model`` — the pipeline and walk-forward search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from medal_predictor.model import build_pipeline, search_hyperparams, train


def _xy(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    # Target loosely follows feature a, so the model has something to learn.
    y = pd.Series((x["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    return x, y


def test_build_pipeline_has_scaler_and_classifier() -> None:
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    assert list(dict(pipe.steps)) == ["scaler", "clf"]


def test_train_fits_and_predicts_proba() -> None:
    x, y = _xy()
    model = train(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_params_override_reaches_classifier() -> None:
    model = train(*_xy(), params={"n_estimators": 17})
    assert model.named_steps["clf"].n_estimators == 17


def test_search_uses_walk_forward_and_returns_best() -> None:
    x, y = _xy(160)
    years = pd.Series(np.repeat([2008, 2012, 2016, 2020], 40))
    search = search_hyperparams(x, y, years, grid={"max_depth": [1, 2]})
    # 4 Olympic years -> 3 walk-forward folds.
    assert search.n_splits_ == 3
    assert search.best_params_["clf__max_depth"] in (1, 2)
