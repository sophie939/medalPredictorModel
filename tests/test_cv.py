"""Tests for ``medal_predictor.cv`` — the walk-forward Olympic splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from medal_predictor.cv import WalkForwardOlympicsSplit


def _years() -> pd.Series:
    # 3 rows per Olympics across four Games.
    return pd.Series([2008, 2008, 2008, 2012, 2012, 2012, 2016, 2016, 2016, 2020, 2020, 2020])


def test_yields_one_fold_per_year_after_the_first() -> None:
    splits = list(WalkForwardOlympicsSplit().split(_years()))
    # Earliest year (2008) is never a test fold; 4 years -> 3 folds.
    assert len(splits) == 3


def test_train_strictly_precedes_test() -> None:
    years = _years()
    yr = np.asarray(years)
    for train_idx, test_idx in WalkForwardOlympicsSplit().split(years):
        max_train_year = yr[train_idx].max()
        test_year = yr[test_idx][0]
        assert (yr[test_idx] == test_year).all()
        assert max_train_year < test_year
        # No overlap between train and test rows.
        assert set(train_idx).isdisjoint(test_idx)


def test_train_set_accumulates() -> None:
    sizes = [len(tr) for tr, _ in WalkForwardOlympicsSplit().split(_years())]
    # 2008 (3) -> 2008+2012 (6) -> 2008+2012+2016 (9).
    assert sizes == [3, 6, 9]


def test_handles_unsorted_input() -> None:
    years = pd.Series([2016, 2008, 2020, 2012])
    test_years = [np.asarray(years)[te][0] for _, te in WalkForwardOlympicsSplit().split(years)]
    assert test_years == [2012, 2016, 2020]
