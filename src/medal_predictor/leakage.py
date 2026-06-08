"""Leakage-safe historical features.

Rewrites the country/sport medal-rate logic from ``notebook_2_eda_features.ipynb``
so a given Olympic year only ever sees data from *earlier* years.
"""

from __future__ import annotations

import pandas as pd


def compute_past_only_rates(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Country and sport medal rates computed from rows strictly before ``year``.

    This is the fix for the target-leakage bug: rates for 2016 must not peek at 2020.
    """
    raise NotImplementedError
