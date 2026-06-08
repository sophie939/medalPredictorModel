"""Feature engineering.

Gender-aware z-scores, BMI, GDP-per-capita and related features. Drops the
synthetic features (HRV, VO2max, blood oxygen, body fat, injury/fitness/risk scores).
"""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model feature matrix ``X`` and binary medal target ``y``.

    Returns ~12 gender-aware features (down from the previous 20).
    """
    raise NotImplementedError
