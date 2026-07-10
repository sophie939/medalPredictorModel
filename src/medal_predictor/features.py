"""Feature engineering.

Gender-aware z-scores, BMI, GDP-per-capita and related features. Drops the
synthetic features (HRV, VO2max, blood oxygen, body fat, injury/fitness/risk scores).
Country/sport medal rates are *not* built here — they leak the target and are
computed leakage-safely in ``leakage.py``.
"""

from __future__ import annotations

import pandas as pd

# Columns carried straight through into the model matrix.
_BASE_FEATURES = ["height", "weight", "bmi", "gdp_per_capita", "age"]
_ZSCORE_FEATURES = ["height_zscore", "weight_zscore"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model feature matrix ``X`` and binary medal target ``y``.

    Expects the gender-parsed frame from ``data.parse_gender``. Returns the
    gender-aware features (height/weight + gender-stratified z-scores, BMI,
    GDP-per-capita, age, and a gender one-hot); the leakage-safe
    country/sport rates are joined on separately.
    """
    out = df.copy()

    out["gdp_per_capita"] = out["gdp"] / out["population"]
    out["bmi"] = out["weight"] / ((out["height"] / 100) ** 2)
    # Real age from date of birth — not the old synthetic 20 + 0.3*(year-2008),
    # which was identical for every athlete in a given Olympics.
    out["age"] = out["participation_year"] - pd.to_datetime(out["dob"]).dt.year

    # Gender-stratified z-scores: standardised within each gender so the mean is
    # ~0 per gender rather than across a physically bimodal mixed population.
    grouped = out.groupby("gender")
    out["height_zscore"] = grouped["height"].transform(lambda s: (s - s.mean()) / s.std())
    out["weight_zscore"] = grouped["weight"].transform(lambda s: (s - s.mean()) / s.std())

    x = out[[*_BASE_FEATURES, *_ZSCORE_FEATURES, "gender"]].copy()
    x = pd.get_dummies(x, columns=["gender"], prefix="gender")
    y = out["has_medal"].copy()
    return x, y
