"""Tests for ``medal_predictor.features`` — the model feature matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from medal_predictor.features import build_features

# Synthetic columns that must never reach the model matrix.
DROPPED = {
    "bodyFat",
    "heartRateVariability",
    "vo2Max",
    "bloodOxygen",
    "injurySeverityScore",
    "fitness_score",
    "risk_score",
}


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    gender = np.where(np.arange(n) % 2 == 0, "Men", "Women")
    # Men taller/heavier on average — the whole point of stratifying.
    height = np.where(gender == "Men", 180, 168) + rng.normal(0, 7, n)
    weight = np.where(gender == "Men", 78, 62) + rng.normal(0, 8, n)
    return pd.DataFrame(
        {
            "gender": gender,
            "height": height,
            "weight": weight,
            "gdp": rng.uniform(1e11, 2e12, n),
            "population": rng.uniform(1e6, 1e8, n),
            "participation_year": rng.choice([2008, 2012, 2016, 2020], n),
            "dob": rng.choice(["1990-01-01", "1985-06-15", "1995-03-20"], n),
            "has_medal": rng.integers(0, 2, n),
            "bodyFat": rng.uniform(5, 25, n),  # synthetic — must be dropped
        }
    )


def test_zscore_mean_is_zero_within_each_gender() -> None:
    x, _ = build_features(_frame())
    df = x.copy()
    df["gender_Men"] = x["gender_Men"]
    for col in ("height_zscore", "weight_zscore"):
        for mask in (x["gender_Men"], ~x["gender_Men"]):
            assert abs(df.loc[mask, col].mean()) < 1e-9


def test_synthetic_features_are_dropped() -> None:
    x, _ = build_features(_frame())
    assert DROPPED.isdisjoint(x.columns)


def test_expected_feature_set_and_target() -> None:
    df = _frame()
    x, y = build_features(df)
    assert set(x.columns) == {
        "height",
        "weight",
        "bmi",
        "gdp_per_capita",
        "age",
        "height_zscore",
        "weight_zscore",
        "gender_Men",
        "gender_Women",
    }
    assert len(x) == len(df)
    assert y.tolist() == df["has_medal"].tolist()


def test_bmi_is_sane() -> None:
    x, _ = build_features(_frame())
    # Sanity range — chiefly guards the cm->m conversion (a unit bug gives BMI ~0.002).
    assert x["bmi"].between(12, 50).all()
