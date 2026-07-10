"""Leakage-safe historical features.

Rewrites the country/sport medal-rate logic from ``notebook_2_eda_features.ipynb``
so a given Olympic year only ever sees data from *earlier* years.

The original notebook computed each country's and sport's medal rate over the
*whole* dataset, then used those rates as features — so a 2016 athlete's
"country medal rate" already baked in 2020 results. That is target leakage: the
feature peeks at the future it is meant to predict, inflating offline scores.
"""

from __future__ import annotations

import pandas as pd

_RATE_COLUMNS = [
    "country_medal_rate",
    "country_avg_ranking",
    "sport_medal_rate",
    "sport_avg_ranking",
]
# Sentinel for a country/sport with no prior-Olympics history (e.g. the earliest
# year, or a first-time entrant). A tree model can learn this "unknown" marker.
_NO_HISTORY_FILL = 0.0


def compute_past_only_rates(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Attach leakage-safe country/sport rates to the athlete-rows of ``year``.

    Rates are computed from rows where ``participation_year < year`` only, so
    e.g. 2016 features never see 2020 (or even 2016 itself). Returns the subset
    of ``df`` for ``year`` with the four rate columns added.
    """
    past = df[df["participation_year"] < year]
    current = df[df["participation_year"] == year].copy()

    country = past.groupby("country").agg(
        country_medal_rate=("has_medal", "mean"),
        country_avg_ranking=("ranking", "mean"),
    )
    sport = past.groupby("sportName").agg(
        sport_medal_rate=("has_medal", "mean"),
        sport_avg_ranking=("ranking", "mean"),
    )

    current = current.merge(country, on="country", how="left")
    current = current.merge(sport, on="sportName", how="left")
    current[_RATE_COLUMNS] = current[_RATE_COLUMNS].fillna(_NO_HISTORY_FILL)
    return current.reset_index(drop=True)
