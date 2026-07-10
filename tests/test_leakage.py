"""Tests for ``medal_predictor.leakage`` — the target-leakage fix."""

from __future__ import annotations

import pandas as pd

from medal_predictor.leakage import compute_past_only_rates


def _frame() -> pd.DataFrame:
    # Country "GBR" wins nothing in 2012 but sweeps medals in 2020.
    rows = [
        # 2012: GBR present, no medals.
        ("GBR", "Rowing", 0, 8, 2012),
        ("GBR", "Rowing", 0, 9, 2012),
        # 2016: GBR present, the rows we'll attach rates to.
        ("GBR", "Rowing", 1, 1, 2016),
        ("GBR", "Rowing", 0, 5, 2016),
        # 2020: GBR wins everything — must stay invisible to 2016.
        ("GBR", "Rowing", 1, 1, 2020),
        ("GBR", "Rowing", 1, 2, 2020),
    ]
    return pd.DataFrame(
        rows, columns=["country", "sportName", "has_medal", "ranking", "participation_year"]
    )


def test_rates_use_only_strictly_earlier_years() -> None:
    out = compute_past_only_rates(_frame(), 2016)
    # 2016 rate must reflect 2012 only (0 medals / 2 rows = 0.0),
    # NOT 2016 itself and NOT the 2020 medal sweep.
    assert (out["country_medal_rate"] == 0.0).all()
    assert (out["sport_medal_rate"] == 0.0).all()


def test_returns_only_current_year_rows() -> None:
    out = compute_past_only_rates(_frame(), 2016)
    assert (out["participation_year"] == 2016).all()
    assert len(out) == 2


def test_earliest_year_has_no_history_sentinel() -> None:
    out = compute_past_only_rates(_frame(), 2012)
    # Nothing precedes 2012, so every rate falls back to the sentinel.
    for col in ("country_medal_rate", "country_avg_ranking", "sport_medal_rate"):
        assert (out[col] == 0.0).all()


def test_rate_reflects_accumulated_past() -> None:
    # By 2020, GBR has 1 medal across 4 prior rows (2012: 0,0; 2016: 1,0) = 0.25.
    out = compute_past_only_rates(_frame(), 2020)
    assert out["country_medal_rate"].iloc[0] == 0.25
