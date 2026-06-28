"""Tests for ``medal_predictor.data`` — the SQL load and gender parser."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from medal_predictor.data import _assign_gender, load_raw, parse_gender

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "olympics.db"

# Expected columns of the raw join, used to guard against accidental query drift.
RAW_COLUMNS = {
    "athleteID",
    "athlete_name",
    "dob",
    "height",
    "weight",
    "country",
    "gdp",
    "population",
    "ranking",
    "participation_year",
    "event_name",
    "sportName",
    "medal_category",
    "has_medal",
}


# --- Pure unit tests: the event-name parser ----------------------------------


@pytest.mark.parametrize(
    ("event_name", "expected"),
    [
        ("Athletics Women's 100m", "Women"),
        ("Athletics Men's 100m", "Men"),
        ("SWIMMING WOMEN'S 200M FREESTYLE", "Women"),  # case-insensitive
        ("Mixed Doubles Badminton", None),  # no clear marker
        ("Youth Olympic Games 5km", None),  # YOG / no marker
    ],
)
def test_assign_gender(event_name: str, expected: str | None) -> None:
    assert _assign_gender(event_name) == expected


def test_assign_gender_prefers_women_over_men_substring() -> None:
    # "men" is a substring of "women"; the women check must win.
    assert _assign_gender("Women's Marathon") == "Women"


def test_parse_gender_drops_unclassified_and_keeps_only_men_women() -> None:
    df = pd.DataFrame(
        {
            "event_name": [
                "Men's Sprint",
                "Women's Sprint",
                "Mixed Relay",  # dropped
                "Youth 1500m",  # dropped
            ],
            "has_medal": [1, 0, 1, 0],
        }
    )

    out = parse_gender(df)

    assert len(out) == 2
    assert set(out["gender"]) == {"Men", "Women"}
    assert out["gender"].notna().all()
    # Index is reset so downstream positional logic is safe.
    assert list(out.index) == [0, 1]


def test_parse_gender_does_not_mutate_input() -> None:
    df = pd.DataFrame({"event_name": ["Men's 100m"], "has_medal": [1]})
    parse_gender(df)
    assert "gender" not in df.columns


# --- Integration tests: the real SQLite database -----------------------------

requires_db = pytest.mark.skipif(not DB_PATH.exists(), reason=f"database not found at {DB_PATH}")


@requires_db
def test_load_raw_returns_expected_columns() -> None:
    df = load_raw(str(DB_PATH))
    assert not df.empty
    assert RAW_COLUMNS.issubset(df.columns)
    # The WHERE clause guarantees these are never null.
    assert df["height"].notna().all()
    assert df["weight"].notna().all()
    assert df["has_medal"].isin([0, 1]).all()


@requires_db
def test_parse_gender_on_real_data_drops_yog_and_balances() -> None:
    raw = load_raw(str(DB_PATH))
    gendered = parse_gender(raw)

    # Some rows are discarded (mixed / Youth Olympic events).
    assert len(gendered) < len(raw)
    assert set(gendered["gender"].unique()) == {"Men", "Women"}

    # The split is roughly balanced — neither gender is a sliver.
    share = gendered["gender"].value_counts(normalize=True)
    assert share["Men"] > 0.30
    assert share["Women"] > 0.30
