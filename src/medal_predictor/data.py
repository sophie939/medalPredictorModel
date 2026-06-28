"""Data loading and cleaning.

Ports the SQL extraction and gender-parsing logic out of ``notebook_1_setup.ipynb``.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

# One row per athlete-event. Mirrors the join in ``notebook_1_setup.ipynb`` but
# is parameterised on the DB path and also pulls ``dob`` (needed downstream for
# ``estimated_age`` in ``features.py``). The synthetic athlete columns are kept
# here so EDA can still inspect them; ``features.py`` drops them from the model.
_EXTRACTION_QUERY = """
SELECT DISTINCT
    a.athleteID,
    a.name AS athlete_name,
    a.dob,
    a.height,
    a.weight,
    a.bodyFat,
    a.heartRateVariability,
    a.vo2Max,
    a.bloodOxygen,
    a.injurySeverityScore,
    c.name AS country,
    cd.gdp,
    cd.population,
    cd.year AS country_year,
    p.ranking,
    p.year AS participation_year,
    se.eventName AS event_name,
    s.sportName,
    CASE
        WHEN p.ranking = 1 THEN 'Gold'
        WHEN p.ranking = 2 THEN 'Silver'
        WHEN p.ranking = 3 THEN 'Bronze'
        ELSE 'No Medal'
    END AS medal_category,
    CASE
        WHEN p.ranking <= 3 THEN 1
        ELSE 0
    END AS has_medal
FROM ATHLETE a
JOIN PARTICIPATES p ON a.athleteID = p.athleteID
JOIN SINGLES_EVENT se ON p.eventID = se.eventID
JOIN SPORT s ON se.sportId = s.sportId
JOIN COUNTRY c ON a.noc = c.noc
JOIN COUNTRY_DETAILS cd ON c.noc = cd.noc AND cd.year = p.year
WHERE p.ranking IS NOT NULL
    AND a.height IS NOT NULL
    AND a.weight IS NOT NULL
    AND cd.gdp IS NOT NULL
    AND cd.population IS NOT NULL
ORDER BY p.year, a.athleteID
"""


def load_raw(db_path: str) -> pd.DataFrame:
    """Load the raw athlete/event rows from the SQLite database.

    Runs the join across ATHLETE, PARTICIPATES, SINGLES_EVENT, SPORT, COUNTRY and
    COUNTRY_DETAILS (see ``notebook_1_setup.ipynb``) and returns one row per
    athlete-event.
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(_EXTRACTION_QUERY, conn)


def _assign_gender(event_name: str) -> str | None:
    """Infer gender from an event name, or ``None`` for mixed/youth events.

    ``women`` is checked before ``men`` because ``men`` is a substring of ``women``.
    """
    name = event_name.lower()
    if "women" in name:
        return "Women"
    if "men" in name:
        return "Men"
    return None


def parse_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Infer athlete gender from event names and drop Youth Olympic (YOG) events.

    Adds a ``gender`` column and drops rows whose event name carries no clear
    ``Men``/``Women`` marker (mixed and Youth Olympic events).
    """
    out = df.copy()
    out["gender"] = out["event_name"].apply(_assign_gender)
    return out[out["gender"].notna()].reset_index(drop=True)
