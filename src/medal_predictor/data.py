"""Data loading and cleaning.

Ports the SQL extraction and gender-parsing logic out of ``notebook_1_setup.ipynb``.
"""

from __future__ import annotations

import pandas as pd


def load_raw(db_path: str) -> pd.DataFrame:
    """Load the raw athlete/event rows from the SQLite database.

    Runs the join across ATHLETE, PARTICIPATES, events, SPORT and country tables
    (see ``notebook_1_setup.ipynb``) and returns one row per athlete-event.
    """
    raise NotImplementedError


def parse_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Infer athlete gender from event names and drop Youth Olympic (YOG) events."""
    raise NotImplementedError
