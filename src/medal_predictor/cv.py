"""Cross-validation that respects time.

Replaces random k-fold CV with walk-forward splitting by Olympic year, so the
model only ever trains on Olympics that happened before the one it is tested on.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class WalkForwardOlympicsSplit:
    """Yield (train, test) index arrays where train years strictly precede the test year.

    e.g. 2008 -> 2012, then 2008+2012 -> 2016, then 2008+2012+2016 -> 2020.

    The first (earliest) Olympic year is never a test fold — it has no prior year
    to train on. Splits are ordered by year so evaluation reads chronologically.
    """

    def split(self, years: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train/test row-index arrays for each successive Olympic year.

        ``years`` is the per-row Olympic year (positional, aligned to ``X``).
        Yielded indices are positions into ``years`` (0..n-1), usable with
        numpy fancy indexing or ``.iloc``.
        """
        positions = np.arange(len(years))
        year_values = np.asarray(years)
        unique_years = np.sort(np.unique(year_values))
        for test_year in unique_years[1:]:
            train_idx = positions[year_values < test_year]
            test_idx = positions[year_values == test_year]
            yield train_idx, test_idx
