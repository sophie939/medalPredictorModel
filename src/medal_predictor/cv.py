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
    """

    def split(self, years: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train/test row-index arrays for each successive Olympic year."""
        raise NotImplementedError
