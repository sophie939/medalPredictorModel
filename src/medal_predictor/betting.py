"""Quant framing: treat model probabilities as bets.

Expected value, Kelly sizing and a simple PnL simulation — the project's
quant-research differentiator.
"""

from __future__ import annotations

import numpy as np


def implied_odds(p: float) -> float:
    """Convert a model probability into fair decimal odds (1 / p)."""
    raise NotImplementedError


def expected_value(p_model: float, p_market: float, stake: float = 1.0) -> float:
    """Expected value of taking a market line at ``p_market`` given ``p_model``."""
    raise NotImplementedError


def kelly_fraction(p_model: float, decimal_odds: float) -> float:
    """Kelly-criterion fraction of bankroll to stake."""
    raise NotImplementedError


def simulate_betting(
    predictions: np.ndarray, y_true: np.ndarray, threshold: float
) -> dict[str, float]:
    """Running PnL from betting 1 unit on every athlete rated above ``threshold``."""
    raise NotImplementedError
