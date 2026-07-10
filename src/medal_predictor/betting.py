"""Quant framing: treat model probabilities as bets.

Expected value, Kelly sizing and a simple PnL simulation — the project's
quant-research differentiator. The idea: a classifier probability is only useful
if it *disagrees* with the market. Here the "market" is the 10.8% base medal
rate, so the model has an edge exactly where it rates an athlete above (or below)
that baseline.
"""

from __future__ import annotations

import numpy as np

# The marginal medal rate over 2008–2020 — the naive "market" line every athlete
# is priced at if you know nothing about them.
BASELINE_MEDAL_RATE = 0.108


def implied_odds(p: float) -> float:
    """Convert a model probability into fair decimal odds (``1 / p``).

    Decimal odds of 1/p mean a 1-unit winning stake returns ``1/p`` units total
    (``1/p - 1`` profit), which is exactly break-even at probability ``p``.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    return 1.0 / p


def expected_value(p_model: float, p_market: float, stake: float = 1.0) -> float:
    """Expected value of taking a market line at ``p_market`` given ``p_model``.

    The market offers fair decimal odds ``1 / p_market``. Betting ``stake`` on the
    outcome wins ``stake * (1/p_market - 1)`` with probability ``p_model`` and
    loses ``stake`` otherwise, which simplifies to::

        EV = stake * (p_model / p_market - 1)

    so EV is positive exactly when the model thinks the event is more likely than
    the market price implies (``p_model > p_market``) — a value bet.
    """
    if not 0.0 < p_market <= 1.0:
        raise ValueError(f"p_market must be in (0, 1], got {p_market}")
    return stake * (p_model / p_market - 1.0)


def kelly_fraction(p_model: float, decimal_odds: float) -> float:
    """Kelly-criterion fraction of bankroll to stake.

    ``f* = (b * p - q) / b`` where ``b = decimal_odds - 1`` (net odds),
    ``p = p_model`` and ``q = 1 - p``. Clipped at 0: a non-positive Kelly fraction
    means there is no edge, so the correct stake is nothing.
    """
    b = decimal_odds - 1.0
    if b <= 0.0:
        return 0.0
    q = 1.0 - p_model
    return max(0.0, (b * p_model - q) / b)


def simulate_betting(
    predictions: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    market_prob: float = BASELINE_MEDAL_RATE,
) -> dict[str, float]:
    """Running PnL from betting 1 unit on every athlete rated above ``threshold``.

    Each selected athlete is bet at the market's fair decimal odds
    (``1 / market_prob``): a true medallist returns ``1/market_prob - 1`` units of
    profit, a non-medallist loses the 1-unit stake. Reports how many bets were
    placed, total/return-per-bet PnL, the hit rate, and a Sharpe-style
    mean/standard-deviation ratio of per-bet PnL.
    """
    predictions = np.asarray(predictions, dtype=float)
    y_true = np.asarray(y_true)
    selected = predictions > threshold
    n_bets = int(selected.sum())
    if n_bets == 0:
        return {"n_bets": 0.0, "total_pnl": 0.0, "roi": 0.0, "hit_rate": 0.0, "sharpe": 0.0}

    payout = implied_odds(market_prob) - 1.0  # net profit on a winning bet
    wins = y_true[selected] == 1
    pnl = np.where(wins, payout, -1.0)
    std = float(pnl.std())
    return {
        "n_bets": float(n_bets),
        "total_pnl": float(pnl.sum()),
        "roi": float(pnl.mean()),  # average PnL per unit staked
        "hit_rate": float(wins.mean()),
        "sharpe": float(pnl.mean() / std) if std > 0 else 0.0,
    }
