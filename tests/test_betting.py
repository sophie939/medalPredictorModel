"""Tests for ``medal_predictor.betting`` — EV, Kelly and the PnL simulation."""

from __future__ import annotations

import numpy as np
import pytest

from medal_predictor.betting import (
    BASELINE_MEDAL_RATE,
    expected_value,
    implied_odds,
    kelly_fraction,
    simulate_betting,
)


def test_implied_odds_is_reciprocal() -> None:
    assert implied_odds(0.25) == pytest.approx(4.0)
    assert implied_odds(1.0) == pytest.approx(1.0)


def test_implied_odds_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        implied_odds(0.0)
    with pytest.raises(ValueError):
        implied_odds(1.5)


def test_expected_value_sign_tracks_edge() -> None:
    # Model more bullish than the market -> positive EV (value bet).
    assert expected_value(0.20, 0.10) > 0
    # Model agrees with the market -> zero EV.
    assert expected_value(0.10, 0.10) == pytest.approx(0.0)
    # Model less bullish than the market -> negative EV.
    assert expected_value(0.05, 0.10) < 0


def test_expected_value_scales_with_stake() -> None:
    assert expected_value(0.20, 0.10, stake=2.0) == pytest.approx(2.0 * expected_value(0.20, 0.10))


def test_kelly_fraction_zero_without_edge() -> None:
    # Fair odds (decimal 1/p) at the true probability -> no edge -> stake nothing.
    assert kelly_fraction(0.10, implied_odds(0.10)) == pytest.approx(0.0)
    # An edge yields a positive, bounded fraction.
    f = kelly_fraction(0.20, implied_odds(0.10))
    assert 0.0 < f <= 1.0


def test_kelly_fraction_never_negative() -> None:
    assert kelly_fraction(0.01, 2.0) == 0.0


def test_simulate_betting_counts_and_pnl() -> None:
    # 4 athletes rated above 0.5; market base rate 0.5 -> even-money payout (+1/-1).
    preds = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    y = np.array([1, 0, 1, 0, 1])
    out = simulate_betting(preds, y, threshold=0.5, market_prob=0.5)
    assert out["n_bets"] == 4.0
    assert out["hit_rate"] == pytest.approx(0.5)
    # 2 wins (+1 each) and 2 losses (-1 each) at even money -> flat.
    assert out["total_pnl"] == pytest.approx(0.0)


def test_simulate_betting_no_qualifying_bets() -> None:
    preds = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 0, 1])
    out = simulate_betting(preds, y, threshold=0.9)
    assert out["n_bets"] == 0.0
    assert out["total_pnl"] == 0.0


def test_baseline_rate_is_sane() -> None:
    assert 0.0 < BASELINE_MEDAL_RATE < 1.0
