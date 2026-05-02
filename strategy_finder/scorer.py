"""
scorer.py — Multi-factor scoring with hard disqualification gates.

A strategy MUST pass every gate to receive a positive score.
Dollar RR ≥ 1.5 is enforced here as the final safety net.
"""

from __future__ import annotations


def score(metrics: dict) -> float:
    """
    Score a strategy's backtest metrics.
    Returns -999 for disqualified strategies, otherwise a composite score.
    """
    # ── Hard gates — instant disqualification ────────────────────────────
    if metrics.get("dollar_rr", 0) < 1.5:
        return -999.0
    if metrics.get("max_drawdown", 100) > 10:
        return -999.0
    if metrics.get("avg_trades_per_month", 0) < 2:
        return -999.0
    if metrics.get("win_rate", 0) < 50:
        return -999.0

    # ── Composite score ──────────────────────────────────────────────────
    return round(
        metrics["cagr"]          * 0.30 +
        metrics["profit_factor"] * 0.25 +
        metrics["sharpe"]        * 0.20 +
       -metrics["max_drawdown"]  * 0.15 +
        metrics["win_rate"]      * 0.10,
        4
    )
