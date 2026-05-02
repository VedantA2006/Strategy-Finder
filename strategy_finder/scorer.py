"""
scorer.py — Professional multi-factor scoring with strict hard gates.

A strategy MUST pass every gate to receive a positive score.
Evaluated after the backtest and robustness pipeline complete.
"""

from __future__ import annotations

from strategy import Strategy


def score(metrics: dict, strategy: Strategy) -> float:
    """
    Score a strategy based on its backtest metrics and robustness results.
    Returns -999 for disqualified strategies, otherwise a composite score.
    """
    # ── Hard gates — instant disqualification ────────────────────────────
    if metrics.get("dollar_rr", 0) < 1.5:
        return -999.0
    if metrics.get("max_drawdown", 100) > 8.0:
        return -999.0
    if metrics.get("avg_trades_per_month", 0) < 3.0:
        return -999.0
    
    wr = metrics.get("win_rate", 0)
    if wr < 50.0 or wr > 78.0:  # Below 50% or overfitted high win rate disqualified
        return -999.0
        
    if metrics.get("total_trades", 0) < 30:
        return -999.0
        
    if strategy.walk_forward_ratio < 0.5:
        return -999.0
        
    if strategy.mc_drawdown_p95 > 15.0:
        return -999.0
        
    if strategy.parameter_sensitivity > 0.30:  # e.g., 0.35 means 35% drop
        return -999.0

    # ── Monthly CAGR gate ─────────────────────────────────────────────────
    cagr = metrics.get("cagr", 0)
    monthly_cagr = ((1 + cagr / 100) ** (1/12) - 1) * 100
    if monthly_cagr < 15.0:
        return -999.0

    # ── Composite score ──────────────────────────────────────────────────
    max_dd = metrics.get("max_drawdown", 1.0)
    if max_dd == 0: max_dd = 1.0
    
    calmar_ratio = cagr / max_dd
    
    # Regime consistency: punish high variance between bull/bear/sideways WR
    regime_wr = [strategy.regime_bull_wr, strategy.regime_bear_wr, strategy.regime_sideways_wr]
    avg_regime_wr = sum(regime_wr) / 3 if any(regime_wr) else 0
    # Penalty if max diff is high (e.g. only works in bull)
    variance_penalty = max(regime_wr) - min(regime_wr) if any(regime_wr) else 0
    # Map to a score 0-10
    regime_consistency = max(0, 10 - (variance_penalty / 5))

    composite = (
        cagr                        * 0.25 +
        calmar_ratio                * 0.20 +
        metrics.get("profit_factor", 0) * 0.15 +
        metrics.get("sharpe", 0)    * 0.15 +
        strategy.walk_forward_ratio * 10.0 * 0.10 +  # scale ratio 0-2 -> 0-20
        regime_consistency          * 0.10 +
       -strategy.mc_drawdown_p95    * 0.05
    )

    return round(composite, 4)
