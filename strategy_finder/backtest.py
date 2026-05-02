"""
backtest.py — Vectorized-loop backtester with ATR-based SL/TP.

CRITICAL RULES (the entire reason this system exists):
  • SL = entry ± (atr_14 × sl_mult)         — ALWAYS, no exceptions
  • TP = entry ± (atr_14 × rr_ratio)        — ALWAYS, no exceptions
  • PnL = risk × (tp_dist / sl_dist)        — price-derived, NEVER risk × RR_RATIO
  • Fees = risk × FEE × 2                   — deducted from every trade
  • Risk = 1.9% of current balance          — compounding
  • ZERO hardcoded percentage SLs anywhere
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy import Strategy

# ─── Constants ───────────────────────────────────────────────────────────────
RISK_PCT = 0.010     # 1.0% of balance risked per trade
FEE      = 0.00055   # 0.055% per side (Bybit taker)
WARMUP   = 200       # skip first N bars for indicator warmup


def backtest(df: pd.DataFrame, strategy: Strategy) -> dict | None:
    """
    Run a full backtest of the given strategy over the enriched DataFrame.

    Returns a metrics dict or None if the strategy is invalid / produces
    no meaningful results.
    """
    # ── Validate RR ratio before doing any work ──────────────────────────
    if strategy.rr_ratio / strategy.sl_mult < 1.5:
        return None

    balance: float = 10_000.0
    equity: list[float] = [balance]
    trades: list[dict] = []
    position: dict | None = None
    cooldown_remaining: int = 0

    # Pre-extract numpy arrays for speed
    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    atr14   = df["atr_14"].values
    atr_pct = df["atr_pct"].values

    for i in range(WARMUP, len(df)):
        row_dict: dict | None = None  # lazy — only build if needed

        # ── TICK OPEN POSITION ───────────────────────────────────────────
        if position is not None:
            direction = position["direction"]

            if direction == "long":
                hit_sl = lows[i] <= position["sl"]
                hit_tp = highs[i] >= position["tp"]
            else:  # short
                hit_sl = highs[i] >= position["sl"]
                hit_tp = lows[i] <= position["tp"]

            if hit_tp or hit_sl:
                # SL takes priority when both hit on the same bar
                is_win = hit_tp and (not hit_sl)

                # CRITICAL: PnL derived from actual price distances
                tp_dist = abs(position["tp"] - position["entry"])
                sl_dist = abs(position["entry"] - position["sl"])
                price_rr = tp_dist / sl_dist  # equals rr_ratio/sl_mult exactly

                if is_win:
                    pnl = (balance * RISK_PCT) * price_rr
                else:
                    pnl = -(balance * RISK_PCT)

                # Deduct round-trip fees
                pnl -= (balance * RISK_PCT) * FEE * 2

                balance += pnl
                if balance <= 0:
                    balance = 0.01  # prevent total wipeout for stats

                trades.append({
                    "entry":     position["entry"],
                    "exit":      position["tp"] if is_win else position["sl"],
                    "pnl":       pnl,
                    "win":       is_win,
                    "bar":       i,
                    "direction": direction,
                })
                equity.append(balance)
                position = None
                cooldown_remaining = strategy.cooldown
                continue

        # ── OPEN NEW POSITION ────────────────────────────────────────────
        if position is None and cooldown_remaining <= 0:
            # ATR gate filter
            if atr_pct[i] < strategy.atr_gate:
                cooldown_remaining = max(0, cooldown_remaining - 1)
                continue

            # Build row dict for eval (lazy, only when we actually check signals)
            if row_dict is None:
                row_dict = df.iloc[i].to_dict()

            try:
                buy_signal = eval(
                    strategy.buy_conditions, {"__builtins__": {}}, row_dict
                )
                sell_signal = eval(
                    strategy.sell_conditions, {"__builtins__": {}}, row_dict
                )
            except Exception:
                cooldown_remaining = max(0, cooldown_remaining - 1)
                continue

            entry = closes[i]
            atr_now = atr14[i]

            if buy_signal and not sell_signal:
                # ── LONG ─────────────────────────────────────────────────
                # ATR-BASED SL AND TP — THE ONLY VALID METHOD
                sl = entry - atr_now * strategy.sl_mult
                tp = entry + atr_now * strategy.rr_ratio
                risk = balance * RISK_PCT
                qty = risk / (entry - sl)  # live-equivalent qty
                position = {
                    "entry": entry, "sl": sl, "tp": tp,
                    "direction": "long", "risk": risk, "qty": qty,
                }

            elif sell_signal and not buy_signal:
                # ── SHORT ────────────────────────────────────────────────
                # ATR-BASED SL AND TP — THE ONLY VALID METHOD
                sl = entry + atr_now * strategy.sl_mult
                tp = entry - atr_now * strategy.rr_ratio
                risk = balance * RISK_PCT
                qty = risk / (sl - entry)  # live-equivalent qty
                position = {
                    "entry": entry, "sl": sl, "tp": tp,
                    "direction": "short", "risk": risk, "qty": qty,
                }

        cooldown_remaining = max(0, cooldown_remaining - 1)

    # ── Compute metrics ──────────────────────────────────────────────────
    if len(trades) < 5:
        return None  # not enough trades to be meaningful

    return compute_metrics(trades, equity, strategy)


def compute_metrics(trades: list[dict], equity: list[float],
                    strategy: Strategy) -> dict:
    """
    Derive all performance metrics from trade list and equity curve.
    """
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]

    total_return_pct = ((equity[-1] / equity[0]) - 1) * 100

    # CAGR — based on 5 years of data
    years = 5.0
    if equity[-1] > 0 and equity[0] > 0:
        cagr = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100
    else:
        cagr = -100.0

    win_rate = (len(wins) / len(trades)) * 100 if trades else 0.0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0.0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Max drawdown (peak-to-trough on equity curve, %)
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    drawdowns = (peak - eq) / peak * 100
    max_drawdown = float(np.max(drawdowns))

    # Sharpe ratio (annualized, using daily returns approximation)
    # Each equity point ≈ one trade; approximate daily returns from equity
    eq_series = pd.Series(equity)
    returns = eq_series.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        # Annualize: ~8760 hourly bars/year, but equity updates per-trade
        # Use sqrt(trades_per_year) scaling
        trades_per_year = len(trades) / years
        sharpe = float(
            (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
        )
    else:
        sharpe = 0.0

    # Average trades per month
    total_months = years * 12
    avg_trades_per_month = len(trades) / total_months

    # Dollar RR = mean(winning_pnl) / mean(abs(losing_pnl))
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
    avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 1.0
    dollar_rr = float(avg_win / avg_loss) if avg_loss > 0 else 0.0

    return {
        "total_return_pct":     round(total_return_pct, 2),
        "cagr":                 round(cagr, 2),
        "win_rate":             round(win_rate, 2),
        "profit_factor":        round(profit_factor, 4),
        "max_drawdown":         round(max_drawdown, 2),
        "sharpe":               round(sharpe, 4),
        "avg_trades_per_month": round(avg_trades_per_month, 2),
        "dollar_rr":            round(dollar_rr, 4),
        "total_trades":         len(trades),
        "wins":                 len(wins),
        "losses":               len(losses),
        "score":                0,  # filled by scorer
        "equity_curve":         [round(e, 2) for e in equity],
    }
