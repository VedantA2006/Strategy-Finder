"""
generator.py — Massive strategy generation and genetic algorithm operators.

Provides a massive ingredient pool for random condition trees, covering
multiple timeframes, momentum lags, session filters, and volume profiles.

Enforces valid RR ratios and biases BUY trees towards bullish logic
(e.g., fast EMA > slow EMA, RSI > 50).
"""

from __future__ import annotations

import random
import re
from typing import Literal

from strategy import Strategy


# ─── Massive Indicator Pool ─────────────────────────────────────────────────

TIMEFRAMES = ["15m", "1h", "4h", "1d"]

def _rnd_tf() -> str:
    # Bias slightly towards 15m and 1h
    return random.choice(["15m", "15m", "1h", "1h", "4h", "1d"])

def _rnd_fast_ema() -> int:
    return random.choice([8, 13, 21])

def _rnd_slow_ema() -> int:
    return random.choice([34, 55, 89, 200])

def _random_single_condition(direction: Literal["buy", "sell"] = "buy") -> str:
    """Generate one comparison expression from the massive ingredient pool."""
    category = random.choices(
        ["ema_crossover", "rsi_thresh", "macd_thresh", "stoch_thresh", 
         "adx_thresh", "bb_crossover", "momentum_roc", "candle_struct", 
         "volume_profile", "price_struct", "regime_filter", "session_filter",
         "supertrend", "vwap_dev", "cmf", "williams_r"],
        weights=[15, 12, 10, 5, 5, 10, 8, 8, 8, 5, 5, 3, 5, 5, 5, 5]
    )[0]

    tf = _rnd_tf()
    pfx = f"tf_{tf}_"

    if category == "ema_crossover":
        # Directional bias: Buy -> Fast > Slow, Sell -> Fast < Slow
        fast = _rnd_fast_ema()
        slow = _rnd_slow_ema()
        op = ">" if direction == "buy" else "<"
        if random.random() < 0.2: # Sometimes invert for mean reversion
            op = "<" if direction == "buy" else ">"
        return f"{pfx}ema_{fast} {op} {pfx}ema_{slow}"

    elif category == "rsi_thresh":
        period = random.choice([7, 14, 21])
        if direction == "buy":
            # Buy logic: either oversold (<30) or strong trend (>50)
            if random.random() < 0.5:
                val = random.randint(20, 40)
                op = "<"
            else:
                val = random.randint(50, 70)
                op = ">"
        else:
            if random.random() < 0.5:
                val = random.randint(60, 80)
                op = ">"
            else:
                val = random.randint(30, 50)
                op = "<"
        return f"{pfx}rsi_{period} {op} {val}"

    elif category == "macd_thresh":
        # Bias around zero, +/- 1000 for BTC scale tolerance
        # A simpler check: > 0 or < 0
        comp = random.choice(["macd_line", "macd_signal", "macd_hist"])
        val = random.uniform(-500, 500)
        op = ">" if direction == "buy" else "<"
        if random.random() < 0.3: val = 0
        return f"{pfx}{comp} {op} {val}"

    elif category == "stoch_thresh":
        comp = random.choice(["stoch_k", "stoch_d"])
        val = random.randint(15, 85)
        op = random.choice([">", "<"])
        return f"{pfx}{comp} {op} {val}"

    elif category == "adx_thresh":
        val = random.randint(20, 40)
        return f"{pfx}adx_14 > {val}"

    elif category == "bb_crossover":
        op = "<" if direction == "buy" else ">"
        # Buy: close < lower band (mean reversion) OR close > middle (trend)
        target = random.choice(["bb_lower", "bb_middle", "bb_upper"])
        if target == "bb_lower" and direction == "buy": op = "<"
        if target == "bb_middle" and direction == "buy": op = ">"
        return f"{pfx}close {op} {pfx}{target}"

    elif category == "momentum_roc":
        # macd_hist > prev_1_macd_hist (turning up)
        ind = random.choice(["rsi_14", "macd_hist", "ema_21", "stoch_k", "adx_14", "cci_20"])
        lag = random.choice(["prev_1", "prev_3"])
        op = ">" if direction == "buy" else "<"
        return f"{pfx}{ind} {op} {pfx}{lag}_{ind}"

    elif category == "candle_struct":
        sub = random.choice(["is_bullish", "is_bearish", "consec_bullish_2", "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "is_engulfing_bull", "is_engulfing_bear", "is_hammer", "is_shooting_star"])
        if sub in ["is_bullish", "is_bearish", "is_engulfing_bull", "is_engulfing_bear", "is_hammer", "is_shooting_star"]:
            val = 1.0 if direction == "buy" and "bull" in sub or sub == "is_hammer" else 0.0
            if "bear" in sub or sub == "is_shooting_star": val = 1.0 if direction == "sell" else 0.0
            return f"{pfx}{sub} == 1"
        elif sub == "consec_bullish_2":
            return f"{pfx}{sub} == 2"
        else:
            op = ">" if random.random() < 0.5 else "<"
            val = round(random.uniform(0.1, 0.8), 2)
            return f"{pfx}{sub} {op} {val}"

    elif category == "volume_profile":
        sub = random.choice(["volume_ratio", "volume_expanding", "obv_slope_5"])
        if sub == "volume_ratio":
            return f"{pfx}volume_ratio > {round(random.uniform(1.2, 2.5), 1)}"
        elif sub == "volume_expanding":
            return f"{pfx}volume_expanding == 1"
        elif sub == "obv_slope_5":
            op = ">" if direction == "buy" else "<"
            return f"{pfx}obv_slope_5 {op} 0"

    elif category == "price_struct":
        # Breakouts / pullbacks
        n = random.choice([10, 20, 50])
        sub = random.choice([f"high_{n}", f"low_{n}", "dist_from_52w_high"])
        if "dist" in sub:
            op = "<" if direction == "buy" else ">"
            return f"{pfx}dist_from_52w_high {op} {random.randint(5, 50)}"
        else:
            op = ">" if direction == "buy" else "<"
            return f"{pfx}close {op} {pfx}{sub}"

    elif category == "regime_filter":
        # e.g. trending day, high volatility
        sub = random.choice(["ema_200_slope", "bb_width", "rsi_14"])
        if sub == "ema_200_slope":
            op = ">" if direction == "buy" else "<"
            return f"tf_1d_ema_200_slope {op} 0"
        elif sub == "bb_width":
            return f"tf_4h_bb_width > {round(random.uniform(0.05, 0.2), 2)}"
        else: # 1d rsi
            op = ">" if direction == "buy" else "<"
            return f"tf_1d_rsi_14 {op} 50"

    elif category == "session_filter":
        # 8 = 8AM UTC (London open), 14 = 2PM UTC (NY open)
        start = random.randint(6, 14)
        end = start + random.randint(4, 8)
        return f"(tf_15m_hour_utc >= {start} and tf_15m_hour_utc <= {end})"

    elif category == "supertrend":
        op = ">" if direction == "buy" else "<"
        return f"{pfx}close {op} {pfx}supertrend_10_3"

    elif category == "vwap_dev":
        op = "<" if direction == "buy" else ">"  # mean reversion
        if random.random() < 0.5: op = ">" if direction == "buy" else "<" # breakout
        val = round(random.uniform(0.01, 0.05), 3)
        op_val = f"-{val}" if op == "<" else f"{val}"
        return f"{pfx}vwap_dev {op} {op_val}"

    elif category == "cmf":
        op = ">" if direction == "buy" else "<"
        val = round(random.uniform(0.05, 0.2), 2)
        if op == "<": val = -val
        return f"{pfx}cmf_20 {op} {val}"

    elif category == "williams_r":
        op = "<" if direction == "buy" else ">"
        val = random.randint(-90, -80) if direction == "buy" else random.randint(-20, -10)
        return f"{pfx}willr_14 {op} {val}"

    # Fallback
    return f"{pfx}rsi_14 {'<' if direction == 'buy' else '>'} 50"


def random_condition_tree(direction: Literal["buy", "sell"] = "buy") -> str:
    """Build a compound boolean expression with 2-6 sub-conditions."""
    n = random.randint(2, 6)
    clauses = set()
    
    # Try multiple times to get unique clauses
    for _ in range(n * 2):
        if len(clauses) >= n:
            break
        clauses.add(_random_single_condition(direction))
        
    clauses_list = list(clauses)
    if not clauses_list:
        clauses_list = [_random_single_condition(direction)]

    # Join with 'and' / 'or'
    parts = [f"({clauses_list[0]})"]
    for c in clauses_list[1:]:
        joiner = random.choice(["and", "and", "or"])  # bias towards 'and'
        parts.append(joiner)
        parts.append(f"({c})")

    return " ".join(parts)


# ─── Genetic Operators ───────────────────────────────────────────────────────

def generate_random_strategy(generation: int, asset: str = "BTCUSDT") -> Strategy:
    """Create a fully random strategy."""
    sl_mult = round(random.uniform(1.0, 4.0), 1)
    min_rr = round(sl_mult * 1.5 + 0.1, 1)
    rr_ratio = round(random.uniform(min_rr, min_rr + 4.0), 1)
    cooldown = random.randint(2, 10)
    atr_gate = round(random.uniform(0.0005, 0.003), 4)
    trail_mult = round(random.uniform(0.0, 2.0), 1) if random.random() < 0.5 else 0.0
    tp1_ratio = round(random.uniform(0.2, 0.8), 2) if random.random() < 0.3 else 0.0

    buy_conditions = random_condition_tree("buy")
    sell_conditions = random_condition_tree("sell")

    return Strategy(
        generation=generation,
        asset=asset,
        sl_mult=sl_mult,
        rr_ratio=rr_ratio,
        cooldown=cooldown,
        atr_gate=atr_gate,
        trail_mult=trail_mult,
        tp1_ratio=tp1_ratio,
        buy_conditions=buy_conditions,
        sell_conditions=sell_conditions,
    )


def build_strategy_from_params(params: list[float], generation: int, asset: str = "BTCUSDT") -> Strategy:
    """Build from ML parameters."""
    sl_mult = round(params[0], 1)
    rr_ratio = round(params[1], 1)
    cooldown = int(round(params[2]))
    atr_gate = round(params[3], 4)
    trail_mult = round(params[4], 1) if params[4] > 0 else 0.0
    tp1_ratio = round(params[5], 2) if params[5] > 0 else 0.0

    min_rr = round(sl_mult * 1.5 + 0.1, 1)
    rr_ratio = max(rr_ratio, min_rr)

    return Strategy(
        generation=generation,
        asset=asset,
        sl_mult=sl_mult,
        rr_ratio=rr_ratio,
        cooldown=cooldown,
        atr_gate=atr_gate,
        trail_mult=trail_mult,
        tp1_ratio=tp1_ratio,
        buy_conditions=random_condition_tree("buy"),
        sell_conditions=random_condition_tree("sell"),
    )


def _split_clauses(cond_str: str) -> tuple[list[str], list[str]]:
    """Splits a tree into clauses and joiners."""
    # This regex matches things inside outer parentheses, assuming simple nesting
    # Since our generator wraps clauses in (), we can split on `) and (` or `) or (`
    parts = re.split(r'\)\s+(and|or)\s+\(', cond_str)
    
    if len(parts) == 1:
        return [cond_str], []
        
    clauses = []
    joiners = []
    
    # parts looks like: ['(cond1', 'and', 'cond2', 'or', 'cond3)']
    clauses.append(parts[0] + ')')
    for i in range(1, len(parts) - 1, 2):
        joiners.append(parts[i])
        clause = '(' + parts[i+1]
        if i + 1 == len(parts) - 1:
            pass # last one already has ) if we matched properly, wait, regex swallowed ) ( 
            # Actually, `re.split` with capturing group keeps the delimiter.
            
    # Better approach for our specific format `(C1) and (C2) or (C3)`
    # We can just tokenize by spaces, but conditions have spaces.
    # We know clauses start with `(` and end with `)`.
    # Let's use a simpler custom parser.
    clauses2 = []
    joiners2 = []
    current_clause = ""
    depth = 0
    tokens = cond_str.split()
    
    for t in tokens:
        if t in ("and", "or") and depth == 0:
            if current_clause:
                clauses2.append(current_clause.strip())
                current_clause = ""
            joiners2.append(t)
        else:
            current_clause += t + " "
            depth += t.count('(')
            depth -= t.count(')')
            
    if current_clause:
        clauses2.append(current_clause.strip())
        
    return clauses2, joiners2


def crossover(parent_a: Strategy, parent_b: Strategy, generation: int) -> Strategy:
    """Two-point crossover on condition strings, inherit params randomly."""
    child = Strategy(
        generation=generation,
        asset=parent_a.asset,
        sl_mult=random.choice([parent_a.sl_mult, parent_b.sl_mult]),
        rr_ratio=random.choice([parent_a.rr_ratio, parent_b.rr_ratio]),
        cooldown=random.choice([parent_a.cooldown, parent_b.cooldown]),
        atr_gate=random.choice([parent_a.atr_gate, parent_b.atr_gate]),
        trail_mult=random.choice([parent_a.trail_mult, parent_b.trail_mult]),
        tp1_ratio=random.choice([parent_a.tp1_ratio, parent_b.tp1_ratio])
    )
    
    # Crossover buy conditions
    a_buy_cl, a_buy_jn = _split_clauses(parent_a.buy_conditions)
    b_buy_cl, b_buy_jn = _split_clauses(parent_b.buy_conditions)
    
    if len(a_buy_cl) > 1 and len(b_buy_cl) > 1:
        # Take first half from A, second half from B
        mid_a = max(1, len(a_buy_cl) // 2)
        mid_b = max(1, len(b_buy_cl) // 2)
        
        new_buy_cl = a_buy_cl[:mid_a] + b_buy_cl[mid_b:]
        new_buy_jn = a_buy_jn[:mid_a] + ["and"] + b_buy_jn[mid_b:]
        new_buy_jn = new_buy_jn[:len(new_buy_cl)-1] # fix length
        
        res = [new_buy_cl[0]]
        for j, c in zip(new_buy_jn, new_buy_cl[1:]):
            res.extend([j, c])
        child.buy_conditions = " ".join(res)
    else:
        child.buy_conditions = random.choice([parent_a.buy_conditions, parent_b.buy_conditions])

    # Crossover sell conditions
    a_sell_cl, a_sell_jn = _split_clauses(parent_a.sell_conditions)
    b_sell_cl, b_sell_jn = _split_clauses(parent_b.sell_conditions)
    
    if len(a_sell_cl) > 1 and len(b_sell_cl) > 1:
        mid_a = max(1, len(a_sell_cl) // 2)
        mid_b = max(1, len(b_sell_cl) // 2)
        
        new_sell_cl = a_sell_cl[:mid_a] + b_sell_cl[mid_b:]
        new_sell_jn = a_sell_jn[:mid_a] + ["and"] + b_sell_jn[mid_b:]
        new_sell_jn = new_sell_jn[:len(new_sell_cl)-1]
        
        res = [new_sell_cl[0]]
        for j, c in zip(new_sell_jn, new_sell_cl[1:]):
            res.extend([j, c])
        child.sell_conditions = " ".join(res)
    else:
        child.sell_conditions = random.choice([parent_a.sell_conditions, parent_b.sell_conditions])

    # Enforce RR
    min_rr = round(child.sl_mult * 1.5 + 0.1, 1)
    if child.rr_ratio < min_rr:
        child.rr_ratio = min_rr

    return child


def mutate_strategy(parent: Strategy, generation: int) -> Strategy:
    """Apply random mutations to a strategy."""
    child = Strategy(
        generation=generation,
        asset=parent.asset,
        sl_mult=parent.sl_mult,
        rr_ratio=parent.rr_ratio,
        cooldown=parent.cooldown,
        atr_gate=parent.atr_gate,
        trail_mult=parent.trail_mult,
        tp1_ratio=parent.tp1_ratio,
        buy_conditions=parent.buy_conditions,
        sell_conditions=parent.sell_conditions,
    )

    # 1. Numeric param nudge
    if random.random() < 0.4:
        child.sl_mult = round(max(1.0, min(5.0, child.sl_mult + random.uniform(-0.3, 0.3))), 1)
    if random.random() < 0.4:
        child.rr_ratio = round(max(1.5, child.rr_ratio + random.uniform(-0.5, 0.5)), 1)
    if random.random() < 0.2:
        child.cooldown = max(2, min(15, child.cooldown + random.randint(-2, 2)))
    if random.random() < 0.2:
        child.atr_gate = round(max(0.0005, min(0.005, child.atr_gate + random.uniform(-0.0005, 0.0005))), 4)
    if random.random() < 0.2:
        child.trail_mult = round(max(0.0, min(3.0, child.trail_mult + random.uniform(-0.2, 0.2))), 1)
    if random.random() < 0.2:
        child.tp1_ratio = round(max(0.0, min(0.8, child.tp1_ratio + random.uniform(-0.1, 0.1))), 2)

    # 2. Mutate conditions
    if random.random() < 0.5:
        child.buy_conditions = _mutate_tree(child.buy_conditions, "buy")
    if random.random() < 0.5:
        child.sell_conditions = _mutate_tree(child.sell_conditions, "sell")

    # Enforce RR
    min_rr = round(child.sl_mult * 1.5 + 0.1, 1)
    if child.rr_ratio < min_rr:
        child.rr_ratio = min_rr

    return child


def _mutate_tree(cond_str: str, direction: Literal["buy", "sell"]) -> str:
    """Mutate a condition tree string."""
    # 10% full regeneration
    if random.random() < 0.10:
        return random_condition_tree(direction)

    clauses, joiners = _split_clauses(cond_str)
    if not clauses:
        return random_condition_tree(direction)

    mutation_type = random.choice(["replace", "insert", "delete", "flip_joiner"])

    if mutation_type == "replace" and clauses:
        idx = random.randint(0, len(clauses) - 1)
        clauses[idx] = f"({_random_single_condition(direction)})"
    
    elif mutation_type == "insert":
        idx = random.randint(0, len(clauses))
        new_clause = f"({_random_single_condition(direction)})"
        clauses.insert(idx, new_clause)
        if idx == 0 and joiners:
            joiners.insert(0, random.choice(["and", "or"]))
        elif idx > 0:
            joiners.insert(idx - 1, random.choice(["and", "or"]))
        else:
            joiners.append("and") # only 1 clause before

    elif mutation_type == "delete" and len(clauses) > 2:
        idx = random.randint(0, len(clauses) - 1)
        clauses.pop(idx)
        if idx < len(joiners):
            joiners.pop(idx)
        elif joiners:
            joiners.pop()

    elif mutation_type == "flip_joiner" and joiners:
        idx = random.randint(0, len(joiners) - 1)
        joiners[idx] = "or" if joiners[idx] == "and" else "and"

    # Reassemble
    if not clauses:
        return random_condition_tree(direction)
        
    res = [clauses[0]]
    for j, c in zip(joiners, clauses[1:]):
        res.extend([j, c])
        
    return " ".join(res)
