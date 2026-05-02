"""
generator.py — Random strategy generation and mutation.

Generates eval-able condition trees from a curated indicator pool.
Always enforces rr_ratio >= sl_mult × 1.5 + 0.1 so dollar RR ≥ 1.5.
"""

from __future__ import annotations

import random
import copy
from strategy import Strategy


# ─── Indicator pool for condition trees ──────────────────────────────────────

# (indicator_name, comparison_type)
# "threshold"  → compare to a numeric constant
# "crossover"  → compare to another indicator
INDICATOR_POOL: list[tuple[str, str]] = [
    # EMAs — crossover comparisons
    ("ema_8",   "crossover"),
    ("ema_13",  "crossover"),
    ("ema_21",  "crossover"),
    ("ema_34",  "crossover"),
    ("ema_55",  "crossover"),
    ("ema_89",  "crossover"),
    ("ema_200", "crossover"),
    # SMAs — crossover
    ("sma_20",  "crossover"),
    ("sma_50",  "crossover"),
    ("sma_200", "crossover"),
    # RSI — threshold
    ("rsi_7",   "threshold"),
    ("rsi_14",  "threshold"),
    ("rsi_21",  "threshold"),
    # MACD — threshold (around zero)
    ("macd_line",   "threshold_macd"),
    ("macd_signal", "threshold_macd"),
    ("macd_hist",   "threshold_macd"),
    # Stochastic — threshold
    ("stoch_k", "threshold"),
    ("stoch_d", "threshold"),
    # ADX — threshold
    ("adx_14",  "threshold_adx"),
    # Volume ratio — threshold
    ("volume_ratio",     "threshold_vol"),
    # Candle body ratio — threshold
    ("candle_body_ratio", "threshold_body"),
    # Bollinger Band comparisons
    ("bb_upper",  "crossover"),
    ("bb_lower",  "crossover"),
    ("bb_middle", "crossover"),
]

# Crossover targets (price-level indicators that can be compared to each other)
PRICE_LEVEL_INDICATORS = [
    "ema_8", "ema_13", "ema_21", "ema_34", "ema_55", "ema_89", "ema_200",
    "sma_20", "sma_50", "sma_200",
    "bb_upper", "bb_middle", "bb_lower",
    "close", "open", "high", "low",
]


# ─── Condition tree generation ───────────────────────────────────────────────

def _random_single_condition() -> str:
    """Generate one comparison expression suitable for eval() on a row dict."""
    ind, ctype = random.choice(INDICATOR_POOL)

    if ctype == "crossover":
        # Compare to another price-level indicator
        targets = [t for t in PRICE_LEVEL_INDICATORS if t != ind]
        other = random.choice(targets)
        op = random.choice([">", "<"])
        return f"{ind} {op} {other}"

    elif ctype == "threshold":
        # RSI / Stochastic style: 0–100
        op = random.choice([">", "<"])
        val = round(random.uniform(15, 85), 1)
        return f"{ind} {op} {val}"

    elif ctype == "threshold_macd":
        op = random.choice([">", "<"])
        val = round(random.uniform(-500, 500), 1)
        return f"{ind} {op} {val}"

    elif ctype == "threshold_adx":
        op = random.choice([">", "<"])
        val = round(random.uniform(15, 50), 1)
        return f"{ind} {op} {val}"

    elif ctype == "threshold_vol":
        op = random.choice([">", "<"])
        val = round(random.uniform(0.5, 3.0), 2)
        return f"{ind} {op} {val}"

    elif ctype == "threshold_body":
        op = random.choice([">", "<"])
        val = round(random.uniform(0.1, 0.9), 2)
        return f"{ind} {op} {val}"

    # Fallback
    return "rsi_14 < 50"


def random_condition_tree() -> str:
    """
    Build a compound boolean expression with 2–5 sub-conditions joined by
    'and' / 'or'.  Returns a valid Python expression string that can be
    eval'd against a row dict.
    """
    n = random.randint(2, 5)
    clauses: list[str] = []
    seen: set[str] = set()

    while len(clauses) < n:
        cond = _random_single_condition()
        # Avoid duplicate clauses
        if cond not in seen:
            seen.add(cond)
            clauses.append(cond)

    # Join with 'and' / 'or' — bias towards 'and' for stricter filters
    parts = [clauses[0]]
    for c in clauses[1:]:
        joiner = random.choice(["and", "and", "or"])  # 2:1 bias for 'and'
        parts.append(joiner)
        parts.append(c)

    expr = " ".join(parts)
    # Wrap each clause in parens for safety
    wrapped = []
    i = 0
    for clause in clauses:
        wrapped.append(f"({clause})")
    joiners = [parts[j] for j in range(1, len(parts), 2)]
    result_parts = [wrapped[0]]
    for k, j in enumerate(joiners):
        result_parts.append(j)
        result_parts.append(wrapped[k + 1])

    return " ".join(result_parts)


# ─── Strategy generation ────────────────────────────────────────────────────

def generate_random_strategy(generation: int) -> Strategy:
    """Create a fully random strategy with valid RR ratio."""
    sl_mult = round(random.uniform(1.0, 4.0), 1)
    min_rr = round(sl_mult * 1.5 + 0.1, 1)
    rr_ratio = round(random.uniform(min_rr, min_rr + 4.0), 1)
    cooldown = random.randint(2, 10)
    atr_gate = round(random.uniform(0.0005, 0.003), 4)

    buy_conditions = random_condition_tree()
    sell_conditions = random_condition_tree()

    return Strategy(
        generation=generation,
        sl_mult=sl_mult,
        rr_ratio=rr_ratio,
        cooldown=cooldown,
        atr_gate=atr_gate,
        buy_conditions=buy_conditions,
        sell_conditions=sell_conditions,
    )


def build_strategy_from_params(params: list[float], generation: int) -> Strategy:
    """Build a strategy from ML-suggested [sl_mult, rr_ratio, cooldown, atr_gate]."""
    sl_mult = round(params[0], 1)
    rr_ratio = round(params[1], 1)
    cooldown = int(round(params[2]))
    atr_gate = round(params[3], 4)

    # Enforce RR constraint
    min_rr = round(sl_mult * 1.5 + 0.1, 1)
    if rr_ratio < min_rr:
        rr_ratio = min_rr

    return Strategy(
        generation=generation,
        sl_mult=sl_mult,
        rr_ratio=rr_ratio,
        cooldown=cooldown,
        atr_gate=atr_gate,
        buy_conditions=random_condition_tree(),
        sell_conditions=random_condition_tree(),
    )


def mutate_strategy(parent: Strategy, generation: int,
                    mutation_rate: float = 0.3) -> Strategy:
    """
    Create a child strategy by randomly tweaking the parent's parameters
    and conditions. Always re-enforces rr_ratio >= sl_mult × 1.5 + 0.1.
    """
    child = Strategy(
        generation=generation,
        sl_mult=parent.sl_mult,
        rr_ratio=parent.rr_ratio,
        cooldown=parent.cooldown,
        atr_gate=parent.atr_gate,
        buy_conditions=parent.buy_conditions,
        sell_conditions=parent.sell_conditions,
    )

    # Mutate numeric params
    if random.random() < mutation_rate:
        child.sl_mult = round(max(1.0, min(4.0,
            child.sl_mult + random.uniform(-0.5, 0.5))), 1)

    if random.random() < mutation_rate:
        child.rr_ratio = round(max(1.0,
            child.rr_ratio + random.uniform(-1.0, 1.0)), 1)

    if random.random() < mutation_rate:
        child.cooldown = max(2, min(10,
            child.cooldown + random.randint(-2, 2)))

    if random.random() < mutation_rate:
        child.atr_gate = round(max(0.0005, min(0.003,
            child.atr_gate + random.uniform(-0.0005, 0.0005))), 4)

    # Mutate conditions — swap one condition for a new random one
    if random.random() < mutation_rate:
        child.buy_conditions = _mutate_condition_string(child.buy_conditions)

    if random.random() < mutation_rate:
        child.sell_conditions = _mutate_condition_string(child.sell_conditions)

    # ENFORCE RR CONSTRAINT — NON-NEGOTIABLE
    min_rr = round(child.sl_mult * 1.5 + 0.1, 1)
    if child.rr_ratio < min_rr:
        child.rr_ratio = min_rr

    return child


def _mutate_condition_string(cond_str: str) -> str:
    """Replace one clause in a condition string, or regenerate entirely."""
    # 30% chance to regenerate entirely
    if random.random() < 0.3:
        return random_condition_tree()

    # Try to split on 'and' / 'or' and replace one clause
    parts = []
    joiners = []
    current = []

    tokens = cond_str.split()
    for tok in tokens:
        if tok in ("and", "or"):
            parts.append(" ".join(current))
            joiners.append(tok)
            current = []
        else:
            current.append(tok)
    if current:
        parts.append(" ".join(current))

    if len(parts) < 2:
        return random_condition_tree()

    # Replace a random clause
    idx = random.randint(0, len(parts) - 1)
    new_clause = f"({_random_single_condition()})"
    parts[idx] = new_clause

    # Reassemble
    result = parts[0]
    for j, p in zip(joiners, parts[1:]):
        result += f" {j} {p}"

    return result
