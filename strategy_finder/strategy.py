"""
strategy.py — Strategy dataclass with full multi-timeframe, multi-asset support.

Every strategy stores ATR-based SL/TP multipliers. There are ZERO hardcoded
percentage stop-losses anywhere in this system.

SL = entry ± (atr_14 × sl_mult)
TP = entry ± (atr_14 × rr_ratio)
rr_ratio MUST be >= sl_mult × 1.5 — enforced at generation, backtest, and scoring.

Extended fields for:
  - Genetic algorithm (fingerprint, parent tracking)
  - Multi-asset (asset tag)
  - Multi-timeframe (n_timeframes_used)
  - Robustness (walk-forward, MC drawdown, parameter sensitivity, regime scores)
  - Condition complexity metrics
"""

from __future__ import annotations

import json
import hashlib
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Strategy:
    # ── Identity ─────────────────────────────────────────────────────────
    id: str = ""
    name: str = ""
    generation: int = 0
    asset: str = "BTCUSDT"           # which asset this was tested on
    parent_a_id: str = ""
    parent_b_id: str = ""

    # ── ATR-based risk parameters ────────────────────────────────────────
    sl_mult: float = 1.5             # SL distance = atr_14 × sl_mult
    rr_ratio: float = 3.0            # TP distance = atr_14 × rr_ratio; MUST >= sl_mult × 1.5
    cooldown: int = 3                # bars to skip after a trade closes
    atr_gate: float = 0.001          # skip trade if atr_pct < atr_gate
    trail_mult: float = 0.0          # trailing stop mult (0 = disabled)
    tp1_ratio: float = 0.0           # partial exit at this fraction of TP (0 = disabled)

    # ── Condition trees (Python eval-able expressions on row dict) ───────
    buy_conditions: str = ""
    sell_conditions: str = ""

    # ── Metrics filled after backtest + robustness pipeline ──────────────
    metrics: dict = field(default_factory=dict)

    # ── Robustness fields (filled by robustness pipeline) ────────────────
    walk_forward_ratio: float = 0.0       # average WF ratio across 5 windows
    mc_drawdown_p95: float = 100.0        # Monte Carlo 95th percentile max DD
    parameter_sensitivity: float = 1.0    # max score drop when nudging params
    regime_bull_wr: float = 0.0           # win rate in bull regime
    regime_bear_wr: float = 0.0           # win rate in bear regime
    regime_sideways_wr: float = 0.0       # win rate in sideways regime
    validation_cagr: float = 0.0          # CAGR on validation split
    p_value: float = 1.0                  # significance of win rate compared to random shuffle
    is_correlated: bool = False           # if highly correlated to existing top strategies
    holdout_cagr: float = 0.0             # Live readiness CAGR (last 6m)
    holdout_win_rate: float = 0.0         # Live readiness win rate (last 6m)
    holdout_trades: int = 0               # Live readiness trades (last 6m)

    # ── Complexity / structure metadata ──────────────────────────────────
    condition_complexity: int = 0         # total clauses in buy + sell trees
    n_timeframes_used: int = 1            # how many distinct timeframes referenced

    # ── Trade log & monthly returns (JSON blobs, stored in DB) ───────────
    monthly_returns_json: str = "[]"
    trade_log_json: str = "[]"
    deep_stats_json: str = "{}"           # For WF window ratios, original_complexity, etc.

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.name:
            self.name = self.generate_name()
        # Auto-compute complexity
        self._update_complexity()

    # ── Complexity computation ───────────────────────────────────────────
    def generate_name(self) -> str:
        """Auto-generate a human-readable name based on strategy characteristics."""
        name_parts = []
        conds = self.buy_conditions.lower()
        
        # 1. Regime
        if "tf_1d_ema_200_slope > 0" in conds: name_parts.append("Bull")
        elif "tf_1d_ema_200_slope < 0" in conds: name_parts.append("Bear")
        else: name_parts.append("Any")
        
        # 2. Indicator
        if "supertrend" in conds: name_parts.append("STrend")
        elif "engulfing" in conds: name_parts.append("Engulf")
        elif "hammer" in conds: name_parts.append("Hammer")
        elif "stoch" in conds: name_parts.append("Stoch")
        elif "bb_" in conds: name_parts.append("BB")
        elif "macd" in conds: name_parts.append("MACD")
        elif "rsi" in conds: name_parts.append("RSI")
        elif "ema" in conds: name_parts.append("EMA")
        elif "vwap" in conds: name_parts.append("VWAP")
        else: name_parts.append("Mix")
        
        # 3. RR
        if self.rr_ratio < 2.5: name_parts.append("TightRR")
        elif self.rr_ratio <= 4.0: name_parts.append("BalRR")
        else: name_parts.append("WideRR")
        
        import string
        import random
        suffix = "".join(random.choices(string.digits, k=3))
        return "-".join(name_parts) + "-" + suffix

    def _update_complexity(self) -> None:
        """Count total clauses and unique timeframes in condition trees."""
        all_conds = f"{self.buy_conditions} {self.sell_conditions}"
        # Count clauses (split on 'and' / 'or')
        tokens = all_conds.split()
        clause_count = sum(1 for t in tokens if t in ("and", "or")) + 1
        if not all_conds.strip():
            clause_count = 0
        self.condition_complexity = clause_count

        # Count unique timeframes referenced
        tf_pattern = re.compile(r'tf_(15m|1h|4h|1d)_')
        tfs = set(tf_pattern.findall(all_conds))
        # If no tf_ prefix found, assume single timeframe
        self.n_timeframes_used = max(1, len(tfs))

    # ── Fingerprint for diversity tracking ───────────────────────────────
    @property
    def fingerprint(self) -> str:
        """Hash of condition strings for diversity enforcement."""
        raw = f"{self.buy_conditions}||{self.sell_conditions}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    # ── RR validation ────────────────────────────────────────────────────
    @property
    def is_valid_rr(self) -> bool:
        """Dollar RR must be >= 1.5:1 — rr_ratio >= sl_mult × 1.5."""
        return self.rr_ratio / self.sl_mult >= 1.5

    # ── Param vector for ML optimizer (14 features) ──────────────────────
    @property
    def params_vector(self) -> list[float]:
        """14-feature vector for Gaussian Process surrogate model."""
        all_conds = f"{self.buy_conditions} {self.sell_conditions}"
        tokens = all_conds.split()
        n_and = tokens.count("and")
        n_or = tokens.count("or")

        # Count unique indicator names (anything that looks like an indicator)
        ind_pattern = re.compile(r'(?:tf_\w+_)?(?:ema|sma|rsi|macd|stoch|adx|atr|bb|cci|willr|mfi|obv|vwap|volume|body|wick|high|low|hour|day)', re.IGNORECASE)
        unique_indicators = len(set(ind_pattern.findall(all_conds)))

        has_volume = 1.0 if any(v in all_conds for v in ["volume", "obv", "mfi", "vwap"]) else 0.0
        has_regime = 1.0 if any(r in all_conds for r in ["tf_1d_adx", "tf_4h_bb_width", "tf_1d_rsi"]) else 0.0
        has_session = 1.0 if any(s in all_conds for s in ["hour_utc", "day_of_week"]) else 0.0

        return [
            self.sl_mult,                    # 0
            self.rr_ratio,                   # 1
            float(self.cooldown),            # 2
            self.atr_gate,                   # 3
            self.trail_mult,                 # 4
            self.tp1_ratio,                  # 5
            float(self.condition_complexity), # 6
            float(n_and),                    # 7
            float(n_or),                     # 8
            float(unique_indicators),        # 9
            float(self.n_timeframes_used),   # 10
            has_volume,                      # 11
            has_regime,                      # 12
            has_session,                     # 13
        ]

    # ── JSON serialization ───────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Strategy:
        metrics = d.get("metrics", {})
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except (json.JSONDecodeError, TypeError):
                metrics = {}
        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            generation=int(d.get("generation", 0)),
            asset=d.get("asset", "BTCUSDT"),
            parent_a_id=d.get("parent_a_id", ""),
            parent_b_id=d.get("parent_b_id", ""),
            sl_mult=float(d.get("sl_mult", 1.5)),
            rr_ratio=float(d.get("rr_ratio", 3.0)),
            cooldown=int(d.get("cooldown", 3)),
            atr_gate=float(d.get("atr_gate", 0.001)),
            trail_mult=float(d.get("trail_mult", 0.0)),
            tp1_ratio=float(d.get("tp1_ratio", 0.0)),
            buy_conditions=d.get("buy_conditions", ""),
            sell_conditions=d.get("sell_conditions", ""),
            metrics=metrics,
            walk_forward_ratio=float(d.get("walk_forward_ratio", 0.0)),
            mc_drawdown_p95=float(d.get("mc_drawdown_p95", 100.0)),
            parameter_sensitivity=float(d.get("parameter_sensitivity", 1.0)),
            regime_bull_wr=float(d.get("regime_bull_wr", 0.0)),
            regime_bear_wr=float(d.get("regime_bear_wr", 0.0)),
            regime_sideways_wr=float(d.get("regime_sideways_wr", 0.0)),
            validation_cagr=float(d.get("validation_cagr", 0.0)),
            condition_complexity=int(d.get("condition_complexity", 0)),
            n_timeframes_used=int(d.get("n_timeframes_used", 1)),
            monthly_returns_json=d.get("monthly_returns_json", "[]"),
            trade_log_json=d.get("trade_log_json", "[]"),
            deep_stats_json=d.get("deep_stats_json", "{}"),
            p_value=float(d.get("p_value", 1.0)),
            is_correlated=bool(d.get("is_correlated", False)),
            holdout_cagr=float(d.get("holdout_cagr", 0.0)),
            holdout_win_rate=float(d.get("holdout_win_rate", 0.0)),
            holdout_trades=int(d.get("holdout_trades", 0)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> Strategy:
        return cls.from_dict(json.loads(s))
