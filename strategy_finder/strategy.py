"""
strategy.py — Strategy dataclass with JSON serialization.

Every strategy stores ATR-based SL/TP multipliers. There are ZERO hardcoded
percentage stop-losses anywhere in this system.

SL = entry ± (atr_14 × sl_mult)
TP = entry ± (atr_14 × rr_ratio)
rr_ratio MUST be >= sl_mult × 1.5 — enforced at generation, backtest, and scoring.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Strategy:
    id: str = ""
    name: str = ""
    generation: int = 0
    sl_mult: float = 1.5          # SL distance = atr_14 × sl_mult
    rr_ratio: float = 3.0         # TP distance = atr_14 × rr_ratio; MUST >= sl_mult × 1.5
    cooldown: int = 3             # bars to skip after a trade closes
    atr_gate: float = 0.001       # skip trade if atr_pct < atr_gate
    buy_conditions: str = ""      # Python eval-able expression on row dict
    sell_conditions: str = ""     # Python eval-able expression on row dict
    metrics: dict = field(default_factory=dict)  # filled after backtest

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.name:
            self.name = f"strat_{self.id[:8]}"

    # ── RR validation ────────────────────────────────────────────────────
    @property
    def is_valid_rr(self) -> bool:
        """Dollar RR must be >= 1.5:1 — rr_ratio >= sl_mult × 1.5."""
        return self.rr_ratio / self.sl_mult >= 1.5

    # ── Param vector for ML optimizer ────────────────────────────────────
    @property
    def params_vector(self) -> list[float]:
        return [self.sl_mult, self.rr_ratio, float(self.cooldown), self.atr_gate]

    # ── JSON serialization ───────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # equity_curve can be large; keep it in metrics
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Strategy:
        # Handle both flat DB rows and nested dicts
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
            sl_mult=float(d.get("sl_mult", 1.5)),
            rr_ratio=float(d.get("rr_ratio", 3.0)),
            cooldown=int(d.get("cooldown", 3)),
            atr_gate=float(d.get("atr_gate", 0.001)),
            buy_conditions=d.get("buy_conditions", ""),
            sell_conditions=d.get("sell_conditions", ""),
            metrics=metrics,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> Strategy:
        return cls.from_dict(json.loads(s))
