"""
runner.py — Autonomous strategy discovery loop.

Generates, backtests, scores, and stores strategies forever.
Uses ML-guided parameter suggestions once enough data is collected.
Run this as a background process alongside the Flask dashboard.
"""

from __future__ import annotations

import sys
import time
import random
import logging

from indicators import load_data
from generator import generate_random_strategy, build_strategy_from_params, mutate_strategy
from backtest import backtest
from scorer import score
from database import StrategyDatabase
from ml_optimizer import StrategyOptimizer

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("runner.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("runner")


def run_forever() -> None:
    """Main discovery loop — runs until interrupted."""
    log.info("=" * 60)
    log.info("Strategy Finder — starting discovery loop")
    log.info("=" * 60)

    # Load 5y 1h data with all indicators computed
    log.info("Loading / fetching 5y 1h BTCUSDT data ...")
    data = load_data()
    log.info(f"Data ready: {len(data):,} rows")

    db = StrategyDatabase()
    optimizer = StrategyOptimizer()
    generation = 0

    # Seed optimizer with existing DB strategies if any
    existing = db.top_n(100)
    for s in existing:
        if s.metrics.get("score", -999) > -999:
            optimizer.record(s.params_vector, s.metrics["score"])
    if existing:
        log.info(f"Seeded optimizer with {len(existing)} existing strategies")

    while True:
        generation += 1
        batch: list = []

        # ── Build the batch ──────────────────────────────────────────────
        if generation < 5 or random.random() < 0.3:
            # Pure random exploration
            batch = [generate_random_strategy(generation) for _ in range(20)]
        else:
            # ML-guided + random mix
            suggested = optimizer.suggest(n=10)
            batch = [build_strategy_from_params(p, generation) for p in suggested]
            batch += [generate_random_strategy(generation) for _ in range(10)]

            # Also mutate top performers
            top = db.top_n(5)
            for parent in top:
                batch.append(mutate_strategy(parent, generation))

        # ── Evaluate the batch ───────────────────────────────────────────
        scored_count = 0
        for s in batch:
            metrics = backtest(data, s)
            if metrics is None:
                continue
            s.metrics = metrics
            s.metrics["score"] = score(metrics)

            if s.metrics["score"] > -999:
                db.save(s)
                optimizer.record(s.params_vector, s.metrics["score"])
                scored_count += 1

        # ── Housekeeping ─────────────────────────────────────────────────
        db.keep_top(100)

        top1 = db.top1()
        best_score = top1.metrics["score"] if top1 else 0.0
        db.log_run(generation, len(batch), best_score)

        log.info(
            f"Gen {generation:>4d} | "
            f"Tested {len(batch):>3d} | "
            f"Scored {scored_count:>3d} | "
            f"DB: {db.count():>3d} | "
            f"Top: {best_score:>8.2f}"
        )

        # Brief pause to prevent CPU spinning
        time.sleep(1)


if __name__ == "__main__":
    try:
        run_forever()
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C)")
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        raise
