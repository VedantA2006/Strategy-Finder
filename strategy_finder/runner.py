"""
runner.py — Professional multi-asset, genetic algorithm discovery engine.

Features:
  - Multi-timeframe data loading across BTC, ETH, SOL.
  - Full robustness pipeline: Monte Carlo DD, parameter sensitivity, regime breakdown.
  - True genetic algorithm with tournament selection, elitism, crossover, mutation, and diversity injection.
  - Multi-asset score bonuses for strategies that generalize.
"""

from __future__ import annotations

import sys
import time
import random
import logging
import copy
import numpy as np

from indicators import load_data, ASSETS
from generator import generate_random_strategy, build_strategy_from_params, mutate_strategy, crossover
from backtest import backtest, _run_engine
from scorer import score
from database import StrategyDatabase
from ml_optimizer import StrategyOptimizer
from strategy import Strategy

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

# ─── Robustness Pipeline ─────────────────────────────────────────────────────

def _monte_carlo_dd(trades: list[dict], balance: float = 10_000.0) -> float:
    """Run 200 random order shuffles on trade list, return 95th percentile Max DD."""
    if not trades: return 100.0
    pnl_array = np.array([t["pnl"] for t in trades])
    max_dds = []
    
    for _ in range(200):
        np.random.shuffle(pnl_array)
        eq = np.zeros(len(pnl_array) + 1)
        eq[0] = balance
        for i in range(len(pnl_array)):
            eq[i+1] = max(0.01, eq[i] + pnl_array[i])
            
        peak = np.maximum.accumulate(eq)
        drawdowns = (peak - eq) / peak * 100
        max_dds.append(np.max(drawdowns))
        
    return float(np.percentile(max_dds, 95))


def _parameter_sensitivity(strategy: Strategy, data: pd.DataFrame, base_score: float) -> float:
    """Nudge parameters slightly, run backtests, calculate max score drop percentage."""
    if base_score <= 0: return 1.0
    
    nudges = [
        (0.2, 0.0), (-0.2, 0.0),
        (0.0, 0.3), (0.0, -0.3)
    ]
    
    worst_drop = 0.0
    for d_sl, d_rr in nudges:
        clone = copy.deepcopy(strategy)
        clone.sl_mult = max(1.0, clone.sl_mult + d_sl)
        clone.rr_ratio = max(1.5, clone.rr_ratio + d_rr)
        min_rr = round(clone.sl_mult * 1.5 + 0.1, 1)
        if clone.rr_ratio < min_rr: clone.rr_ratio = min_rr
        
        # Fast full run to get metric score
        res = _run_engine(data, clone, "val")
        if res is None:
            worst_drop = 1.0
            break
            
        clone.metrics = res["metrics"]
        clone.metrics["score"] = score(res["metrics"], clone)
        
        s_score = clone.metrics["score"]
        if s_score < 0:
            worst_drop = 1.0
            break
            
        drop = (base_score - s_score) / base_score
        if drop > worst_drop:
            worst_drop = drop
            
    return worst_drop


def _regime_breakdown(strategy: Strategy, data: pd.DataFrame, trades: list[dict]) -> None:
    """Calculate win rate during Bull, Bear, and Sideways regimes based on 1D EMA slope."""
    if not trades: return
    
    bull_wins, bull_total = 0, 0
    bear_wins, bear_total = 0, 0
    side_wins, side_total = 0, 0
    
    # Pre-map timestamps to EMA slopes for fast lookup
    # 1D EMA slope is broadcasted to 15m index in data
    # We can just look up the row nearest the trade exit_time
    
    # For speed, just iter over trades and find row in df
    # since df is indexed, this can be slow. 
    # Better: just convert trades to DataFrame, merge with data
    df_t = pd.DataFrame(trades)
    df_t['exit_time'] = pd.to_datetime(df_t['exit_time'])
    
    # We need ema slope at trade entry
    entry_times = pd.to_datetime(df_t['entry_time'])
    
    # Map to df index
    # We'll just do a dirty loop for simplicity, but use searchsorted on timestamps
    ts_array = data["timestamp"].values
    slope_array = data["tf_1d_ema_200_slope"].values
    
    for _, t in df_t.iterrows():
        # Find closest index
        idx = np.searchsorted(ts_array, t['entry_time'])
        if idx >= len(slope_array): idx = -1
        slope = slope_array[idx]
        
        if slope > 0.1:  # Bull
            bull_total += 1
            if t['win']: bull_wins += 1
        elif slope < -0.1: # Bear
            bear_total += 1
            if t['win']: bear_wins += 1
        else: # Sideways
            side_total += 1
            if t['win']: side_wins += 1
            
    strategy.regime_bull_wr = (bull_wins / bull_total * 100) if bull_total > 0 else 0
    strategy.regime_bear_wr = (bear_wins / bear_total * 100) if bear_total > 0 else 0
    strategy.regime_sideways_wr = (side_wins / side_total * 100) if side_total > 0 else 0


def run_robustness(strategy: Strategy, data: pd.DataFrame, base_res: dict) -> bool:
    """Run full robustness suite. Return True if it passes internal checks."""
    metrics = base_res["metrics"]
    trades = base_res["trades"]
    
    strategy.mc_drawdown_p95 = _monte_carlo_dd(trades)
    
    # Base score for sensitivity
    base_score = score(metrics, strategy)
    if base_score <= 0: return False
    
    strategy.parameter_sensitivity = _parameter_sensitivity(strategy, data, base_score)
    _regime_breakdown(strategy, data, trades)
    
    return True


# ─── Genetic Algorithm ───────────────────────────────────────────────────────

def tournament_selection(population: list[Strategy], k: int = 5) -> Strategy:
    """Pick k random strategies, return the best one."""
    contenders = random.sample(population, min(k, len(population)))
    return max(contenders, key=lambda s: s.metrics.get("score", -999))


def run_forever() -> None:
    """Main discovery loop — runs until interrupted."""
    log.info("=" * 60)
    log.info("Strategy Finder — Professional Discovery Engine")
    log.info("=" * 60)

    # 1. Load Data for All Assets
    asset_data = {}
    for asset in ASSETS:
        log.info(f"Loading data for {asset} ...")
        asset_data[asset] = load_data(asset)
    log.info("All market data loaded.")

    db = StrategyDatabase()
    optimizer = StrategyOptimizer()
    generation = 0
    
    # Maintain live population
    population: list[Strategy] = []

    # Seed from DB
    existing = db.top_n(50)
    if existing:
        population = existing
        for s in existing:
            optimizer.record(s.asset, s.params_vector, s.metrics.get("score", 0))
        log.info(f"Seeded population with {len(existing)} existing strategies.")

    while True:
        generation += 1
        batch: list[Strategy] = []

        # ── 2. Build Generation Batch ────────────────────────────────────
        
        # Elitism: top 5 survive exactly as they are
        population.sort(key=lambda s: s.metrics.get("score", -999), reverse=True)
        elites = copy.deepcopy(population[:5])
        batch.extend(elites)
        
        # Diversity check
        unique_fps = len(set(s.fingerprint for s in population))
        diversity = unique_fps / max(1, len(population))
        if diversity < 0.6 and len(population) >= 10:
            log.info(f"Diversity low ({diversity:.2f}). Injecting 10 randoms.")
            for _ in range(10):
                batch.append(generate_random_strategy(generation, random.choice(ASSETS)))
                
        # Fill remainder of batch to reach 50
        while len(batch) < 50:
            asset = random.choice(ASSETS)
            
            # 10% pure random
            if random.random() < 0.1:
                batch.append(generate_random_strategy(generation, asset))
                continue
                
            # 30% ML Suggested
            if random.random() < 0.3:
                params = optimizer.suggest(asset, n=1)[0]
                batch.append(build_strategy_from_params(params, generation, asset))
                continue
                
            # 60% Genetic Crossover + Mutation
            if len(population) >= 5:
                p1 = tournament_selection(population)
                p2 = tournament_selection(population)
                child = crossover(p1, p2, generation)
                child.asset = asset
                child = mutate_strategy(child, generation)
                batch.append(child)
            else:
                batch.append(generate_random_strategy(generation, asset))

        # ── 3. Evaluate Batch ────────────────────────────────────────────
        next_population: list[Strategy] = []
        scored_count = 0
        scores_this_gen = []

        for s in batch:
            data = asset_data[s.asset]
            
            # Main Backtest (includes Walk Forward)
            res = backtest(data, s)
            if res is None:
                continue
                
            # Robustness Pipeline
            if not run_robustness(s, data, res):
                continue
                
            # Final Score
            s.metrics = res["metrics"]
            s.metrics["score"] = score(s.metrics, s)
            
            if s.metrics["score"] > -999:
                # Potential Cross-Asset Test for high scorers
                if s.metrics["score"] > 5.0 and s.asset == "BTCUSDT":
                    # Check if it works on ETH
                    eth_res = backtest(asset_data["ETHUSDT"], s)
                    if eth_res and score(eth_res["metrics"], s) > 0:
                        # Bonus!
                        s.metrics["score"] *= 1.2
                        log.info(f"Cross-Asset Bonus Applied! {s.id}")
                
                db.save(s)
                optimizer.record(s.asset, s.params_vector, s.metrics["score"])
                next_population.append(s)
                scores_this_gen.append(s.metrics["score"])
                scored_count += 1

        # ── 4. Housekeeping ──────────────────────────────────────────────
        population = next_population
        if not population:
            # Extinction event! Seed randomly
            population = [generate_random_strategy(generation, random.choice(ASSETS)) for _ in range(10)]
            
        db.keep_top(100)

        best_score = max(scores_this_gen) if scores_this_gen else 0.0
        mean_score = sum(scores_this_gen) / len(scores_this_gen) if scores_this_gen else 0.0
        
        # Diversity for DB log
        unique_fps = len(set(s.fingerprint for s in population))
        div = unique_fps / max(1, len(population))
        
        db.log_run(generation, len(batch), best_score, mean_score, div)

        log.info(
            f"Gen {generation:>4d} | "
            f"Tested {len(batch):>3d} | "
            f"Scored {scored_count:>3d} | "
            f"Div: {div:.2f} | "
            f"Top: {best_score:>8.2f} | "
            f"DB: {db.count():>3d}"
        )

        time.sleep(1)


if __name__ == "__main__":
    try:
        run_forever()
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C)")
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        raise
