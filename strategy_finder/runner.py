"""
runner.py - Professional multi-asset, genetic algorithm discovery engine.

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
import pandas as pd

HOF_SCORE_THRESHOLD = 10.0

from indicators import load_data, ASSETS
from generator import generate_random_strategy, build_strategy_from_params, mutate_strategy, crossover, AdaptiveMutationConfig, update_category_weights
from backtest import backtest, _run_engine
from scorer import score
from database import StrategyDatabase
from ml_optimizer import StrategyOptimizer
from strategy import Strategy
import multiprocessing
import requests




import datetime

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def emit_event(queue, event_name, payload):
    if queue:
        queue.put({"event": event_name, "data": payload})

def prune_conditions(strategy: Strategy, data: pd.DataFrame) -> Strategy:
    import copy
    from generator import _split_clauses
    recent_data = data.iloc[-17280:].copy() if len(data) > 17280 else data.copy()
    
    for direction in ["buy", "sell"]:
        while True:
            conds = getattr(strategy, f"{direction}_conditions")
            clauses, joiners = _split_clauses(conds)
            if len(clauses) <= 1:
                break
                
            base_res = _run_engine(recent_data, strategy, "val")
            if not base_res: break
            base_score = score(base_res["metrics"], strategy)
            
            best_pruned_score = -999.0
            best_pruned_conds = None
            
            for i in range(len(clauses)):
                new_c = clauses[:]
                new_j = joiners[:]
                new_c.pop(i)
                if i < len(new_j): new_j.pop(i)
                elif new_j: new_j.pop()
                
                res_str = [new_c[0]]
                for j, c in zip(new_j, new_c[1:]): res_str.extend([j, c])
                test_conds = " ".join(res_str)
                
                test_s = copy.deepcopy(strategy)
                setattr(test_s, f"{direction}_conditions", test_conds)
                
                res = _run_engine(recent_data, test_s, "val")
                if res:
                    s_score = score(res["metrics"], test_s)
                    if s_score >= best_pruned_score:
                        best_pruned_score = s_score
                        best_pruned_conds = test_conds
                        
            if best_pruned_score >= base_score * 0.98 and best_pruned_conds:
                setattr(strategy, f"{direction}_conditions", best_pruned_conds)
            else:
                break
                
    return strategy

def run_holdout_eval(strategy: Strategy, full_data: pd.DataFrame) -> None:
    from indicators import HOLDOUT_CUTOFF
    holdout_data = full_data[full_data["timestamp"] > HOLDOUT_CUTOFF].copy()
    if len(holdout_data) < 200: return
    
    res = _run_engine(holdout_data, strategy, "full")
    if res:
        strategy.holdout_cagr = res["metrics"]["cagr"]
        strategy.holdout_win_rate = res["metrics"]["win_rate"]
        strategy.holdout_trades = res["metrics"]["total_trades"]
        # Real p-value only for holdout candidates
        n_trades = res["metrics"]["total_trades"]
        actual_wins = res["metrics"]["wins"]
        if n_trades > 0:
            sim_wins = np.random.binomial(n_trades, 0.5, size=1000)
            strategy.p_value = float(np.mean(sim_wins >= actual_wins))
        else:
            strategy.p_value = 1.0

def _evaluate_worker(args: tuple) -> Strategy | None:
    s, data, eth_data, event_queue = args
    
    emit_event(event_queue, "strategy_event", {
        "phase": "created",
        "id": s.id[:8],
        "name": s.name,
        "asset": s.asset,
        "generation": s.generation,
        "origin": getattr(s, "_origin", "random"),
        "buy_conditions": s.buy_conditions,
        "sell_conditions": s.sell_conditions,
        "sl_mult": s.sl_mult,
        "rr_ratio": s.rr_ratio,
        "trail_mult": s.trail_mult,
        "tp1_ratio": s.tp1_ratio,
        "atr_gate": s.atr_gate,
        "cooldown": s.cooldown,
        "complexity": s.condition_complexity,
        "n_timeframes": s.n_timeframes_used,
        "timestamp": now_iso()
    })
    
    # Fast reject filter (last 6 months, ~17280 bars of 15m)
    recent_data = data.iloc[-17280:].copy() if len(data) > 17280 else data.copy()
    quick_res = _run_engine(recent_data, s, "val")
    if quick_res is None:
        emit_event(event_queue, "strategy_event", {"phase": "rejected", "id": s.id[:8], "reason": "engine_failed", "value": 0, "timestamp": now_iso()})
        return None
        
    qm = quick_res["metrics"]
    if qm["max_drawdown"] > 12.0: 
        emit_event(event_queue, "strategy_event", {"phase": "rejected", "id": s.id[:8], "reason": "drawdown", "value": qm["max_drawdown"], "timestamp": now_iso()})
        return None
    if qm["win_rate"] < 42.0 or qm["win_rate"] > 85.0: 
        emit_event(event_queue, "strategy_event", {"phase": "rejected", "id": s.id[:8], "reason": "win_rate", "value": qm["win_rate"], "timestamp": now_iso()})
        return None
    if qm["avg_trades_per_month"] < 1.5: 
        emit_event(event_queue, "strategy_event", {"phase": "rejected", "id": s.id[:8], "reason": "trade_freq", "value": qm["avg_trades_per_month"], "timestamp": now_iso()})
        return None
    if qm["total_trades"] < 5: 
        emit_event(event_queue, "strategy_event", {"phase": "rejected", "id": s.id[:8], "reason": "trade_count", "value": qm["total_trades"], "timestamp": now_iso()})
        return None

    # Quick score check — only prune if promising
    quick_score = score(qm, s)
    import json
    orig_complexity = s.condition_complexity
    if quick_score > 5.0:
        s = prune_conditions(s, data)
        s._update_complexity()
        try:
            ds = json.loads(s.deep_stats_json)
            ds["original_complexity"] = orig_complexity
            s.deep_stats_json = json.dumps(ds)
        except: pass

    # Full backtest
    res = backtest(data, s)
    if res is None:
        emit_event(event_queue, "strategy_event", {"phase": "backtest_failed", "id": s.id[:8], "timestamp": now_iso()})
        return None
        
    # Robustness pipeline
    passed, reason = run_robustness(s, data, res)
    if not passed:
        emit_event(event_queue, "strategy_event", {"phase": "robustness_failed", "id": s.id[:8], "reason": reason, "value": getattr(s, "mc_drawdown_p95", 0), "threshold": 0, "timestamp": now_iso()})
        return None
        
    s.metrics = res["metrics"]
    s.metrics["score"] = score(s.metrics, s)
    
    if s.metrics["score"] > -999:
        if s.metrics["score"] > HOF_SCORE_THRESHOLD and s.asset == "BTCUSDT" and eth_data is not None:
            eth_res = backtest(eth_data, s)
            if eth_res and score(eth_res["metrics"], s) > 0:
                s.metrics["score"] *= 1.2
        return s
    
    emit_event(event_queue, "strategy_event", {"phase": "scored_out", "id": s.id[:8], "score": s.metrics["score"], "reason": "score_failed", "timestamp": now_iso()})
    return None

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


def run_robustness(strategy: Strategy, data: pd.DataFrame, base_res: dict) -> tuple[bool, str]:
    """Run full robustness suite. Return True if it passes internal checks."""
    metrics = base_res["metrics"]
    trades = base_res["trades"]
    
    strategy.mc_drawdown_p95 = _monte_carlo_dd(trades)
    if strategy.mc_drawdown_p95 > 15.0:
        return False, "mc_drawdown"
        
    base_score = score(metrics, strategy)
    if base_score <= 0: return False, "base_score"
    
    strategy.parameter_sensitivity = _parameter_sensitivity(strategy, data, base_score)
    if strategy.parameter_sensitivity > 0.30:
        return False, "parameter_sensitivity"
        
    _regime_breakdown(strategy, data, trades)
    return True, ""


# ─── Genetic Algorithm ───────────────────────────────────────────────────────

def tournament_selection(population: list[Strategy], k: int = 5) -> Strategy:
    """Pick k random strategies, return the best one."""
    contenders = random.sample(population, min(k, len(population)))
    return max(contenders, key=lambda s: s.metrics.get("score", -999))


import threading
def event_drainer(q):
    import requests
    while True:
        try:
            item = q.get()
            if item is None: break
            try:
                requests.post("http://localhost:5000/api/emit", json=item, timeout=0.5)
            except:
                pass
        except:
            pass

def run_forever() -> None:
    """Main discovery loop - runs until interrupted."""
    log.info("=" * 60)
    log.info("Strategy Finder - Professional Discovery Engine")
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
    
    m = multiprocessing.Manager()
    event_queue = m.Queue()
    drainer = threading.Thread(target=event_drainer, args=(event_queue,), daemon=True)
    drainer.start()
    
    current_all_time_best = db.top1().metrics.get("score", 0) if db.top1() else 0
    
    # Replay GP memory
    obs = db.load_gp_observations()
    replayed = 0
    for asset, records in obs.items():
        for params, p_score in records:
            optimizer.record(asset, params, p_score)
            replayed += 1
    log.info(f"Loaded {replayed} past GP observations from memory.")
    
    mut_config = AdaptiveMutationConfig()
    
    # Maintain live population
    population: list[Strategy] = []

    # Seed from DB
    existing = db.top_n(50)
    if existing:
        population = existing
        log.info(f"Seeded population with {len(existing)} existing strategies.")

    while True:
        generation += 1
        batch: list[Strategy] = []
        mutations_attempted = 0
        mutations_improved = 0
        update_category_weights(db)

        # ── 2. Build Generation Batch ────────────────────────────────────
        
        # Elitism: top 5 survive exactly as they are
        population.sort(key=lambda s: s.metrics.get("score", -999), reverse=True)
        elites = copy.deepcopy(population[:5])
        for e in elites:
            e._origin = "elite"
            e.generation = generation
        batch.extend(elites)
        
        # Diversity check
        unique_fps = len(set(s.fingerprint for s in population))
        diversity = unique_fps / max(1, len(population))
        if diversity < 0.6 and len(population) >= 10:
            log.info(f"Diversity low ({diversity:.2f}). Injecting 10 randoms.")
            for _ in range(10):
                c = generate_random_strategy(generation, random.choice(ASSETS))
                c._origin = "random"
                batch.append(c)
                
        # Fill remainder of batch to reach 200
        while len(batch) < 200:
            asset = random.choice(ASSETS)
            
            # 10% pure random
            if db.count() > 200 and random.random() < 0.15:
                top_templates = db.conn.execute("SELECT clause, direction FROM condition_templates ORDER BY times_in_top20 DESC LIMIT 5").fetchall()
                if top_templates:
                    templ = random.choice(top_templates)
                    child = generate_random_strategy(generation, asset)
                    if templ["direction"] == "buy":
                        child.buy_conditions = f"({templ['clause'].strip('() ')}) and ({child.buy_conditions.strip('() ')})"
                    else:
                        child.sell_conditions = f"({templ['clause'].strip('() ')}) and ({child.sell_conditions.strip('() ')})"
                    child._origin = "random"
                    batch.append(child)
                    continue

            if random.random() < 0.1:
                c = generate_random_strategy(generation, asset)
                c._origin = "random"
                batch.append(c)
                continue
                
            # 30% ML Suggested
            if random.random() < 0.3:
                params = optimizer.suggest(asset, n=1)[0]
                c = build_strategy_from_params(params, generation, asset)
                c._origin = "ml_suggested"
                batch.append(c)
                continue
                
            # 60% Genetic Crossover + Mutation
            if len(population) >= 5:
                p1 = tournament_selection(population)
                p2 = tournament_selection(population)
                child = crossover(p1, p2, generation)
                child.asset = asset
                child = mutate_strategy(child, generation, mut_config)
                child._parent_score = p1.metrics.get("score", 0)
                child._origin = "crossover"
                batch.append(child)
                mutations_attempted += 1
            else:
                c = generate_random_strategy(generation, asset)
                c._origin = "random"
                batch.append(c)

        # ── 3. Evaluate Batch ────────────────────────────────────────────
        next_population: list[Strategy] = []
        scored_count = 0
        scores_this_gen = []
        
        pool_size = multiprocessing.cpu_count()
        tasks = [(s, asset_data[s.asset], asset_data.get("ETHUSDT") if s.asset == "BTCUSDT" else None, event_queue) for s in batch]
        
        with multiprocessing.Pool(pool_size) as pool:
            for s_res in pool.imap_unordered(_evaluate_worker, tasks):
                if s_res is not None:
                    db.save_gp_observation(s_res.asset, s_res.params_vector, s_res.metrics["score"])
                    optimizer.record(s_res.asset, s_res.params_vector, s_res.metrics["score"])
                    
                    if hasattr(s_res, "_parent_score"):
                        if s_res.metrics.get("score", 0) > s_res._parent_score:
                            mutations_improved += 1
                    
                    if s_res.metrics.get("score", 0) > HOF_SCORE_THRESHOLD:
                        run_holdout_eval(s_res, asset_data[s_res.asset])
                        
                    is_new_best = s_res.metrics.get("score", 0) > current_all_time_best
                    if is_new_best: current_all_time_best = s_res.metrics.get("score", 0)
                    
                    emit_event(event_queue, "strategy_event", {
                        "phase": "saved",
                        "id": s_res.id[:8],
                        "name": s_res.name,
                        "score": s_res.metrics["score"],
                        "cagr": s_res.metrics["cagr"],
                        "win_rate": s_res.metrics["win_rate"],
                        "max_drawdown": s_res.metrics["max_drawdown"],
                        "sharpe": s_res.metrics["sharpe"],
                        "profit_factor": s_res.metrics["profit_factor"],
                        "avg_monthly_return": s_res.metrics.get("avg_monthly_return", 0),
                        "walk_forward_ratio": getattr(s_res, "walk_forward_ratio", 0),
                        "mc_drawdown_p95": getattr(s_res, "mc_drawdown_p95", 0),
                        "is_new_best": is_new_best,
                        "timestamp": now_iso()
                    })
                    
                    # HoF event
                    if s_res.metrics.get("score", 0) > HOF_SCORE_THRESHOLD: # simplified HoF gate
                        emit_event(event_queue, "strategy_event", {
                            "phase": "hall_of_fame",
                            "id": s_res.id[:8],
                            "name": s_res.name,
                            "score": s_res.metrics["score"],
                            "timestamp": now_iso()
                        })
                    
                    db.save(s_res)
                    try:
                        from telegram_alerts import send_strategy_alert
                        send_strategy_alert(s_res)
                    except Exception:
                        pass
                    next_population.append(s_res)
                    scores_this_gen.append(s_res.metrics["score"])
                    scored_count += 1

        # ── 4. Housekeeping ──────────────────────────────────────────────
        success_rate = mutations_improved / max(1, mutations_attempted)
        if success_rate > 0.25:
            mut_config.sl_mult_step = max(0.05, mut_config.sl_mult_step * 0.85)
            mut_config.rr_ratio_step = max(0.1, mut_config.rr_ratio_step * 0.85)
            mut_config.cooldown_step = max(1, int(mut_config.cooldown_step * 0.85))
            mut_config.atr_gate_step = max(0.0001, mut_config.atr_gate_step * 0.85)
            mut_config.trail_mult_step = max(0.05, mut_config.trail_mult_step * 0.85)
            mut_config.condition_mutate_prob = max(0.1, mut_config.condition_mutate_prob * 0.85)
            mut_config.param_mutate_prob = max(0.1, mut_config.param_mutate_prob * 0.85)
        elif success_rate < 0.10:
            mut_config.sl_mult_step = min(1.5, mut_config.sl_mult_step * 1.20)
            mut_config.rr_ratio_step = min(2.0, mut_config.rr_ratio_step * 1.20)
            mut_config.cooldown_step = min(10, int(mut_config.cooldown_step * 1.20) + 1)
            mut_config.atr_gate_step = min(0.002, mut_config.atr_gate_step * 1.20)
            mut_config.trail_mult_step = min(1.0, mut_config.trail_mult_step * 1.20)
            mut_config.condition_mutate_prob = min(0.9, mut_config.condition_mutate_prob * 1.20)
            mut_config.param_mutate_prob = min(0.9, mut_config.param_mutate_prob * 1.20)

        # Meta-learning: Category stats & Condition templates
        import datetime
        top20 = db.top_n(20)
        from generator import _split_clauses
        category_keywords = ["ema_", "rsi_", "macd_", "stoch_", "adx_", "bb_", "volume_", "obv_", "body_ratio", "wick_", "dist_from", "hour_utc", "ema_200_slope", "bb_width"]
        
        # Reset top20 stats
        db.conn.execute("UPDATE category_stats SET appearances_top20 = 0")
        
        for st in next_population:
            conds = st.buy_conditions + " " + st.sell_conditions
            for cat in category_keywords:
                if cat in conds:
                    db.conn.execute("INSERT OR IGNORE INTO category_stats (category, appearances_top20, appearances_total, updated_at) VALUES (?, 0, 0, ?)", (cat, datetime.datetime.utcnow().isoformat()))
                    db.conn.execute("UPDATE category_stats SET appearances_total = appearances_total + 1 WHERE category = ?", (cat,))
        
        for st in top20:
            conds = st.buy_conditions + " " + st.sell_conditions
            for cat in category_keywords:
                if cat in conds:
                    db.conn.execute("UPDATE category_stats SET appearances_top20 = appearances_top20 + 1 WHERE category = ?", (cat,))
                    
            b_c, _ = _split_clauses(st.buy_conditions)
            s_c, _ = _split_clauses(st.sell_conditions)
            for c in b_c:
                db.conn.execute('''
                    INSERT INTO condition_templates (clause, direction, times_in_top20, avg_score_when_present, updated_at)
                    VALUES (?, 'buy', 1, ?, ?)
                    ON CONFLICT(clause) DO UPDATE SET
                    times_in_top20 = times_in_top20 + 1,
                    avg_score_when_present = ((avg_score_when_present * times_in_top20) + ?) / (times_in_top20 + 1),
                    updated_at = ?
                ''', (c, st.metrics.get("score", 0), datetime.datetime.utcnow().isoformat(), st.metrics.get("score", 0), datetime.datetime.utcnow().isoformat()))
            for c in s_c:
                db.conn.execute('''
                    INSERT INTO condition_templates (clause, direction, times_in_top20, avg_score_when_present, updated_at)
                    VALUES (?, 'sell', 1, ?, ?)
                    ON CONFLICT(clause) DO UPDATE SET
                    times_in_top20 = times_in_top20 + 1,
                    avg_score_when_present = ((avg_score_when_present * times_in_top20) + ?) / (times_in_top20 + 1),
                    updated_at = ?
                ''', (c, st.metrics.get("score", 0), datetime.datetime.utcnow().isoformat(), st.metrics.get("score", 0), datetime.datetime.utcnow().isoformat()))
        db.conn.commit()

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
            f"MutSR: {success_rate:.2f} | "
            f"DB: {db.count():>3d}"
        )
        
        # Emit update to Flask dashboard
        try:
            requests.post("http://localhost:5000/api/emit", json={
                "event": "generation_update",
                "data": {
                    "generation": generation,
                    "best_score": round(best_score, 2),
                    "mean_score": round(mean_score, 2),
                    "diversity": round(div, 2),
                    "db_count": db.count()
                }
            }, timeout=1)
        except requests.RequestException:
            pass

        time.sleep(1)


if __name__ == "__main__":
    try:
        run_forever()
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C)")
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        raise
