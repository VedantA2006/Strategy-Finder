import sys
import re

with open("runner.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add prune_conditions and run_holdout_eval
extra_funcs = """

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
"""

content = content.replace("def _evaluate_worker(args: tuple) -> Strategy | None:", extra_funcs + "\ndef _evaluate_worker(args: tuple) -> Strategy | None:")

# 2. Modify _evaluate_worker to use prune_conditions
evaluate_worker_orig = """    # Full backtest
    res = backtest(data, s)"""
evaluate_worker_new = """    import json
    orig_complexity = s.condition_complexity
    s = prune_conditions(s, data)
    s._update_complexity()
    try:
        ds = json.loads(s.deep_stats_json)
        ds["original_complexity"] = orig_complexity
        s.deep_stats_json = json.dumps(ds)
    except: pass

    # Full backtest
    res = backtest(data, s)"""
content = content.replace(evaluate_worker_orig, evaluate_worker_new)

# 3. Import mut_config and category weights
content = content.replace("from generator import generate_random_strategy, build_strategy_from_params, mutate_strategy, crossover", 
                          "from generator import generate_random_strategy, build_strategy_from_params, mutate_strategy, crossover, AdaptiveMutationConfig, update_category_weights")

# 4. Modify run_forever:
run_forever_orig_start = """    db = StrategyDatabase()
    optimizer = StrategyOptimizer()
    generation = 0
    
    # Maintain live population"""

run_forever_new_start = """    db = StrategyDatabase()
    optimizer = StrategyOptimizer()
    generation = 0
    
    # Replay GP memory
    obs = db.load_gp_observations()
    replayed = 0
    for asset, records in obs.items():
        for params, p_score in records:
            optimizer.record(asset, params, p_score)
            replayed += 1
    log.info(f"Loaded {replayed} past GP observations from memory.")
    
    mut_config = AdaptiveMutationConfig()
    
    # Maintain live population"""
content = content.replace(run_forever_orig_start, run_forever_new_start)

run_forever_seed = """    existing = db.top_n(50)
    if existing:
        population = existing
        for s in existing:
            optimizer.record(s.asset, s.params_vector, s.metrics.get("score", 0))"""
run_forever_seed_new = """    existing = db.top_n(50)
    if existing:
        population = existing"""
content = content.replace(run_forever_seed, run_forever_seed_new)

content = content.replace("batch: list[Strategy] = []", "batch: list[Strategy] = []\n        mutations_attempted = 0\n        mutations_improved = 0\n        update_category_weights(db)")

gen_random = """            if random.random() < 0.1:
                batch.append(generate_random_strategy(generation, asset))
                continue"""
gen_random_new = """            if db.count() > 200 and random.random() < 0.15:
                top_templates = db.conn.execute("SELECT clause, direction FROM condition_templates ORDER BY times_in_top20 DESC LIMIT 5").fetchall()
                if top_templates:
                    templ = random.choice(top_templates)
                    child = generate_random_strategy(generation, asset)
                    if templ["direction"] == "buy":
                        child.buy_conditions = f"({templ['clause']}) and {child.buy_conditions}"
                    else:
                        child.sell_conditions = f"({templ['clause']}) and {child.sell_conditions}"
                    batch.append(child)
                    continue

            if random.random() < 0.1:
                batch.append(generate_random_strategy(generation, asset))
                continue"""
content = content.replace(gen_random, gen_random_new)

cross_mut = """                child = crossover(p1, p2, generation)
                child.asset = asset
                child = mutate_strategy(child, generation)
                batch.append(child)"""
cross_mut_new = """                child = crossover(p1, p2, generation)
                child.asset = asset
                child = mutate_strategy(child, generation, mut_config)
                child._parent_score = p1.metrics.get("score", 0)
                batch.append(child)
                mutations_attempted += 1"""
content = content.replace(cross_mut, cross_mut_new)

eval_worker = """            for s_res in pool.imap_unordered(_evaluate_worker, tasks):
                if s_res is not None:
                    db.save(s_res)
                    optimizer.record(s_res.asset, s_res.params_vector, s_res.metrics["score"])
                    next_population.append(s_res)
                    scores_this_gen.append(s_res.metrics["score"])
                    scored_count += 1"""
eval_worker_new = """            for s_res in pool.imap_unordered(_evaluate_worker, tasks):
                if s_res is not None:
                    db.save_gp_observation(s_res.asset, s_res.params_vector, s_res.metrics["score"])
                    optimizer.record(s_res.asset, s_res.params_vector, s_res.metrics["score"])
                    
                    if hasattr(s_res, "_parent_score"):
                        if s_res.metrics.get("score", 0) > s_res._parent_score:
                            mutations_improved += 1
                    
                    # Update Holdout stats if adding to HoF (score > best of top 20 or top 20 not full)
                    # For simplicity, if score > 5, run holdout
                    if s_res.metrics.get("score", 0) > 5.0:
                        run_holdout_eval(s_res, asset_data[s_res.asset])
                    
                    db.save(s_res)
                    next_population.append(s_res)
                    scores_this_gen.append(s_res.metrics["score"])
                    scored_count += 1"""
content = content.replace(eval_worker, eval_worker_new)

housekeeping = """        # ── 4. Housekeeping ──────────────────────────────────────────────
        population = next_population"""
housekeeping_new = """        # ── 4. Housekeeping ──────────────────────────────────────────────
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

        population = next_population"""
content = content.replace(housekeeping, housekeeping_new)

log_line = """            f"Top: {best_score:>8.2f} | "
            f"DB: {db.count():>3d}"
        )"""
log_line_new = """            f"Top: {best_score:>8.2f} | "
            f"MutSR: {success_rate:.2f} | "
            f"DB: {db.count():>3d}"
        )"""
content = content.replace(log_line, log_line_new)

with open("runner.py", "w", encoding="utf-8") as f:
    f.write(content)
