"""
database.py — MongoDB persistence for Strategy Finder.

Features:
  - MongoDB backend for multi-asset and robustness metrics.
  - Hall of Fame collection to permanently protect top strategies.
  - Generation stats collection with diversity tracking.
  - GP observations with capped replay.
  - Condition templates with upsert + running average.
"""

from __future__ import annotations

import os
import json
import datetime
import pathlib
from typing import Optional

from pymongo import MongoClient, DESCENDING, ASCENDING
from strategy import Strategy

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://geminivedant5_db_user:Vedant06@ac-g3mnn9n-shard-00-00.gg0xqcv.mongodb.net:27017,ac-g3mnn9n-shard-00-01.gg0xqcv.mongodb.net:27017,ac-g3mnn9n-shard-00-02.gg0xqcv.mongodb.net:27017/?ssl=true&replicaSet=atlas-1jkcvg-shard-0&authSource=admin&appName=Cluster0")
DB_NAME = "strategy_finder"


class _ResultShim:
    """Shim to emulate SQLite cursor result for runner.py meta-learning code."""
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return [_DictRow(r) for r in self._rows]

    def fetchone(self):
        return _DictRow(self._rows[0]) if self._rows else None

    def __iter__(self):
        return iter([_DictRow(r) for r in self._rows])


class _DictRow(dict):
    """Dict subclass that supports both dict key access and sqlite3.Row-style access."""
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)

    def keys(self):
        return super().keys()


class StrategyDatabase:
    """MongoDB wrapper for strategy storage — drop-in replacement for SQLite version."""

    def __init__(self, mongo_uri: str = MONGO_URI, db_name: str = DB_NAME):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]

        # Collections
        self.strategies_col = self.db["strategies"]
        self.hof_col = self.db["hall_of_fame"]
        self.gp_col = self.db["gp_observations"]
        self.cat_stats_col = self.db["category_stats"]
        self.cond_templates_col = self.db["condition_templates"]
        self.gen_stats_col = self.db["generation_stats"]
        self.events_col = self.db["events"]

        self._create_indexes()

    # Expose conn-like access for runner.py meta-learning that calls db.conn.execute(...)
    # This shim provides a compatible interface
    class _ConnShim:
        def __init__(self, db_ref):
            self._db = db_ref

        def execute(self, sql: str, params=None):
            """Shim that intercepts SQL-style calls from runner.py meta-learning and maps to MongoDB."""
            sql_lower = sql.strip().lower()

            if sql_lower.startswith("select clause, direction from condition_templates"):
                rows = list(self._db.cond_templates_col.find(
                    {}, {"clause": 1, "direction": 1, "_id": 0}
                ).sort("times_in_top20", -1).limit(5))
                return _ResultShim(rows)

            elif sql_lower.startswith("update category_stats set appearances_top20 = 0"):
                self._db.cat_stats_col.update_many({}, {"$set": {"appearances_top20": 0}})
                return _ResultShim([])

            elif sql_lower.startswith("insert or ignore into category_stats"):
                if params:
                    self._db.cat_stats_col.update_one(
                        {"category": params[0]},
                        {"$setOnInsert": {"category": params[0], "appearances_top20": 0, "appearances_total": 0, "updated_at": params[1]}},
                        upsert=True
                    )
                return _ResultShim([])

            elif sql_lower.startswith("update category_stats set appearances_total"):
                if params:
                    self._db.cat_stats_col.update_one(
                        {"category": params[0]},
                        {"$inc": {"appearances_total": 1}}
                    )
                return _ResultShim([])

            elif sql_lower.startswith("update category_stats set appearances_top20 = appearances_top20 + 1"):
                if params:
                    self._db.cat_stats_col.update_one(
                        {"category": params[0]},
                        {"$inc": {"appearances_top20": 1}}
                    )
                return _ResultShim([])

            elif sql_lower.startswith("insert into condition_templates"):
                if params:
                    clause = params[0]
                    score_val = params[1]
                    updated_at = params[2]
                    direction = "buy"
                    if "'sell'" in sql_lower or (len(params) > 3 and "sell" in str(params)):
                        direction = "sell"
                    # Determine direction from SQL
                    if "?, 'buy'" in sql.lower():
                        direction = "buy"
                    elif "?, 'sell'" in sql.lower():
                        direction = "sell"
                    
                    existing = self._db.cond_templates_col.find_one({"clause": clause})
                    if existing:
                        old_count = existing.get("times_in_top20", 0)
                        old_avg = existing.get("avg_score_when_present", 0.0)
                        new_avg = ((old_avg * old_count) + score_val) / (old_count + 1)
                        self._db.cond_templates_col.update_one(
                            {"clause": clause},
                            {"$set": {"avg_score_when_present": new_avg, "updated_at": updated_at},
                             "$inc": {"times_in_top20": 1}}
                        )
                    else:
                        self._db.cond_templates_col.insert_one({
                            "clause": clause,
                            "direction": direction,
                            "times_in_top20": 1,
                            "avg_score_when_present": score_val,
                            "updated_at": updated_at
                        })
                return _ResultShim([])

            elif sql_lower.startswith("select * from category_stats"):
                rows = list(self._db.cat_stats_col.find({}, {"_id": 0}))
                return _ResultShim(rows)

            return _ResultShim([])

        def commit(self):
            pass  # MongoDB auto-commits

    @property
    def conn(self):
        """Compatibility shim for runner.py meta-learning code that calls db.conn.execute(...)"""
        return self._ConnShim(self)

    def _create_indexes(self) -> None:
        self.strategies_col.create_index([("score", DESCENDING)])
        self.strategies_col.create_index([("is_correlated", ASCENDING)])
        self.strategies_col.create_index([("asset", ASCENDING)])
        self.strategies_col.create_index([("created_at", DESCENDING)])
        self.strategies_col.create_index([("is_hof", ASCENDING), ("score", DESCENDING)])
        self.cond_templates_col.create_index([("clause", ASCENDING)], unique=True)
        # TTL index: events auto-expire after 24 hours
        self.events_col.create_index([("created_at", ASCENDING)], expireAfterSeconds=86400)

    # ── Events (engine → frontend via MongoDB) ─────────────────────────────

    def emit_event(self, event: str, data: dict) -> None:
        """Write an event to the events collection for the frontend to consume."""
        self.events_col.insert_one({
            "event": event,
            "data": data,
            "created_at": datetime.datetime.utcnow(),
            "consumed": False
        })

    # ── Save / Retrieve ──────────────────────────────────────────────────

    def save(self, s: Strategy) -> None:
        """Save or update a strategy in MongoDB."""
        params = {
            "sl_mult": s.sl_mult,
            "rr_ratio": s.rr_ratio,
            "cooldown": s.cooldown,
            "atr_gate": s.atr_gate,
            "trail_mult": s.trail_mult,
            "tp1_ratio": s.tp1_ratio,
        }

        doc = {
            "id": s.id,
            "name": s.name,
            "generation": s.generation,
            "asset": s.asset,
            "params_json": json.dumps(params),
            "buy_conditions": s.buy_conditions,
            "sell_conditions": s.sell_conditions,
            "total_return": s.metrics.get("total_return_pct", 0),
            "cagr": s.metrics.get("cagr", 0),
            "win_rate": s.metrics.get("win_rate", 0),
            "dollar_rr": s.metrics.get("dollar_rr", 0),
            "profit_factor": s.metrics.get("profit_factor", 0),
            "max_drawdown": s.metrics.get("max_drawdown", 0),
            "max_dd_duration": s.metrics.get("max_dd_duration", 0),
            "sharpe": s.metrics.get("sharpe", 0),
            "trades_per_month": s.metrics.get("avg_trades_per_month", 0),
            "score": s.metrics.get("score", -999),
            "walk_forward_ratio": s.walk_forward_ratio,
            "mc_drawdown_p95": s.mc_drawdown_p95,
            "parameter_sensitivity": s.parameter_sensitivity,
            "regime_bull_wr": s.regime_bull_wr,
            "regime_bear_wr": s.regime_bear_wr,
            "regime_sideways_wr": s.regime_sideways_wr,
            "validation_cagr": getattr(s, "validation_cagr", 0),
            "p_value": getattr(s, "p_value", 0.0),
            "is_correlated": int(getattr(s, "is_correlated", False)),
            "avg_monthly_return": s.metrics.get("avg_monthly_return", 0),
            "overfit_score": s.metrics.get("overfit_score", 0),
            "condition_complexity": s.condition_complexity,
            "n_timeframes_used": s.n_timeframes_used,
            "equity_curve": json.dumps(s.metrics.get("equity_curve", [])),
            "monthly_returns_json": getattr(s, "monthly_returns_json", "[]"),
            "trade_log_json": getattr(s, "trade_log_json", "[]"),
            "parent_a_id": getattr(s, "parent_a_id", ""),
            "parent_b_id": getattr(s, "parent_b_id", ""),
            "holdout_cagr": getattr(s, "holdout_cagr", 0.0),
            "holdout_win_rate": getattr(s, "holdout_win_rate", 0.0),
            "holdout_trades": getattr(s, "holdout_trades", 0),
            "deep_stats_json": getattr(s, "deep_stats_json", "{}"),
            "is_hof": False,
            "created_at": datetime.datetime.utcnow().isoformat(),
        }

        self.strategies_col.replace_one({"id": s.id}, doc, upsert=True)
        self._update_hall_of_fame()

    def _update_hall_of_fame(self) -> None:
        """Maintain top 20 all-time highest scoring valid strategies."""
        rows = list(self.strategies_col.find(
            {"score": {"$gt": -999}},
            {"id": 1, "_id": 0}
        ).sort("score", DESCENDING).limit(20))

        for row in rows:
            self.hof_col.update_one(
                {"id": row["id"]},
                {"$setOnInsert": {"id": row["id"], "added_at": datetime.datetime.utcnow().isoformat()}},
                upsert=True
            )

    def get(self, strategy_id: str) -> Optional[Strategy]:
        """Fetch a single strategy by ID."""
        row = self.strategies_col.find_one({"id": strategy_id})
        if row is None:
            return None
        return self._doc_to_strategy(row)

    def top_n(self, n: int = 20, asset: str = None) -> list[Strategy]:
        """Return top N valid strategies ordered by score descending."""
        query = {"score": {"$gt": -999}}
        if asset:
            query["asset"] = asset
        rows = list(self.strategies_col.find(query).sort("score", DESCENDING).limit(n))
        return [self._doc_to_strategy(r) for r in rows]

    def top1(self) -> Optional[Strategy]:
        """Return the single best strategy by score."""
        results = self.top_n(1)
        return results[0] if results else None

    def count(self) -> int:
        """Total strategies in the database."""
        return self.strategies_col.count_documents({})

    def keep_top(self, n: int = 100) -> None:
        """Prune the database, keeping top strategies and Hall of Fame."""
        # Get IDs to keep: top 20 by score + top 80 non-correlated + HoF
        top_20_ids = [r["id"] for r in self.strategies_col.find(
            {"score": {"$gt": -999}}, {"id": 1, "_id": 0}
        ).sort("score", DESCENDING).limit(20)]

        top_80_ids = [r["id"] for r in self.strategies_col.find(
            {"score": {"$gt": -999}, "is_correlated": 0}, {"id": 1, "_id": 0}
        ).sort("score", DESCENDING).limit(80)]

        hof_ids = [r["id"] for r in self.hof_col.find({}, {"id": 1, "_id": 0})]

        keep_ids = set(top_20_ids + top_80_ids + hof_ids)

        if keep_ids:
            self.strategies_col.delete_many({"id": {"$nin": list(keep_ids)}})

    # ── GP Memory ────────────────────────────────────────────────────────

    def save_gp_observation(self, asset: str, params_vector: list[float], score: float) -> None:
        self.gp_col.insert_one({
            "asset": asset,
            "params_json": json.dumps(params_vector),
            "score": score,
            "recorded_at": datetime.datetime.utcnow().isoformat()
        })

    def load_gp_observations(self) -> dict[str, list[tuple[list[float], float]]]:
        rows = list(self.gp_col.find({}).sort("_id", DESCENDING).limit(5000))
        obs: dict[str, list[tuple[list[float], float]]] = {}
        for r in rows:
            asset = r["asset"]
            if asset not in obs:
                obs[asset] = []
            obs[asset].append((json.loads(r["params_json"]), r["score"]))
        return obs

    # ── Run logging ──────────────────────────────────────────────────────

    def log_run(self, generation: int, tested: int, best_score: float, mean_score: float, diversity: float) -> None:
        self.gen_stats_col.insert_one({
            "generation": generation,
            "tested": tested,
            "best_score": best_score,
            "mean_score": mean_score,
            "diversity": diversity,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        # Keep only last 1000 generation stats
        total = self.gen_stats_col.count_documents({})
        if total > 1000:
            oldest = list(self.gen_stats_col.find({}).sort("_id", ASCENDING).limit(total - 1000))
            if oldest:
                oldest_ids = [r["_id"] for r in oldest]
                self.gen_stats_col.delete_many({"_id": {"$in": oldest_ids}})

    def recent_logs(self, n: int = 20) -> list[dict]:
        rows = list(self.gen_stats_col.find({}, {"_id": 0}).sort("_id", DESCENDING).limit(n))
        return rows

    def today_stats(self) -> dict:
        """Return stats for today's run session."""
        today = datetime.date.today().isoformat()
        pipeline = [
            {"$match": {"timestamp": {"$gte": today}}},
            {"$group": {
                "_id": None,
                "runs": {"$sum": 1},
                "total_tested": {"$sum": "$tested"},
                "best_today": {"$max": "$best_score"}
            }}
        ]
        result = list(self.gen_stats_col.aggregate(pipeline))
        if result:
            return {
                "runs": result[0].get("runs", 0),
                "total_tested": result[0].get("total_tested", 0),
                "best_today": result[0].get("best_today", 0),
            }
        return {"runs": 0, "total_tested": 0, "best_today": 0}

    # ── Internal ─────────────────────────────────────────────────────────

    def _doc_to_strategy(self, row: dict) -> Strategy:
        """Convert a MongoDB document to a Strategy object."""
        params = json.loads(row.get("params_json", "{}"))
        equity_curve = json.loads(row.get("equity_curve", "[]"))
        deep_stats = json.loads(row.get("deep_stats_json", "{}")) if row.get("deep_stats_json") else {}

        s = Strategy(
            id=row["id"],
            name=row["name"],
            generation=row["generation"],
            asset=row["asset"],
            sl_mult=params.get("sl_mult", 1.5),
            rr_ratio=params.get("rr_ratio", 3.0),
            cooldown=params.get("cooldown", 3),
            atr_gate=params.get("atr_gate", 0.001),
            trail_mult=params.get("trail_mult", 0.0),
            tp1_ratio=params.get("tp1_ratio", 0.0),
            buy_conditions=row["buy_conditions"],
            sell_conditions=row["sell_conditions"],

            walk_forward_ratio=row.get("walk_forward_ratio", 0),
            mc_drawdown_p95=row.get("mc_drawdown_p95", 0),
            parameter_sensitivity=row.get("parameter_sensitivity", 0),
            regime_bull_wr=row.get("regime_bull_wr", 0),
            regime_bear_wr=row.get("regime_bear_wr", 0),
            regime_sideways_wr=row.get("regime_sideways_wr", 0),
            validation_cagr=row.get("validation_cagr", 0),
            p_value=row.get("p_value", 0.0),
            is_correlated=bool(row.get("is_correlated", 0)),
            condition_complexity=row.get("condition_complexity", 0),
            n_timeframes_used=row.get("n_timeframes_used", 0),
            monthly_returns_json=row.get("monthly_returns_json", "[]"),
            trade_log_json=row.get("trade_log_json", "[]"),
            parent_a_id=row.get("parent_a_id", ""),
            parent_b_id=row.get("parent_b_id", ""),
            holdout_cagr=row.get("holdout_cagr", 0.0),
            holdout_win_rate=row.get("holdout_win_rate", 0.0),
            holdout_trades=row.get("holdout_trades", 0),
            deep_stats_json=row.get("deep_stats_json", "{}"),

            metrics={
                "total_return_pct":     row.get("total_return", 0),
                "cagr":                 row.get("cagr", 0),
                "win_rate":             row.get("win_rate", 0),
                "dollar_rr":            row.get("dollar_rr", 0),
                "profit_factor":        row.get("profit_factor", 0),
                "max_drawdown":         row.get("max_drawdown", 0),
                "max_dd_duration":      row.get("max_dd_duration", 0),
                "sharpe":               row.get("sharpe", 0),
                "avg_trades_per_month": row.get("trades_per_month", 0),
                "p_value":              row.get("p_value", 0.0),
                "score":                row.get("score", -999),
                "avg_monthly_return":   row.get("avg_monthly_return", 0.0),
                "overfit_score":        row.get("overfit_score", 0.0),
                "equity_curve":         equity_curve,
                **deep_stats
            },
        )
        return s

    def close(self) -> None:
        self.client.close()
