"""
database.py — SQLite persistence for Strategy Finder.

Features:
  - Extended schema for multi-asset and robustness metrics.
  - Hall of Fame table to permanently protect top strategies.
  - Generation stats table with diversity tracking.
"""

from __future__ import annotations

import json
import sqlite3
import datetime
import pathlib
from typing import Optional

from strategy import Strategy


DB_PATH = pathlib.Path(__file__).parent / "strategies.db"


class StrategyDatabase:
    """Thread-safe SQLite wrapper for strategy storage."""

    def __init__(self, db_path: str | pathlib.Path = DB_PATH):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS strategies (
                id              TEXT PRIMARY KEY,
                name            TEXT,
                generation      INTEGER,
                asset           TEXT,
                params_json     TEXT,
                buy_conditions  TEXT,
                sell_conditions TEXT,
                
                -- Standard Backtest Metrics
                total_return    REAL,
                cagr            REAL,
                win_rate        REAL,
                dollar_rr       REAL,
                profit_factor   REAL,
                max_drawdown    REAL,
                max_dd_duration INTEGER,
                sharpe          REAL,
                trades_per_month REAL,
                score           REAL,
                
                -- Robustness Metrics
                walk_forward_ratio REAL,
                mc_drawdown_p95 REAL,
                parameter_sensitivity REAL,
                regime_bull_wr REAL,
                regime_bear_wr REAL,
                regime_sideways_wr REAL,
                validation_cagr REAL,
                p_value REAL DEFAULT 1.0,
                is_correlated INTEGER DEFAULT 0,
                avg_monthly_return REAL DEFAULT 0.0,
                overfit_score REAL DEFAULT 0.0,
                
                -- Complexity Metrics
                condition_complexity INTEGER,
                n_timeframes_used INTEGER,
                
                -- JSON Blobs
                equity_curve    TEXT,
                monthly_returns_json TEXT,
                trade_log_json  TEXT,
                deep_stats_json TEXT,
                
                created_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS hall_of_fame (
                id TEXT PRIMARY KEY,
                added_at TEXT
            );

            CREATE TABLE IF NOT EXISTS generation_stats (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                generation  INTEGER,
                tested      INTEGER,
                best_score  REAL,
                mean_score  REAL,
                diversity   REAL,
                timestamp   TEXT
            );

            CREATE TABLE IF NOT EXISTS gp_observations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                asset       TEXT,
                params_json TEXT,
                score       REAL,
                recorded_at TEXT
            );
            
            CREATE TABLE IF NOT EXISTS category_stats (
                category    TEXT PRIMARY KEY,
                appearances_top20   INTEGER DEFAULT 0,
                appearances_total   INTEGER DEFAULT 0,
                updated_at  TEXT
            );
            
            CREATE TABLE IF NOT EXISTS condition_templates (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                clause      TEXT UNIQUE,
                times_in_top20  INTEGER DEFAULT 0,
                avg_score_when_present  REAL DEFAULT 0.0,
                direction   TEXT,
                updated_at  TEXT
            );
        """)
        # Safe alter for existing DBs
        try:
            self.conn.execute("ALTER TABLE strategies ADD COLUMN p_value REAL DEFAULT 1.0")
            self.conn.execute("ALTER TABLE strategies ADD COLUMN is_correlated INTEGER DEFAULT 0")
            self.conn.execute("ALTER TABLE strategies ADD COLUMN avg_monthly_return REAL DEFAULT 0.0")
            self.conn.execute("ALTER TABLE strategies ADD COLUMN overfit_score REAL DEFAULT 0.0")
            self.conn.execute("ALTER TABLE strategies ADD COLUMN deep_stats_json TEXT")
        except sqlite3.OperationalError:
            pass
            
        try:
            self.conn.execute("ALTER TABLE strategies ADD COLUMN parent_a_id TEXT DEFAULT ''")
            self.conn.execute("ALTER TABLE strategies ADD COLUMN parent_b_id TEXT DEFAULT ''")
            self.conn.execute("ALTER TABLE strategies ADD COLUMN holdout_cagr REAL DEFAULT 0.0")
            self.conn.execute("ALTER TABLE strategies ADD COLUMN holdout_win_rate REAL DEFAULT 0.0")
            self.conn.execute("ALTER TABLE strategies ADD COLUMN holdout_trades INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
            
        self.conn.commit()

    # ── CRUD ─────────────────────────────────────────────────────────────

    def save(self, s: Strategy) -> None:
        """Insert or replace a strategy with its metrics."""
        # Correlation check
        s.is_correlated = False
        if s.trade_log_json and s.metrics.get("score", -999) > -999:
            try:
                top10 = self.top_n(10)
                if top10:
                    import pandas as pd
                    new_trades = json.loads(s.trade_log_json)
                    if new_trades:
                        df_new = pd.DataFrame(new_trades)
                        df_new['exit_time'] = pd.to_datetime(df_new['exit_time'])
                        df_new.set_index('exit_time', inplace=True)
                        daily_new = df_new['pnl'].resample('D').sum().fillna(0)
                        
                        for top_s in top10:
                            if not top_s.trade_log_json or top_s.trade_log_json == "[]": continue
                            top_trades = json.loads(top_s.trade_log_json)
                            if not top_trades: continue
                            df_top = pd.DataFrame(top_trades)
                            df_top['exit_time'] = pd.to_datetime(df_top['exit_time'])
                            df_top.set_index('exit_time', inplace=True)
                            daily_top = df_top['pnl'].resample('D').sum().fillna(0)
                            
                            idx = daily_new.index.intersection(daily_top.index)
                            if len(idx) > 30:
                                corr = daily_new.loc[idx].corr(daily_top.loc[idx])
                                if corr > 0.85:
                                    s.is_correlated = True
                                    break
            except Exception as e:
                pass

        m = s.metrics
        params = {
            "sl_mult": s.sl_mult,
            "rr_ratio": s.rr_ratio,
            "cooldown": s.cooldown,
            "atr_gate": s.atr_gate,
            "trail_mult": s.trail_mult,
            "tp1_ratio": s.tp1_ratio,
        }
        
        avg_monthly = m.get("avg_monthly_return", 0.0)
        overfit_score = 1.0 - s.walk_forward_ratio if s.walk_forward_ratio else 0.0
        
        deep_stats = {k: v for k, v in m.items() if k not in ("equity_curve",)}
        
        self.conn.execute("""
            INSERT OR REPLACE INTO strategies
            (id, name, generation, asset, parent_a_id, parent_b_id, params_json, buy_conditions, sell_conditions,
             total_return, cagr, win_rate, dollar_rr, profit_factor,
             max_drawdown, max_dd_duration, sharpe, trades_per_month, score,
             walk_forward_ratio, mc_drawdown_p95, parameter_sensitivity,
             regime_bull_wr, regime_bear_wr, regime_sideways_wr, validation_cagr,
             p_value, is_correlated, avg_monthly_return, overfit_score,
             condition_complexity, n_timeframes_used,
             holdout_cagr, holdout_win_rate, holdout_trades,
             equity_curve, monthly_returns_json, trade_log_json, deep_stats_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?)
        """, (
            s.id, s.name, s.generation, s.asset, s.parent_a_id, s.parent_b_id,
            json.dumps(params),
            s.buy_conditions, s.sell_conditions,
            
            m.get("total_return_pct", 0),
            m.get("cagr", 0),
            m.get("win_rate", 0),
            m.get("dollar_rr", 0),
            m.get("profit_factor", 0),
            m.get("max_drawdown", 0),
            m.get("max_dd_duration", 0),
            m.get("sharpe", 0),
            m.get("avg_trades_per_month", 0),
            m.get("score", -999),
            
            s.walk_forward_ratio,
            s.mc_drawdown_p95,
            s.parameter_sensitivity,
            s.regime_bull_wr,
            s.regime_bear_wr,
            s.regime_sideways_wr,
            s.validation_cagr,
            
            m.get("p_value", 1.0),
            1 if s.is_correlated else 0,
            avg_monthly,
            overfit_score,
            
            s.condition_complexity,
            s.n_timeframes_used,
            
            s.holdout_cagr,
            s.holdout_win_rate,
            s.holdout_trades,
            
            json.dumps(m.get("equity_curve", [])),
            s.monthly_returns_json,
            s.trade_log_json,
            json.dumps(deep_stats),
            datetime.datetime.utcnow().isoformat(),
        ))
        self.conn.commit()

        # Check if it should be in Hall of Fame (top 20)
        self._update_hall_of_fame()

    def _update_hall_of_fame(self) -> None:
        """Maintain top 20 all-time highest scoring valid strategies."""
        # Get top 20
        rows = self.conn.execute("""
            SELECT id FROM strategies 
            WHERE score > -999 
            ORDER BY score DESC LIMIT 20
        """).fetchall()
        top_ids = [r["id"] for r in rows]

        # Insert new ones
        for strat_id in top_ids:
            self.conn.execute("""
                INSERT OR IGNORE INTO hall_of_fame (id, added_at)
                VALUES (?, ?)
            """, (strat_id, datetime.datetime.utcnow().isoformat()))
        
        # We NEVER remove from Hall of Fame per requirements!
        self.conn.commit()

    # ── GP Memory ────────────────────────────────────────────────────────
    
    def save_gp_observation(self, asset: str, params_vector: list[float], score: float) -> None:
        self.conn.execute("""
            INSERT INTO gp_observations (asset, params_json, score, recorded_at)
            VALUES (?, ?, ?, ?)
        """, (asset, json.dumps(params_vector), score, datetime.datetime.utcnow().isoformat()))
        self.conn.commit()

    def load_gp_observations(self) -> dict[str, list[tuple[list[float], float]]]:
        rows = self.conn.execute("SELECT asset, params_json, score FROM gp_observations ORDER BY id DESC LIMIT 5000").fetchall()
        obs = {}
        for r in rows:
            asset = r["asset"]
            if asset not in obs:
                obs[asset] = []
            obs[asset].append((json.loads(r["params_json"]), r["score"]))
        return obs

    def get(self, strategy_id: str) -> Optional[Strategy]:
        """Fetch a single strategy by ID."""
        row = self.conn.execute(
            "SELECT * FROM strategies WHERE id = ?", (strategy_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_strategy(row)

    def top_n(self, n: int = 20, asset: str = None) -> list[Strategy]:
        """Return top N valid strategies ordered by score descending."""
        query = "SELECT * FROM strategies WHERE score > -999"
        params = []
        if asset:
            query += " AND asset = ?"
            params.append(asset)
        query += " ORDER BY score DESC LIMIT ?"
        params.append(n)
        
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_strategy(r) for r in rows]

    def top1(self) -> Optional[Strategy]:
        """Return the single best strategy by score."""
        results = self.top_n(1)
        return results[0] if results else None

    def count(self) -> int:
        """Total strategies in the database."""
        row = self.conn.execute("SELECT COUNT(*) FROM strategies").fetchone()
        return row[0]

    def keep_top(self, n: int = 100) -> None:
        """
        Prune the database, keeping only the top N valid strategies,
        BUT never delete any strategy that is in the Hall of Fame.
        """
        self.conn.execute(f"""
            DELETE FROM strategies
            WHERE id NOT IN (
                SELECT id FROM strategies WHERE score > -999 ORDER BY score DESC LIMIT 20
            )
            AND id NOT IN (
                SELECT id FROM strategies WHERE score > -999 AND is_correlated = 0 ORDER BY score DESC LIMIT 80
            )
            AND id NOT IN (SELECT id FROM hall_of_fame)
        """)
        self.conn.commit()

    # ── Run logging ──────────────────────────────────────────────────────

    def log_run(self, generation: int, tested: int, best_score: float, mean_score: float, diversity: float) -> None:
        self.conn.execute("""
            INSERT INTO generation_stats (generation, tested, best_score, mean_score, diversity, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (generation, tested, best_score, mean_score, diversity,
              datetime.datetime.utcnow().isoformat()))
        self.conn.execute("DELETE FROM generation_stats WHERE id NOT IN (SELECT id FROM generation_stats ORDER BY id DESC LIMIT 1000)")
        self.conn.commit()

    def recent_logs(self, n: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM generation_stats ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def today_stats(self) -> dict:
        """Return stats for today's run session."""
        today = datetime.date.today().isoformat()
        row = self.conn.execute("""
            SELECT COUNT(*) as runs, SUM(tested) as total_tested,
                   MAX(best_score) as best_today
            FROM generation_stats WHERE timestamp >= ?
        """, (today,)).fetchone()
        return {
            "runs": row["runs"] or 0,
            "total_tested": row["total_tested"] or 0,
            "best_today": row["best_today"] or 0,
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _row_to_strategy(self, row: sqlite3.Row) -> Strategy:
        """Convert a database row to a Strategy object."""
        params = json.loads(row["params_json"]) if row["params_json"] else {}
        equity_curve = json.loads(row["equity_curve"]) if row["equity_curve"] else []
        deep_stats = json.loads(row["deep_stats_json"]) if "deep_stats_json" in row.keys() and row["deep_stats_json"] else {}

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
            
            walk_forward_ratio=row["walk_forward_ratio"],
            mc_drawdown_p95=row["mc_drawdown_p95"],
            parameter_sensitivity=row["parameter_sensitivity"],
            regime_bull_wr=row["regime_bull_wr"],
            regime_bear_wr=row["regime_bear_wr"],
            regime_sideways_wr=row["regime_sideways_wr"],
            validation_cagr=row["validation_cagr"],
            p_value=row["p_value"],
            is_correlated=bool(row["is_correlated"]),
            condition_complexity=row["condition_complexity"],
            n_timeframes_used=row["n_timeframes_used"],
            monthly_returns_json=row["monthly_returns_json"],
            trade_log_json=row["trade_log_json"],
            parent_a_id=row.keys() and "parent_a_id" in row.keys() and row["parent_a_id"] or "",
            parent_b_id=row.keys() and "parent_b_id" in row.keys() and row["parent_b_id"] or "",
            holdout_cagr=row.keys() and "holdout_cagr" in row.keys() and row["holdout_cagr"] or 0.0,
            holdout_win_rate=row.keys() and "holdout_win_rate" in row.keys() and row["holdout_win_rate"] or 0.0,
            holdout_trades=row.keys() and "holdout_trades" in row.keys() and row["holdout_trades"] or 0,
            deep_stats_json=row.keys() and "deep_stats_json" in row.keys() and row["deep_stats_json"] or "{}",
            
            metrics={
                "total_return_pct":     row["total_return"],
                "cagr":                 row["cagr"],
                "win_rate":             row["win_rate"],
                "dollar_rr":            row["dollar_rr"],
                "profit_factor":        row["profit_factor"],
                "max_drawdown":         row["max_drawdown"],
                "max_dd_duration":      row["max_dd_duration"],
                "sharpe":               row["sharpe"],
                "avg_trades_per_month": row["trades_per_month"],
                "p_value":              row["p_value"],
                "score":                row["score"],
                "avg_monthly_return":   row["avg_monthly_return"] if "avg_monthly_return" in row.keys() else 0.0,
                "overfit_score":        row["overfit_score"] if "overfit_score" in row.keys() else 0.0,
                "equity_curve":         equity_curve,
                **deep_stats
            },
        )
        return s

    def close(self) -> None:
        self.conn.close()
