"""
database.py — SQLite persistence for discovered strategies.

Stores all strategy params, conditions, and backtest metrics.
Provides leaderboard queries, pruning, and run logging.
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
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS strategies (
                id              TEXT PRIMARY KEY,
                name            TEXT,
                generation      INTEGER,
                params_json     TEXT,
                buy_conditions  TEXT,
                sell_conditions TEXT,
                total_return    REAL,
                cagr            REAL,
                win_rate        REAL,
                dollar_rr       REAL,
                profit_factor   REAL,
                max_drawdown    REAL,
                sharpe          REAL,
                trades_per_month REAL,
                score           REAL,
                equity_curve    TEXT,
                created_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS run_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                generation  INTEGER,
                tested      INTEGER,
                best_score  REAL,
                timestamp   TEXT
            );
        """)
        self.conn.commit()

    # ── CRUD ─────────────────────────────────────────────────────────────

    def save(self, s: Strategy) -> None:
        """Insert or replace a strategy with its metrics."""
        m = s.metrics
        params = {
            "sl_mult": s.sl_mult,
            "rr_ratio": s.rr_ratio,
            "cooldown": s.cooldown,
            "atr_gate": s.atr_gate,
        }
        self.conn.execute("""
            INSERT OR REPLACE INTO strategies
            (id, name, generation, params_json, buy_conditions, sell_conditions,
             total_return, cagr, win_rate, dollar_rr, profit_factor,
             max_drawdown, sharpe, trades_per_month, score, equity_curve, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            s.id, s.name, s.generation,
            json.dumps(params),
            s.buy_conditions, s.sell_conditions,
            m.get("total_return_pct", 0),
            m.get("cagr", 0),
            m.get("win_rate", 0),
            m.get("dollar_rr", 0),
            m.get("profit_factor", 0),
            m.get("max_drawdown", 0),
            m.get("sharpe", 0),
            m.get("avg_trades_per_month", 0),
            m.get("score", -999),
            json.dumps(m.get("equity_curve", [])),
            datetime.datetime.utcnow().isoformat(),
        ))
        self.conn.commit()

    def get(self, strategy_id: str) -> Optional[Strategy]:
        """Fetch a single strategy by ID."""
        row = self.conn.execute(
            "SELECT * FROM strategies WHERE id = ?", (strategy_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_strategy(row)

    def top_n(self, n: int = 20) -> list[Strategy]:
        """Return top N valid strategies ordered by score descending."""
        rows = self.conn.execute(
            "SELECT * FROM strategies WHERE score > -999 AND max_drawdown <= 10 AND win_rate >= 50 ORDER BY score DESC LIMIT ?", (n,)
        ).fetchall()
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
        """Prune the database, keeping only the top N valid strategies."""
        self.conn.execute(f"""
            DELETE FROM strategies
            WHERE id NOT IN (
                SELECT id FROM strategies WHERE score > -999 AND max_drawdown <= 10 AND win_rate >= 50 ORDER BY score DESC LIMIT ?
            )
        """, (n,))
        self.conn.commit()

    # ── Run logging ──────────────────────────────────────────────────────

    def log_run(self, generation: int, tested: int, best_score: float) -> None:
        self.conn.execute("""
            INSERT INTO run_log (generation, tested, best_score, timestamp)
            VALUES (?, ?, ?, ?)
        """, (generation, tested, best_score,
              datetime.datetime.utcnow().isoformat()))
        self.conn.commit()

    def recent_logs(self, n: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM run_log ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def today_stats(self) -> dict:
        """Return stats for today's run session."""
        today = datetime.date.today().isoformat()
        row = self.conn.execute("""
            SELECT COUNT(*) as runs, SUM(tested) as total_tested,
                   MAX(best_score) as best_today
            FROM run_log WHERE timestamp >= ?
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

        s = Strategy(
            id=row["id"],
            name=row["name"],
            generation=row["generation"],
            sl_mult=params.get("sl_mult", 1.5),
            rr_ratio=params.get("rr_ratio", 3.0),
            cooldown=params.get("cooldown", 3),
            atr_gate=params.get("atr_gate", 0.001),
            buy_conditions=row["buy_conditions"],
            sell_conditions=row["sell_conditions"],
            metrics={
                "total_return_pct":     row["total_return"],
                "cagr":                 row["cagr"],
                "win_rate":             row["win_rate"],
                "dollar_rr":            row["dollar_rr"],
                "profit_factor":        row["profit_factor"],
                "max_drawdown":         row["max_drawdown"],
                "sharpe":               row["sharpe"],
                "avg_trades_per_month": row["trades_per_month"],
                "score":                row["score"],
                "equity_curve":         equity_curve,
            },
        )
        return s

    def close(self) -> None:
        self.conn.close()
