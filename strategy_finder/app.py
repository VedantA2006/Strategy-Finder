"""
app.py — Flask dashboard for the strategy discovery system.

Routes:
  /               — Leaderboard (top 20 strategies by score)
  /strategy/<id>  — Detail page with equity curve chart
  /live           — Runner status and recent log lines
"""

from __future__ import annotations

import json
import pathlib

from flask import Flask, render_template, abort

from database import StrategyDatabase

app = Flask(
    __name__,
    template_folder=str(pathlib.Path(__file__).parent / "templates"),
    static_folder=str(pathlib.Path(__file__).parent / "static"),
)


def get_db() -> StrategyDatabase:
    return StrategyDatabase()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Leaderboard — top 20 strategies ranked by score."""
    db = get_db()
    strategies = db.top_n(20)
    db.close()
    return render_template("index.html", strategies=strategies)


@app.route("/strategy/<strategy_id>")
def strategy_detail(strategy_id: str):
    """Detail page for a single strategy with equity curve chart."""
    db = get_db()
    s = db.get(strategy_id)
    db.close()
    if s is None:
        abort(404)
    equity_json = json.dumps(s.metrics.get("equity_curve", []))
    return render_template("strategy.html", s=s, equity_json=equity_json)


@app.route("/live")
def live_status():
    """Runner status page with live generation stats and log tail."""
    db = get_db()
    top1 = db.top1()
    stats = db.today_stats()
    logs = db.recent_logs(20)
    total = db.count()
    db.close()

    # Read last 20 lines from runner.log if it exists
    log_lines: list[str] = []
    log_path = pathlib.Path(__file__).parent / "runner.log"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            log_lines = [line.strip() for line in all_lines[-20:]]

    return render_template(
        "live.html",
        top1=top1,
        stats=stats,
        logs=logs,
        log_lines=log_lines,
        total=total,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
