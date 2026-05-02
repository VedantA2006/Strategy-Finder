"""
app.py — Professional Flask dashboard for Strategy Finder.

Routes:
  /               — Leaderboard with filters
  /strategy/<id>  — Detail page with drawdown, heatmap, and regime breakdown
  /live           — Live runner status with charts
  /api/top        — JSON endpoint for hybrid_system live engine
  /api/export/<id>— Python code export
"""

from __future__ import annotations

import json
import pathlib

from flask import Flask, render_template, abort, request, jsonify

from database import StrategyDatabase

app = Flask(
    __name__,
    template_folder=str(pathlib.Path(__file__).parent / "templates"),
    static_folder=str(pathlib.Path(__file__).parent / "static"),
)

def get_db() -> StrategyDatabase:
    return StrategyDatabase()


# ─── Page Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Leaderboard — top strategies with filtering."""
    asset_filter = request.args.get("asset", None)
    
    db = get_db()
    strategies = db.top_n(20, asset=asset_filter)
    
    # Get sparkline data (mean and best scores from recent generations)
    logs = db.recent_logs(50)
    logs.reverse() # chronological
    
    spark_labels = [l["generation"] for l in logs]
    spark_best = [l["best_score"] for l in logs]
    spark_mean = [l["mean_score"] for l in logs]
    
    db.close()
    
    return render_template(
        "index.html", 
        strategies=strategies,
        current_asset=asset_filter,
        spark_labels=json.dumps(spark_labels),
        spark_best=json.dumps(spark_best),
        spark_mean=json.dumps(spark_mean)
    )


@app.route("/strategy/<strategy_id>")
def strategy_detail(strategy_id: str):
    """Detail page with advanced charts and metrics."""
    db = get_db()
    s = db.get(strategy_id)
    db.close()
    
    if s is None:
        abort(404)
        
    equity_json = json.dumps(s.metrics.get("equity_curve", []))
    
    # Calculate drawdown curve from equity
    eq = s.metrics.get("equity_curve", [])
    dd_curve = []
    if eq:
        peak = eq[0]
        for val in eq:
            if val > peak: peak = val
            dd_curve.append(round((val - peak) / peak * 100, 2))
    dd_json = json.dumps(dd_curve)
    
    monthly = json.loads(s.monthly_returns_json) if s.monthly_returns_json else []
    
    return render_template(
        "strategy.html", 
        s=s, 
        equity_json=equity_json,
        dd_json=dd_json,
        monthly_returns=monthly
    )


@app.route("/live")
def live_status():
    """Runner status page with live charts."""
    db = get_db()
    top1 = db.top1()
    stats = db.today_stats()
    logs = db.recent_logs(20)
    total = db.count()
    
    # For chart
    chart_logs = db.recent_logs(50)
    chart_logs.reverse()
    chart_labels = json.dumps([l["generation"] for l in chart_logs])
    chart_best = json.dumps([l["best_score"] for l in chart_logs])
    
    # Latest diversity
    latest_div = logs[0]["diversity"] if logs else 0.0
    
    # Estimated throughput (strategies / hour) based on last 20 gens
    throughput = 0
    if len(logs) >= 2:
        import datetime
        try:
            t1 = datetime.datetime.fromisoformat(logs[-1]["timestamp"])
            t2 = datetime.datetime.fromisoformat(logs[0]["timestamp"])
            dt_hours = (t2 - t1).total_seconds() / 3600
            total_tested = sum(l["tested"] for l in logs[:-1])
            if dt_hours > 0:
                throughput = int(total_tested / dt_hours)
        except:
            pass
            
    db.close()

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
        chart_labels=chart_labels,
        chart_best=chart_best,
        latest_div=latest_div,
        throughput=throughput
    )


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route("/api/top")
def api_top():
    """JSON endpoint for the hybrid live engine to consume top strategies."""
    db = get_db()
    strategies = db.top_n(10)
    db.close()
    
    return jsonify([s.to_dict() for s in strategies])


@app.route("/api/export/<strategy_id>")
def api_export(strategy_id: str):
    """Returns a Python code string ready for hybrid_system.py."""
    db = get_db()
    s = db.get(strategy_id)
    db.close()
    
    if not s:
        return jsonify({"error": "Not found"}), 404
        
    code = f'''# Auto-exported from Strategy Finder
# ID: {s.id}
# Asset: {s.asset}
# Score: {s.metrics.get("score", 0)}

class StrategyConfig:
    SL_MULT = {s.sl_mult}
    RR_RATIO = {s.rr_ratio}
    COOLDOWN = {s.cooldown}
    ATR_GATE = {s.atr_gate}

    @staticmethod
    def buy_conditions(row: dict) -> bool:
        return eval(
            "{s.buy_conditions}", 
            {{"__builtins__": {{}}}}, 
            row
        )

    @staticmethod
    def sell_conditions(row: dict) -> bool:
        return eval(
            "{s.sell_conditions}", 
            {{"__builtins__": {{}}}}, 
            row
        )
'''
    return jsonify({"code": code})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
