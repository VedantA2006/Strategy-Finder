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

from flask import Flask, render_template, abort, request, jsonify, Response, redirect
from flask_socketio import SocketIO
import csv
import io

from database import StrategyDatabase

app = Flask(
    __name__,
    template_folder=str(pathlib.Path(__file__).parent / "templates"),
    static_folder=str(pathlib.Path(__file__).parent / "static"),
)
socketio = SocketIO(app, cors_allowed_origins="*")

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
    
    heatmap_data = {}
    for m in monthly:
        yr, mo = m["month"].split("-")
        if yr not in heatmap_data: heatmap_data[yr] = {}
        heatmap_data[yr][mo] = m.get("return_pct", 0)
    
    return render_template(
        "strategy.html", 
        s=s, 
        equity_json=equity_json,
        dd_json=dd_json,
        monthly_returns=monthly,
        heatmap_data=heatmap_data
    )

@app.route("/strategy/<strategy_id>/export")
def strategy_export_json(strategy_id: str):
    """Returns a JSON file formatted as a live-trading config."""
    db = get_db()
    s = db.get(strategy_id)
    db.close()
    
    if not s:
        abort(404)
        
    config = {
        "id": s.id,
        "asset": s.asset,
        "sl_mult": s.sl_mult,
        "rr_ratio": s.rr_ratio,
        "trail_mult": s.trail_mult,
        "tp1_ratio": s.tp1_ratio,
        "cooldown": s.cooldown,
        "atr_gate": s.atr_gate,
        "buy_conditions": s.buy_conditions,
        "sell_conditions": s.sell_conditions
    }
    return jsonify(config)

@app.route("/strategy/<strategy_id>/download/backtest")
def download_backtest_csv(strategy_id: str):
    db = get_db()
    s = db.get(strategy_id)
    db.close()
    if not s or not s.trade_log_json:
        abort(404)
        
    trades = json.loads(s.trade_log_json)
    if not trades:
        abort(404)
        
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["trade_no", "direction", "entry_time", "exit_time", "duration_hours", 
                     "entry_price", "sl_price", "tp_price", "exit_price", "exit_reason", 
                     "pnl_usd", "pnl_pct", "balance_before", "balance_after", "atr_at_entry", 
                     "sl_mult_used", "rr_ratio_used", "regime", "win"])
    
    for i, t in enumerate(trades):
        pnl_pct = (t["pnl"] / t["balance_before"] * 100) if "balance_before" in t and t["balance_before"] > 0 else 0.0
        writer.writerow([
            i+1, t.get("direction", ""), t.get("entry_time", ""), t.get("exit_time", ""), round(t.get("duration_hours", 0), 2),
            round(t.get("entry_price", 0), 4), round(t.get("sl", 0), 4), round(t.get("tp", 0), 4), round(t.get("exit_price", 0), 4),
            t.get("exit_reason", ""), round(t.get("pnl", 0), 2), round(pnl_pct, 2),
            round(t.get("balance_before", 0), 2), round(t.get("balance_after", 0), 2),
            round(t.get("atr_entry", 0), 4), t.get("sl_mult_used", 0), t.get("rr_ratio_used", 0),
            t.get("regime", ""), 1 if t.get("win") else 0
        ])
        
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-disposition": f"attachment; filename=strategy_{strategy_id}_trades.csv"})

@app.route("/strategy/<strategy_id>/download/report")
def download_report(strategy_id: str):
    db = get_db()
    s = db.get(strategy_id)
    db.close()
    if not s: abort(404)
    
    equity = s.metrics.get("equity_curve", [])
    svg_path = ""
    if equity:
        max_eq = max(equity)
        min_eq = min(equity)
        range_eq = max_eq - min_eq if max_eq != min_eq else 1
        width = 800
        height = 300
        points = []
        for i, val in enumerate(equity):
            x = (i / max(1, len(equity) - 1)) * width
            y = height - ((val - min_eq) / range_eq) * height
            points.append(f"{x},{y}")
        svg_path = " ".join(points)
        
    monthly = json.loads(s.monthly_returns_json) if s.monthly_returns_json else []
    heatmap_data = {}
    for m in monthly:
        yr, mo = m["month"].split("-")
        if yr not in heatmap_data: heatmap_data[yr] = {}
        heatmap_data[yr][mo] = m.get("return_pct", 0)
        
    return render_template("report.html", s=s, svg_path=svg_path, heatmap_data=heatmap_data)

@app.route("/compare")
def compare_strategies():
    ids = request.args.get("ids", "")
    if not ids: return redirect("/")
    id_list = [x.strip() for x in ids.split(",")][:5]
    
    db = get_db()
    strategies = [db.get(sid) for sid in id_list]
    strategies = [s for s in strategies if s is not None]
    db.close()
    
    if len(strategies) < 2: return redirect("/")
    
    return render_template("compare.html", strategies=strategies)

@app.route("/lineage/<strategy_id>")
def strategy_lineage(strategy_id: str):
    db = get_db()
    
    def fetch_ancestry(sid: str, depth: int = 0) -> dict | None:
        if depth >= 4 or not sid: return None
        s = db.get(sid)
        if not s: return None
        
        node = {
            "id": s.id,
            "name": s.name,
            "score": round(s.metrics.get("score", 0), 2),
            "generation": s.generation
        }
        
        children = []
        if s.parent_a_id:
            pa = fetch_ancestry(s.parent_a_id, depth + 1)
            if pa: children.append(pa)
        if s.parent_b_id and s.parent_b_id != s.parent_a_id:
            pb = fetch_ancestry(s.parent_b_id, depth + 1)
            if pb: children.append(pb)
            
        if children:
            node["parents"] = children
            
        return node
        
    tree = fetch_ancestry(strategy_id)
    db.close()
    
    if not tree:
        abort(404)
        
    return jsonify(tree)



@app.route("/feed")
def creation_feed():
    return render_template("creation_feed.html")

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

@app.route("/api/emit", methods=["POST"])
def api_emit():
    data = request.json
    if data:
        socketio.emit(data.get("event"), data.get("data"))
    return jsonify({"status": "ok"})

@app.route("/api/top")
def api_top():
    """JSON endpoint for the hybrid live engine to consume top strategies."""
    db = get_db()
    strategies = db.top_n(10)
    db.close()
    
    return jsonify([s.to_dict() for s in strategies])


@app.route("/strategy/<strategy_id>/download/code")
def download_code(strategy_id):
    s = db.get(strategy_id)
    if not s:
        return "Strategy not found", 404
        
    import datetime
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    content = render_template("backtest_script.py.jinja", s=s, datetime=now_str)
    
    return Response(
        content,
        mimetype="text/x-python",
        headers={"Content-disposition": f"attachment; filename=backtest_{s.name}.py"}
    )

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
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
