# Strategy Finder — Autonomous Trading Strategy Discovery System

An autonomous system that generates, backtests, scores, and evolves BTCUSDT trading strategies using ATR-based SL/TP, ML-guided parameter optimization, and a real-time Flask dashboard.

---

## 🚨 Critical Design Decision: ATR-Based SL/TP Only

The **only** valid SL/TP method in this system is ATR-based:

| Formula | Calculation |
|---------|-------------|
| **Stop Loss (Long)** | `entry - (ATR_14 × sl_mult)` |
| **Stop Loss (Short)** | `entry + (ATR_14 × sl_mult)` |
| **Take Profit (Long)** | `entry + (ATR_14 × rr_ratio)` |
| **Take Profit (Short)** | `entry - (ATR_14 × rr_ratio)` |
| **PnL (Win)** | `risk × (tp_dist / sl_dist)` — price-derived |
| **PnL (Loss)** | `-risk` |
| **Live Qty** | `riskUSDT / sl_dist_in_price` |

**Why?** The previous implementation used a fixed 1% SL in backtests but ATR-based SL in live trading. This made all backtest results completely meaningless. This system enforces identical formulas everywhere — backtest and live are guaranteed to match.

**There are ZERO hardcoded percentage stop-losses anywhere in this codebase.**

---

## 📁 File Structure

```
strategy_finder/
├── runner.py           # Autonomous discovery loop
├── generator.py        # Random strategy generation & mutation
├── backtest.py         # ATR-based backtester (the critical file)
├── indicators.py       # Compute all indicators on 5y data
├── strategy.py         # Strategy dataclass
├── scorer.py           # Multi-factor scoring with hard gates
├── ml_optimizer.py     # RandomForest parameter optimization
├── database.py         # SQLite persistence
├── app.py              # Flask dashboard
├── templates/
│   ├── index.html      # Leaderboard (top 20)
│   ├── strategy.html   # Strategy detail + equity chart
│   └── live.html       # Runner status page
├── static/style.css    # Premium dark-mode stylesheet
├── data/
│   └── btcusdt_1h_5y.parquet  # Cached 5y OHLCV data
├── strategies.db       # SQLite database (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

### 1. Install Dependencies

```bash
cd strategy_finder
pip install -r requirements.txt
```

### 2. Run the Discovery Engine

```bash
python runner.py
```

This will:
- Fetch 5 years of 1h BTCUSDT data from Bybit (cached to `data/btcusdt_1h_5y.parquet`)
- Compute all indicators
- Start generating, backtesting, and scoring strategies in an infinite loop
- Save the top 100 strategies to `strategies.db`

### 3. Run the Dashboard (in a separate terminal)

```bash
python app.py
```

Open http://localhost:5000 in your browser.

| Page | URL | Refresh |
|------|-----|---------|
| Leaderboard | `/` | Every 30s |
| Strategy Detail | `/strategy/<id>` | Manual |
| Live Status | `/live` | Every 10s |

### 4. Run Both Simultaneously

**Terminal 1:**
```bash
python runner.py
```

**Terminal 2:**
```bash
python app.py
```

---

## 🔄 Daily Data Refresh (Cron)

Set up a daily cron job to refresh the cached data:

```bash
# Linux/Mac — add to crontab -e
0 2 * * * cd /path/to/strategy_finder && python -c "from indicators import refresh_data; refresh_data()"

# Windows — Task Scheduler
# Action: python -c "from indicators import refresh_data; refresh_data()"
# Trigger: Daily at 2:00 AM
# Start in: C:\path\to\strategy_finder
```

---

## ⚙️ System Parameters

| Parameter | Value | Enforced In |
|-----------|-------|-------------|
| Risk per trade | 1.0% of balance (compounding) | `backtest.py` |
| Fee per side | 0.055% (Bybit taker) | `backtest.py` |
| Min Dollar RR | 1.5:1 | `generator.py`, `backtest.py`, `scorer.py` |
| Max Drawdown gate | 10% | `scorer.py` |
| Min Win Rate gate | 50% | `scorer.py` |
| Min Trades/Month | 2 | `scorer.py` |
| Data | 5y 1h Bybit BTCUSDT | `indicators.py` |
| SL formula | `entry ± (ATR_14 × sl_mult)` | `backtest.py` |
| TP formula | `entry ± (ATR_14 × rr_ratio)` | `backtest.py` |

---

## 🧠 ML Optimizer

Once 50+ strategies have been evaluated, the `StrategyOptimizer` fits a RandomForest on `(params_vector, score)` pairs and suggests promising parameter combinations for the next generation. This accelerates discovery by focusing on high-scoring parameter regions.

The optimizer always enforces `rr_ratio >= sl_mult × 1.5 + 0.1`.

---

## 📊 Scoring Formula

```python
score = (
    cagr          × 0.30 +
    profit_factor × 0.25 +
    sharpe        × 0.20 +
   -max_drawdown  × 0.15 +
    win_rate      × 0.10
)
```

**Disqualification gates (score = -999):**
- Dollar RR < 1.5
- Max Drawdown > 10%
- Trades per month < 2
- Win Rate < 50%

---

## 🔒 Backtest ↔ Live Parity Guarantee

Every design decision in this system ensures backtest results translate directly to live performance:

1. **SL/TP**: Both use `ATR_14 × multiplier` — no fixed percentages
2. **Position sizing**: `qty = risk_usd / sl_distance` — identical formula
3. **PnL**: Derived from actual price distances, not the RR ratio constant
4. **Fees**: `risk × 0.00055 × 2` deducted from every trade
5. **Risk**: Always 1.0% of current balance, compounding

When deploying a discovered strategy to live trading, use the same `sl_mult`, `rr_ratio`, and `atr_gate` values. The position size formula is:

```
qty = (account_balance × 0.010) / abs(entry_price - sl_price)
```

This is identical to the backtest sizing.
