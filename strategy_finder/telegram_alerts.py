# Setup instructions:
# 1. Create a bot via @BotFather on Telegram, copy the token
# 2. Get your chat ID by messaging @userinfobot on Telegram
# 3. Set environment variables before running:
#    export TELEGRAM_BOT_TOKEN="your_token_here"
#    export TELEGRAM_CHAT_ID="your_chat_id_here"

import os
import logging
import requests
import numpy as np
from strategy import Strategy

log = logging.getLogger("telegram_alerts")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8640865202:AAHw3_WwwXzEh5VZvyhbUe9ri7ZA9GDsLgE")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "5508995431")


def send_strategy_alert(strategy: Strategy) -> None:
    """Send a Telegram alert for a strategy that passed all gates."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        annual_cagr = strategy.metrics.get("cagr", 0)
        monthly_cagr = ((1 + annual_cagr / 100) ** (1/12) - 1) * 100

        # Threshold checks — return silently if any fail
        if strategy.metrics.get("max_drawdown", 100) >= 10.0:
            return
        if strategy.metrics.get("win_rate", 0) < 50.0:
            return
        if monthly_cagr < 2.0:
            return

        score = strategy.metrics.get("score", 0)
        avg_monthly = strategy.metrics.get("avg_monthly_return", 0)
        win_rate = strategy.metrics.get("win_rate", 0)
        profit_factor = strategy.metrics.get("profit_factor", 0)
        sharpe = strategy.metrics.get("sharpe", 0)
        max_dd = strategy.metrics.get("max_drawdown", 0)
        mc_dd = getattr(strategy, "mc_drawdown_p95", 0)
        param_sens = getattr(strategy, "parameter_sensitivity", 0)
        wf_ratio = getattr(strategy, "walk_forward_ratio", 0)

        message = (
            f"\U0001f7e2 *New Strategy Passed*\n\n"
            f"*Name:* {strategy.name}\n"
            f"*Asset:* {strategy.asset}\n"
            f"*Score:* {score:.2f}\n\n"
            f"\U0001f4ca *Performance*\n"
            f"- Monthly CAGR: {monthly_cagr:.2f}%\n"
            f"- Annual CAGR: {annual_cagr:.1f}%\n"
            f"- Avg Monthly Return: {avg_monthly:.1f}%\n"
            f"- Win Rate: {win_rate:.1f}%\n"
            f"- Profit Factor: {profit_factor:.2f}\n"
            f"- Sharpe: {sharpe:.2f}\n\n"
            f"\U0001f6e1 *Risk*\n"
            f"- Max Drawdown: {max_dd:.1f}%\n"
            f"- MC Drawdown P95: {mc_dd:.1f}%\n"
            f"- Parameter Sensitivity: {param_sens:.2f}\n"
            f"- Walk Forward Ratio: {wf_ratio:.2f}\n\n"
            f"\u2699\ufe0f *Parameters*\n"
            f"- SL Mult: {strategy.sl_mult} | RR Ratio: {strategy.rr_ratio}\n"
            f"- Cooldown: {strategy.cooldown} bars | ATR Gate: {strategy.atr_gate}\n"
            f"- Trail Mult: {strategy.trail_mult} | TP1 Ratio: {strategy.tp1_ratio}\n\n"
            f"\U0001f4cb *Conditions*\n"
            f"Buy: `{strategy.buy_conditions}`\n"
            f"Sell: `{strategy.sell_conditions}`\n\n"
            f"\U0001f9ec Generation: {strategy.generation} | Complexity: {strategy.condition_complexity}"
        )

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }, timeout=10)

        if resp.status_code == 200:
            log.info(f"Telegram alert sent for {strategy.name}")
        else:
            log.warning(f"Telegram alert failed: {resp.status_code} {resp.text}")

        # Send standalone backtest file
        _send_backtest_file(strategy)

    except Exception as e:
        log.error(f"Telegram alert error: {e}")


def _generate_backtest_script(strategy: Strategy) -> str:
    """Generate a complete standalone backtest script for the strategy."""
    script = f'''#!/usr/bin/env python3
"""
Standalone Backtest Script
Strategy: {strategy.name}
Asset: {strategy.asset}
ID: {strategy.id}
Generated: {__import__("datetime").datetime.utcnow().isoformat()}Z
"""

import time
import datetime
import pathlib
import numpy as np
import pandas as pd
import requests as req

# ─── Strategy Parameters ─────────────────────────────────────────────────────
ASSET       = "{strategy.asset}"
SL_MULT     = {strategy.sl_mult}
RR_RATIO    = {strategy.rr_ratio}
COOLDOWN    = {strategy.cooldown}
ATR_GATE    = {strategy.atr_gate}
TRAIL_MULT  = {strategy.trail_mult}
TP1_RATIO   = {strategy.tp1_ratio}

BUY_CONDITIONS  = """{strategy.buy_conditions}"""
SELL_CONDITIONS = """{strategy.sell_conditions}"""

# ─── Risk ─────────────────────────────────────────────────────────────────────
RISK_PCT    = 0.01      # 1% of balance per trade
FEE         = 0.00055   # 0.055% per side (Bybit taker)
SLIPPAGE    = 0.0005    # 0.05% slippage
WARMUP      = 200
INITIAL_BAL = 10_000.0

# ─── Data Fetch ──────────────────────────────────────────────────────────────

TIMEFRAMES = ["15", "60", "240", "D"]
TF_MAP = {{"15": "15m", "60": "1h", "240": "4h", "D": "1d"}}


def fetch_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch klines from Bybit V5 REST API with parquet caching."""
    cache_dir = pathlib.Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{{symbol}}_{{interval}}.parquet"

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 6:
            return pd.read_parquet(cache_file)

    all_data = []
    end_time = int(time.time() * 1000)

    for _ in range(10):
        url = "https://api.bybit.com/v5/market/kline"
        params = {{"category": "linear", "symbol": symbol, "interval": interval, "limit": limit, "end": end_time}}
        resp = req.get(url, params=params, timeout=15)
        data = resp.json().get("result", {{}}).get("list", [])
        if not data:
            break
        all_data.extend(data)
        end_time = int(data[-1][0]) - 1
        time.sleep(0.1)

    if not all_data:
        raise ValueError(f"No data for {{symbol}} {{interval}}")

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df.to_parquet(cache_file)
    return df


def compute_indicators(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute all indicators for a single timeframe using ta library."""
    import ta

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # EMAs
    for p in [8, 13, 21, 34, 55, 89, 200]:
        df[f"{{prefix}}_ema_{{p}}"] = ta.trend.ema_indicator(c, window=p)

    # SMAs
    for p in [20, 50, 200]:
        df[f"{{prefix}}_sma_{{p}}"] = ta.trend.sma_indicator(c, window=p)

    # RSI
    for p in [7, 14, 21]:
        df[f"{{prefix}}_rsi_{{p}}"] = ta.momentum.rsi(c, window=p)

    # MACD
    macd = ta.trend.MACD(c)
    df[f"{{prefix}}_macd_line"] = macd.macd()
    df[f"{{prefix}}_macd_signal"] = macd.macd_signal()
    df[f"{{prefix}}_macd_hist"] = macd.macd_diff()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df[f"{{prefix}}_stoch_k"] = stoch.stoch()
    df[f"{{prefix}}_stoch_d"] = stoch.stoch_signal()

    # ADX
    adx = ta.trend.ADXIndicator(h, l, c)
    df[f"{{prefix}}_adx_14"] = adx.adx()

    # ATR
    atr = ta.volatility.AverageTrueRange(h, l, c)
    df[f"{{prefix}}_atr_14"] = atr.average_true_range()
    df[f"{{prefix}}_atr_pct"] = df[f"{{prefix}}_atr_14"] / c

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(c)
    df[f"{{prefix}}_bb_upper"] = bb.bollinger_hband()
    df[f"{{prefix}}_bb_middle"] = bb.bollinger_mavg()
    df[f"{{prefix}}_bb_lower"] = bb.bollinger_lband()
    df[f"{{prefix}}_bb_width"] = bb.bollinger_wband()

    # CCI, Williams %R, MFI, CMF
    df[f"{{prefix}}_cci_20"] = ta.trend.cci(h, l, c, window=20)
    df[f"{{prefix}}_willr_14"] = ta.momentum.williams_r(h, l, c, lbp=14)
    df[f"{{prefix}}_mfi_14"] = ta.volume.money_flow_index(h, l, c, v, window=14)
    df[f"{{prefix}}_cmf_20"] = ta.volume.chaikin_money_flow(h, l, c, v, window=20)

    # OBV slope
    obv = ta.volume.on_balance_volume(c, v)
    df[f"{{prefix}}_obv_slope_5"] = obv.diff(5)

    # VWAP deviation
    vwap = (v * (h + l + c) / 3).cumsum() / v.cumsum()
    df[f"{{prefix}}_vwap_dev"] = (c - vwap) / vwap

    # Supertrend (simplified)
    df[f"{{prefix}}_supertrend_10_3"] = ta.trend.ema_indicator(c, window=10)

    # Candle structure
    body = abs(c - df["open"])
    candle_range = h - l
    candle_range = candle_range.replace(0, np.nan)
    df[f"{{prefix}}_body_ratio"] = body / candle_range
    df[f"{{prefix}}_upper_wick_ratio"] = (h - pd.concat([c, df["open"]], axis=1).max(axis=1)) / candle_range
    df[f"{{prefix}}_lower_wick_ratio"] = (pd.concat([c, df["open"]], axis=1).min(axis=1) - l) / candle_range
    df[f"{{prefix}}_is_bullish"] = (c > df["open"]).astype(float)
    df[f"{{prefix}}_is_bearish"] = (c < df["open"]).astype(float)
    df[f"{{prefix}}_consec_bullish_2"] = (df[f"{{prefix}}_is_bullish"].rolling(2).sum()).fillna(0)
    df[f"{{prefix}}_consec_bullish_3"] = (df[f"{{prefix}}_is_bullish"].rolling(3).sum()).fillna(0)

    # Volume
    vol_sma = v.rolling(20).mean()
    df[f"{{prefix}}_volume_ratio"] = v / vol_sma
    df[f"{{prefix}}_volume_expanding"] = (v > vol_sma * 1.5).astype(float)

    # Highs/Lows
    for n in [10, 20, 50]:
        df[f"{{prefix}}_high_{{n}}"] = h.rolling(n).max()
        df[f"{{prefix}}_low_{{n}}"] = l.rolling(n).min()

    df[f"{{prefix}}_dist_from_52w_high"] = ((h.rolling(252).max() - c) / h.rolling(252).max() * 100).fillna(0)

    # EMA 200 slope
    ema200 = df[f"{{prefix}}_ema_200"]
    df[f"{{prefix}}_ema_200_slope"] = (ema200 - ema200.shift(5)) / ema200.shift(5) * 100

    # Lags
    for ind in ["rsi_14", "macd_hist", "ema_21", "stoch_k", "adx_14", "cci_20"]:
        col = f"{{prefix}}_{{ind}}"
        if col in df.columns:
            df[f"{{prefix}}_prev_1_{{ind}}"] = df[col].shift(1)
            df[f"{{prefix}}_prev_3_{{ind}}"] = df[col].shift(3)

    # Session
    if "15m" in prefix:
        df[f"{{prefix}}_hour_utc"] = df["timestamp"].dt.hour
        df[f"{{prefix}}_day_of_week"] = df["timestamp"].dt.dayofweek

    # Engulfing / Hammer / Shooting Star (simplified)
    df[f"{{prefix}}_is_engulfing_bull"] = 0.0
    df[f"{{prefix}}_is_engulfing_bear"] = 0.0
    df[f"{{prefix}}_is_hammer"] = ((df[f"{{prefix}}_lower_wick_ratio"] > 0.6) & (df[f"{{prefix}}_body_ratio"] < 0.3)).astype(float)
    df[f"{{prefix}}_is_shooting_star"] = ((df[f"{{prefix}}_upper_wick_ratio"] > 0.6) & (df[f"{{prefix}}_body_ratio"] < 0.3)).astype(float)

    df[f"{{prefix}}_close"] = c
    df[f"{{prefix}}_open"] = df["open"]
    df[f"{{prefix}}_high"] = h
    df[f"{{prefix}}_low"] = l
    df[f"{{prefix}}_volume"] = v

    return df


def load_data():
    """Load and merge all timeframes."""
    dfs = {{}}
    for tf_api, tf_label in TF_MAP.items():
        df = fetch_klines(ASSET, tf_api)
        prefix = f"tf_{{tf_label}}"
        df = compute_indicators(df, prefix)
        dfs[tf_label] = df

    base = dfs["15m"][["timestamp"] + [c for c in dfs["15m"].columns if c.startswith("tf_15m_")]].copy()

    for tf in ["1h", "4h", "1d"]:
        tf_df = dfs[tf][["timestamp"] + [c for c in dfs[tf].columns if c.startswith(f"tf_{{tf}}_")]].copy()
        base = pd.merge_asof(base.sort_values("timestamp"), tf_df.sort_values("timestamp"), on="timestamp", direction="backward")

    base = base.ffill().bfill()
    print(f"Data loaded: {{len(base)}} rows x {{len(base.columns)}} columns")
    return base


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest():
    df = load_data()
    row_dicts = df.to_dict(orient="records")

    closes = df["tf_15m_close"].values
    opens  = df["tf_15m_open"].values
    highs  = df["tf_15m_high"].values
    lows   = df["tf_15m_low"].values
    atr14  = df["tf_15m_atr_14"].values
    atr_pct = df["tf_15m_atr_pct"].values

    balance = INITIAL_BAL
    equity = [balance]
    trades = []
    position = None
    cooldown_remaining = 0

    for i in range(WARMUP, len(df)):
        if position is not None:
            d = position["direction"]
            ep = position["entry_price"]
            sl = position["sl"]
            tp = position["tp"]
            qty = position["qty"]

            c_open, c_high, c_low, c_close = opens[i], highs[i], lows[i], closes[i]

            # Trailing stop
            if TRAIL_MULT > 0:
                if d == "long":
                    new_sl = c_close - atr14[i] * TRAIL_MULT
                    if new_sl > sl:
                        sl = new_sl
                        position["sl"] = sl
                else:
                    new_sl = c_close + atr14[i] * TRAIL_MULT
                    if new_sl < sl:
                        sl = new_sl
                        position["sl"] = sl

            exit_price = None
            if d == "long":
                if c_open <= sl: exit_price = c_open
                elif c_open >= tp: exit_price = c_open
                elif c_low <= sl: exit_price = sl
                elif c_high >= tp: exit_price = tp
            else:
                if c_open >= sl: exit_price = c_open
                elif c_open <= tp: exit_price = c_open
                elif c_high >= sl: exit_price = sl
                elif c_low <= tp: exit_price = tp

            if exit_price is not None:
                if d == "long":
                    actual_exit = exit_price * (1 - SLIPPAGE)
                    pnl = (actual_exit - ep) * qty
                else:
                    actual_exit = exit_price * (1 + SLIPPAGE)
                    pnl = (ep - actual_exit) * qty

                fees = (qty * ep + qty * actual_exit) * FEE
                pnl -= fees
                is_win = pnl > 0

                balance += pnl
                if balance <= 0:
                    balance = 0.01
                trades.append({{"pnl": pnl, "win": is_win}})
                equity.append(balance)
                position = None
                cooldown_remaining = COOLDOWN
                continue

        if position is None and cooldown_remaining <= 0:
            if atr_pct[i] < ATR_GATE:
                cooldown_remaining = max(0, cooldown_remaining - 1)
                continue

            row = row_dicts[i]
            try:
                buy_sig = eval(BUY_CONDITIONS, {{"__builtins__": {{}}}}, row)
                sell_sig = eval(SELL_CONDITIONS, {{"__builtins__": {{}}}}, row)
            except:
                cooldown_remaining = max(0, cooldown_remaining - 1)
                continue

            entry_base = closes[i]
            atr_now = atr14[i]

            if buy_sig and not sell_sig:
                actual_entry = entry_base * (1 + SLIPPAGE)
                sl = actual_entry - atr_now * SL_MULT
                tp = actual_entry + atr_now * RR_RATIO
                risk = balance * RISK_PCT
                qty = risk / (actual_entry - sl) if actual_entry > sl else 0
                if qty > 0:
                    position = {{"entry_price": actual_entry, "sl": sl, "tp": tp, "direction": "long", "qty": qty}}

            elif sell_sig and not buy_sig:
                actual_entry = entry_base * (1 - SLIPPAGE)
                sl = actual_entry + atr_now * SL_MULT
                tp = actual_entry - atr_now * RR_RATIO
                risk = balance * RISK_PCT
                qty = risk / (sl - actual_entry) if sl > actual_entry else 0
                if qty > 0:
                    position = {{"entry_price": actual_entry, "sl": sl, "tp": tp, "direction": "short", "qty": qty}}

        cooldown_remaining = max(0, cooldown_remaining - 1)

    # ─── Results ─────────────────────────────────────────────────────────
    if not trades:
        print("No trades generated.")
        return

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    total_return = ((balance / INITIAL_BAL) - 1) * 100
    days = len(df) * 15 / (60 * 24)
    years = max(days / 365.25, 0.1)
    cagr = ((balance / INITIAL_BAL) ** (1 / years) - 1) * 100
    monthly_cagr = ((1 + cagr / 100) ** (1/12) - 1) * 100
    wr = len(wins) / len(trades) * 100
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    max_dd = np.max(dd)

    print(f"\\n{'='*60}")
    print(f"BACKTEST RESULTS — {{ASSET}}")
    print(f"{'='*60}")
    print(f"Total Trades:      {{len(trades)}}")
    print(f"Win Rate:          {{wr:.1f}}%")
    print(f"Annual CAGR:       {{cagr:.1f}}%")
    print(f"Monthly CAGR:      {{monthly_cagr:.2f}}%")
    print(f"Profit Factor:     {{pf:.2f}}")
    print(f"Max Drawdown:      {{max_dd:.1f}}%")
    print(f"Final Balance:     ${{balance:,.2f}}")
    print(f"Total Return:      {{total_return:.1f}}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_backtest()
'''
    return script


def _send_backtest_file(strategy: Strategy) -> None:
    """Send the standalone backtest script as a Telegram document."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        content = _generate_backtest_script(strategy)
        filename = f"backtest_{strategy.name}_{strategy.id[:8]}.py"

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        resp = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": f"\U0001f4ce Standalone backtest for {strategy.name} \u2014 run with: python {filename}"
        }, files={
            "document": (filename, content.encode("utf-8"), "text/plain")
        }, timeout=15)

        if resp.status_code == 200:
            log.info(f"Backtest file sent for {strategy.name}")
        else:
            log.warning(f"Backtest file send failed: {resp.status_code}")

    except Exception as e:
        log.error(f"Backtest file error: {e}")
