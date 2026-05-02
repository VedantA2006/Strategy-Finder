"""
indicators.py — Compute all indicators on the full 5-year 1h BTCUSDT DataFrame.

Fetches data from Bybit public REST API, caches to data/btcusdt_1h_5y.parquet,
computes every required indicator once, and stores the enriched DataFrame in
module-level CACHED_DF for reuse across backtest runs.
"""

import os
import time
import datetime
import pathlib

import numpy as np
import pandas as pd
import requests
import ta

# ─── Module-level cache ─────────────────────────────────────────────────────
CACHED_DF: pd.DataFrame | None = None
DATA_DIR = pathlib.Path(__file__).parent / "data"
PARQUET_PATH = DATA_DIR / "btcusdt_1h_5y.parquet"


# ─── Bybit data fetching ────────────────────────────────────────────────────
def _fetch_bybit_klines(symbol: str = "BTCUSDT", interval: str = "60",
                        years: int = 5) -> pd.DataFrame:
    """Fetch historical klines from Bybit V5 public REST API with pagination."""
    url = "https://api.bybit.com/v5/market/kline"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - years * 365 * 24 * 60 * 60 * 1000
    all_rows: list[list] = []
    cursor_end = end_ms

    print(f"[indicators] Fetching {years}y of {symbol} {interval}m data from Bybit ...")

    while cursor_end > start_ms:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": str(start_ms),
            "end": str(cursor_end),
            "limit": "1000",
        }
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(f"Bybit API failed after 5 retries: {exc}")
                time.sleep(2 ** attempt)

        rows = data.get("result", {}).get("list", [])
        if not rows:
            break

        all_rows.extend(rows)
        # Bybit returns newest-first; last element is the oldest in this page
        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break
        cursor_end = oldest_ts - 1
        time.sleep(0.12)  # rate-limit politeness

    if not all_rows:
        raise RuntimeError("No data returned from Bybit API.")

    # Bybit kline columns: [startTime, open, high, low, close, volume, turnover]
    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df.drop(columns=["turnover"], inplace=True)

    print(f"[indicators] Fetched {len(df):,} candles "
          f"({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
    return df


def _load_or_fetch() -> pd.DataFrame:
    """Load parquet cache if fresh (< 24 h old), else re-fetch from Bybit."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if PARQUET_PATH.exists():
        age_h = (time.time() - PARQUET_PATH.stat().st_mtime) / 3600
        if age_h < 24:
            print(f"[indicators] Using cached parquet ({age_h:.1f} h old)")
            return pd.read_parquet(PARQUET_PATH)
        print(f"[indicators] Cache stale ({age_h:.1f} h), re-fetching ...")

    df = _fetch_bybit_klines()
    df.to_parquet(PARQUET_PATH, index=False)
    return df


# ─── Indicator computation ───────────────────────────────────────────────────
def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all required indicators to the raw OHLCV DataFrame."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # EMAs
    for span in [8, 13, 21, 34, 55, 89, 200]:
        df[f"ema_{span}"] = ta.trend.EMAIndicator(c, window=span).ema_indicator()

    # SMAs
    for span in [20, 50, 200]:
        df[f"sma_{span}"] = ta.trend.SMAIndicator(c, window=span).sma_indicator()

    # RSI
    for period in [7, 14, 21]:
        df[f"rsi_{period}"] = ta.momentum.RSIIndicator(c, window=period).rsi()

    # MACD (12, 26, 9)
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Stochastic (14, 3)
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ATR — THE ONLY SL/TP DISTANCE SOURCE
    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["atr_pct"] = df["atr_14"] / c

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    # ADX
    df["adx_14"] = ta.trend.ADXIndicator(h, l, c, window=14).adx()

    # Volume ratio
    vol_sma = ta.trend.SMAIndicator(v.astype(float), window=20).sma_indicator()
    df["volume_ratio"] = v / vol_sma

    # Candle body ratio
    body = (c - df["open"]).abs()
    wick = h - l
    df["candle_body_ratio"] = np.where(wick > 0, body / wick, 0.0)

    # Drop warmup NaN rows
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[indicators] Indicators computed — {len(df):,} rows remain after NaN drop")
    return df


# ─── Public API ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load (or fetch) data, compute indicators, and cache in module global."""
    global CACHED_DF
    if CACHED_DF is not None:
        return CACHED_DF
    raw = _load_or_fetch()
    CACHED_DF = _compute_indicators(raw)
    return CACHED_DF


def refresh_data() -> pd.DataFrame:
    """Force re-fetch from Bybit and recompute indicators. For daily cron."""
    global CACHED_DF
    if PARQUET_PATH.exists():
        PARQUET_PATH.unlink()
    CACHED_DF = None
    return load_data()
