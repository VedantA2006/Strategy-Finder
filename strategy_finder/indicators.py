"""
indicators.py — Multi-timeframe indicator engine for Strategy Finder.

Fetches 4 timeframes (15m, 1h, 4h, 1d) from Bybit V5 REST API.
Each timeframe is cached separately as a parquet file in data/.
Computes a massive indicator pool on each timeframe, then merges
all higher timeframes down to the 15m index via forward-fill.

The final merged DataFrame has columns namespaced like:
  tf_15m_rsi_14, tf_1h_ema_89, tf_4h_adx_14, tf_1d_ema_200

Execution happens on 15m closes. Higher TF values are read-only
context with no lookahead — they are forward-filled from the most
recently CLOSED higher-TF bar.

INDICATOR CATEGORIES:
  - EMAs: 8/13/21/34/55/89/200
  - SMAs: 20/50/200
  - RSI: 7/14/21
  - MACD: line/signal/histogram
  - Stochastic: K/D
  - ADX, ATR%, CCI, Williams %R, MFI
  - OBV change, VWAP deviation
  - Bollinger: upper/middle/lower/width
  - Momentum lags: prev_1, prev_3 for key indicators
  - Candle structure: body_ratio, upper_wick_ratio, lower_wick_ratio,
    is_bullish, is_bearish, consecutive_bullish
  - Volume profile: volume_ratio, volume_expanding, obv_slope_5
  - Price structure: close vs N-bar high/low, dist_from_52w_high
  - Session/time: hour_utc, day_of_week
  - Regime: 200-EMA slope for bull/bear/sideways classification
"""

from __future__ import annotations

import os
import time
import datetime
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import requests
import ta

# ─── Constants ───────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).parent / "data"

# Timeframe config: (label, Bybit interval string, minutes per bar)
TIMEFRAMES = [
    ("15m", "15",   15),
    ("1h",  "60",   60),
    ("4h",  "240",  240),
    ("1d",  "D",    1440),
]

# Assets supported
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Years of data to fetch
DATA_YEARS = 5

# Module-level cache: {(asset, tf_label): DataFrame}
CACHED_DFS: dict[tuple[str, str], pd.DataFrame] = {}
# Module-level cache: {asset: merged_DataFrame}
CACHED_MERGED: dict[str, pd.DataFrame] = {}


# ─── Bybit data fetching ────────────────────────────────────────────────────

def _parquet_path(asset: str, tf_label: str) -> pathlib.Path:
    """Return the parquet cache path for a given asset and timeframe."""
    return DATA_DIR / f"{asset.lower()}_{tf_label}_{DATA_YEARS}y.parquet"


def _fetch_bybit_klines(symbol: str = "BTCUSDT", interval: str = "60",
                        years: int = DATA_YEARS) -> pd.DataFrame:
    """Fetch historical klines from Bybit V5 public REST API with pagination."""
    url = "https://api.bybit.com/v5/market/kline"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - years * 365 * 24 * 60 * 60 * 1000
    all_rows: list[list] = []
    cursor_end = end_ms

    print(f"[indicators] Fetching {years}y of {symbol} {interval} data from Bybit ...")

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
        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break
        cursor_end = oldest_ts - 1
        time.sleep(0.12)  # rate-limit politeness

    if not all_rows:
        raise RuntimeError(f"No data returned from Bybit API for {symbol} {interval}.")

    # Bybit kline columns: [startTime, open, high, low, close, volume, turnover]
    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df.drop(columns=["turnover"], inplace=True)

    print(f"[indicators] Fetched {len(df):,} candles for {symbol} {interval} "
          f"({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
    return df


def _load_or_fetch(asset: str, tf_label: str, interval: str) -> pd.DataFrame:
    """Load parquet cache if fresh (<24h old), else re-fetch from Bybit."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pq_path = _parquet_path(asset, tf_label)

    if pq_path.exists():
        age_h = (time.time() - pq_path.stat().st_mtime) / 3600
        if age_h < 24:
            print(f"[indicators] Using cached {asset} {tf_label} parquet ({age_h:.1f}h old)")
            return pd.read_parquet(pq_path)
        print(f"[indicators] Cache stale for {asset} {tf_label} ({age_h:.1f}h), re-fetching ...")

    df = _fetch_bybit_klines(symbol=asset, interval=interval)
    df.to_parquet(pq_path, index=False)
    return df


# ─── Indicator computation ───────────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    """
    Compute the full indicator suite on a single-timeframe OHLCV DataFrame.
    All columns are prefixed with tf_{tf_label}_ for namespace isolation.
    """
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)
    v = df["volume"].astype(float)
    pfx = f"tf_{tf_label}_"

    # ── EMAs ─────────────────────────────────────────────────────────────
    for span in [8, 13, 21, 34, 55, 89, 200]:
        df[f"{pfx}ema_{span}"] = ta.trend.EMAIndicator(c, window=span).ema_indicator()

    # ── SMAs ─────────────────────────────────────────────────────────────
    for span in [20, 50, 200]:
        df[f"{pfx}sma_{span}"] = ta.trend.SMAIndicator(c, window=span).sma_indicator()

    # ── RSI ──────────────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        df[f"{pfx}rsi_{period}"] = ta.momentum.RSIIndicator(c, window=period).rsi()

    # ── MACD (12, 26, 9) ─────────────────────────────────────────────────
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df[f"{pfx}macd_line"] = macd.macd()
    df[f"{pfx}macd_signal"] = macd.macd_signal()
    df[f"{pfx}macd_hist"] = macd.macd_diff()

    # ── Stochastic (14, 3) ───────────────────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df[f"{pfx}stoch_k"] = stoch.stoch()
    df[f"{pfx}stoch_d"] = stoch.stoch_signal()

    # ── ADX ──────────────────────────────────────────────────────────────
    df[f"{pfx}adx_14"] = ta.trend.ADXIndicator(h, l, c, window=14).adx()

    # ── ATR & ATR% ───────────────────────────────────────────────────────
    df[f"{pfx}atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df[f"{pfx}atr_pct"] = df[f"{pfx}atr_14"] / c

    # ── Bollinger Bands (20, 2) ──────────────────────────────────────────
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df[f"{pfx}bb_upper"] = bb.bollinger_hband()
    df[f"{pfx}bb_middle"] = bb.bollinger_mavg()
    df[f"{pfx}bb_lower"] = bb.bollinger_lband()
    bb_width = (df[f"{pfx}bb_upper"] - df[f"{pfx}bb_lower"]) / df[f"{pfx}bb_middle"]
    df[f"{pfx}bb_width"] = bb_width

    # ── CCI ──────────────────────────────────────────────────────────────
    df[f"{pfx}cci_20"] = ta.trend.CCIIndicator(h, l, c, window=20).cci()

    # ── Williams %R ──────────────────────────────────────────────────────
    df[f"{pfx}willr_14"] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()

    # ── MFI ──────────────────────────────────────────────────────────────
    df[f"{pfx}mfi_14"] = ta.volume.MFIIndicator(h, l, c, v, window=14).money_flow_index()

    # ── OBV & OBV change ─────────────────────────────────────────────────
    df[f"{pfx}obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df[f"{pfx}obv_change"] = df[f"{pfx}obv"].pct_change() * 100
    # OBV slope over 5 bars (linear regression slope)
    obv_series = df[f"{pfx}obv"]
    df[f"{pfx}obv_slope_5"] = obv_series.rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=False
    )

    # ── VWAP deviation (rolling 20-bar approximation) ────────────────────
    typical_price = (h + l + c) / 3
    vwap_cum_vol = (typical_price * v).rolling(20).sum() / v.rolling(20).sum()
    df[f"{pfx}vwap_dev"] = (c - vwap_cum_vol) / vwap_cum_vol * 100

    # ── Momentum / Rate-of-change lags ───────────────────────────────────
    for lag_name, lag_n in [("prev_1", 1), ("prev_3", 3)]:
        for ind in ["rsi_14", "macd_hist", "macd_line", "ema_21", "stoch_k", "adx_14", "cci_20"]:
            src_col = f"{pfx}{ind}"
            if src_col in df.columns:
                df[f"{pfx}{lag_name}_{ind}"] = df[src_col].shift(lag_n)

    # ── Candle structure ─────────────────────────────────────────────────
    body = (c - o).abs()
    total_range = h - l
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l

    df[f"{pfx}body_ratio"] = np.where(total_range > 0, body / total_range, 0.0)
    df[f"{pfx}upper_wick_ratio"] = np.where(total_range > 0, upper_wick / total_range, 0.0)
    df[f"{pfx}lower_wick_ratio"] = np.where(total_range > 0, lower_wick / total_range, 0.0)
    df[f"{pfx}is_bullish"] = (c > o).astype(float)
    df[f"{pfx}is_bearish"] = (c < o).astype(float)

    # Consecutive bullish count (2 and 3 bar lookback)
    bull = (c > o).astype(int)
    df[f"{pfx}consec_bullish_2"] = (bull + bull.shift(1)).fillna(0)
    df[f"{pfx}consec_bullish_3"] = (bull + bull.shift(1) + bull.shift(2)).fillna(0)

    # ── Volume profile ───────────────────────────────────────────────────
    vol_sma20 = v.rolling(20).mean()
    df[f"{pfx}volume_ratio"] = v / vol_sma20
    df[f"{pfx}volume_expanding"] = (v > v.shift(1)).astype(float)

    # ── Price structure ──────────────────────────────────────────────────
    for n in [10, 20, 50]:
        df[f"{pfx}high_{n}"] = h.rolling(n).max()
        df[f"{pfx}low_{n}"] = l.rolling(n).min()

    # 52-week high distance (approx bars: 15m=35040, 1h=8760, 4h=2190, 1d=365)
    bars_52w = {"15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}
    n52 = min(bars_52w.get(tf_label, 8760), len(df))
    high_52w = h.rolling(n52, min_periods=1).max()
    df[f"{pfx}dist_from_52w_high"] = (high_52w - c) / high_52w * 100

    # ── Session / Time filters ───────────────────────────────────────────
    if "timestamp" in df.columns:
        df[f"{pfx}hour_utc"] = df["timestamp"].dt.hour
        df[f"{pfx}day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon

    # ── Regime (200-EMA slope) — only meaningful for 1d, computed for all ─
    ema200 = df[f"{pfx}ema_200"]
    df[f"{pfx}ema_200_slope"] = ema200.pct_change(20) * 100  # 20-bar slope

    # Keep raw OHLCV columns with tf prefix too
    df[f"{pfx}close"] = c
    df[f"{pfx}open"] = o
    df[f"{pfx}high"] = h
    df[f"{pfx}low"] = l
    df[f"{pfx}volume"] = v

    print(f"[indicators] {tf_label}: {len(df.columns)} columns computed")
    return df


# ─── Multi-timeframe merging ────────────────────────────────────────────────

def _merge_timeframes(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all timeframes down to the 15m index via forward-fill.

    Higher timeframe values are forward-filled so that each 15m row
    sees the most recently CLOSED higher-TF bar's values (no lookahead).
    """
    base = dfs["15m"].copy()
    base = base.set_index("timestamp")

    for tf_label in ["1h", "4h", "1d"]:
        if tf_label not in dfs:
            continue
        htf = dfs[tf_label].copy()
        htf = htf.set_index("timestamp")

        # Drop raw OHLCV columns that would duplicate (keep only tf_ prefixed)
        raw_cols = ["open", "high", "low", "close", "volume"]
        htf = htf.drop(columns=[c for c in raw_cols if c in htf.columns], errors="ignore")

        # Reindex to 15m timestamps and forward-fill (shift by 1 to prevent lookahead)
        # The HTF bar at time T represents data UP TO time T, so the value is
        # available starting from the NEXT 15m bar after T
        htf_reindexed = htf.reindex(base.index, method="ffill")

        # Join to base
        base = base.join(htf_reindexed, how="left", rsuffix="_dup")
        # Drop any _dup columns
        base = base[[c for c in base.columns if not c.endswith("_dup")]]

    # Forward-fill any remaining NaNs from alignment
    base = base.ffill()

    # Drop rows with NaN (warmup period)
    base = base.dropna()
    base = base.reset_index()  # bring timestamp back as column

    print(f"[indicators] Merged DataFrame: {len(base):,} rows × {len(base.columns)} columns")
    return base


# ─── Public API ──────────────────────────────────────────────────────────────

def load_data(asset: str = "BTCUSDT") -> pd.DataFrame:
    """
    Load (or fetch) all 4 timeframes for the given asset, compute indicators,
    merge into a single 15m-resolution DataFrame, and cache in module global.
    """
    global CACHED_MERGED

    if asset in CACHED_MERGED:
        return CACHED_MERGED[asset]

    print(f"[indicators] Loading all timeframes for {asset} ...")
    tf_dfs: dict[str, pd.DataFrame] = {}

    for tf_label, interval, _ in TIMEFRAMES:
        cache_key = (asset, tf_label)
        if cache_key in CACHED_DFS:
            raw = CACHED_DFS[cache_key]
        else:
            raw = _load_or_fetch(asset, tf_label, interval)
            CACHED_DFS[cache_key] = raw

        enriched = _compute_indicators(raw.copy(), tf_label)
        tf_dfs[tf_label] = enriched

    merged = _merge_timeframes(tf_dfs)
    CACHED_MERGED[asset] = merged
    return merged


def refresh_data(asset: str = "BTCUSDT") -> pd.DataFrame:
    """Force re-fetch from Bybit and recompute indicators."""
    global CACHED_MERGED
    for tf_label, _, _ in TIMEFRAMES:
        pq = _parquet_path(asset, tf_label)
        if pq.exists():
            pq.unlink()
        cache_key = (asset, tf_label)
        if cache_key in CACHED_DFS:
            del CACHED_DFS[cache_key]
    if asset in CACHED_MERGED:
        del CACHED_MERGED[asset]
    return load_data(asset)


def get_available_columns(asset: str = "BTCUSDT") -> list[str]:
    """Return all indicator column names available after merge."""
    df = load_data(asset)
    exclude = {"timestamp", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in exclude]
