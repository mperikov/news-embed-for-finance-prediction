"""Daily market data from Yahoo Finance (via yfinance) -> long OHLCV table.

Each ticker is cached in data/raw/market/<ticker>.parquet and re-downloaded only when
the cache is older than one day (or `force=True`).
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "_").replace("=", "_").replace(".", "_")
    return C.RAW_MARKET_DIR / f"{safe}.parquet"


def _is_fresh(path: Path, max_age_days: int = 1) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400


def _cache_covers(df: pd.DataFrame, ticker: str, start: str) -> bool:
    """True if the cached history reaches back to max(requested start, ticker birth)."""
    first = C.TICKERS.get(ticker, {}).get("first", "1900")
    first_ts = pd.Timestamp(first if len(str(first)) >= 8 else f"{first}-01-01")
    want = max(pd.Timestamp(start), first_ts)
    return pd.Timestamp(df["date"].min()) <= want + pd.Timedelta(days=60)


def download_ticker(ticker: str, start: str, end: str | None = None, retries: int = 3) -> pd.DataFrame:
    """Full daily history of one ticker as a tidy frame with PRICE_COLUMNS."""
    import yfinance as yf

    end_excl = (pd.Timestamp(end or date.today()) + timedelta(days=1)).strftime("%Y-%m-%d")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end_excl, auto_adjust=False, actions=False)
            break
        except Exception as e:  # network hiccups / Yahoo throttling
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"yfinance failed for {ticker}: {last_err}")

    if hist is None or hist.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    hist = hist.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                                "Adj Close": "adj_close", "Volume": "volume"})
    if "adj_close" not in hist:  # indices / FX have no dividends: adjusted == close
        hist["adj_close"] = hist["close"]
    idx = pd.DatetimeIndex(hist.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = hist[["open", "high", "low", "close", "adj_close", "volume"]].copy()
    out.insert(0, "ticker", ticker)
    out.insert(0, "date", idx.normalize())
    out = out.reset_index(drop=True)
    out = out.dropna(subset=["close"])
    out = out[~out["date"].duplicated(keep="last")]
    return out[PRICE_COLUMNS]


def build_market(start: str = C.START_DATE, end: str | None = C.END_DATE,
                 tickers: dict[str, dict] | None = None, force: bool = False) -> pd.DataFrame:
    """Download (or load from cache) every ticker in `tickers` and return one long table."""
    tickers = tickers or C.TICKERS
    C.RAW_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        path = _cache_path(ticker)
        df = pd.read_parquet(path) if path.exists() else None
        if df is not None and _is_fresh(path) and not force and _cache_covers(df, ticker, start):
            pass
        else:
            try:
                df = download_ticker(ticker, start, end)
            except Exception as e:
                if path.exists():
                    log.warning("%s: download failed (%s); using stale cache", ticker, e)
                    df = pd.read_parquet(path)
                else:
                    log.error("%s: download failed (%s); ticker skipped", ticker, e)
                    continue
            df.to_parquet(path, index=False)
        log.info("%-9s %5d rows  %s -> %s", ticker, len(df), df["date"].min().date(), df["date"].max().date())
        frames.append(df)
    if not frames:
        raise RuntimeError("no market data could be downloaded")
    prices = pd.concat(frames, ignore_index=True)
    prices = prices[prices["date"] >= pd.Timestamp(start)]
    if end:
        prices = prices[prices["date"] <= pd.Timestamp(end)]
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def coverage(prices: pd.DataFrame) -> pd.DataFrame:
    """First/last date and number of observations per ticker."""
    g = prices.groupby("ticker")["date"]
    cov = pd.DataFrame({"first": g.min(), "last": g.max(), "n_obs": g.size()})
    cov["name"] = [C.TICKERS.get(t, {}).get("name", "") for t in cov.index]
    cov["group"] = [C.TICKERS.get(t, {}).get("group", "") for t in cov.index]
    return cov[["name", "group", "first", "last", "n_obs"]].sort_values(["group", "first"])
