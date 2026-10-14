"""Put news and prices on one clock.

* trading calendar  = days on which the S&P 500 has a close (NYSE sessions)
* information window of trading day D = (close of D-1, close of D]  (close = 16:00 New York)
  An article published at time t belongs to the first trading day whose close is >= t,
  so it can only influence returns that are realised after it was public:
  weekend / holiday / after-hours news is carried to the next session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .news import article_text


def trading_calendar(prices: pd.DataFrame, start=None, end=None) -> pd.DatetimeIndex:
    d = prices.loc[prices["ticker"] == C.CALENDAR_TICKER, "date"].drop_duplicates().sort_values()
    if start is not None:
        d = d[d >= pd.Timestamp(start)]
    if end is not None:
        d = d[d <= pd.Timestamp(end)]
    return pd.DatetimeIndex(d.to_numpy(), name="date")


def session_close_utc(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    hh, mm = (int(x) for x in C.NYSE_CLOSE.split(":"))
    local = pd.DatetimeIndex(calendar).tz_localize(None) + pd.Timedelta(hours=hh, minutes=mm)
    return local.tz_localize(C.MARKET_TZ).tz_convert("UTC")


def assign_trading_date(pub_time_utc: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Trading day whose information window contains each publication time (NaT if after the last close)."""
    closes = session_close_utc(calendar).as_unit("ns").asi8
    t = pd.DatetimeIndex(pub_time_utc).tz_convert("UTC").as_unit("ns").asi8
    idx = np.searchsorted(closes, t, side="left")
    inside = idx < len(calendar)
    out = np.full(len(t), np.datetime64("NaT", "ns"))
    out[inside] = calendar.as_unit("ns").to_numpy()[idx[inside]]
    return pd.Series(out, index=pub_time_utc.index, name="trading_date")


def prices_wide(prices: pd.DataFrame, calendar: pd.DatetimeIndex, column: str = "adj_close") -> pd.DataFrame:
    wide = prices.pivot(index="date", columns="ticker", values=column).reindex(calendar)
    wide.columns.name = None
    return wide[[t for t in C.TICKERS if t in wide.columns]]


def build_daily(news: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per trading day: S&P 500 close / log-return, article counts, concatenated finance texts."""
    spx = (prices[prices["ticker"] == C.CALENDAR_TICKER].set_index("date")["adj_close"]).reindex(calendar)
    daily = pd.DataFrame(index=calendar)
    daily["spx_close"] = spx
    daily["spx_logret"] = np.log(spx).diff()
    daily["n_articles"] = news.groupby("trading_date").size().reindex(calendar).fillna(0).astype(int)
    fin = news[news["is_finance"]]
    daily["n_finance"] = fin.groupby("trading_date").size().reindex(calendar).fillna(0).astype(int)
    texts = article_text(fin).groupby(fin["trading_date"]).agg("\n".join)
    daily["finance_text"] = texts.reindex(calendar).fillna("")
    return daily.reset_index()


def last_complete_price_date(prices: pd.DataFrame) -> pd.Timestamp:
    """Last calendar-ticker date, excluding today if the New York session has not closed yet."""
    last = prices.loc[prices["ticker"] == C.CALENDAR_TICKER, "date"].max()
    now_ny = pd.Timestamp.now(tz=C.MARKET_TZ)
    if last.normalize() == now_ny.normalize().tz_localize(None) and now_ny.strftime("%H:%M") < "16:30":
        last = prices.loc[(prices["ticker"] == C.CALENDAR_TICKER) & (prices["date"] < last), "date"].max()
    return last
