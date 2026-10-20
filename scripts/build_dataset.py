"""Build (or just locate) the aligned news + market dataset.

    from scripts.build_dataset import ensure_dataset
    paths = ensure_dataset()            # downloads + builds on first call, no-op afterwards

CLI:  python -m scripts.build_dataset [--force] [--start 1990-01-01] [--end 2025-12-31]
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import align as A
from . import config as C
from . import market as M
from . import news as N

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                            datefmt="%H:%M:%S", stream=sys.stdout)
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3", "yfinance", "peewee", "kagglehub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def ensure_dataset(force: bool = False, start: str = C.START_DATE, end: str | None = C.END_DATE,
                   verbose: bool = True) -> dict[str, Path]:
    """Return paths of the processed files, building them first if they are missing."""
    if verbose:
        _setup_logging()
    if not force and all(p.exists() for p in C.PROCESSED_FILES.values()):
        if _processed_covers(start, end):
            log.info("dataset found in %s - reusing (pass force=True to rebuild)", C.PROCESSED_DIR)
            return dict(C.PROCESSED_FILES)
        log.info("processed files exist but do not cover %s -> %s; rebuilding", start, end or "today")
    return build(start=start, end=end, force=force)


def _processed_covers(start: str, end: str | None) -> bool:
    try:
        meta = json.loads(C.PROCESSED_FILES["meta"].read_text(encoding="utf-8"))
        have = meta.get("period") or {}
        if not have.get("start"):
            return False
        # first trading day may be a few calendar days after START_DATE (New Year / weekend)
        if pd.Timestamp(have["start"]) > pd.Timestamp(start) + pd.Timedelta(days=5):
            return False
        if end and have.get("end") and pd.Timestamp(have["end"]) < pd.Timestamp(end) - pd.Timedelta(days=14):
            return False
        return True
    except Exception:
        return False


def build(start: str = C.START_DATE, end: str | None = C.END_DATE, force: bool = False) -> dict[str, Path]:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. sources ------------------------------------------------------------------------
    news, news_end, news_info = N.build_news(start, end, force=force)
    prices = M.build_market(start, end, force=force)

    # 2. common period: both series must be complete on every day we keep --------------
    end_date = min(news_end, A.last_complete_price_date(prices))
    if end:
        end_date = min(end_date, pd.Timestamp(end))
    calendar = A.trading_calendar(prices, start, end_date)
    log.info("common period %s -> %s: %d trading days", calendar[0].date(), calendar[-1].date(), len(calendar))

    # 3. align news to trading days -----------------------------------------------------
    news["trading_date"] = A.assign_trading_date(news["pub_time_utc"], calendar)
    news = news.dropna(subset=["trading_date"]).reset_index(drop=True)  # published after the last close
    prices = prices[prices["date"] <= end_date].reset_index(drop=True)
    wide = A.prices_wide(prices, calendar)
    daily = A.build_daily(news, prices, calendar)

    # 4. write ---------------------------------------------------------------------------
    news.to_parquet(C.PROCESSED_FILES["news"], index=False)
    prices.to_parquet(C.PROCESSED_FILES["prices"], index=False)
    wide.to_parquet(C.PROCESSED_FILES["prices_wide"])
    daily.to_parquet(C.PROCESSED_FILES["daily"], index=False)

    cov = M.coverage(prices)
    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "period": {"start": str(calendar[0].date()), "end": str(calendar[-1].date()), "trading_days": int(len(calendar))},
        "news": {
            **news_info,
            "articles": int(len(news)),
            "finance_articles": int(news["is_finance"].sum()),
            "articles_by_flag_source": {k: int(v) for k, v in news["flag_source"].value_counts().items()},
            "first_pub_time_utc": news["pub_time_utc"].min().isoformat(),
            "last_pub_time_utc": news["pub_time_utc"].max().isoformat(),
            "articles_per_trading_day": round(len(news) / len(calendar), 1),
            "finance_articles_per_trading_day": round(float(news["is_finance"].sum()) / len(calendar), 1),
        },
        "market": {
            "source": "yahoo finance (yfinance)",
            "tickers": {t: {"name": C.TICKERS[t]["name"], "group": C.TICKERS[t]["group"],
                            "first": str(cov.loc[t, "first"].date()), "last": str(cov.loc[t, "last"].date()),
                            "n_obs": int(cov.loc[t, "n_obs"])} for t in cov.index},
            "core_tickers": [t for t in C.CORE_TICKERS if t in cov.index],
        },
        "files": {k: str(v.relative_to(C.ROOT)) for k, v in C.PROCESSED_FILES.items()},
    }
    C.PROCESSED_FILES["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("done: %d articles (%d finance), %d tickers, %d trading days -> %s",
             len(news), news["is_finance"].sum(), cov.shape[0], len(calendar), C.PROCESSED_DIR)
    return dict(C.PROCESSED_FILES)


def load(paths: dict[str, Path] | None = None) -> dict[str, pd.DataFrame | dict]:
    """Read every processed file into memory."""
    paths = paths or C.PROCESSED_FILES
    return {
        "news": pd.read_parquet(paths["news"]),
        "prices": pd.read_parquet(paths["prices"]),
        "prices_wide": pd.read_parquet(paths["prices_wide"]),
        "daily": pd.read_parquet(paths["daily"]),
        "meta": json.loads(Path(paths["meta"]).read_text(encoding="utf-8")),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild even if processed files exist")
    ap.add_argument("--start", default=C.START_DATE)
    ap.add_argument("--end", default=C.END_DATE)
    args = ap.parse_args(argv)
    ensure_dataset(force=args.force, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
