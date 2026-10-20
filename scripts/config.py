"""Central configuration: paths, period, instruments and the finance-news filter."""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]

try:  # optional: read NYT_API_KEY from <root>/.env
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without developer mode
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_KAGGLE_DIR = RAW_DIR / "kaggle_nyt"
RAW_HF_DIR = RAW_DIR / "hf_nyt"
RAW_NYT_API_DIR = RAW_DIR / "nyt_api"
RAW_MARKET_DIR = RAW_DIR / "market"

PROCESSED_FILES = {
    "news": PROCESSED_DIR / "news_articles.parquet",   # one row per article, with trading_date
    "prices": PROCESSED_DIR / "market_prices.parquet",  # long OHLCV table, one row per (date, ticker)
    "prices_wide": PROCESSED_DIR / "prices_wide.parquet",  # adjusted close, trading days x tickers
    "daily": PROCESSED_DIR / "daily.parquet",           # one row per trading day: news + S&P 500
    "meta": PROCESSED_DIR / "meta.json",
}
EMBED_DIR = PROCESSED_DIR / "embeddings"  # cached raw embeddings (sentence-transformers, …)
OVERNIGHT_FILE = PROCESSED_DIR / "overnight_finance_text.parquet"

# --------------------------------------------------------------------------- period
START_DATE = "1990-01-01"
END_DATE: str | None = None  # None -> today; news and prices are cut to their common range

# --------------------------------------------------------------------------- news
# 1. Mirror of the NYT Archive API with full metadata (headline, abstract, lead paragraph,
#    section, desk, keywords, ...) for every article since 2000-01-01; its daily updates
#    stopped in 2023 (detected automatically). CC0, ~0.9 GB zip / 4.6 GB csv.
KAGGLE_NYT_DATASET = "aryansingh0909/nyt-articles-21m-2000-present"
# 2. Keyless fill for the months after that: headline + abstract for every NYT article
#    1927 -> today (yearly parquet, ~7 MB per recent year). No section metadata.
HF_NYT_DATASET = "bguzzo2k/nyt_100y_news_headlines"
# 3. Optional: with a (free) key the fill is taken from the NYT Archive API instead,
#    with full metadata and up to yesterday.
NYT_API_KEY = os.environ.get("NYT_API_KEY") or None
NYT_API_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
NYT_API_SLEEP_SECONDS = 12.5  # rate limit: 5 requests / minute, 500 / day

# Rows without metadata (HF fill) get `keep` / `is_finance` from a text classifier trained on
# the labelled rows of the preceding years (see textflags.py).
TEXT_MODEL_TRAIN_YEARS = 8

# Material that is not news (dropped from the article table).
DROP_TYPE_OF_MATERIAL = {
    "Paid Death Notice", "Obituary", "Obituary (Obit)", "Obituary; Biography", "Biography; Obituary",
    "Correction", "Editors' Note", "Caption", "Op-Ed; Caption", "Caption; Op-Ed",
    "Schedule", "List", "Statistics", "Results Listing", "recipe", "Quote", "Question",
    "Letter", "Review", "Slideshow", "First Chapter", "Chronology", "Postscript",
}
# Culture / lifestyle / service sections: no information about markets, ~40% of all rows.
# (World, U.S., Business Day, Technology, Opinion, New York, Washington, Science, Health,
#  Climate, Real Estate, Education, Week in Review, The Upshot, Briefing, Archives, Blogs ... are kept.)
DROP_SECTIONS = {
    "Sports", "Arts", "Theater", "Movies", "Books", "Book Review", "Style", "Fashion & Style",
    "T Magazine", "T Brand", "Magazine", "Food", "Dining & Wine", "Travel", "Home & Garden",
    "Great Homes & Destinations", "Automobiles", "Crosswords & Games", "xword", "Corrections",
    "Obituaries", "Paid Death Notices", "Classifieds", "Multimedia/Photos", "Multimedia", "Slideshows",
    "Podcasts", "Video", "Lens", "Watching", "The Learning Network", "Learning", "Parenting", "Well",
    "Smarter Living", "At Home", "Weddings", "Society", "Times Insider", "Reader Center", "Admin",
    "Today\u2019s Paper", "Today's Paper", "Universal", "Neediest Cases", "Giving", "Booming",
    "UrbanEye", "Critic's Choice", "Guides", "Guide", "Gift Guide", "The New York Times Presents",
    "timestalks", "Feeds", "Polls", "Weather", "membercenter", "Topics", "Times Topics", "Homepage",
    "en Español", "Headway", "Burst", "College",
}

# "Finance" flag = business/markets/economy coverage. Precision over recall: the business
# section and desks, business url paths, or a markets/macro *subject* keyword.
FINANCE_SECTIONS = {"Business Day", "Business", "Your Money", "DealBook", "Economy", "Job Market"}
FINANCE_NEWS_DESKS = {
    "Business", "Business Day", "Business/Financial Desk", "Financial Desk", "Financial",
    "Money and Business/Financial Desk", "SundayBusiness", "Sunday Business", "Business Day Sunday",
    "Business / World Business", "Business / Media & Advertising", "Money", "Your Money",
    "DealBook", "Dealbook", "Economy", "Markets", "Personal Investing", "Mutual Funds",
    "Retirement", "Small Business", "Wealth", "Energy", "Media", "Job Market",
}
FINANCE_KEYWORDS = {
    "Stocks and Bonds", "Economic Conditions and Trends", "Interest Rates", "United States Economy",
    "Federal Reserve System", "Banking and Financial Institutions", "Credit and Debt",
    "Mergers, Acquisitions and Divestitures", "Company Reports", "Corporate Profits",
    "International Trade and World Market", "Currency", "Foreign Exchange", "Oil (Petroleum) and Gasoline",
    "Inflation (Economics)", "Recession and Depression", "Unemployment", "Hedge Funds", "Mutual Funds",
    "Mortgages", "Stock Exchanges", "Futures and Options Trading", "Commodities", "Gold",
    "Customs (Tariff)", "Securities and Commodities Violations", "Treasury Department",
    "Finances", "Consumer Behavior", "Retail Stores and Trade", "Gross Domestic Product",
    "Government Bonds", "Credit Ratings", "Dow Jones Stock Average", "Standard & Poor's 500-Stock Index",
    "Nasdaq Composite Index", "Bitcoin (Currency)", "Virtual Currency",
}

# --------------------------------------------------------------------------- market
NYSE_CLOSE = "16:00"           # local New York time; news after the close belongs to the next trading day
NYSE_OPEN = "09:30"            # overnight window is (previous close, this open]
MARKET_TZ = "America/New_York"
CALENDAR_TICKER = "^GSPC"      # trading calendar = days on which the S&P 500 has a close

# Yahoo Finance tickers. All are US-traded / US-calendar instruments with daily history
# covering (most of) 2000 -> today. `first` is the approximate first available date.
TICKERS: dict[str, dict] = {
    # US equity indices
    "^GSPC": dict(name="S&P 500", group="equity_index", first="1927"),
    "^DJI": dict(name="Dow Jones Industrial Average", group="equity_index", first="1992"),
    "^IXIC": dict(name="Nasdaq Composite", group="equity_index", first="1971"),
    "^RUT": dict(name="Russell 2000", group="equity_index", first="1987"),
    # Broad ETFs (total-return via adj_close)
    "SPY": dict(name="SPDR S&P 500 ETF", group="broad_etf", first="1993"),
    "QQQ": dict(name="Invesco Nasdaq-100 ETF", group="broad_etf", first="1999"),
    # SPDR Select Sector ETFs (since 1998-12)
    "XLF": dict(name="Financials", group="sector_etf", first="1998"),
    "XLK": dict(name="Technology", group="sector_etf", first="1998"),
    "XLE": dict(name="Energy", group="sector_etf", first="1998"),
    "XLV": dict(name="Health Care", group="sector_etf", first="1998"),
    "XLY": dict(name="Consumer Discretionary", group="sector_etf", first="1998"),
    "XLP": dict(name="Consumer Staples", group="sector_etf", first="1998"),
    "XLI": dict(name="Industrials", group="sector_etf", first="1998"),
    "XLB": dict(name="Materials", group="sector_etf", first="1998"),
    "XLU": dict(name="Utilities", group="sector_etf", first="1998"),
    # Volatility and Treasury yields (CBOE indices; levels, not prices)
    "^VIX": dict(name="CBOE Volatility Index", group="volatility", first="1990"),
    "^IRX": dict(name="13-week T-bill yield", group="rates", first="1960"),
    "^FVX": dict(name="5-year Treasury yield", group="rates", first="1962"),
    "^TNX": dict(name="10-year Treasury yield", group="rates", first="1962"),
    "^TYX": dict(name="30-year Treasury yield", group="rates", first="1977"),
    # Bond ETFs (shorter history)
    "TLT": dict(name="iShares 20+ Year Treasury ETF", group="bond_etf", first="2002-07"),
    "LQD": dict(name="iShares IG Corporate Bond ETF", group="bond_etf", first="2002-07"),
    "HYG": dict(name="iShares High Yield Corporate Bond ETF", group="bond_etf", first="2007-04"),
    # FX
    "DX-Y.NYB": dict(name="US Dollar Index", group="fx", first="1971"),
    "JPY=X": dict(name="USD/JPY", group="fx", first="1996"),
    "EURUSD=X": dict(name="EUR/USD", group="fx", first="2003-12"),
    # Commodities (front-month continuous futures)
    "GC=F": dict(name="Gold futures", group="commodity", first="2000-08"),
    "CL=F": dict(name="WTI crude oil futures", group="commodity", first="2000-08"),
    "HG=F": dict(name="Copper futures", group="commodity", first="2000-09"),
}

# Tickers whose history covers the whole period; used for the "core" NaN-free panel.
CORE_TICKERS = [t for t, m in TICKERS.items() if int(m["first"][:4]) <= int(START_DATE[:4])]
