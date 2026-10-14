# News embeddings for finance prediction — dataset

Daily news + market dataset on a single NYSE trading calendar, **2000 → today**, built for the model

* news embeddings `ε_t`,
* latent dynamics `x_{t+1} = A x_t + ε_t` (contracting `A`),
* observed prices `p_t = <g, x_t>`.

Open **`dataset.ipynb`** and run the first cell: it downloads the raw data on the first run
(~1.3 GB, 10–15 min) and reuses `data/processed/` afterwards. Everything else is in `scripts/`.

```bash
pip install -r requirements.txt
python -m scripts.build_dataset            # same as the first notebook cell; --force rebuilds
```

## What is in the dataset

| file (`data/processed/`) | rows | content |
|---|---|---|
| `news_articles.parquet` | ≈1.24 M | one row per NYT article: `pub_time_utc`, `headline`, `abstract`, `lead_paragraph`, `section`, `subsection`, `news_desk`, `type_of_material`, `document_type`, `source`, `keywords` (` \| `-joined), `word_count`, `web_url`, `is_finance`, `flag_source`, **`trading_date`** |
| `market_prices.parquet` | ≈190 k | long OHLCV table: `date, ticker, open, high, low, close, adj_close, volume` for 29 instruments |
| `prices_wide.parquet` | 6.5 k × 29 | `adj_close` on the trading calendar (index) × tickers — the observed `p_t` |
| `daily.parquet` | 6.5 k | per trading day: `spx_close`, `spx_logret`, `n_articles`, `n_finance`, `finance_text` (all finance articles of the day as `headline. abstract` lines) |
| `meta.json` | | sources, period, per-ticker coverage, text-model metrics |

Period: first trading day of 2000 → the last day on which **both** news and prices are complete
(news lag behind prices by a few days to months depending on the source, see below).

### Time alignment (the important part)

* Trading calendar = days with an S&P 500 close.
* The information window of trading day `D` is `(close of D−1, close of D]`, close = 16:00 New York time.
  An article is assigned to the first trading day whose close is **after** its publication time,
  so weekend / holiday / after-hours news goes to the next session and every article in
  `trading_date = D` was public before the close of `D` → no look-ahead when `ε_D` explains the return of `D`.
* Print-era articles (2000–2006) carry a date-only timestamp (midnight New York) and are therefore
  attributed to that day's session (the print edition is available before the open).
* Not modelled: the few early-close sessions per year (13:00 close).

### News

Source: The New York Times, via mirrors of the official [Archive API](https://developer.nytimes.com/docs/archive-product/1/overview)
(all NYT articles, ≈100–330 per day, every day since 2000).

1. **Kaggle `aryansingh0909/nyt-articles-21m-2000-present`** (CC0): full metadata — headline, abstract,
   lead paragraph, section, news desk, keywords, url. Anonymous download (no Kaggle account needed).
   Its daily updates degraded in 2023; completeness is verified **month by month** against source 2.
2. **Hugging Face `bguzzo2k/nyt_100y_news_headlines`**: headline + abstract for every article
   1927 → today (same origin, yearly parquet). Used as the completeness reference and, without an API key,
   to fill the months where source 1 is short (currently 2023-05 → 2025-12).
   These rows have no section metadata: `keep` (news vs. culture/lifestyle/service material) and
   `is_finance` are imputed with a text classifier (hashed tf-idf + logistic regression) trained on the
   labelled rows of the preceding 8 years; held-out accuracy ≈0.90 / 0.94 (see `meta.json → news.text_model`).
   Such rows are marked `flag_source = "text_model"` (vs. `"metadata"`), so they can be excluded.
3. **NYT Archive API** (optional, free key → `NYT_API_KEY` in `.env` or the environment): with a key the
   incomplete months are fetched from the API instead (full metadata, up to yesterday) — one request per
   month, 12.5 s apart because of the public rate limit (5/min, 500/day).

Cleaning: paid notices, corrections, obituaries, letters, reviews, schedules, lists, recipes, crosswords
and the culture / lifestyle / service sections (Sports, Arts, Style, Food, Travel, Real Estate ads, …) are
dropped; rows without headline or text are dropped. Everything else — World, U.S., Business, Technology,
Politics, Opinion, New York, Science, Health, Climate, … — is kept, so the finance subset can be redefined.

`is_finance` (business / markets / economy coverage, ≈55 per trading day): business sections or desks,
business url paths (`/business/`, `/your-money/`, `/dealbook/`, DealBook and Economix blogs) or a markets /
macro subject keyword (`Stocks and Bonds`, `Interest Rates`, `Federal Reserve System`, `United States
Economy`, `Oil (Petroleum) and Gasoline`, …). Definitions live in `scripts/config.py`.

### Market

Yahoo Finance (via `yfinance`), daily, US-traded / US-calendar instruments:

| group | tickers |
|---|---|
| equity indices | `^GSPC` S&P 500, `^DJI`, `^IXIC`, `^RUT` |
| broad / sector ETFs | `SPY`, `QQQ`; `XLF XLK XLE XLV XLY XLP XLI XLB XLU` (SPDR sectors, since 1998) |
| volatility, rates | `^VIX`; `^IRX` 13w, `^FVX` 5y, `^TNX` 10y, `^TYX` 30y Treasury yields |
| bond ETFs | `TLT` (2002-07), `LQD` (2002-07), `HYG` (2007-04) |
| FX | `DX-Y.NYB` dollar index, `JPY=X`, `EURUSD=X` (2003-12) |
| commodities | `GC=F` gold, `CL=F` WTI, `HG=F` copper (front-month futures, 2000-08) |

`adj_close` is split- and dividend-adjusted (total return) for ETFs and equals `close` for indices, yields
and FX. 24 "core" tickers cover the whole period (`meta.json → market.core_tickers`); the rest start later
and are `NaN` before their first date. Add tickers in `scripts/config.py::TICKERS`.

## Layout

```
dataset.ipynb            minimal notebook: download-or-reuse, overview, coverage, plots, example day, checks
scripts/
  config.py              paths, period, tickers, cleaning / finance-flag definitions
  news.py                Kaggle mirror parser, HF reference + fill, NYT API client, cleaning, is_finance
  textflags.py           text classifier imputing keep / is_finance for rows without metadata
  market.py              yfinance downloader with per-ticker cache
  align.py               trading calendar, news -> trading_date, prices_wide, daily table
  build_dataset.py       ensure_dataset() used by the notebook + CLI
data/raw/                downloads (gitignored): Kaggle csv, HF parquet, API json, per-ticker parquet
data/processed/          the dataset (gitignored, rebuilt by the scripts)
```

## Known limitations

* NYT only — one (US, general-interest) outlet; business coverage is dense but not a wire service.
  The article text is headline + abstract (+ lead paragraph for the metadata rows), not full text.
* 2023-05 → 2025-12 (without an API key): no section metadata, flags imputed by a classifier
  (`flag_source = "text_model"`). Use the NYT API key for exact metadata.
* Yahoo Finance history is unofficial; occasional revisions of adjusted prices are possible.
  Rebuild with `--force` to refresh.
