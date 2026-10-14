"""NYT article metadata -> one normalized article table.

Two sources with identical output schema (NEWS_COLUMNS):

1. Kaggle mirror of the NYT Archive API (default, no credentials, CC0):
   every NYT article since 2000-01-01 with headline, abstract, lead paragraph, section,
   news desk, keywords, ... ; ~0.9 GB zip -> 4.6 GB csv, parsed once and cached as parquet.
2. NYT Archive API (optional, needs NYT_API_KEY): extends the mirror up to today,
   one request per month, cached as data/raw/nyt_api/YYYY-MM.json.gz.
"""
from __future__ import annotations

import ast
import gzip
import json
import logging
import re
import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)

NEWS_COLUMNS = [
    "article_id", "pub_time_utc", "headline", "abstract", "lead_paragraph",
    "section", "subsection", "news_desk", "type_of_material", "document_type",
    "source", "keywords", "word_count", "web_url",
]
KAGGLE_COLUMNS = {
    "_id": "article_id", "pub_date": "pub_time_utc", "headline": "headline", "abstract": "abstract",
    "lead_paragraph": "lead_paragraph", "section_name": "section", "subsection_name": "subsection",
    "news_desk": "news_desk", "type_of_material": "type_of_material", "document_type": "document_type",
    "source": "source", "keywords": "keywords", "word_count": "word_count", "web_url": "web_url",
}
_NORMALIZED_CACHE = C.RAW_KAGGLE_DIR / "normalized.parquet"
_MISSING = {"", "nan", "None", "NaN", "null"}


# ----------------------------------------------------------------------------- helpers
def _literal(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


# A quoted Python string literal: 'single' or "double" quoted, with backslash escapes.
_PY_STR = r"""(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")"""
_MAIN_RE = re.compile(r"'main':\s*" + _PY_STR)
_VALUE_RE = re.compile(r"'value':\s*" + _PY_STR)


def _unquote(single: str | None, double: str | None) -> str:
    """Decode one regex match of _PY_STR (literal_eval only when there are escapes).

    Works for both `match.group()` (None for the unused alternative) and `findall()` ('' instead).
    """
    if single:
        raw, quote = single, "'"
    else:
        raw, quote = (double or ""), '"'
    if "\\" not in raw:
        return raw
    val = _literal(quote + raw + quote)
    return val if isinstance(val, str) else raw


def _headline_main(s) -> str:
    """`headline` is stored as the repr of a dict -> return its 'main' field."""
    if not isinstance(s, str) or s in _MISSING:
        return ""
    m = _MAIN_RE.search(s)
    if m:
        return _unquote(m.group(1), m.group(2))
    d = _literal(s)  # slow path: unusual formatting
    if isinstance(d, dict):
        return str(d.get("main") or d.get("print_headline") or "")
    return ""


def _keyword_values(s) -> str:
    """`keywords` is the repr of a list of {'name','value',...} dicts -> 'v1 | v2 | ...'."""
    if not isinstance(s, str) or s in _MISSING or s == "[]":
        return ""
    vals = [_unquote(a, b) for a, b in _VALUE_RE.findall(s)]
    return " | ".join(v for v in vals if v)


def _clean_str(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("")
    return s.where(~s.isin(_MISSING), "").astype(str)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes and column order of a frame that already has NEWS_COLUMNS."""
    df = df.copy()
    for c in ["article_id", "headline", "abstract", "lead_paragraph", "section", "subsection",
              "news_desk", "type_of_material", "document_type", "source", "keywords", "web_url"]:
        df[c] = _clean_str(df[c]).str.strip()
    if not pd.api.types.is_datetime64_any_dtype(df["pub_time_utc"]):
        df["pub_time_utc"] = pd.to_datetime(df["pub_time_utc"], utc=True, errors="coerce", format="mixed")
    df["pub_time_utc"] = df["pub_time_utc"].dt.tz_convert("UTC")
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").round().astype("Int32")
    df = df.dropna(subset=["pub_time_utc"])
    df.loc[df["article_id"] == "", "article_id"] = df["web_url"]
    return df[NEWS_COLUMNS]


# ----------------------------------------------------------------------------- Kaggle mirror
def download_kaggle(force: bool = False) -> Path:
    """Download the Kaggle mirror (anonymous access is enough) and return the csv path."""
    C.RAW_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    csvs = list(C.RAW_KAGGLE_DIR.rglob("*.csv"))
    if csvs and not force:
        return csvs[0]
    import kagglehub

    log.info("downloading %s from Kaggle (~0.9 GB zip, 4.6 GB csv)", C.KAGGLE_NYT_DATASET)
    kagglehub.dataset_download(C.KAGGLE_NYT_DATASET, output_dir=str(C.RAW_KAGGLE_DIR), force_download=force)
    csvs = list(C.RAW_KAGGLE_DIR.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no csv found in {C.RAW_KAGGLE_DIR}")
    return csvs[0]


def parse_kaggle_csv(csv_path: Path, chunksize: int = 200_000) -> pd.DataFrame:
    """Stream the 4.6 GB csv and normalize it to NEWS_COLUMNS (several minutes)."""
    frames = []
    reader = pd.read_csv(csv_path, usecols=list(KAGGLE_COLUMNS), dtype=str,
                         keep_default_na=False, chunksize=chunksize)
    t0 = time.time()
    for i, chunk in enumerate(reader):
        chunk = chunk.rename(columns=KAGGLE_COLUMNS)
        chunk["headline"] = chunk["headline"].map(_headline_main)
        chunk["keywords"] = chunk["keywords"].map(_keyword_values)
        frames.append(_finalize(chunk))
        log.info("parsed %d rows (%.0f s)", (i + 1) * chunksize, time.time() - t0)
    return pd.concat(frames, ignore_index=True)


def load_kaggle(force: bool = False) -> pd.DataFrame:
    if _NORMALIZED_CACHE.exists() and not force:
        return pd.read_parquet(_NORMALIZED_CACHE)
    df = parse_kaggle_csv(download_kaggle(force=False))
    df.to_parquet(_NORMALIZED_CACHE, index=False)
    return df


# ----------------------------------------------------------------------------- NYT Archive API
def fetch_month(year: int, month: int, api_key: str, max_age_days: float | None = None) -> list[dict]:
    """Raw docs for one month, cached on disk (re-fetched when the cache is older than `max_age_days`).

    Sleeps after each call to respect the public rate limit (5 requests / minute).
    """
    C.RAW_NYT_API_DIR.mkdir(parents=True, exist_ok=True)
    path = C.RAW_NYT_API_DIR / f"{year}-{month:02d}.json.gz"
    fresh = path.exists() and (max_age_days is None or time.time() - path.stat().st_mtime < max_age_days * 86400)
    if fresh:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)["response"]["docs"]
    url = C.NYT_API_URL.format(year=year, month=month)
    for attempt in range(6):
        r = requests.get(url, params={"api-key": api_key}, timeout=180)
        if r.status_code == 429:  # per-minute or per-day quota
            log.warning("NYT API 429 for %d-%02d, sleeping 65 s", year, month)
            time.sleep(65)
            continue
        r.raise_for_status()
        data = r.json()
        break
    else:
        raise RuntimeError(f"NYT API: too many 429 responses for {year}-{month:02d}")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    log.info("NYT API %d-%02d: %d docs", year, month, len(data["response"]["docs"]))
    time.sleep(C.NYT_API_SLEEP_SECONDS)
    return data["response"]["docs"]


def api_docs_to_frame(docs: list[dict]) -> pd.DataFrame:
    rows = []
    for d in docs:
        rows.append({
            "article_id": d.get("_id") or d.get("uri") or d.get("web_url"),
            "pub_time_utc": d.get("pub_date"),
            "headline": (d.get("headline") or {}).get("main") or (d.get("headline") or {}).get("print_headline"),
            "abstract": d.get("abstract"),
            "lead_paragraph": d.get("lead_paragraph"),
            "section": d.get("section_name"),
            "subsection": d.get("subsection_name"),
            "news_desk": d.get("news_desk"),
            "type_of_material": d.get("type_of_material"),
            "document_type": d.get("document_type"),
            "source": d.get("source"),
            "keywords": " | ".join(k.get("value", "") for k in (d.get("keywords") or []) if k.get("value")),
            "word_count": d.get("word_count"),
            "web_url": d.get("web_url"),
        })
    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    return _finalize(df) if len(df) else df


def load_api(months, api_key: str) -> pd.DataFrame:
    """The given months from the Archive API (the running month is re-fetched once a day)."""
    months = list(months)
    log.info("NYT Archive API: %d month(s) (%s .. %s), ~%.0f min at the public rate limit",
             len(months), months[0], months[-1], len(months) * C.NYT_API_SLEEP_SECONDS / 60)
    current = pd.Timestamp.today().to_period("M")
    frames = [api_docs_to_frame(fetch_month(m.year, m.month, api_key, max_age_days=1 if m == current else None))
              for m in months]
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- HF mirror (headline + abstract)
def iter_hf(years) -> Iterator[tuple[int, pd.DataFrame]]:
    """(year, frame) for the yearly parquet files of the HF mirror (date, headline, abstract),
    normalized to NEWS_COLUMNS. Metadata columns are empty; `article_id` = timestamp + headline.
    """
    from huggingface_hub import hf_hub_download

    for year in years:
        try:
            path = hf_hub_download(C.HF_NYT_DATASET, f"yearly_parquet/nyt_{year}.parquet", repo_type="dataset",
                                   local_dir=str(C.RAW_HF_DIR))
        except Exception as e:  # year not published yet
            log.info("HF mirror: no file for %d (%s)", year, type(e).__name__)
            continue
        df = pd.read_parquet(path).rename(columns={"date": "pub_time_utc"})
        for c in NEWS_COLUMNS:
            if c not in df:
                df[c] = ""
        df["source"] = "The New York Times"
        df["word_count"] = pd.NA
        df = _finalize(df)
        df["article_id"] = "hf:" + df["pub_time_utc"].dt.strftime("%Y%m%d%H%M%S") + ":" + df["headline"].str[:80]
        yield year, df


# ----------------------------------------------------------------------------- cleaning / flags
def article_text(news: pd.DataFrame) -> pd.Series:
    """`headline. abstract` (lead paragraph when the abstract is empty) - the unit fed to an embedding model."""
    body = news["abstract"].where(news["abstract"].str.len() > 0, news["lead_paragraph"])
    head = news["headline"].str.rstrip(".?!")
    return (head + ". " + body).str.strip()


def _keyword_regex(values: set[str]) -> str:
    """Match a whole keyword (keywords are joined by ' | '), case-insensitively."""
    alts = "|".join(sorted(re.escape(v) for v in values))
    return rf"(?i)(?:^|\| )(?:{alts})(?= \||$)"


_FINANCE_URL_RE = (r"nytimes\.com/\d{4}/\d{2}/\d{2}/(?:business|your-money|dealbook)/"
                   r"|//(?:dealbook|economix)\.blogs\.nytimes\.com/")


def is_finance(df: pd.DataFrame) -> pd.Series:
    """Business / markets / economy coverage: business section or desk, business url path,
    or a markets/macro subject keyword (see config.FINANCE_*). Needs metadata."""
    sec = df["section"].isin(C.FINANCE_SECTIONS)
    desk = df["news_desk"].isin(C.FINANCE_NEWS_DESKS)
    url = df["web_url"].str.contains(_FINANCE_URL_RE, regex=True)
    kw = df["keywords"].str.contains(_keyword_regex(C.FINANCE_KEYWORDS), regex=True)
    return (sec | desk | url | kw).fillna(False).astype(bool)


def keep_mask(df: pd.DataFrame) -> pd.Series:
    """News-like material according to metadata (type, section, document type)."""
    keep = ~df["type_of_material"].isin(C.DROP_TYPE_OF_MATERIAL)
    keep &= ~df["section"].isin(C.DROP_SECTIONS)
    keep &= ~df["headline"].str.startswith("Paid Notice")
    keep &= df["document_type"].isin({"article", "blogpost", "multimedia", ""})
    return keep.fillna(False).astype(bool)


def has_text(df: pd.DataFrame) -> pd.Series:
    return (df["headline"].str.len() > 0) & ((df["abstract"].str.len() > 0) | (df["lead_paragraph"].str.len() > 0))


def _month(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert(None).dt.to_period("M")


def _monthly_counts(df: pd.DataFrame, months: pd.PeriodIndex) -> pd.Series:
    return _month(df["pub_time_utc"]).value_counts().reindex(months, fill_value=0)


def _last_complete_month(last_ts: pd.Timestamp) -> pd.Period:
    """Month of `last_ts`, unless the data stops more than a day before that month's end."""
    last_ts = last_ts.tz_convert(None)
    m = last_ts.to_period("M")
    return m if last_ts >= m.to_timestamp(how="end") - pd.Timedelta(days=1) else m - 1


def _match_key(df: pd.DataFrame) -> pd.Series:
    """Identity of an article across mirrors: publication timestamp + normalized headline."""
    return (df["pub_time_utc"].dt.strftime("%Y%m%d%H%M%S") + "|"
            + df["headline"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip())


def _cut(df: pd.DataFrame, lo: pd.Timestamp | None = None, hi: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rows with lo <= pub_time_utc <= hi (dates are inclusive, interpreted in UTC)."""
    m = pd.Series(True, index=df.index)
    if lo is not None:
        m &= df["pub_time_utc"] >= pd.Timestamp(lo).tz_localize("UTC")
    if hi is not None:
        m &= df["pub_time_utc"] < pd.Timestamp(hi).tz_localize("UTC") + pd.Timedelta(days=1)
    return df[m]


def build_news(start: str = C.START_DATE, end: str | None = C.END_DATE, force: bool = False,
               min_coverage: float = 0.95) -> tuple[pd.DataFrame, pd.Timestamp, dict]:
    """Cleaned article table for [start, end] with `is_finance` and `flag_source` columns.

    Returns (news, last date with complete coverage, info dict for meta.json).

    * The Kaggle mirror (full metadata) is the base. Its monthly article counts are checked
      against the HF mirror (same origin, known complete); a month with < `min_coverage` of
      the reference count is incomplete (the Kaggle mirror stopped updating in 2023).
    * Incomplete months are completed from the NYT Archive API when NYT_API_KEY is set
      (full metadata), otherwise with the HF rows missing from the base (headline + abstract
      only; `keep` / `is_finance` imputed from text, see textflags.py).
    """
    start_ts, end_date = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp(date.today())
    months = pd.period_range(start_ts, end_date, freq="M")
    base = load_kaggle(force=force)
    base["flag_source"] = "metadata"
    base_month = _month(base["pub_time_utc"])
    n_base = base_month.value_counts().reindex(months, fill_value=0)

    # compare month by month with the reference mirror; keep its rows missing from the base
    n_ref, ref_last_ts, hf_rows = pd.Series(0, index=months), None, []
    for year, hf in iter_hf(range(start_ts.year, end_date.year + 1)):
        hf_month = _month(hf["pub_time_utc"])
        counts = hf_month.value_counts().reindex(months, fill_value=0)  # a few rows carry bogus dates: ignored
        n_ref += counts
        ref_last_ts = max(ref_last_ts, hf["pub_time_utc"].max()) if ref_last_ts is not None else hf["pub_time_utc"].max()
        short = months[((counts > 0) & (n_base < min_coverage * counts)).to_numpy()]
        if len(short):
            base_keys = _match_key(base[base_month.isin(short)])
            hf_rows.append(hf[hf_month.isin(short) & ~_match_key(hf).isin(base_keys)])
        log.info("HF mirror %d: %d articles, %d month(s) where the Kaggle mirror is short", year, len(hf), len(short))
    if ref_last_ts is None:
        raise RuntimeError(f"reference mirror {C.HF_NYT_DATASET} unavailable; cannot verify news completeness")
    ref_end = _last_complete_month(ref_last_ts)
    incomplete = ((n_base < min_coverage * n_ref) | (months > ref_end)).to_numpy()  # beyond the reference: unknown
    log.info("Kaggle mirror: %d articles; %d of %d months below %.0f%% of the reference count (%s ...)",
             len(base), int(incomplete.sum()), len(months), 100 * min_coverage,
             ", ".join(str(m) for m in months[incomplete][:4]))
    info = {"sources": [f"kaggle:{C.KAGGLE_NYT_DATASET} (complete months: {int((~incomplete).sum())})"],
            "reference": f"hf:{C.HF_NYT_DATASET} (complete until {ref_end})",
            "incomplete_months": [str(m) for m in months[incomplete]]}

    # rows that complete the incomplete months -------------------------------------------
    fill = None
    if incomplete.any() and C.NYT_API_KEY:
        api = load_api(months[incomplete], C.NYT_API_KEY)
        fill = api[~api["article_id"].isin(base["article_id"])].copy()
        fill["flag_source"] = "metadata"
        reliable_end = min(end_date, pd.Timestamp(date.today()) - pd.Timedelta(days=1))
        info["sources"].append(f"nytimes archive api ({int(incomplete.sum())} months, {len(fill)} articles)")
    else:
        reliable_end = min(end_date, ref_end.to_timestamp(how="end").normalize())
        if hf_rows:
            fill = pd.concat(hf_rows, ignore_index=True)
            fill = _cut(fill, hi=reliable_end).copy()
            fill["flag_source"] = "text_model"
            info["sources"].append(f"hf:{C.HF_NYT_DATASET} ({len(fill)} articles added)")
    del hf_rows

    # flags for rows with metadata -------------------------------------------------------
    base_keep, base_fin = keep_mask(base), is_finance(base)
    news = base[base_keep & has_text(base)].copy()
    news["is_finance"] = base_fin[news.index]

    # rows without metadata: impute the same flags from text -----------------------------
    if fill is not None and len(fill):
        fill = fill[has_text(fill)].copy()
        if fill["flag_source"].iloc[0] == "metadata":
            fill_keep, fill_fin = keep_mask(fill), is_finance(fill)
        else:
            from .textflags import fit_and_predict

            # train on the labelled rows of the years just before the fill period (same era, same style)
            train_from = fill["pub_time_utc"].min() - pd.DateOffset(years=C.TEXT_MODEL_TRAIN_YEARS)
            labelled = has_text(base) & (base["pub_time_utc"] >= train_from)
            labels = pd.DataFrame({"keep": base_keep, "is_finance": base_fin})[labelled]
            pred, info["text_model"] = fit_and_predict(article_text(base[labelled]), labels, article_text(fill))
            info["text_model"]["train_period"] = f"{train_from.date()} -> {fill['pub_time_utc'].min().date()}"
            fill_keep, fill_fin = pred["keep"], pred["is_finance"]
        fill = fill[fill_keep]
        fill["is_finance"] = fill_fin[fill.index]
        news = pd.concat([news, fill], ignore_index=True)

    news = _cut(news, lo=start_ts, hi=min(end_date, reliable_end))
    news = news.drop_duplicates("article_id").sort_values("pub_time_utc", kind="stable").reset_index(drop=True)
    info["incomplete_months"] = [m for m in info["incomplete_months"] if m <= str(reliable_end.to_period("M"))]
    return news, reliable_end, info
