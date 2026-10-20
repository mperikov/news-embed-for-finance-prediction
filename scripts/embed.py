"""Daily news embeddings ε_t, then a train-only reduction down to d.

Raw daily vector (notebook ``EMBEDDING``):

* ``tfidf``   — TfidfVectorizer fitted on train texts only
* ``hashing`` — HashingVectorizer + train-only IDF (no vocabulary)
* ``sbert``   — sentence-transformer; cached under data/processed/embeddings/

Reduction (notebook ``REDUCE``), always fit on train only:

* ``pca`` — PCA / TruncatedSVD (unsupervised)
* ``pls`` — PLS onto train returns (or a return panel); needs ``y``
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .news import article_text

log = logging.getLogger(__name__)

METHODS = ("tfidf", "hashing", "sbert", "sentiment")
REDUCERS = ("pca", "pls")
TEXT_SOURCES = ("finance", "all")
SBERT_POOLS = ("daily", "mean")


def daily_texts(daily: pd.DataFrame, news: pd.DataFrame | None = None, source: str = "finance") -> pd.Series:
    """One string per trading day, aligned to ``daily['date']``."""
    if source not in TEXT_SOURCES:
        raise ValueError(f"TEXT_SOURCE must be one of {TEXT_SOURCES}, got {source!r}")
    dates = pd.DatetimeIndex(daily["date"])
    if source == "finance":
        return daily["finance_text"].fillna("").astype(str)
    if news is None:
        raise ValueError("news frame is required when TEXT_SOURCE='all'")
    texts = article_text(news).groupby(news["trading_date"]).agg("\n".join)
    return texts.reindex(dates).fillna("").astype(str)


def fit_embeddings(
    texts: pd.Series,
    train_mask: np.ndarray,
    method: str = "tfidf",
    d: int = 16,
    *,
    reduce: str = "pca",
    y: np.ndarray | None = None,
    dates: pd.DatetimeIndex | None = None,
    tfidf: dict | None = None,
    hashing: dict | None = None,
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sbert_batch_size: int = 64,
    sbert_pool: str = "daily",
    sbert_max_articles: int = 16,
    with_sentiment: bool = False,
    force: bool = False,
) -> tuple[np.ndarray, dict]:
    """Return ε of shape (T, d) and a small info dict (explained variance, …)."""
    if method not in METHODS:
        raise ValueError(f"EMBEDDING must be one of {METHODS}, got {method!r}")
    if reduce not in REDUCERS:
        raise ValueError(f"REDUCE must be one of {REDUCERS}, got {reduce!r}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape != (len(texts),):
        raise ValueError("train_mask must be a 1-d boolean mask as long as texts")
    if reduce == "pls" and y is None:
        raise ValueError("REDUCE='pls' needs y (train returns, or a T×m panel)")

    raw, raw_info = _raw_embeddings(
        texts, train_mask, method,
        dates=dates, tfidf=tfidf, hashing=hashing,
        sbert_model=sbert_model, sbert_batch_size=sbert_batch_size,
        sbert_pool=sbert_pool, sbert_max_articles=sbert_max_articles, force=force,
    )
    eps, red_info = _reduce(raw, train_mask, d, reduce=reduce, y=y)
    if with_sentiment and method != "sentiment":
        sent, _ = _sentiment(texts)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(sent[train_mask])
        eps = np.hstack([eps, sc.transform(sent)])
        red_info = {**red_info, "with_sentiment": True, "d_total": int(eps.shape[1])}
    info = {
        "method": method,
        "reduce": reduce,
        "d": int(eps.shape[1]),
        "n_days": int(eps.shape[0]),
        "n_train": int(train_mask.sum()),
        "raw_dim": int(raw.shape[1]) if raw.ndim == 2 else None,
        **raw_info,
        **red_info,
    }
    log.info("embeddings %s -> %s d=%d (train=%d days, raw_dim=%s)",
             method, red_info.get("pca", reduce), info["d"], info["n_train"], info["raw_dim"])
    return eps, info


def _raw_embeddings(texts, train_mask, method, *, dates, tfidf, hashing,
                    sbert_model, sbert_batch_size, sbert_pool, sbert_max_articles, force):
    if method == "tfidf":
        return _tfidf(texts, train_mask, tfidf or {})
    if method == "hashing":
        return _hashing(texts, train_mask, hashing or {})
    if method == "sentiment":
        return _sentiment(texts)
    return _sbert(texts, dates, sbert_model, sbert_batch_size, sbert_pool,
                  sbert_max_articles, force)


def _tfidf(texts: pd.Series, train_mask: np.ndarray, params: dict):
    from sklearn.feature_extraction.text import TfidfVectorizer

    kw = dict(lowercase=True, stop_words="english", ngram_range=(1, 2),
              min_df=5, max_df=0.5, max_features=50_000, sublinear_tf=True, norm="l2")
    kw.update(params)
    vec = TfidfVectorizer(**kw)
    vec.fit(texts.to_numpy()[train_mask])
    X = vec.transform(texts.to_numpy())
    info = {"vocab": int(len(vec.vocabulary_)), "vectorizer": "TfidfVectorizer",
            **{k: _jsonable(v) for k, v in kw.items()}}
    return X, info


def _hashing(texts: pd.Series, train_mask: np.ndarray, params: dict):
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

    kw = dict(ngram_range=(1, 2), n_features=2**18, alternate_sign=False, norm=None)
    kw.update(params)
    n_features = kw.pop("n_features")
    hasher = HashingVectorizer(n_features=n_features, dtype=np.float32, **kw)
    hashed = hasher.transform(texts.to_numpy())
    tfidf = TfidfTransformer(sublinear_tf=True)
    tfidf.fit(hashed[train_mask])
    X = tfidf.transform(hashed)
    info = {"vocab": int(n_features), "vectorizer": "HashingVectorizer+Tfidf",
            "n_features": int(n_features), **{k: _jsonable(v) for k, v in kw.items()}}
    return X, info


# Compact Loughran–McDonald-style lists (finance polarity). Enough for a daily score.
_POS = {
    "gain", "gains", "gained", "profit", "profits", "growth", "grew", "rise", "rises", "rose",
    "rising", "rally", "rallied", "surge", "surged", "strong", "stronger", "beat", "beats",
    "upgrade", "upgraded", "recovery", "recover", "boom", "bull", "bullish", "optimistic",
    "record", "outperform", "outperformed", "expand", "expansion", "improve", "improved",
    "improvement", "positive", "advance", "advanced", "high", "higher", "highest",
}
_NEG = {
    "loss", "losses", "lost", "decline", "declined", "declines", "drop", "dropped", "fell",
    "fall", "falling", "crash", "crisis", "recession", "bankruptcy", "default", "risk",
    "risks", "weak", "weaker", "plunge", "plunged", "fear", "fears", "cut", "cuts", "slump",
    "turmoil", "inflation", "hawkish", "downgrade", "downgraded", "miss", "missed", "layoff",
    "layoffs", "unemployment", "deficit", "warning", "selloff", "sell-off", "bear", "bearish",
    "collapse", "collapses", "worst", "low", "lower", "lowest", "negative", "concern",
    "concerns", "slowdown", "volatile", "volatility",
}


def _sentiment(texts: pd.Series):
    """3 daily features (no collinear pos+neg+net): log length, neg share, polarity."""
    rows = []
    for blob in texts:
        toks = re.findall(r"[a-z]+", str(blob).lower())
        n = max(len(toks), 1)
        pos = sum(w in _POS for w in toks)
        neg = sum(w in _NEG for w in toks)
        rows.append((
            np.log1p(len(toks)),
            neg / n,
            (pos - neg) / (pos + neg + 1.0),
        ))
    X = np.asarray(rows, dtype=np.float64)
    return X, {"vectorizer": "lexicon_sentiment", "vocab": len(_POS) + len(_NEG),
               "names": ["log_len", "neg_share", "polarity"]}


TOPICS = {
    "energy": {
        "oil", "crude", "opec", "petroleum", "gasoline", "energy", "refinery", "barrel",
        "wti", "brent", "saudi", "fuel", "pipeline", "gas", "exxon", "chevron",
    },
    "fed": {
        "fed", "fomc", "inflation", "rates", "powell", "yellen", "bernanke", "greenspan",
        "treasury", "yield", "hike", "hawkish", "dovish", "ecb", "interest",
    },
    "banks": {
        "bank", "banks", "credit", "loan", "mortgage", "bailout", "deposit", "lending",
        "lehman", "jpmorgan", "citigroup", "goldman",
    },
}


def topic_shares(texts: pd.Series) -> tuple[np.ndarray, list[str]]:
    names = list(TOPICS)
    rows = []
    for blob in texts:
        toks = re.findall(r"[a-z]+", str(blob).lower())
        n = max(len(toks), 1)
        rows.append([sum(w in TOPICS[k] for w in toks) / n for k in names])
    return np.asarray(rows, dtype=np.float64), names


def novelty(eps: np.ndarray) -> np.ndarray:
    """1 − cosine(ε_t, ε_{t−1}); 0 on the first day."""
    e = np.asarray(eps, dtype=np.float64)
    n = np.linalg.norm(e, axis=1)
    n = np.maximum(n, 1e-12)
    out = np.zeros(len(e), dtype=np.float64)
    if len(e) > 1:
        out[1:] = 1.0 - np.clip((e[1:] * e[:-1]).sum(1) / (n[1:] * n[:-1]), -1.0, 1.0)
    return out[:, None]


def build_news_features(
    daily: pd.DataFrame,
    texts: pd.Series,
    train_mask: np.ndarray,
    *,
    use_sbert: bool = True,
    use_topics: bool = True,
    count_from_text: bool = False,
    d_sbert: int = 6,
    dates=None,
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[np.ndarray, list[str], dict]:
    """Standardized next-day news features: sentiment + log n_finance + novelty [+ SBERT PCA]."""
    from sklearn.preprocessing import StandardScaler

    parts, names = [], []
    sent, sinfo = _sentiment(texts)
    parts.append(sent)
    names += list(sinfo["names"])
    if count_from_text:
        nfin = np.log1p(texts.str.count("\n").to_numpy(dtype=np.float64) + (texts.str.len() > 0).to_numpy(dtype=np.float64))
        nfin = nfin[:, None]
        nname = "log_n_lines"
    else:
        nfin = np.log1p(daily["n_finance"].to_numpy(dtype=np.float64))[:, None]
        nname = "log_n_finance"
    parts.append(nfin)
    names.append(nname)
    if use_topics:
        top, tnames = topic_shares(texts)
        parts.append(top)
        names += tnames

    info = {"sentiment": sinfo}
    if use_sbert:
        try:
            z, zinfo = fit_embeddings(
                texts, train_mask, method="sbert", d=d_sbert, reduce="pca", dates=dates,
                sbert_model=sbert_model, sbert_pool="daily",
            )
            parts.append(z)
            names += [f"sbert{i+1}" for i in range(z.shape[1])]
            info["sbert"] = {k: zinfo[k] for k in ("method", "d", "raw_dim", "pca") if k in zinfo}
        except Exception as exc:
            log.warning("sbert skipped: %s", exc)
            info["sbert"] = {"error": str(exc)}

    X = np.hstack(parts)
    X = np.hstack([X, novelty(X)])
    names.append("novelty")

    sc = StandardScaler()
    sc.fit(X[train_mask])
    X = sc.transform(X).astype(np.float64)
    info["names"] = names
    info["d"] = X.shape[1]
    return X, names, info


def _sbert(texts: pd.Series, dates: pd.DatetimeIndex | None, model_name: str,
           batch_size: int, pool: str, max_articles: int, force: bool):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "EMBEDDING='sbert' needs sentence-transformers: pip install sentence-transformers"
        ) from e
    if pool not in SBERT_POOLS:
        raise ValueError(f"SBERT_POOL must be one of {SBERT_POOLS}, got {pool!r}")

    dates = pd.DatetimeIndex(dates if dates is not None else texts.index)
    cache = _sbert_cache_path(model_name, dates, pool, max_articles)
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=False)
        cached_dates = np.asarray(z["dates"]).astype(str)
        if list(cached_dates) == [str(d.date()) for d in dates]:
            log.info("sbert cache hit %s shape=%s", cache.name, z["emb"].shape)
            return z["emb"].astype(np.float32), {
                "vectorizer": model_name, "cache": str(cache),
                "sbert_pool": pool, "sbert_max_articles": int(max_articles),
            }

    model = SentenceTransformer(model_name)
    if pool == "daily":
        log.info("encoding %d daily blobs with %s (truncated to model window)", len(texts), model_name)
        emb = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        emb = np.asarray(emb, dtype=np.float32)
    else:
        emb = _sbert_mean_pool(model, texts, batch_size, max_articles)
    C.EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, emb=emb, dates=np.asarray([str(d.date()) for d in dates]))
    return emb, {
        "vectorizer": model_name, "cache": str(cache),
        "sbert_pool": pool, "sbert_max_articles": int(max_articles),
    }


def _sbert_mean_pool(model, texts: pd.Series, batch_size: int, max_articles: int) -> np.ndarray:
    """Encode up to ``max_articles`` lines per day (headline. abstract) and average."""
    lines, owners = [], []
    for i, blob in enumerate(texts):
        arts = [ln.strip() for ln in str(blob).split("\n") if ln.strip()][:max_articles]
        if not arts:
            arts = [""]
        lines.extend(arts)
        owners.extend([i] * len(arts))
    log.info("encoding %d articles (%d days, cap=%d/day) for mean-pool sbert",
             len(lines), len(texts), max_articles)
    piece = model.encode(lines, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    piece = np.asarray(piece, dtype=np.float32)
    out = np.zeros((len(texts), piece.shape[1]), dtype=np.float32)
    n = np.zeros(len(texts), dtype=np.float32)
    for vec, i in zip(piece, owners):
        out[i] += vec
        n[i] += 1
    n = np.maximum(n, 1.0)
    return out / n[:, None]


def _sbert_cache_path(model_name: str, dates: pd.DatetimeIndex, pool: str, max_articles: int) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    span = f"{dates[0].date()}_{dates[-1].date()}_{len(dates)}"
    extra = f"{pool}" + (f"{max_articles}" if pool == "mean" else "")
    return C.EMBED_DIR / f"sbert_{safe}_{span}_{extra}.npz"


def _reduce(raw, train_mask: np.ndarray, d: int, reduce: str, y: np.ndarray | None):
    if reduce == "pls":
        return _pls(raw, train_mask, d, y)
    return _pca(raw, train_mask, d)


def _pca(raw, train_mask: np.ndarray, d: int) -> tuple[np.ndarray, dict]:
    """PCA on dense raw embeddings, TruncatedSVD on sparse tf-idf/hashing (same role)."""
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    n_train, n_features = int(train_mask.sum()), int(raw.shape[1])
    k = int(min(d, n_features, max(n_train - 1, 1)))
    sparse = hasattr(raw, "tocsr")
    if sparse:
        reducer = TruncatedSVD(n_components=k, random_state=0)
        reducer.fit(raw[train_mask])
        z = reducer.transform(raw)
        ev = np.asarray(reducer.explained_variance_ratio_, dtype=float)
        name = "TruncatedSVD"
    else:
        raw = np.asarray(raw, dtype=np.float64)
        reducer = PCA(n_components=k, random_state=0)
        reducer.fit(raw[train_mask])
        z = reducer.transform(raw)
        ev = np.asarray(reducer.explained_variance_ratio_, dtype=float)
        name = "PCA"

    scaler = StandardScaler()
    scaler.fit(z[train_mask])
    z = scaler.transform(z).astype(np.float64)
    return z, {
        "pca": name,
        "explained_variance_ratio": ev,
        "explained_variance_sum": float(ev.sum()),
    }


def _pls(raw, train_mask: np.ndarray, d: int, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Supervised reduction: optional SVD on sparse tf-idf, then PLS onto train y."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    Y = np.asarray(y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.shape[0] != raw.shape[0]:
        raise ValueError(f"y has {Y.shape[0]} rows, embeddings have {raw.shape[0]}")

    svd_k = None
    if hasattr(raw, "tocsr"):
        svd_k = int(min(256, raw.shape[1] - 1, max(int(train_mask.sum()) - 1, 2)))
        svd = TruncatedSVD(n_components=svd_k, random_state=0)
        svd.fit(raw[train_mask])
        Z = svd.transform(raw)
    else:
        Z = np.asarray(raw, dtype=np.float64)

    ok = train_mask & np.isfinite(Y).all(axis=1)
    k = int(min(d, Z.shape[1], max(int(ok.sum()) - 1, 1)))
    k = max(k, 1)
    pls = PLSRegression(n_components=k, scale=True)
    pls.fit(Z[ok], Y[ok])
    z = np.asarray(pls.transform(Z), dtype=np.float64)
    scaler = StandardScaler()
    scaler.fit(z[train_mask])
    z = scaler.transform(z).astype(np.float64)
    return z, {
        "pca": f"PLS" + (f"+SVD{svd_k}" if svd_k else ""),
        "explained_variance_ratio": np.asarray([], dtype=float),
        "explained_variance_sum": float("nan"),
        "pls_targets": int(Y.shape[1]),
        "pls_n_train": int(ok.sum()),
    }


def _jsonable(v):
    if isinstance(v, tuple):
        return list(v)
    return v
