"""Impute metadata-based flags from text for articles that come without metadata.

The Kaggle mirror carries section / desk / keywords for every article up to 2023-04; the
HF mirror used for later months only has headline + abstract. To keep `is_finance` and the
section-based cleaning consistent over the whole period, two linear classifiers
(hashed tf-idf of `headline. abstract` -> logistic regression) are fitted on the labelled
rows and applied to the unlabelled ones. Held-out metrics are returned and stored in meta.json.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TARGETS = ("keep", "is_finance")


def fit_and_predict(train_text: pd.Series, train_labels: pd.DataFrame, apply_text: pd.Series,
                    max_train: int = 400_000, holdout: float = 0.05, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Return predicted TARGETS for `apply_text` and held-out metrics for each target."""
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score

    rng = np.random.default_rng(seed)
    if len(train_text) > max_train:
        pick = np.sort(rng.choice(len(train_text), max_train, replace=False))
        train_text, train_labels = train_text.iloc[pick], train_labels.iloc[pick]
    is_test = rng.random(len(train_text)) < holdout

    hasher = HashingVectorizer(ngram_range=(1, 2), n_features=2**20, alternate_sign=False, norm=None,
                               dtype=np.float32)
    tfidf = TfidfTransformer(sublinear_tf=True)
    X_train = tfidf.fit_transform(hasher.transform(train_text[~is_test]))
    X_test = tfidf.transform(hasher.transform(train_text[is_test]))
    X_apply = tfidf.transform(hasher.transform(apply_text))

    preds, metrics = {}, {}
    for target in TARGETS:
        y = train_labels[target].to_numpy(dtype=bool)
        clf = LogisticRegression(C=4.0, max_iter=1000, solver="liblinear")
        clf.fit(X_train, y[~is_test])
        # decision threshold such that the predicted positive rate equals the labelled base rate:
        # keeps the article flow (e.g. finance articles per day) comparable across the two sources
        p_test = clf.predict_proba(X_test)[:, 1]
        thr = float(np.quantile(p_test, 1 - y[~is_test].mean()))
        y_hat = p_test >= thr
        metrics[target] = {
            "accuracy": round(float((y_hat == y[is_test]).mean()), 4),
            "f1": round(float(f1_score(y[is_test], y_hat)), 4),
            "precision": round(float(precision_score(y[is_test], y_hat)), 4),
            "recall": round(float(recall_score(y[is_test], y_hat)), 4),
            "threshold": round(thr, 4), "base_rate": round(float(y.mean()), 4),
            "n_train": int((~is_test).sum()), "n_test": int(is_test.sum()),
        }
        preds[target] = clf.predict_proba(X_apply)[:, 1] >= thr
        log.info("textflags %-10s held-out accuracy=%.3f f1=%.3f precision=%.3f recall=%.3f (n=%d)", target,
                 *(metrics[target][k] for k in ("accuracy", "f1", "precision", "recall")), is_test.sum())
    return pd.DataFrame(preds, index=apply_text.index), metrics
