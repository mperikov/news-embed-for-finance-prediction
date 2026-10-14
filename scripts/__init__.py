"""Data pipeline for the *News embeddings for finance prediction* project.

Modules
-------
config         paths, date range, tickers, finance filter
news           NYT article metadata (Kaggle mirror of the NYT Archive API + optional live API extension)
market         daily prices from Yahoo Finance (yfinance)
align          trading calendar, news -> trading-day assignment, aligned daily table
build_dataset  `ensure_dataset()` entry point used by the notebook, and a CLI
"""
