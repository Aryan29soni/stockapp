"""Bulk BUY/SELL/HOLD technical signals for many symbols at once (for the
app's filter chips), computed from one batched history download rather than
one request per symbol."""

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

from . import indicators as ind
from .data import _yf_symbol  # reuse the same .NS/.BO suffix logic

_SIGNAL_CACHE: dict[str, tuple[float, dict]] = {}
_SIGNAL_CACHE_TTL_SECONDS = 60 * 15  # signals only move once per trading day

_CHUNK_SIZE = 75  # yfinance's bulk download scales roughly linearly with
# symbol count even with threads=True, so splitting a big request into a few
# chunks fetched concurrently is noticeably faster than one huge call.


def _download_chunk(yf_symbols: list[str]) -> pd.DataFrame:
    return yf.download(
        yf_symbols, period="6mo", group_by="ticker", threads=True,
        progress=False, auto_adjust=True,
    )


def get_bulk_signals(symbols: list[str], exchange: str = "NSE") -> dict[str, dict]:
    """Returns {symbol: {"signal": "BUY"|"SELL"|"HOLD", "reasons": [...]}} for
    as many of the requested symbols as have enough history."""
    now = time.time()
    results: dict[str, dict] = {}
    need_fetch = []

    for symbol in symbols:
        cache_key = f"{symbol}:{exchange}"
        cached = _SIGNAL_CACHE.get(cache_key)
        if cached and now - cached[0] < _SIGNAL_CACHE_TTL_SECONDS:
            results[symbol] = cached[1]
        else:
            need_fetch.append(symbol)

    if need_fetch:
        yf_symbols = [_yf_symbol(s, exchange) for s in need_fetch]
        chunks = [yf_symbols[i : i + _CHUNK_SIZE] for i in range(0, len(yf_symbols), _CHUNK_SIZE)]

        if len(chunks) == 1:
            dfs = [_download_chunk(chunks[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                dfs = list(pool.map(_download_chunk, chunks))

        # Map each symbol to whichever chunk's dataframe it belongs to.
        symbol_to_df: dict[str, pd.DataFrame] = {}
        for chunk, df in zip(chunks, dfs):
            for yf_symbol in chunk:
                symbol_to_df[yf_symbol] = df

        for symbol, yf_symbol in zip(need_fetch, yf_symbols):
            try:
                # yfinance always returns ticker-keyed columns here since we
                # pass a list (even a list of one) with group_by="ticker".
                df = symbol_to_df[yf_symbol]
                sub = df[yf_symbol]
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(sub) < 20:
                    continue
                computed = ind.compute_indicators(sub)
                entry = {"signal": computed["signal"], "reasons": computed["reasons"]}
                _SIGNAL_CACHE[f"{symbol}:{exchange}"] = (now, entry)
                results[symbol] = entry
            except (KeyError, ValueError):
                continue

    return results
