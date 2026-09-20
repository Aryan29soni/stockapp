"""Bulk BUY/SELL/HOLD technical signals for many symbols at once (for the
app's filter chips), computed from one batched history download rather than
one request per symbol."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

from . import indicators as ind
from .cache import TTLCache
from .data import _yf_symbol  # reuse the same .NS/.BO suffix logic
from .timeouts import DataProviderTimeout, call_with_timeout

# signals only move once per trading day; bounded to 1,500 since every
# distinct symbol viewed (there are 1,400+ NSE stocks) adds a key (see cache.py).
_SIGNAL_CACHE: TTLCache[dict] = TTLCache(ttl_seconds=60 * 15, max_entries=1500)

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
    results: dict[str, dict] = {}
    need_fetch = []

    for symbol in symbols:
        cached = _SIGNAL_CACHE.get(f"{symbol}:{exchange}")
        if cached is not None:
            results[symbol] = cached
        else:
            need_fetch.append(symbol)

    if need_fetch:
        yf_symbols = [_yf_symbol(s, exchange) for s in need_fetch]
        chunks = [yf_symbols[i : i + _CHUNK_SIZE] for i in range(0, len(yf_symbols), _CHUNK_SIZE)]

        # A hung chunk (Yahoo slow/rate-limiting) must not take the whole
        # bulk request down with it - skip that chunk's symbols rather than
        # blocking every caller of /signals or /news behind it (see
        # timeouts.py).
        def _fetch_chunk_safe(chunk: list[str]) -> pd.DataFrame | None:
            try:
                return call_with_timeout(_download_chunk, chunk, timeout=20)
            except DataProviderTimeout:
                return None

        if len(chunks) == 1:
            dfs = [_fetch_chunk_safe(chunks[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                dfs = list(pool.map(_fetch_chunk_safe, chunks))

        # Map each symbol to whichever chunk's dataframe it belongs to (chunks
        # that timed out are simply absent, so those symbols get skipped below).
        symbol_to_df: dict[str, pd.DataFrame] = {}
        for chunk, df in zip(chunks, dfs):
            if df is None:
                continue
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
                _SIGNAL_CACHE.set(f"{symbol}:{exchange}", entry)
                results[symbol] = entry
            except (KeyError, ValueError):
                continue

    return results
