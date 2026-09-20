"""Fetches OHLCV data and live-ish quotes for Indian (NSE/BSE) tickers via yfinance."""

import pandas as pd
import yfinance as yf

from .cache import TTLCache

# max_entries bounds: with 1,400+ NSE symbols x several history periods,
# an unbounded cache here is what was actually OOM-killing the backend on
# Render's free 512MB tier - see cache.py. 300 history entries (full
# DataFrames) and 1500 quotes (small dicts) comfortably covers a full
# session's worth of browsing without growing forever.
_HISTORY_CACHE: TTLCache[pd.DataFrame] = TTLCache(ttl_seconds=60 * 5, max_entries=300)
_QUOTE_CACHE: TTLCache[dict] = TTLCache(ttl_seconds=10, max_entries=1500)


def _yf_symbol(symbol: str, exchange: str = "NSE") -> str:
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    return f"{symbol.strip().upper()}{suffix}"


def get_history(symbol: str, period: str = "6mo", exchange: str = "NSE") -> pd.DataFrame:
    """Returns a DataFrame indexed by date with columns Open/High/Low/Close/Volume."""
    cache_key = f"{symbol}:{period}:{exchange}"
    cached = _HISTORY_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    ticker = yf.Ticker(_yf_symbol(symbol, exchange))
    df = ticker.history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for symbol '{symbol}' on {exchange}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    _HISTORY_CACHE.set(cache_key, df)
    return df.copy()


_INDEX_TICKERS = {"NSE": "^NSEI", "BSE": "^BSESN"}  # Nifty 50 / Sensex


def get_index_history(period: str = "2y", exchange: str = "NSE") -> pd.DataFrame:
    """Broad market index history, used as a relative-momentum feature for forecasts."""
    cache_key = f"INDEX:{exchange}:{period}"
    cached = _HISTORY_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    ticker = yf.Ticker(_INDEX_TICKERS.get(exchange.upper(), "^NSEI"))
    df = ticker.history(period=period, auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    _HISTORY_CACHE.set(cache_key, df)
    return df.copy()


def get_quote(symbol: str, exchange: str = "NSE") -> dict:
    """Live-ish quote via yfinance's lightweight fast_info (no full history download)."""
    cache_key = f"{symbol}:{exchange}"
    cached = _QUOTE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ticker = yf.Ticker(_yf_symbol(symbol, exchange))
    try:
        info = ticker.fast_info
        price = float(info["lastPrice"])
        prev_close = float(info["previousClose"])
        day_high = float(info["dayHigh"])
        day_low = float(info["dayLow"])
        volume = int(info["lastVolume"])
    except (KeyError, TypeError, ValueError):
        # Fallback for symbols fast_info doesn't cover well
        df = get_history(symbol, period="5d", exchange=exchange)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        price = float(last["Close"])
        prev_close = float(prev["Close"])
        day_high = float(last["High"])
        day_low = float(last["Low"])
        volume = int(last["Volume"])

    if not price or price != price:  # zero or NaN
        raise ValueError(f"No live price available for symbol '{symbol}' on {exchange}")

    change = price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close else 0.0

    quote = {
        "symbol": symbol.upper(),
        "price": round(price, 2),
        "change": round(change, 2),
        "changePercent": round(change_pct, 2),
        "dayHigh": round(day_high, 2),
        "dayLow": round(day_low, 2),
        "volume": volume,
        "asOf": pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S"),
    }
    _QUOTE_CACHE.set(cache_key, quote)
    return quote


def _bulk_fetch_quotes(symbols: list[str], exchange: str) -> dict[str, dict]:
    """One batched request for many symbols at once - far faster than N individual
    fast_info calls (yfinance's chart API supports fetching many tickers together)."""
    if not symbols:
        return {}

    yf_symbols = [_yf_symbol(s, exchange) for s in symbols]
    df = yf.download(
        yf_symbols, period="5d", group_by="ticker", threads=True,
        progress=False, auto_adjust=True,
    )
    now_str = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S")

    results: dict[str, dict] = {}
    for symbol, yf_symbol in zip(symbols, yf_symbols):
        try:
            # yfinance always returns ticker-keyed columns here since we pass
            # a list (even a list of one) with group_by="ticker" - there is
            # no "flat columns" case to special-case for a single symbol.
            sub = df[yf_symbol]
            sub = sub.dropna(subset=["Close"])
            if len(sub) == 0:
                continue
            last = sub.iloc[-1]
            prev = sub.iloc[-2] if len(sub) > 1 else last
            price = float(last["Close"])
            prev_close = float(prev["Close"])
            if not price or price != price:  # zero or NaN
                continue
            change = price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close else 0.0
            results[symbol] = {
                "symbol": symbol.upper(),
                "price": round(price, 2),
                "change": round(change, 2),
                "changePercent": round(change_pct, 2),
                "dayHigh": round(float(last["High"]), 2),
                "dayLow": round(float(last["Low"]), 2),
                "volume": int(last["Volume"]),
                "asOf": now_str,
            }
        except (KeyError, ValueError, TypeError):
            continue
    return results


def get_quotes(symbols: list[str], exchange: str = "NSE") -> list[dict]:
    """Fetch multiple live quotes, batching the network call and reusing the
    same short-lived cache as get_quote (for list/watchlist polling)."""
    quotes: dict[str, dict] = {}
    need_fetch = []

    for symbol in symbols:
        cached = _QUOTE_CACHE.get(f"{symbol}:{exchange}")
        if cached is not None:
            quotes[symbol] = cached
        else:
            need_fetch.append(symbol)

    if need_fetch:
        fetched = _bulk_fetch_quotes(need_fetch, exchange)
        for symbol, quote in fetched.items():
            _QUOTE_CACHE.set(f"{symbol}:{exchange}", quote)
            quotes[symbol] = quote

    return [quotes[s] for s in symbols if s in quotes]
