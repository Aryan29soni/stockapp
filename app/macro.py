"""Market-wide context beyond a single stock's own price history: India VIX
(market-wide fear/volatility), USDINR and Brent crude (macro drivers that
move whole sectors at once - a rupee move hits importers/exporters
differently, oil hits OMCs/aviation/paints), and sector-index relative
strength for the few sectors Yahoo Finance actually has multi-year daily
history for.

Checked empirically before building this: most NSE sector index tickers
yfinance exposes (Auto, FMCG, Metal, Energy, Realty, Media, PSU Bank, Fin
Service) only return a single live snapshot with no history, so they can't
be backtested against. Only Bank (^NSEBANK), IT (^CNXIT) and Pharma
(^CNXPHARMA) have real ~2y daily series. Sectors outside that set simply
don't get a sector-relative-strength feature (they still get the broad
Nifty-relative and macro features).
"""

import time

import pandas as pd
import yfinance as yf

_CACHE: dict[str, tuple[float, pd.Series]] = {}
_CACHE_TTL_SECONDS = 60 * 30  # these move slowly; no need to refetch often

_MACRO_TICKERS = {
    "vix": "^INDIAVIX",
    "usdinr": "USDINR=X",
    "crude": "BZ=F",
}

_SECTOR_INDEX_TICKERS = {
    "financial services": "^NSEBANK",
    "banking": "^NSEBANK",
    "it": "^CNXIT",
    "information technology": "^CNXIT",
    "pharma": "^CNXPHARMA",
    "healthcare": "^CNXPHARMA",
    "healthcare services": "^CNXPHARMA",
}


def _fetch_series(ticker: str, period: str) -> pd.Series | None:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df.empty:
            return None
        return df["Close"]
    except Exception:
        return None


def _cached_series(cache_key: str, ticker: str, period: str) -> pd.Series | None:
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    series = _fetch_series(ticker, period)
    if series is not None:
        _CACHE[cache_key] = (now, series)
    return series


def get_macro_history(period: str = "2y") -> dict[str, pd.Series]:
    """Returns whichever of {"vix", "usdinr", "crude"} fetched successfully -
    callers should treat missing keys as "skip this feature", not an error."""
    result = {}
    for name, ticker in _MACRO_TICKERS.items():
        series = _cached_series(f"{name}:{period}", ticker, period)
        if series is not None:
            result[name] = series
    return result


def get_sector_index_history(sector: str | None, period: str = "2y") -> pd.Series | None:
    if not sector:
        return None
    ticker = _SECTOR_INDEX_TICKERS.get(sector.strip().lower())
    if not ticker:
        return None
    return _cached_series(f"sector:{ticker}:{period}", ticker, period)


def align_to_calendar(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Reindexes a series from a DIFFERENT exchange/timezone (e.g. USDINR's
    forex-market timestamps, crude's US-market timestamps) onto the stock's
    own NSE trading calendar by calendar date, not exact timestamp - a plain
    .reindex() across timezones would silently match almost nothing and
    produce an all-NaN column, which would then drop every training row."""
    s = series.copy()
    s.index = s.index.tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")]
    target_dates = target_index.tz_localize(None).normalize()
    aligned = s.reindex(target_dates).ffill()
    aligned.index = target_index
    return aligned
