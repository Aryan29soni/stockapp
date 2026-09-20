"""Earnings-surprise features: whether the company's most recently reported
quarter beat or missed analyst estimates, and how long ago that was.

Built from yfinance's actual earnings-ANNOUNCEMENT dates (not fiscal
period-end dates), so at each historical trading day we only ever use
earnings that had genuinely already been reported by that day - no
lookahead. Point-in-time fundamentals (P/E, debt/equity, etc.) were
considered too: yfinance's `Ticker.info` only exposes their CURRENT value,
not a historical time series, so for a model trained on one stock's own
past history they'd just be a constant - zero predictive signal. Earnings
surprises are the one fundamentals-adjacent data point yfinance gives us
with real historical dates attached.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from .cache import TTLCache
from .data import _yf_symbol
from .timeouts import call_with_timeout

# reported quarterly, so no need to refetch often. "No earnings data found"
# is itself cached as None (see cache.py's TTLCache.has()) so a symbol with
# no data doesn't get re-fetched from yfinance on every single call.
_CACHE: TTLCache[pd.DataFrame | None] = TTLCache(ttl_seconds=60 * 60 * 6, max_entries=1500)


def _fetch_earnings_dates(symbol: str, exchange: str) -> pd.DataFrame | None:
    try:
        ticker = yf.Ticker(_yf_symbol(symbol, exchange))
        raw = call_with_timeout(ticker.get_earnings_dates, limit=20, timeout=10)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    raw = raw.dropna(subset=["Reported EPS", "Surprise(%)"])
    if raw.empty:
        return None
    raw = raw.sort_index()
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    return raw


def get_earnings_dates_cached(symbol: str, exchange: str) -> pd.DataFrame | None:
    cache_key = f"{symbol}:{exchange}"
    if _CACHE.has(cache_key):
        return _CACHE.get(cache_key)
    result = _fetch_earnings_dates(symbol, exchange)
    _CACHE.set(cache_key, result)
    return result


def build_earnings_features(
    trading_dates: pd.DatetimeIndex, earnings: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Returns a DataFrame indexed like trading_dates with
    'earnings_surprise_pct' and 'days_since_earnings' (both using only the
    most recent report strictly BEFORE each date), or None if there's no
    usable earnings history."""
    if earnings is None or len(earnings) == 0:
        return None

    report_dates = earnings.index.tz_localize(None) if earnings.index.tz is not None else earnings.index
    dates_arr = report_dates.to_numpy()
    surprise_arr = earnings["Surprise(%)"].to_numpy()
    target_dates = trading_dates.tz_localize(None) if trading_dates.tz is not None else trading_dates

    surprise_aligned = np.full(len(target_dates), np.nan)
    days_since = np.full(len(target_dates), np.nan)

    for i, d in enumerate(target_dates.to_numpy()):
        mask = dates_arr < d
        if not mask.any():
            continue
        last_idx = np.where(mask)[0][-1]
        surprise_aligned[i] = surprise_arr[last_idx]
        days_since[i] = (d - dates_arr[last_idx]) / np.timedelta64(1, "D")

    feat = pd.DataFrame(
        {"earnings_surprise_pct": surprise_aligned, "days_since_earnings": days_since},
        index=trading_dates,
    )
    return feat
