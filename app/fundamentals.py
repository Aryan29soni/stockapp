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

import time

import numpy as np
import pandas as pd
import yfinance as yf

from .data import _yf_symbol

_CACHE: dict[str, tuple[float, pd.DataFrame | None]] = {}
_CACHE_TTL_SECONDS = 60 * 60 * 6  # reported quarterly; no need to refetch often


def _fetch_earnings_dates(symbol: str, exchange: str) -> pd.DataFrame | None:
    try:
        raw = yf.Ticker(_yf_symbol(symbol, exchange)).get_earnings_dates(limit=20)
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
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    result = _fetch_earnings_dates(symbol, exchange)
    _CACHE[cache_key] = (now, result)
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
