"""Side-by-side comparison of 2+ stocks: normalized price series, latest
technical indicators/signal per symbol, and pairwise return correlation."""

import numpy as np
import pandas as pd
import yfinance as yf

from . import indicators as ind
from . import stocks as stocks_module
from .data import _yf_symbol

MAX_COMPARE_SYMBOLS = 5


def compare_stocks(symbols: list[str], period: str = "6mo", exchange: str = "NSE") -> dict:
    symbols = symbols[:MAX_COMPARE_SYMBOLS]
    if len(symbols) < 2:
        raise ValueError("Provide at least 2 symbols to compare")

    yf_symbols = [_yf_symbol(s, exchange) for s in symbols]
    df = yf.download(
        yf_symbols, period=period, group_by="ticker", threads=True,
        progress=False, auto_adjust=True,
    )

    per_symbol = {}
    close_by_symbol: dict[str, pd.Series] = {}

    for symbol, yf_symbol in zip(symbols, yf_symbols):
        # yfinance always returns ticker-keyed columns here since we pass a
        # list with group_by="ticker" (MAX_COMPARE_SYMBOLS requires >= 2
        # anyway, but keep this consistent with data.py/signals.py).
        sub = df[yf_symbol]
        sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(sub) < 2:
            continue

        close = sub["Close"]
        base = float(close.iloc[0])
        normalized_pct = ((close / base) - 1) * 100

        computed = ind.compute_indicators(sub)
        meta = stocks_module.get_stock_meta(symbol)

        per_symbol[symbol] = {
            "symbol": symbol,
            "name": meta["name"],
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in sub.index],
                "normalizedPercent": [round(float(v), 2) for v in normalized_pct],
            },
            "latest": computed["latest"],
            "signal": computed["signal"],
            "totalReturnPercent": round(float(normalized_pct.iloc[-1]), 2),
        }
        close_by_symbol[symbol] = close

    if len(per_symbol) < 2:
        raise ValueError("Not enough data to compare these symbols")

    # Pairwise correlation of daily returns, aligned on shared trading dates.
    returns_df = pd.DataFrame({s: c.pct_change() for s, c in close_by_symbol.items()}).dropna()
    corr = returns_df.corr()
    correlation_matrix = {
        a: {b: (None if a == b else round(float(corr.loc[a, b]), 2)) for b in corr.columns}
        for a in corr.index
    }

    return {
        "period": period,
        "stocks": [per_symbol[s] for s in symbols if s in per_symbol],
        "correlationMatrix": correlation_matrix,
    }
