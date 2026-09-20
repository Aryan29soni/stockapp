"""Technical indicators computed from OHLCV data, plus a rule-based signal."""

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    rsi14 = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    bb_upper, bb_mid, bb_lower = bollinger_bands(close)

    latest = {
        "close": round(float(close.iloc[-1]), 2),
        "sma20": _safe_round(sma20.iloc[-1]),
        "sma50": _safe_round(sma50.iloc[-1]),
        "ema12": _safe_round(ema12.iloc[-1]),
        "ema26": _safe_round(ema26.iloc[-1]),
        "rsi14": _safe_round(rsi14.iloc[-1]),
        "macd": _safe_round(macd_line.iloc[-1]),
        "macdSignal": _safe_round(signal_line.iloc[-1]),
        "macdHistogram": _safe_round(hist.iloc[-1]),
        "bollingerUpper": _safe_round(bb_upper.iloc[-1]),
        "bollingerMid": _safe_round(bb_mid.iloc[-1]),
        "bollingerLower": _safe_round(bb_lower.iloc[-1]),
    }

    series = {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "close": [round(float(v), 2) for v in close],
        "sma20": [_safe_round(v) for v in sma20],
        "sma50": [_safe_round(v) for v in sma50],
        "rsi14": [_safe_round(v) for v in rsi14],
        "macd": [_safe_round(v) for v in macd_line],
        "macdSignal": [_safe_round(v) for v in signal_line],
        "bollingerUpper": [_safe_round(v) for v in bb_upper],
        "bollingerLower": [_safe_round(v) for v in bb_lower],
    }

    signal, reasons = _rule_based_signal(latest)

    return {
        "latest": latest,
        "series": series,
        "signal": signal,
        "reasons": reasons,
    }


def _safe_round(value, digits: int = 2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def _rule_based_signal(latest: dict) -> tuple[str, list[str]]:
    """Simple, transparent rule-based buy/sell/hold signal from indicator confluence."""
    score = 0
    reasons = []

    rsi14 = latest["rsi14"]
    if rsi14 is not None:
        if rsi14 < 30:
            score += 1
            reasons.append(f"RSI is {rsi14} (oversold, <30) - bullish signal")
        elif rsi14 > 70:
            score -= 1
            reasons.append(f"RSI is {rsi14} (overbought, >70) - bearish signal")

    if latest["sma20"] is not None and latest["sma50"] is not None:
        if latest["sma20"] > latest["sma50"]:
            score += 1
            reasons.append("20-day SMA is above 50-day SMA - short-term uptrend")
        else:
            score -= 1
            reasons.append("20-day SMA is below 50-day SMA - short-term downtrend")

    if latest["macd"] is not None and latest["macdSignal"] is not None:
        if latest["macd"] > latest["macdSignal"]:
            score += 1
            reasons.append("MACD is above its signal line - bullish momentum")
        else:
            score -= 1
            reasons.append("MACD is below its signal line - bearish momentum")

    if latest["close"] is not None and latest["bollingerLower"] is not None and latest["bollingerUpper"] is not None:
        if latest["close"] <= latest["bollingerLower"]:
            score += 1
            reasons.append("Price is at/below the lower Bollinger Band - potential bounce")
        elif latest["close"] >= latest["bollingerUpper"]:
            score -= 1
            reasons.append("Price is at/above the upper Bollinger Band - potential pullback")

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, reasons
