"""Feature-based ML price forecast: a small ensemble validated by a
multi-fold, walk-forward backtest.

Trains RandomForest + GradientBoosting regressors on technical-indicator
features (including the stock's momentum relative to the Nifty 50 index) to
predict the forward `horizon`-day return, and averages their predictions.
Accuracy is measured across several chronological folds - each trained only
on data strictly before its test window - so the reported numbers reflect
genuine out-of-sample performance aggregated over more evaluations than a
single train/test split would give.

Still a statistical model trained on a few hundred daily bars, not a
crystal ball - it has no notion of news, earnings, or macro events. The API
response always carries a disclaimer.
"""

import threading

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from . import fundamentals as fundamentals_module
from . import indicators as ind
from . import macro as macro_module
from .cache import TTLCache

# Each forecast trains 2 sklearn models x 4 times (3 backtest folds + 1 final
# refit) - real CPU/memory work, not just a lookup. On Render's free tier
# (a fraction of one shared CPU, 512MB RAM), several of these running at once
# - e.g. a user opening a few stocks in a row while the cache is cold - was
# almost certainly what actually caused the repeated OOM kills, not any
# single request's cost alone. This serializes training system-wide so at
# most one fit runs at a time; everything else just waits its turn instead of
# piling up concurrently. A timed-out waiter falls back to the cheap
# trend-line estimate rather than hanging forever.
_TRAINING_LOCK = threading.Semaphore(1)
_TRAINING_LOCK_TIMEOUT_SECONDS = 25

DISCLAIMER = (
    "This forecast comes from a small ensemble of machine-learning models "
    "trained only on this stock's own recent price/volume history, technical "
    "indicators, its momentum relative to the Nifty 50, and (for Banking, IT "
    "and Pharma stocks, where a real historical sector index is available) "
    "its momentum relative to its own sector. It is NOT financial advice, "
    "has no knowledge of news, earnings, or macro events, and its "
    "backtested accuracy below reflects past, not future, performance. Do "
    "not use this as your sole basis for any investment decision."
)

# Measured directly: walk-forward backtested across 10 stocks and all four
# horizon options before shipping. 1-month and 3-month average ~49% directional
# accuracy (a coin flip); 6-month averages ~53%; 1-year averages ~60% but
# swung from 19% to 83% stock-to-stock, meaning that average mostly reflects
# whether a stock happened to be mid-trend during the backtest window, not a
# stable skill. Technical/momentum features (RSI, MACD, moving averages,
# short-term returns) just don't carry much information that far out, and
# longer horizons are also validated on fewer, more overlapping backtest
# windows, making their accuracy figures noisier than the short-horizon ones.
LONG_HORIZON_CAVEAT = (
    " Forecasts 3 months or further out are meaningfully less reliable than "
    "shorter ones: the model's indicators lose predictive power well before "
    "then, and the backtested accuracy shown is based on fewer independent "
    "test windows, so treat it as a rough directional guess, not something "
    "to plan around."
)
LONG_HORIZON_THRESHOLD_DAYS = 63


def _disclaimer_for(horizon_days: int, extra: str = "") -> str:
    text = DISCLAIMER
    if horizon_days >= LONG_HORIZON_THRESHOLD_DAYS:
        text += LONG_HORIZON_CAVEAT
    return text + extra

MIN_ROWS_FOR_ML = 150  # ~7 months of trading days
MIN_TRAIN_ROWS = 60
RANDOM_STATE = 42
N_BACKTEST_FOLDS = 3

# the analysis endpoint's signal and the /predict endpoint both need this
# same fit; caching means switching the price-history chart's period
# (which doesn't affect this 5y-history model) or loading both sections of
# a stock's page doesn't retrain it twice. Bounded (see cache.py) since
# symbol x exchange x 4 horizons is a lot of distinct keys over a session.
_FORECAST_CACHE: TTLCache[dict] = TTLCache(ttl_seconds=60 * 15, max_entries=400)


def _build_features(
    df: pd.DataFrame,
    index_df: pd.DataFrame | None,
    sector_index: pd.Series | None = None,
    macro: dict[str, pd.Series] | None = None,
    earnings_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]

    sma10 = ind.sma(close, 10)
    sma20 = ind.sma(close, 20)
    sma50 = ind.sma(close, 50)
    rsi14 = ind.rsi(close, 14)
    _, _, macd_hist = ind.macd(close)
    bb_upper, _, bb_lower = ind.bollinger_bands(close)

    feat = pd.DataFrame(index=df.index)
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)
    ret_20 = close.pct_change(20)
    feat["ret_5"] = ret_5
    feat["ret_10"] = ret_10
    feat["ret_20"] = ret_20
    feat["sma10_ratio"] = close / sma10 - 1
    feat["sma20_ratio"] = close / sma20 - 1
    feat["sma50_ratio"] = close / sma50 - 1
    feat["rsi14"] = rsi14
    feat["macd_hist_norm"] = macd_hist / close
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    feat["bb_pct_b"] = (close - bb_lower) / bb_width
    feat["vol_ratio"] = volume / volume.rolling(20).mean()
    feat["volatility_20"] = close.pct_change().rolling(20).std()

    if index_df is not None and len(index_df) > 20:
        idx_close = index_df["Close"].reindex(df.index).ffill()
        idx_ret_5 = idx_close.pct_change(5)
        idx_ret_10 = idx_close.pct_change(10)
        idx_ret_20 = idx_close.pct_change(20)
        feat["nifty_ret_10"] = idx_ret_10
        feat["relative_strength_5"] = ret_5 - idx_ret_5
        feat["relative_strength_10"] = ret_10 - idx_ret_10
        feat["relative_strength_20"] = ret_20 - idx_ret_20

    # Only a few sectors have sector-index history deep enough to backtest
    # against (see macro.py) - when unavailable this column is simply never
    # added, rather than added full of NaN (which would drop every row).
    if sector_index is not None and len(sector_index) > 20:
        sector_close = macro_module.align_to_calendar(sector_index, df.index)
        sector_ret_10 = sector_close.pct_change(10)
        feat["sector_relative_strength_10"] = ret_10 - sector_ret_10

    if macro:
        if "vix" in macro:
            vix = macro_module.align_to_calendar(macro["vix"], df.index)
            feat["vix_level"] = vix
            feat["vix_change_5"] = vix.pct_change(5)
        if "usdinr" in macro:
            usdinr = macro_module.align_to_calendar(macro["usdinr"], df.index)
            feat["usdinr_ret_10"] = usdinr.pct_change(10)
        if "crude" in macro:
            crude = macro_module.align_to_calendar(macro["crude"], df.index)
            feat["crude_ret_10"] = crude.pct_change(10)

    if earnings_features is not None:
        feat = feat.join(earnings_features)

    return feat


def _make_models() -> list:
    return [
        RandomForestRegressor(
            n_estimators=250, max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05, min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
    ]


def _ensemble_fit_predict(train_X, train_y, predict_X):
    """Fits both model types on train_X/train_y and returns the mean of their
    predictions on predict_X (a simple, robust two-model ensemble)."""
    preds = []
    for model in _make_models():
        model.fit(train_X, train_y)
        preds.append(model.predict(predict_X))
    return np.mean(preds, axis=0)


def _points_from_path(
    last_close: float,
    future_dates,
    predicted_return: float,
    err_low: float,
    err_high: float,
) -> list[dict]:
    n = len(future_dates)
    points = []
    for i, d in enumerate(future_dates, start=1):
        frac = i / n
        point_return = predicted_return * frac
        band_scale = np.sqrt(frac)
        predicted_price = last_close * np.exp(point_return)
        upper = last_close * np.exp(point_return + abs(err_high) * band_scale)
        lower = last_close * np.exp(point_return - abs(err_low) * band_scale)
        points.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "predicted": round(float(predicted_price), 2),
                "upper": round(float(max(upper, lower, predicted_price)), 2),
                "lower": round(float(min(upper, lower, predicted_price)), 2),
            }
        )
    return points


def _trend(predicted_return: float) -> str:
    if predicted_return > 0.005:
        return "UPWARD"
    if predicted_return < -0.005:
        return "DOWNWARD"
    return "FLAT"


def forecast(
    df: pd.DataFrame,
    horizon_days: int = 10,
    index_df: pd.DataFrame | None = None,
    sector_index: pd.Series | None = None,
    macro: dict[str, pd.Series] | None = None,
    earnings: pd.DataFrame | None = None,
) -> dict:
    if len(df) >= MIN_ROWS_FOR_ML:
        result = _ml_forecast(df, horizon_days, index_df, sector_index, macro, earnings)
        if result is not None:
            return result
    return _trend_fallback_forecast(df, horizon_days)


def forecast_cached(
    symbol: str,
    exchange: str,
    df: pd.DataFrame,
    horizon_days: int,
    index_df: pd.DataFrame | None,
    sector_index: pd.Series | None = None,
    macro: dict[str, pd.Series] | None = None,
    earnings: pd.DataFrame | None = None,
) -> dict:
    cache_key = f"{symbol}:{exchange}:{horizon_days}"
    cached = _FORECAST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = forecast(
        df, horizon_days=horizon_days, index_df=index_df,
        sector_index=sector_index, macro=macro, earnings=earnings,
    )
    _FORECAST_CACHE.set(cache_key, result)
    return result


def _walk_forward_backtest(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Runs N_BACKTEST_FOLDS chronological train/test splits, each trained
    only on data before its test window, and returns the concatenated
    (predictions, actuals) across all folds' test sets."""
    n = len(X)
    fold_size = max(15, int(n * 0.15))
    all_preds, all_actuals = [], []

    for i in range(N_BACKTEST_FOLDS, 0, -1):
        test_end = n - (i - 1) * fold_size
        test_start = test_end - fold_size
        if test_start < MIN_TRAIN_ROWS:
            continue
        train_X, test_X = X[:test_start], X[test_start:test_end]
        train_y, test_y = y[:test_start], y[test_start:test_end]
        if len(test_X) == 0:
            continue
        preds = _ensemble_fit_predict(train_X, train_y, test_X)
        all_preds.append(preds)
        all_actuals.append(test_y)

    if not all_preds:
        return None
    return np.concatenate(all_preds), np.concatenate(all_actuals)


def _ml_forecast(
    df: pd.DataFrame,
    horizon_days: int,
    index_df: pd.DataFrame | None,
    sector_index: pd.Series | None = None,
    macro: dict[str, pd.Series] | None = None,
    earnings: pd.DataFrame | None = None,
) -> dict | None:
    close = df["Close"]
    earnings_features = fundamentals_module.build_earnings_features(df.index, earnings)
    features = _build_features(df, index_df, sector_index, macro, earnings_features)
    target = np.log(close.shift(-horizon_days) / close)

    data = features.copy()
    data["target"] = target
    data = data.dropna()

    if len(data) < MIN_TRAIN_ROWS + 15:
        return None

    feature_cols = list(features.columns)
    X = data[feature_cols].to_numpy()
    y = data["target"].to_numpy()

    latest_row = features.iloc[[-1]][feature_cols].to_numpy()
    if np.isnan(latest_row).any():
        return None

    if not _TRAINING_LOCK.acquire(timeout=_TRAINING_LOCK_TIMEOUT_SECONDS):
        return None  # too much concurrent demand right now - caller falls back to the trend estimate
    try:
        backtest = _walk_forward_backtest(X, y)
        if backtest is None:
            return None
        test_preds, test_actuals = backtest

        errors = test_preds - test_actuals
        mae = float(np.mean(np.abs(errors)))
        directional_accuracy = float(np.mean(np.sign(test_preds) == np.sign(test_actuals)))
        err_low, err_high = (float(v) for v in np.percentile(errors, [10, 90]))

        # Final ensemble refit on all labeled data (including backtest folds)
        # for the live forecast itself - backtest accuracy above was measured
        # on models that never saw their own test fold.
        predicted_return = float(_ensemble_fit_predict(X, y, latest_row)[0])
    finally:
        _TRAINING_LOCK.release()

    last_close = float(close.iloc[-1])
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=horizon_days + 1)[1:]
    points = _points_from_path(last_close, future_dates, predicted_return, err_low, err_high)

    return {
        "method": "ml_ensemble",
        "lastClose": round(last_close, 2),
        "horizonDays": horizon_days,
        "trend": _trend(predicted_return),
        "expectedChangePercent": round((np.exp(predicted_return) - 1) * 100, 2),
        "confidence": round(max(0.0, min(1.0, directional_accuracy)), 2),
        "backtestedDirectionalAccuracyPercent": round(directional_accuracy * 100, 1),
        "backtestedMaeReturnPercent": round(mae * 100, 2),
        "backtestSamples": int(len(test_preds)),
        "points": points,
        "disclaimer": _disclaimer_for(horizon_days),
    }


def _trend_fallback_forecast(df: pd.DataFrame, horizon_days: int) -> dict:
    """Simple linear-trend fallback for symbols with too little history for ML."""
    lookback_days = 60
    close = df["Close"].tail(lookback_days).reset_index(drop=True)
    n = len(close)
    if n < 10:
        raise ValueError("Not enough history to compute a forecast")

    log_prices = np.log(close.to_numpy())
    x = np.arange(n)
    slope, intercept = np.polyfit(x, log_prices, deg=1)
    fitted = slope * x + intercept
    residuals = log_prices - fitted
    residual_std = float(np.std(residuals)) if n > 1 else 0.0

    predicted_return = float(slope) * horizon_days
    last_close = float(close.iloc[-1])
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=horizon_days + 1)[1:]

    band = residual_std * np.sqrt(horizon_days) * 1.64  # ~90% band, matches ML's 10/90 pctile
    points = _points_from_path(last_close, future_dates, predicted_return, band, band)

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((log_prices - np.mean(log_prices)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "method": "trend_fallback",
        "lastClose": round(last_close, 2),
        "horizonDays": horizon_days,
        "trend": _trend(predicted_return),
        "expectedChangePercent": round((np.exp(predicted_return) - 1) * 100, 2),
        "confidence": round(max(0.0, min(1.0, r_squared)), 2),
        "backtestedDirectionalAccuracyPercent": None,
        "backtestedMaeReturnPercent": None,
        "backtestSamples": 0,
        "points": points,
        "disclaimer": _disclaimer_for(
            horizon_days,
            extra=" (Not enough price history yet for the ML model, so this uses a "
            "simpler trend-line estimate instead.)",
        ),
    }
