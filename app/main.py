import asyncio
import logging

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

load_dotenv()


class _SuppressLogo404(logging.Filter):
    """/logo/{symbol} 404s constantly and by design for any symbol without a
    configured LOGO_DEV_API_KEY or a curated domain - that's the expected
    signal telling the mobile app to fall back to its colored-initial badge,
    not an error worth a log line every time someone scrolls the stock list."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not ("/logo/" in message and " 404 " in message)


logging.getLogger("uvicorn.access").addFilter(_SuppressLogo404())

from . import auth, compare, data, indicators, logos, macro, news, predict, signals, social, stocks  # noqa: E402
from .db import (  # noqa: E402
    PortfolioHolding,
    PortfolioTransaction,
    User,
    WatchlistItem,
    get_db,
    init_db,
)
from .timeouts import DataProviderTimeout  # noqa: E402

app = FastAPI(title="Indian Stock Analysis & Prediction API", version="0.1.0")

# Render's free tier is ~512MB with a fraction of one CPU. predict.py's
# _TRAINING_LOCK bounds concurrent ML training, and timeouts.py bounds how
# long any single yfinance call can hang, but neither stops several
# DIFFERENT heavy requests (a few stocks' 5y history downloads, a news scan
# across 150 symbols, a forecast) from simply running at the same time and
# adding up to more memory than the instance has - confirmed by testing this
# directly: 3 concurrent /predict calls plus a /news call was enough to
# crash the instance even with both of those fixes in place. This caps how
# many requests the whole app processes at once; anything beyond that queues
# briefly, then fails fast with a 503 rather than piling on and taking the
# whole instance down for everyone.
_MAX_CONCURRENT_REQUESTS = 4
_CONCURRENCY_QUEUE_TIMEOUT_SECONDS = 20
_REQUEST_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)


@app.middleware("http")
async def limit_concurrency(request: Request, call_next):
    # Render's own liveness probe hits /health constantly - it must never be
    # queued behind heavy requests, or Render can conclude the instance
    # itself is unhealthy and restart it, which defeats the point of this
    # limiter.
    if request.url.path == "/health":
        return await call_next(request)
    try:
        await asyncio.wait_for(_REQUEST_SEMAPHORE.acquire(), timeout=_CONCURRENCY_QUEUE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is busy right now. Please try again in a moment."},
        )
    try:
        return await call_next(request)
    finally:
        _REQUEST_SEMAPHORE.release()


# Registered AFTER limit_concurrency (and must stay that way): Starlette
# wraps middleware in reverse-registration order, so whichever is added LAST
# ends up OUTERMOST. CORS needs to be outermost so it can attach headers to
# every response, including limit_concurrency's own 503 short-circuit that
# never reaches the route/CORS layer otherwise - confirmed live in
# production, where that 503 came back with no CORS header at all and the
# browser reported it as an opaque "CORS policy" failure instead of a
# readable 503, defeating the whole point of returning a clean error.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DataProviderTimeout)
async def data_provider_timeout_handler(request: Request, exc: DataProviderTimeout) -> JSONResponse:
    # Catches a slow/rate-limiting Yahoo Finance response for every endpoint
    # in one place, rather than every route needing its own try/except - see
    # timeouts.py for why this matters (a hung yfinance call used to be able
    # to take the whole backend down, not just fail one request).
    return JSONResponse(
        status_code=503,
        content={"detail": "Market data provider is responding slowly right now. Please try again in a moment."},
    )


init_db()


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class PreferencesRequest(BaseModel):
    themePreference: str | None = None


class SymbolRequest(BaseModel):
    symbol: str


class MergeWatchlistRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)


class DepositRequest(BaseModel):
    amount: float = Field(gt=0, le=100_000_000)


class TradeRequest(BaseModel):
    symbol: str
    quantity: int = Field(gt=0)


def _user_payload(user: User) -> dict:
    return {"id": user.id, "username": user.username, "themePreference": user.theme_preference}


@app.post("/auth/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    username = auth.validate_username(body.username)
    auth.validate_password(body.password)
    user = User(username=username, password_hash=auth.hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="That username is already taken.")
    db.refresh(user)
    return {"token": auth.create_token(user.id), "user": _user_payload(user)}


@app.post("/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    username = auth.normalize_username(body.username)
    auth.check_rate_limit(username)
    user = db.scalar(select(User).where(User.username == username))
    if not user or not auth.verify_password(body.password, user.password_hash):
        auth.record_failed_attempt(username)
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    auth.clear_attempts(username)
    return {"token": auth.create_token(user.id), "user": _user_payload(user)}


@app.get("/auth/me")
def me(user: User = Depends(auth.get_current_user)):
    return _user_payload(user)


@app.put("/auth/preferences")
def update_preferences(
    body: PreferencesRequest, user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)
):
    user.theme_preference = body.themePreference
    db.commit()
    db.refresh(user)
    return _user_payload(user)


@app.get("/watchlist")
def get_watchlist(user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    items = db.scalars(select(WatchlistItem).where(WatchlistItem.user_id == user.id)).all()
    return {"symbols": [i.symbol for i in items]}


@app.post("/watchlist")
def add_watchlist_item(
    body: SymbolRequest, user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)
):
    symbol = body.symbol.strip().upper()
    exists = db.scalar(
        select(WatchlistItem).where(WatchlistItem.user_id == user.id, WatchlistItem.symbol == symbol)
    )
    if not exists:
        db.add(WatchlistItem(user_id=user.id, symbol=symbol))
        db.commit()
    items = db.scalars(select(WatchlistItem).where(WatchlistItem.user_id == user.id)).all()
    return {"symbols": [i.symbol for i in items]}


@app.delete("/watchlist/{symbol}")
def remove_watchlist_item(
    symbol: str, user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)
):
    db.execute(
        delete(WatchlistItem).where(WatchlistItem.user_id == user.id, WatchlistItem.symbol == symbol.upper())
    )
    db.commit()
    items = db.scalars(select(WatchlistItem).where(WatchlistItem.user_id == user.id)).all()
    return {"symbols": [i.symbol for i in items]}


@app.post("/watchlist/merge")
def merge_watchlist(
    body: MergeWatchlistRequest, user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)
):
    """Unions the caller's locally-stored symbols (from before they were
    logged in) into their server watchlist, rather than overwriting either
    side - called once right after login."""
    existing = {
        i.symbol for i in db.scalars(select(WatchlistItem).where(WatchlistItem.user_id == user.id)).all()
    }
    for symbol in body.symbols:
        upper = symbol.strip().upper()
        if upper and upper not in existing:
            db.add(WatchlistItem(user_id=user.id, symbol=upper))
            existing.add(upper)
    db.commit()
    return {"symbols": sorted(existing)}


def _holding_payload(holding: PortfolioHolding, exchange: str = "NSE") -> dict:
    try:
        current_price = data.get_quote(holding.symbol, exchange=exchange)["price"]
    except ValueError:
        current_price = None
    cost_basis = holding.quantity * holding.avg_buy_price
    market_value = holding.quantity * current_price if current_price is not None else None
    unrealized_pnl = market_value - cost_basis if market_value is not None else None
    unrealized_pnl_percent = (
        (unrealized_pnl / cost_basis * 100) if unrealized_pnl is not None and cost_basis > 0 else None
    )
    return {
        "symbol": holding.symbol,
        "quantity": holding.quantity,
        "avgBuyPrice": round(holding.avg_buy_price, 2),
        "currentPrice": round(current_price, 2) if current_price is not None else None,
        "costBasis": round(cost_basis, 2),
        "marketValue": round(market_value, 2) if market_value is not None else None,
        "unrealizedPnl": round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
        "unrealizedPnlPercent": round(unrealized_pnl_percent, 2) if unrealized_pnl_percent is not None else None,
    }


def _transaction_payload(tx: PortfolioTransaction) -> dict:
    return {
        "id": tx.id,
        "type": tx.type,
        "symbol": tx.symbol,
        "quantity": tx.quantity,
        "pricePerShare": round(tx.price_per_share, 2) if tx.price_per_share is not None else None,
        "amount": round(tx.amount, 2),
        "createdAt": tx.created_at.isoformat() if tx.created_at else None,
    }


@app.get("/portfolio")
def get_portfolio(
    exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    holdings = db.scalars(
        select(PortfolioHolding).where(PortfolioHolding.user_id == user.id, PortfolioHolding.quantity > 0)
    ).all()
    holding_payloads = [_holding_payload(h, exchange) for h in holdings]
    holdings_value = sum(h["marketValue"] for h in holding_payloads if h["marketValue"] is not None)
    transactions = db.scalars(
        select(PortfolioTransaction)
        .where(PortfolioTransaction.user_id == user.id)
        .order_by(PortfolioTransaction.created_at.desc())
        .limit(50)
    ).all()
    return {
        "cashBalance": round(user.paper_cash_balance, 2),
        "holdingsValue": round(holdings_value, 2),
        "totalValue": round(user.paper_cash_balance + holdings_value, 2),
        "holdings": holding_payloads,
        "transactions": [_transaction_payload(t) for t in transactions],
    }


@app.post("/portfolio/deposit")
def deposit_funds(
    body: DepositRequest, user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)
):
    """Dummy fund source - no payment processor, no real money. Adds
    straight to the user's paper_cash_balance so the forecast/signal
    features can be tested with a simulated portfolio."""
    user.paper_cash_balance += body.amount
    db.add(PortfolioTransaction(user_id=user.id, type="DEPOSIT", amount=body.amount))
    db.commit()
    return {"cashBalance": round(user.paper_cash_balance, 2)}


@app.post("/portfolio/buy")
def buy_stock(
    body: TradeRequest,
    exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    symbol = body.symbol.strip().upper()
    try:
        price = data.get_quote(symbol, exchange=exchange)["price"]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    total_cost = price * body.quantity
    if total_cost > user.paper_cash_balance:
        raise HTTPException(status_code=400, detail="Not enough virtual cash for this trade.")

    holding = db.scalar(
        select(PortfolioHolding).where(PortfolioHolding.user_id == user.id, PortfolioHolding.symbol == symbol)
    )
    if holding and holding.quantity > 0:
        new_quantity = holding.quantity + body.quantity
        holding.avg_buy_price = (
            holding.avg_buy_price * holding.quantity + price * body.quantity
        ) / new_quantity
        holding.quantity = new_quantity
    elif holding:
        holding.quantity = body.quantity
        holding.avg_buy_price = price
    else:
        holding = PortfolioHolding(
            user_id=user.id, symbol=symbol, quantity=body.quantity, avg_buy_price=price
        )
        db.add(holding)

    user.paper_cash_balance -= total_cost
    db.add(
        PortfolioTransaction(
            user_id=user.id, type="BUY", symbol=symbol, quantity=body.quantity,
            price_per_share=price, amount=-total_cost,
        )
    )
    db.commit()
    return {"cashBalance": round(user.paper_cash_balance, 2), "holding": _holding_payload(holding, exchange)}


@app.post("/portfolio/sell")
def sell_stock(
    body: TradeRequest,
    exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    symbol = body.symbol.strip().upper()
    holding = db.scalar(
        select(PortfolioHolding).where(PortfolioHolding.user_id == user.id, PortfolioHolding.symbol == symbol)
    )
    if not holding or holding.quantity < body.quantity:
        raise HTTPException(status_code=400, detail="Not enough shares held to sell that quantity.")

    try:
        price = data.get_quote(symbol, exchange=exchange)["price"]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    proceeds = price * body.quantity
    holding.quantity -= body.quantity
    user.paper_cash_balance += proceeds
    db.add(
        PortfolioTransaction(
            user_id=user.id, type="SELL", symbol=symbol, quantity=body.quantity,
            price_per_share=price, amount=proceeds,
        )
    )
    db.commit()
    return {"cashBalance": round(user.paper_cash_balance, 2), "holding": _holding_payload(holding, exchange)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stocks")
def list_stocks(q: str = Query("", description="Search by symbol or name")):
    return {"results": stocks.search_stocks(q)}


@app.get("/stocks/{symbol}/quote")
def quote(symbol: str, exchange: str = Query("NSE", pattern="^(NSE|BSE)$")):
    try:
        q = data.get_quote(symbol, exchange=exchange)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    meta = stocks.get_stock_meta(symbol)
    return {**q, "name": meta["name"], "sector": meta["sector"], "exchange": exchange}


@app.get("/quotes")
def quotes(
    symbols: str = Query(..., description="Comma-separated symbols, e.g. RELIANCE,TCS"),
    exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:300]
    results = data.get_quotes(symbol_list, exchange=exchange)
    by_symbol = {q["symbol"]: q for q in results}
    return {
        "results": [
            {**by_symbol[s], "name": stocks.get_stock_meta(s)["name"]}
            for s in symbol_list
            if s in by_symbol
        ]
    }


@app.get("/news")
def news_endpoint(
    limit: int = Query(20, ge=1, le=50),
    exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
):
    return news.get_news_feed(limit=limit, exchange=exchange)


@app.get("/social/trending")
def social_trending(limit: int = Query(20, ge=1, le=50)):
    return social.get_trending(limit=limit)


@app.get("/logo/{symbol}")
def logo(symbol: str):
    name = stocks.get_stock_meta(symbol).get("name")
    redirect_url = logos.get_logo_redirect_url(symbol, name)
    if not redirect_url:
        raise HTTPException(status_code=404, detail="No logo available for this symbol")
    return RedirectResponse(redirect_url)


@app.get("/compare")
def compare_endpoint(
    symbols: str = Query(..., description="Comma-separated symbols, 2-5, e.g. RELIANCE,TCS"),
    period: str = Query("6mo", pattern="^(1mo|3mo|6mo|1y|2y)$"),
    exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    try:
        return compare.compare_stocks(symbol_list, period=period, exchange=exchange)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/signals")
def bulk_signals(
    symbols: str = Query(..., description="Comma-separated symbols, e.g. RELIANCE,TCS"),
    exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:300]
    return {"results": signals.get_bulk_signals(symbol_list, exchange=exchange)}


@app.get("/stocks/{symbol}/history")
def history(
    symbol: str,
    period: str = Query("6mo", pattern="^(1mo|3mo|6mo|1y|2y|5y)$"),
    exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
):
    try:
        df = data.get_history(symbol, period=period, exchange=exchange)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "symbol": symbol.upper(),
        "period": period,
        "candles": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": round(float(r.Open), 2),
                "high": round(float(r.High), 2),
                "low": round(float(r.Low), 2),
                "close": round(float(r.Close), 2),
                "volume": int(r.Volume),
            }
            for d, r in df.iterrows()
        ],
    }


@app.get("/stocks/{symbol}/analysis")
def analysis(
    symbol: str,
    period: str = Query("6mo", pattern="^(1mo|3mo|6mo|1y|2y|5y)$"),
    exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
):
    try:
        df = data.get_history(symbol, period=period, exchange=exchange)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    result = indicators.compute_indicators(df)
    # This stays the fast, rule-based signal (indicator confluence) so the
    # page's price/indicator sections render immediately. The single-stock
    # view upgrades this to a walk-forward-backtested ML call once the
    # /predict forecast (already fetched for that section) has loaded - see
    # deriveSignal() in mobile/src/utils/mlSignal.ts. Training a second model
    # here just to answer synchronously would block this normally-instant
    # endpoint on a ~7s fit.
    return {"symbol": symbol.upper(), **result}


def _forecast_context(symbol: str, exchange: str) -> dict:
    """Best-effort extra context for the forecast model beyond the stock's
    own price history.

    Sector-relative-strength is the only addition wired in live: an A/B
    backtest across 10 stocks showed it's the only one of {sector, market
    macro (VIX/USDINR/crude), earnings-surprise} that improved average
    backtested directional accuracy (52.4% -> 53.7%). Macro was roughly a
    wash (52.5%), earnings-surprise made it WORSE on average (51.0%) - with
    only ~450 rows of history per stock split across 3 backtest folds, extra
    features mostly gave the model more room to fit noise. Combining all
    three was worse than doing nothing (50.6%). macro.py and fundamentals.py
    still exist and are tested (see their A/B results in the ml_signal
    history) in case a future iteration with more data or cross-stock
    pooling can use them profitably.
    """
    try:
        index_df = data.get_index_history(period="5y", exchange=exchange)
    except Exception:
        index_df = None
    try:
        sector = stocks.get_stock_meta(symbol).get("sector")
        sector_index = macro.get_sector_index_history(sector, period="5y")
    except Exception:
        sector_index = None
    return {"index_df": index_df, "sector_index": sector_index}


@app.get("/stocks/{symbol}/predict")
def prediction(
    symbol: str,
    horizon: int = Query(10, ge=1, le=252),
    exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
):
    try:
        # 5y (not 2y) so longer horizons - up to 252 trading days out - have
        # enough runway for a meaningful walk-forward backtest; see predict.py
        # for what was actually validated at each horizon before shipping.
        df = data.get_history(symbol, period="5y", exchange=exchange)
        ctx = _forecast_context(symbol.upper(), exchange)
        result = predict.forecast_cached(symbol.upper(), exchange, df, horizon, **ctx)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"symbol": symbol.upper(), **result}
