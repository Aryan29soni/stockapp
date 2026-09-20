"""Storage for the one thing in this app that needs to persist beyond a
single request: user accounts and their watchlists. Everything else
(quotes, indicators, forecasts, news, social) is fetched live and cached
in memory - this is the only real database.

Defaults to a local SQLite file for local development (it's a file, needs
no separate server process, and is trivially backed up by copying app.db).
Set DATABASE_URL (e.g. to a Neon/Postgres connection string) in production
hosts like Render, whose free web services have an ephemeral filesystem -
a local SQLite file there would be wiped on every restart/spin-down.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

_DB_PATH = Path(__file__).parent / "data" / "app.db"

_database_url = os.getenv("DATABASE_URL")
if _database_url:
    # Some hosts (Render, Heroku-style) hand out "postgres://", but
    # SQLAlchemy's psycopg2 dialect requires "postgresql://".
    if _database_url.startswith("postgres://"):
        _database_url = _database_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(_database_url)
else:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(32), unique=True, nullable=False, index=True)
    password_hash = Column(String(60), nullable=False)  # bcrypt hash is always 60 chars
    theme_preference = Column(String(10), nullable=True)  # 'light' | 'dark' | 'system' | None
    # Paper-trading virtual cash - never real money, added via a dummy
    # "deposit" endpoint with no payment processor behind it. Starts at 0
    # until the user adds funds, so an empty portfolio isn't mistaken for
    # a broken one.
    paper_cash_balance = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    watchlist_items = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")
    portfolio_holdings = relationship(
        "PortfolioHolding", back_populates="user", cascade="all, delete-orphan"
    )
    portfolio_transactions = relationship(
        "PortfolioTransaction", back_populates="user", cascade="all, delete-orphan"
    )


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    __table_args__ = (UniqueConstraint("user_id", "symbol", name="uq_user_symbol"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    added_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="watchlist_items")


class PortfolioHolding(Base):
    """A user's current paper-trading position in one symbol. avg_buy_price
    is a running weighted-average cost basis, updated on every BUY and left
    unchanged on SELL (standard average-cost accounting) - it's what P&L is
    measured against, not the live price history."""

    __tablename__ = "portfolio_holdings"
    __table_args__ = (UniqueConstraint("user_id", "symbol", name="uq_portfolio_user_symbol"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Integer, nullable=False, default=0)
    avg_buy_price = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="portfolio_holdings")


class PortfolioTransaction(Base):
    """Append-only ledger of every paper-trading action, so a user can
    review what they did and when against how the forecast/signal called
    it at the time."""

    __tablename__ = "portfolio_transactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(String(10), nullable=False)  # 'DEPOSIT' | 'BUY' | 'SELL'
    symbol = Column(String(20), nullable=True)  # null for DEPOSIT
    quantity = Column(Integer, nullable=True)  # null for DEPOSIT
    price_per_share = Column(Float, nullable=True)  # null for DEPOSIT
    amount = Column(Float, nullable=False)  # cash delta: +deposit, -buy cost, +sell proceeds
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="portfolio_transactions")


def _add_missing_columns() -> None:
    """create_all() only creates missing TABLES, not missing COLUMNS on
    tables that already exist - which matters here because the production
    database already had a `users` table before paper_cash_balance was
    added. No Alembic for a two-table personal project; this covers the
    one case that needs it without risking the real data a full
    drop/recreate would."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    existing_columns = {c["name"] for c in inspector.get_columns("users")}
    if "paper_cash_balance" not in existing_columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN paper_cash_balance FLOAT DEFAULT 0.0"))


def init_db() -> None:
    Base.metadata.create_all(engine)
    _add_missing_columns()


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
