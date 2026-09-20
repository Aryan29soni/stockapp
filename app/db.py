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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    watchlist_items = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    __table_args__ = (UniqueConstraint("user_id", "symbol", name="uq_user_symbol"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    added_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="watchlist_items")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
