"""Shared TTL cache for the module-level caches scattered across this
backend (data.py, predict.py, signals.py, news.py, fundamentals.py,
macro.py, social.py).

Every one of them used to be a plain dict that checked a timestamp on
lookup but never removed an entry once it went stale - the TTL only
gated whether a cached value was still considered *fresh*, not whether it
still lived in memory. With 1,400+ NSE symbols, several history periods,
and four forecast horizons all being distinct cache keys, that grows
without bound over the process's uptime. On Render's free tier (512MB),
that unbounded growth was the actual cause of the backend being OOM-killed
- not request volume or model training, which don't retain memory between
requests. This caps each cache's size and evicts the oldest entry (by
insertion order) once it's full, same trade-off these caches already made
by using a TTL in the first place: bounded memory over permanently-fresh
data.
"""

import time
from typing import Generic, TypeVar

T = TypeVar("T")

_MISSING = object()  # sentinel distinct from a legitimately-cached None
# (e.g. fundamentals.py caches "this symbol has no earnings data" as None -
# get() returning None for that would be indistinguishable from "not
# cached", defeating the cache for every such symbol on every call).


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: float, max_entries: int):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, tuple[float, T]] = {}

    def get(self, key: str, default=None):
        entry = self._store.get(key)
        if entry is None:
            return default
        cached_at, value = entry
        if time.time() - cached_at >= self.ttl_seconds:
            return default
        return value

    def has(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def set(self, key: str, value: T) -> None:
        if key not in self._store and len(self._store) >= self.max_entries:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[key] = (time.time(), value)
