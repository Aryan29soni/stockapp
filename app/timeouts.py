"""Hard wall-clock timeout for yfinance/Yahoo Finance calls.

yfinance's own HTTP client defaults to a 30s timeout per individual HTTP
request, but one logical call (e.g. .history() or .news) can issue several
requests in sequence (cookie/crumb negotiation, retries, the actual data
fetch), so a single call can still take minutes if Yahoo is slow or
rate-limiting this server's IP - and a plain try/except does nothing against
a call that's merely slow, not failing. This is what actually took the
backend down: worker threads stuck for minutes on a slow Yahoo response,
queuing up every other request (including plain health checks) behind them
until the whole instance became unresponsive - a much bigger problem than
the ML-training concurrency this module's sibling fix (predict.py's
_TRAINING_LOCK) addresses.

Running the call on a small dedicated thread pool and giving up on IT well
before that pileup can happen bounds the damage: the original call may keep
running in the background until it finally returns or errors, but it no
longer blocks the request that was waiting on it.
"""

import concurrent.futures
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_TIMEOUT_SECONDS = 12

# A separate pool from FastAPI's own request threadpool, sized generously -
# a stuck call parks a thread here indefinitely (Python can't forcibly kill
# a running thread), so this needs enough headroom that a run of slow Yahoo
# responses doesn't also starve this pool itself.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix="yf-timeout")


class DataProviderTimeout(Exception):
    """Raised when a yfinance call exceeds its allotted time. Deliberately
    its own type (not TimeoutError) so it's unambiguous at every call site -
    existing `except Exception` blocks in the best-effort fetchers (news,
    signals, fundamentals, macro) still catch it and degrade gracefully,
    while call sites that don't wrap it (data.py's own functions) let it
    propagate up to main.py's dedicated exception handler, which turns it
    into a clean 503 instead of an unbounded hang."""


def call_with_timeout(fn: Callable[..., T], *args, timeout: float = DEFAULT_TIMEOUT_SECONDS, **kwargs) -> T:
    future = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise DataProviderTimeout(
            f"{getattr(fn, '__name__', fn)} timed out after {timeout}s"
        ) from None
