"""'Stocks in the news' feed: recent per-stock headlines from Yahoo Finance,
tagged with a sentiment score and combined with the same rule-based
technical signal used elsewhere in the app to produce one overall tilt -
e.g. "BUY" only when both the recent news tone AND the technical indicators
agree, "HOLD" when they disagree.

Sentiment blends two signals:
1. A finance-tuned weighted keyword lexicon (with negation handling) - this
   alone already beats a generic sentiment model on finance headlines, since
   general models don't know that e.g. "downgrade" or "sell-off" are
   decisively negative in this domain.
2. VADER (Hutto & Gilbert 2014), a general-purpose, lexicon-and-rule-based
   sentiment intensity analyzer with real negation/degree-modifier/contrast
   handling - used as a fallback when the finance lexicon finds no keyword
   match at all (rather than silently calling every such headline neutral),
   and as a disagreement check that dampens the keyword score when the two
   signals point in opposite directions (a sign the headline is ambiguous).

This is still not full natural-language understanding of financial meaning
- no model here reads for context like "beats a lowered estimate" - but VADER
is a real, independently-validated NLU sentiment model, not a hand-rolled
keyword count. The API response always carries a disclaimer.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import yfinance as yf

from . import sentiment as sentiment_module
from . import signals as signals_module
from . import stocks as stocks_module
from .cache import TTLCache
from .data import _yf_symbol
from .timeouts import call_with_timeout

DISCLAIMER = (
    "Sentiment blends a finance-tuned keyword lexicon (with negation "
    "handling) with VADER, a general-purpose NLU sentiment model, "
    "aggregated across a stock's recent articles. The combined tilt "
    "cross-checks that sentiment against the same rule-based technical "
    "signal used elsewhere in the app; it is NOT investment advice or a "
    "real recommendation. Always read the article and do your own research. "
    "This feed refreshes automatically roughly every 10 minutes."
)

_NEWS_CACHE: TTLCache[tuple[list[dict], str]] = TTLCache(ttl_seconds=60 * 10, max_entries=10)
# short TTL so it feels live; yfinance's own feed doesn't update much faster
# than this anyway. Only ever 2 real keys (NSE/BSE) so size isn't a growth
# risk here, but using the same shared cache (see cache.py) for consistency.

_NEWS_LOOKBACK_HOURS = 120
_MAX_SYMBOLS_SCANNED = 150
_MAX_ARTICLES_PER_SYMBOL = 3


def _news_lean(score: int) -> str:
    if score >= 2:
        return "BUY"
    if score <= -2:
        return "SELL"
    return "HOLD"


def _combine(news_lean: str, technical_signal: str | None) -> tuple[str, str, str]:
    """Cross-checks the news lean against the technical signal.
    Returns (combinedTilt, confidence, rationale)."""
    if technical_signal is None:
        return news_lean, "LOW", "Only recent-news sentiment is available (no technical signal)."
    if news_lean == technical_signal and news_lean != "HOLD":
        return news_lean, "HIGH", f"Recent news tone and the technical signal both point to {news_lean}."
    if news_lean == "HOLD" and technical_signal != "HOLD":
        return technical_signal, "MODERATE", f"News tone is neutral; technicals lean {technical_signal}."
    if technical_signal == "HOLD" and news_lean != "HOLD":
        return news_lean, "MODERATE", f"Technicals are neutral; recent news tone leans {news_lean}."
    if news_lean != technical_signal and "HOLD" not in (news_lean, technical_signal):
        return "HOLD", "LOW", f"News tone ({news_lean}) and technicals ({technical_signal}) disagree - mixed signal."
    return "HOLD", "LOW", "No strong signal from either news tone or technicals."


def _fetch_news_list(ticker: "yf.Ticker") -> list[dict]:
    return ticker.news or []


def _fetch_one(symbol: str, exchange: str) -> tuple[str, list[dict]]:
    try:
        raw = call_with_timeout(_fetch_news_list, yf.Ticker(_yf_symbol(symbol, exchange)), timeout=10)
    except Exception:
        return symbol, []
    return symbol, raw


def get_news_feed(limit: int = 20, exchange: str = "NSE") -> dict:
    cache_key = f"NEWS:{exchange}"
    cached = _NEWS_CACHE.get(cache_key)
    if cached is not None:
        items, generated_at = cached
    else:
        symbols = stocks_module.POPULAR_SYMBOLS[:_MAX_SYMBOLS_SCANNED]
        with ThreadPoolExecutor(max_workers=25) as pool:
            fetched = list(pool.map(lambda s: _fetch_one(s, exchange), symbols))

        cutoff = datetime.now(timezone.utc) - timedelta(hours=_NEWS_LOOKBACK_HOURS)
        per_symbol_articles: dict[str, list[dict]] = {}

        for symbol, articles in fetched:
            parsed = []
            for article in articles:
                content = article.get("content", article)
                pub_date_str = content.get("pubDate") or content.get("displayTime")
                if not pub_date_str:
                    continue
                try:
                    pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if pub_date < cutoff:
                    continue
                title = content.get("title", "")
                summary = content.get("summary", "") or content.get("description", "")
                provider = content.get("provider", {}) or {}
                canonical = content.get("canonicalUrl", {}) or {}
                parsed.append(
                    {
                        "title": title,
                        "summary": summary,
                        "source": provider.get("displayName", "Yahoo Finance"),
                        "url": canonical.get("url") or content.get("previewUrl"),
                        "pubDate": pub_date,
                        "score": sentiment_module.score_text(f"{title} {summary}"),
                    }
                )
            if parsed:
                parsed.sort(key=lambda a: a["pubDate"], reverse=True)
                per_symbol_articles[symbol] = parsed[:_MAX_ARTICLES_PER_SYMBOL]

        # Cross-check against the same rule-based technical signal used
        # elsewhere in the app, computed in one batched call.
        technical_signals = signals_module.get_bulk_signals(list(per_symbol_articles.keys()), exchange)

        items = []
        for symbol, articles in per_symbol_articles.items():
            meta = stocks_module.get_stock_meta(symbol)
            latest = articles[0]
            aggregate_score = sum(a["score"] for a in articles)
            news_lean = _news_lean(aggregate_score)
            technical_entry = technical_signals.get(symbol) or {}
            technical_signal = technical_entry.get("signal")
            combined_tilt, confidence, rationale = _combine(news_lean, technical_signal)

            summary = latest["summary"]
            items.append(
                {
                    "symbol": symbol,
                    "name": meta["name"],
                    "headline": latest["title"],
                    "summary": (summary[:220] + "…") if len(summary) > 220 else summary,
                    "source": latest["source"],
                    "url": latest["url"],
                    "publishedAt": latest["pubDate"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "articleCount": len(articles),
                    "sentiment": sentiment_module.sentiment_label(aggregate_score),
                    "sentimentScore": aggregate_score,
                    "newsLean": news_lean,
                    "technicalSignal": technical_signal,
                    # The specific indicator readings behind the technical
                    # call (e.g. "RSI indicates oversold conditions") - same
                    # list the single-stock page already shows, so the "why"
                    # here is a real reason, not just a restated label.
                    "technicalReasons": technical_entry.get("reasons", []),
                    # Per-article tone, so "why" for the news lean can point
                    # at which specific headlines drove it rather than just
                    # restating the aggregate score.
                    "newsBreakdown": [
                        {
                            "headline": a["title"],
                            "source": a["source"],
                            "sentiment": sentiment_module.sentiment_label(a["score"]),
                        }
                        for a in articles
                    ],
                    "combinedTilt": combined_tilt,
                    "confidence": confidence,
                    "rationale": rationale,
                    "_pubDateSort": latest["pubDate"],
                }
            )

        items.sort(key=lambda it: it["_pubDateSort"], reverse=True)
        for it in items:
            del it["_pubDateSort"]
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _NEWS_CACHE.set(cache_key, (items, generated_at))

    return {"results": items[:limit], "disclaimer": DISCLAIMER, "generatedAt": generated_at}
