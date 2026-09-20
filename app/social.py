""""What's being talked about today" - aggregates real stock-symbol mentions
from Indian finance discussion on Reddit, YouTube, Discord and Telegram,
scores each mention with the same sentiment engine used for the News tab
(see sentiment.py), and ranks symbols by how much they're being discussed
today.

All four sources are either official APIs or content Telegram itself serves
publicly - no scraping behind a login wall and no bot-detection bypass.
(StockTwits and X/Twitter were evaluated and rejected: StockTwits' public
endpoints now sit behind a Cloudflare bot challenge, and X removed its free
tier in Feb 2026, moving to pay-per-request pricing.)

- Reddit: OAuth2 client-credentials ("app-only") grant, read-only access to
  public posts from a handful of major Indian stock-market subreddits - IF
  you have credentials. As of Reddit's Nov 2025 Responsible Builder Policy,
  reddit.com/prefs/apps no longer issues new credentials to personal/hobby
  projects at all (confirmed: it redirects to the policy page instead of the
  old "create app" form, and Reddit's own longtime bot developers report
  personal scripts don't qualify for the manual-approval exception). This
  only does anything if REDDIT_CLIENT_ID/SECRET are from before that date.
  Its terms also prohibit commercial use/redistribution regardless.
- YouTube: official Data API v3 (free, 10k quota units/day - comment reads
  cost ~1 unit/call), pulling recent comments from major Indian finance
  news/education channels. YouTube's API terms don't carry Reddit's blanket
  commercial-use ban.
- Discord: official bot API (free, no approval needed under Discord's
  10,000-user privileged-intent threshold), reading recent messages from
  whichever server text channels the bot has actually been added to. Unlike
  the other three sources, this can't ship with a working default - there's
  no way to add a bot to a community server we don't own without that
  server's admin inviting it, so DISCORD_CHANNEL_IDS is a user-supplied list
  (see .env.example) and this source does nothing until you add at least one.
- Telegram: the public web preview every Telegram channel gets at
  t.me/s/{channel} (built for link-unfurling, so it's plain server-rendered
  HTML with no login, bot, or phone number required). This is fetching a
  page Telegram itself serves to anyone, but it's an unofficial, undocumented
  use of that page (not a licensed API), so it's more fragile than the
  others - it could change or get rate-limited without notice. Only pulls
  from verified official channels of registered news outlets (CNBC-TV18,
  Moneycontrol), not anonymous "tip" channels - many of those are unregistered
  advisory operations that are a regulatory minefield in Indian markets, and
  deliberately not something this app sources from. Note this means Telegram
  content here is published market news/commentary, not crowd chatter like
  the other three sources - closer in spirit to the News tab, but still
  useful as an early-signal feed since these channels post faster than a full
  article.

All four fail soft: without the relevant credentials/config in the
environment, that source is silently skipped (not an error), so the endpoint
works with whichever sources are configured - or returns an empty,
clearly-labeled result if none are.

Mention detection is regex/lexicon-based, not a trained NER model: it always
matches cashtag-style "$SYMBOL"/"#SYMBOL", and separately matches a bare
all-caps SYMBOL as a whole word, skipping a short stoplist of symbols that
collide with common English words (ACE, IDEA, RAIN, SAIL, STAR) to cut false
positives. This is a precision/recall tradeoff, not verified NLU - treat
mention counts as a rough "how much is this being talked about" signal.
"""

import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

from . import sentiment as sentiment_module
from . import stocks as stocks_module

DISCLAIMER = (
    "Mention counts and sentiment are pulled from whichever of Reddit, "
    "YouTube, Discord and Telegram are configured, scored with the same "
    "keyword+VADER sentiment engine used on the News tab. Symbol matching is "
    "heuristic, not verified - a mention means the ticker text showed up in "
    "a post, comment or message, not that it was necessarily about that "
    "stock. Telegram content here is published news/commentary from "
    "registered outlets, not crowd chatter like the other sources. This is "
    "retail chatter and market commentary, not analysis; it is NOT "
    "investment advice."
)

_REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
_REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
_REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "indian-stock-app/0.1 (personal use)")

_YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

_DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
_DISCORD_CHANNEL_IDS = [c.strip() for c in os.environ.get("DISCORD_CHANNEL_IDS", "").split(",") if c.strip()]

# Verified official Telegram channels of registered news outlets only - see
# the module docstring for why we don't pull from "tip" channels.
_TELEGRAM_CHANNELS = ["cnbc_tv18", "moneycontrolcom"]

_SUBREDDITS = ["IndianStreetBets", "IndiaInvestments", "StockMarketIndia", "IndianStockMarket"]

# A mix of channel @handles (resolved via YouTube's forHandle lookup) and one
# raw channel ID (ET Now's handle lookup was unreliable) covering major
# Indian business-news and finance-education channels.
_YOUTUBE_HANDLES = ["CNBC-TV18", "moneycontrol", "PranjalKamra", "AssetYogi", "CARachanaRanade"]
_YOUTUBE_EXTRA_CHANNEL_IDS = {"ETNow": "UCI_mwTKUhicNzFrhm33MzBQ"}

_VIDEOS_PER_CHANNEL = 3
_COMMENTS_PER_VIDEO = 50

_CACHE_TTL_SECONDS = 60 * 30  # discussion volume moves slowly enough that 30min is fine
_cache: dict[str, tuple[float, dict]] = {}

_MENTION_STOPLIST = {"ACE", "IDEA", "RAIN", "SAIL", "STAR"}

_reddit_token: tuple[float, str] | None = None  # (expiresAt, token)


def _build_symbol_index() -> dict[str, str]:
    """token (as it would appear in text) -> canonical symbol."""
    index: dict[str, str] = {}
    for s in stocks_module.ALL_STOCKS:
        symbol = s["symbol"]
        if len(symbol) >= 3 and symbol not in _MENTION_STOPLIST and symbol.isalpha():
            index[symbol] = symbol
    return index


_SYMBOL_INDEX = _build_symbol_index()
_CASHTAG_RE = re.compile(r"[$#]([A-Za-z]{2,15})")
_WORD_RE = re.compile(r"\b[A-Z]{3,15}\b")


def _extract_mentions(text: str) -> set[str]:
    if not text:
        return set()
    found: set[str] = set()
    for m in _CASHTAG_RE.finditer(text):
        sym = m.group(1).upper()
        if sym in _SYMBOL_INDEX:
            found.add(sym)
    for m in _WORD_RE.finditer(text):
        sym = m.group(0)
        if sym in _SYMBOL_INDEX:
            found.add(sym)
    return found


def _reddit_access_token() -> str | None:
    global _reddit_token
    if not (_REDDIT_CLIENT_ID and _REDDIT_CLIENT_SECRET):
        return None
    now = time.time()
    if _reddit_token and now < _reddit_token[0]:
        return _reddit_token[1]
    try:
        resp = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(_REDDIT_CLIENT_ID, _REDDIT_CLIENT_SECRET),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": _REDDIT_USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        token = payload["access_token"]
        _reddit_token = (now + payload.get("expires_in", 3600) - 60, token)
        return token
    except Exception:
        return None


def _fetch_reddit_items() -> list[dict]:
    """Returns [{text, url, source: 'reddit'}] from today's posts (title +
    selftext) across the configured subreddits."""
    token = _reddit_access_token()
    if not token:
        return []
    headers = {"Authorization": f"bearer {token}", "User-Agent": _REDDIT_USER_AGENT}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    def fetch_one(subreddit: str) -> list[dict]:
        try:
            resp = requests.get(
                f"https://oauth.reddit.com/r/{subreddit}/new",
                headers=headers,
                params={"limit": 50},
                timeout=10,
            )
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
        except Exception:
            return []
        items = []
        for child in children:
            post = child.get("data", {})
            created = datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc)
            if created < cutoff:
                continue
            text = f"{post.get('title', '')} {post.get('selftext', '')}"
            permalink = post.get("permalink")
            items.append(
                {
                    "text": text,
                    "url": f"https://reddit.com{permalink}" if permalink else None,
                    "title": post.get("title", ""),
                    "source": "reddit",
                    "sourceLabel": f"r/{subreddit}",
                }
            )
        return items

    with ThreadPoolExecutor(max_workers=len(_SUBREDDITS)) as pool:
        results = list(pool.map(fetch_one, _SUBREDDITS))
    return [item for batch in results for item in batch]


def _youtube_get(path: str, params: dict) -> dict | None:
    try:
        resp = requests.get(
            f"https://www.googleapis.com/youtube/v3/{path}",
            params={**params, "key": _YOUTUBE_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _resolve_channel_id(handle: str) -> str | None:
    data = _youtube_get("channels", {"part": "id", "forHandle": handle})
    items = (data or {}).get("items") or []
    return items[0]["id"] if items else None


def _recent_video_ids(channel_id: str) -> list[str]:
    data = _youtube_get(
        "search",
        {
            "part": "id",
            "channelId": channel_id,
            "order": "date",
            "type": "video",
            "maxResults": _VIDEOS_PER_CHANNEL,
        },
    )
    items = (data or {}).get("items") or []
    return [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]


def _fetch_video_comments(video_id: str) -> list[dict]:
    data = _youtube_get(
        "commentThreads",
        {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": _COMMENTS_PER_VIDEO,
            "order": "relevance",
            "textFormat": "plainText",
        },
    )
    items = (data or {}).get("items") or []
    out = []
    for it in items:
        snippet = it.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
        text = snippet.get("textDisplay", "")
        if not text:
            continue
        out.append(
            {
                "text": text,
                "url": f"https://www.youtube.com/watch?v={video_id}&lc={it.get('id', '')}",
                "title": text[:80],
                "source": "youtube",
                "sourceLabel": "YouTube comment",
            }
        )
    return out


def _fetch_youtube_items() -> list[dict]:
    if not _YOUTUBE_API_KEY:
        return []

    channel_ids = dict(_YOUTUBE_EXTRA_CHANNEL_IDS)
    with ThreadPoolExecutor(max_workers=len(_YOUTUBE_HANDLES)) as pool:
        resolved = list(pool.map(_resolve_channel_id, _YOUTUBE_HANDLES))
    for handle, cid in zip(_YOUTUBE_HANDLES, resolved):
        if cid:
            channel_ids[handle] = cid

    with ThreadPoolExecutor(max_workers=max(1, len(channel_ids))) as pool:
        video_id_batches = list(pool.map(_recent_video_ids, channel_ids.values()))
    video_ids = [vid for batch in video_id_batches for vid in batch]

    with ThreadPoolExecutor(max_workers=8) as pool:
        comment_batches = list(pool.map(_fetch_video_comments, video_ids))
    return [item for batch in comment_batches for item in batch]


_discord_guild_ids: dict[str, str] = {}  # channel_id -> guild_id, resolved once and cached


def _discord_headers() -> dict:
    return {"Authorization": f"Bot {_DISCORD_BOT_TOKEN}"}


def _discord_guild_id(channel_id: str) -> str | None:
    if channel_id in _discord_guild_ids:
        return _discord_guild_ids[channel_id]
    try:
        resp = requests.get(
            f"https://discord.com/api/v10/channels/{channel_id}", headers=_discord_headers(), timeout=10
        )
        resp.raise_for_status()
        guild_id = resp.json().get("guild_id")
    except Exception:
        return None
    if guild_id:
        _discord_guild_ids[channel_id] = guild_id
    return guild_id


def _fetch_discord_channel(channel_id: str) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    try:
        resp = requests.get(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers=_discord_headers(),
            params={"limit": 100},
            timeout=10,
        )
        resp.raise_for_status()
        messages = resp.json()
    except Exception:
        return []

    guild_id = _discord_guild_id(channel_id)
    items = []
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        try:
            created = datetime.fromisoformat(msg["timestamp"])
        except (KeyError, ValueError):
            continue
        if created < cutoff:
            continue
        jump_url = (
            f"https://discord.com/channels/{guild_id}/{channel_id}/{msg['id']}"
            if guild_id
            else None
        )
        items.append(
            {
                "text": content,
                "url": jump_url,
                "title": content[:80],
                "source": "discord",
                "sourceLabel": "Discord",
            }
        )
    return items


def _fetch_discord_items() -> list[dict]:
    if not (_DISCORD_BOT_TOKEN and _DISCORD_CHANNEL_IDS):
        return []
    with ThreadPoolExecutor(max_workers=len(_DISCORD_CHANNEL_IDS)) as pool:
        batches = list(pool.map(_fetch_discord_channel, _DISCORD_CHANNEL_IDS))
    return [item for batch in batches for item in batch]


def _fetch_telegram_channel(channel: str) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    try:
        resp = requests.get(f"https://t.me/s/{channel}", timeout=10)
        resp.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    items = []
    for wrapper in soup.select("div.tgme_widget_message"):
        post_id = wrapper.get("data-post")
        time_el = wrapper.select_one("time[datetime]")
        text_el = wrapper.select_one(".tgme_widget_message_text")
        if not (post_id and time_el and text_el):
            continue
        try:
            created = datetime.fromisoformat(time_el["datetime"])
        except ValueError:
            continue
        if created < cutoff:
            continue
        text = text_el.get_text(separator=" ").strip()
        if not text:
            continue
        items.append(
            {
                "text": text,
                "url": f"https://t.me/{post_id}",
                "title": text[:80],
                "source": "telegram",
                "sourceLabel": f"Telegram: {channel}",
            }
        )
    return items


def _fetch_telegram_items() -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(_TELEGRAM_CHANNELS)) as pool:
        batches = list(pool.map(_fetch_telegram_channel, _TELEGRAM_CHANNELS))
    return [item for batch in batches for item in batch]


def get_trending(limit: int = 20) -> dict:
    cache_key = "trending"
    now = time.time()
    cached = _cache.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    sources_configured = []
    all_items: list[dict] = []
    if _REDDIT_CLIENT_ID and _REDDIT_CLIENT_SECRET:
        sources_configured.append("reddit")
        all_items.extend(_fetch_reddit_items())
    if _YOUTUBE_API_KEY:
        sources_configured.append("youtube")
        all_items.extend(_fetch_youtube_items())
    if _DISCORD_BOT_TOKEN and _DISCORD_CHANNEL_IDS:
        sources_configured.append("discord")
        all_items.extend(_fetch_discord_items())
    # No credentials needed - always attempted.
    sources_configured.append("telegram")
    all_items.extend(_fetch_telegram_items())

    per_symbol: dict[str, dict] = defaultdict(lambda: {"mentionCount": 0, "scoreSum": 0, "samples": []})
    for item in all_items:
        mentions = _extract_mentions(item["text"])
        if not mentions:
            continue
        score = sentiment_module.score_text(item["text"])
        for symbol in mentions:
            bucket = per_symbol[symbol]
            bucket["mentionCount"] += 1
            bucket["scoreSum"] += score
            if len(bucket["samples"]) < 3 and item.get("url"):
                bucket["samples"].append(
                    {"title": item["title"], "url": item["url"], "source": item["source"], "sourceLabel": item["sourceLabel"]}
                )

    results = []
    for symbol, bucket in per_symbol.items():
        meta = stocks_module.get_stock_meta(symbol)
        avg_score = bucket["scoreSum"] / bucket["mentionCount"]
        results.append(
            {
                "symbol": symbol,
                "name": meta["name"],
                "sector": meta.get("sector"),
                "mentionCount": bucket["mentionCount"],
                "sentiment": sentiment_module.sentiment_label(round(avg_score)),
                "sentimentScore": round(avg_score, 2),
                "links": bucket["samples"],
            }
        )
    results.sort(key=lambda r: r["mentionCount"], reverse=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "results": results[:limit],
        "sourcesConfigured": sources_configured,
        "disclaimer": DISCLAIMER,
        "generatedAt": generated_at,
    }
    _cache[cache_key] = (now, payload)
    return payload
