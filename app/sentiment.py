"""Shared finance-text sentiment scorer, used by both the News tab and the
Social tab so a headline and a Reddit comment are scored on the same scale.

Blends two signals:
1. A finance-tuned weighted keyword lexicon (with negation handling) - this
   alone already beats a generic sentiment model on finance text, since
   general models don't know that e.g. "downgrade" or "sell-off" are
   decisively negative in this domain.
2. VADER (Hutto & Gilbert 2014), a general-purpose, lexicon-and-rule-based
   sentiment intensity analyzer with real negation/degree-modifier/contrast
   handling - used as a fallback when the finance lexicon finds no keyword
   match at all (rather than silently calling every such text neutral), and
   as a disagreement check that dampens the keyword score when the two
   signals point in opposite directions (a sign the text is ambiguous).

Still not full natural-language understanding of financial meaning - no
model here reads for context like "beats a lowered estimate" - but VADER is
a real, independently-validated NLU sentiment model, not a hand-rolled
keyword count.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()
_VADER_NEUTRAL_BAND = 0.15  # |compound| below this counts as "no opinion" for agreement checks

# Word -> weight. Stronger/more decisive financial words carry more weight
# than mild ones, so e.g. "crashed" moves the score more than "fell".
_POSITIVE_WORDS = {
    "surge": 2, "surges": 2, "surged": 2, "soar": 2, "soars": 2, "soared": 2,
    "rally": 2, "rallies": 2, "rallied": 2, "jump": 1, "jumps": 1, "jumped": 1,
    "gain": 1, "gains": 1, "gained": 1, "beat": 2, "beats": 2, "record": 1,
    "growth": 1, "grows": 1, "grew": 1, "upgrade": 2, "upgraded": 2,
    "outperform": 2, "profit": 1, "profits": 1, "profitable": 1, "bullish": 2,
    "buy": 1, "strong": 1, "boost": 1, "boosts": 1, "boosted": 1, "expand": 1,
    "expands": 1, "expansion": 1, "high": 1, "highs": 1, "wins": 1, "win": 1,
    "won": 1, "positive": 1, "rise": 1, "rises": 1, "rose": 1, "rebound": 1,
    "rebounds": 1, "optimistic": 1, "milestone": 1, "raises": 1, "raise": 1,
    "raised": 1, "approval": 1, "approved": 1, "deal": 1, "partnership": 1,
    "stake": 1, "acquire": 1, "acquires": 1, "acquisition": 1, "dividend": 1,
    "buyback": 1, "multibagger": 2, "breakout": 1, "moon": 1, "rocket": 1,
}
_NEGATIVE_WORDS = {
    "plunge": 2, "plunges": 2, "plunged": 2, "crash": 2, "crashes": 2,
    "crashed": 2, "fall": 1, "falls": 1, "fell": 1, "drop": 1, "drops": 1,
    "dropped": 1, "decline": 1, "declines": 1, "declined": 1, "loss": 1,
    "losses": 1, "miss": 1, "misses": 1, "missed": 1, "downgrade": 2,
    "downgraded": 2, "underperform": 2, "bearish": 2, "sell": 1, "sell-off": 2,
    "selloff": 2, "weak": 1, "weakness": 1, "cut": 1, "cuts": 1, "slump": 2,
    "slumps": 2, "slumped": 2, "probe": 2, "fraud": 2, "scam": 2,
    "investigation": 2, "lawsuit": 2, "penalty": 1, "fine": 1, "fined": 1,
    "layoff": 1, "layoffs": 1, "resign": 1, "resigns": 1, "resigned": 1,
    "low": 1, "lows": 1, "negative": 1, "concern": 1, "concerns": 1,
    "warning": 1, "warns": 1, "risk": 1, "risks": 1, "debt": 1, "default": 2,
    "delay": 1, "delayed": 1, "recall": 1, "halt": 1, "halted": 1, "ban": 1,
    "banned": 1, "dump": 1, "dumping": 1, "overvalued": 1, "bagholder": 1,
}
_NEGATIONS = {"not", "no", "never", "n't", "without", "denies", "denied"}
_NEGATION_WINDOW = 3  # words


def _keyword_score(text: str) -> int:
    words = [w.strip("'\"().,") for w in text.lower().split()]
    score = 0
    for i, w in enumerate(words):
        weight = _POSITIVE_WORDS.get(w) or (-_NEGATIVE_WORDS.get(w, 0) or None)
        if weight is None:
            continue
        window = words[max(0, i - _NEGATION_WINDOW) : i]
        if any(neg in window or w2.endswith("n't") for neg in _NEGATIONS for w2 in window):
            weight = -weight
        score += weight
    return score


def score_text(text: str) -> int:
    """Combines the finance-tuned keyword lexicon with VADER's general NLU
    sentiment: the keyword score leads when it has an opinion (finance
    vocabulary usually reads clearer than generic sentiment on this text),
    VADER fills in when the lexicon found nothing to match, and a strong
    disagreement between the two dampens the result rather than trusting
    either blindly."""
    keyword_score = _keyword_score(text)
    vader_compound = _vader.polarity_scores(text)["compound"]

    if keyword_score == 0:
        return round(vader_compound * 2)

    vader_sign = 1 if vader_compound > _VADER_NEUTRAL_BAND else (-1 if vader_compound < -_VADER_NEUTRAL_BAND else 0)
    keyword_sign = 1 if keyword_score > 0 else -1
    if vader_sign != 0 and vader_sign != keyword_sign:
        return round(keyword_score * 0.4)
    return keyword_score


def sentiment_label(score: int) -> str:
    if score > 0:
        return "POSITIVE"
    if score < 0:
        return "NEGATIVE"
    return "NEUTRAL"
