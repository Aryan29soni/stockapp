"""NSE-listed stock universe for search/browse.

Backed by a bundled JSON file (~1400 symbols, merged from the full NSE
equity ticker list and Nifty 500 sector/industry data) so search covers
far more than the handful of large-caps shown by default.
"""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "nse_stocks.json"

with open(_DATA_PATH, encoding="utf-8") as _f:
    ALL_STOCKS: list[dict] = json.load(_f)

_BY_SYMBOL = {s["symbol"]: s for s in ALL_STOCKS}

# The Nifty 100 + Nifty 200 constituents, topped up with the next ~100 Nifty
# 500 names - 300 NSE stocks by free-float market cap, shown by default
# before the user searches. Source: niftyindices.com official constituent
# lists, cross-checked against the bundled dataset / yfinance as of Sep 2026.
POPULAR_SYMBOLS = [
    "ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "AMBUJACEM", "APOLLOHOSP",
    "ASIANPAINT", "DMART", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BANKBARODA",
    "BEL", "BPCL", "BHARTIARTL", "BOSCHLTD", "BRITANNIA", "CGPOWER", "CANBK", "CHOLAFIN",
    "CIPLA", "COALINDIA", "CUMMINSIND", "DLF", "DIVISLAB", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GAIL", "GODREJCP", "GRASIM", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HAL", "HINDUNILVR", "HINDZINC", "HYUNDAI", "ICICIBANK", "ITC", "INDHOTEL", "IOC",
    "IRFC", "INFY", "INDIGO", "JSWSTEEL", "JINDALSTEL", "JIOFIN", "KOTAKBANK", "LTM",
    "LT", "LODHA", "M&M", "MARUTI", "MAXHEALTH", "MAZDOCK", "MUTHOOTFIN", "NTPC",
    "NESTLEIND", "ONGC", "PIDILITIND", "PFC", "POWERGRID", "PNB", "RECLTD", "RELIANCE",
    "SBILIFE", "MOTHERSON", "SHREECEM", "SHRIRAMFIN", "ENRIN", "SIEMENS", "SOLARINDS", "SBIN",
    "SUNPHARMA", "TVSMOTOR", "TATACAP", "TCS", "TATACONSUM", "TMCV", "TMPV", "TATAPOWER",
    "TATASTEEL", "TECHM", "TITAN", "TORNTPHARM", "TRENT", "ULTRACEMCO", "UNIONBANK", "UNITDSPR",
    "VBL", "VEDL", "WIPRO", "ZYDUSLIFE", "360ONE", "APLAPOLLO", "AUBANK", "ATGL",
    "ABCAPITAL", "ALKEM", "ASHOKLEY", "ASTRAL", "AUROPHARMA", "BSE", "BANKINDIA", "BDL",
    "BHARATFORG", "BHEL", "GROWW", "BIOCON", "BLUESTARCO", "COCHINSHIP", "COFORGE", "COLPAL",
    "CONCOR", "COROMANDEL", "DABUR", "DIXON", "EXIDEIND", "NYKAA", "FEDERALBNK", "FORTIS",
    "GVT&D", "GMRAIRPORT", "GLENMARK", "GODFRYPHLP", "GODREJPROP", "HAVELLS", "HEROMOTOCO", "HINDPETRO",
    "POWERINDIA", "HUDCO", "ICICIGI", "ICICIAMC", "IDFCFIRSTB", "INDIANB", "IRCTC", "IREDA",
    "INDUSTOWER", "INDUSINDBK", "NAUKRI", "JSWENERGY", "JUBLFOOD", "KEI", "KPITTECH", "KALYANKJIL",
    "LTF", "LGEINDIA", "LICHSGFIN", "LAURUSLABS", "LENSKART", "LUPIN", "MRF", "M&MFIN",
    "MANKIND", "MARICO", "MFSL", "MOTILALOFS", "MPHASIS", "MCX", "NHPC", "NMDC",
    "NATIONALUM", "OBEROIRLTY", "OIL", "PAYTM", "OFSS", "POLICYBZR", "PIIND", "PAGEIND",
    "PATANJALI", "PERSISTENT", "PHOENIXLTD", "POLYCAB", "PREMIERENE", "PRESTIGE", "RADICO", "RVNL",
    "SBICARD", "SRF", "SAIL", "SUPREMEIND", "SUZLON", "SWIGGY", "TATACOMM", "TATAELXSI",
    "TATAINVEST", "TIINDIA", "UPL", "VMM", "IDEA", "VOLTAS", "WAAREEENER", "YESBANK",
    "3MINDIA", "ACC", "ACMESOLAR", "AIAENG", "AWL", "AADHARHFC", "AARTIIND", "AAVAS",
    "ABBOTINDIA", "ACE", "ACUTAAS", "ABFRL", "ABLBL", "ABREL", "ABSLAMC", "CPPLUS",
    "AEGISLOG", "AEGISVOPAK", "AFCONS", "AFFLE", "AJANTPHARM", "ABDL", "ARE&M", "AMBER",
    "ANANDRATHI", "ANANTRAJ", "ANGELONE", "ANTHEM", "ANURAS", "APARINDS", "APOLLOTYRE", "APTUS",
    "ASAHIINDIA", "ASTERDM", "ATHERENERG", "ATUL", "AIIL", "BEML", "BLS", "BAJAJHFL",
    "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "MAHABANK", "BATAINDIA", "BAYERCROP", "BELRISE", "BERGEPAINT",
    "BHARTIHEXA", "BIKAJI", "BSOFT", "BLUEDART", "BLUEJET", "BBTC", "FIRSTCRY", "BRIGADE",
    "MAPMYINDIA", "CCL", "CESC", "CIEINDIA", "CRISIL", "CANFINHOME", "CANHLIFE", "CAPLIPOINT",
    "CGCL", "CARBORUNIV", "CARTRADE", "CASTROLIND", "CEATLTD", "CEMPRO", "CENTRALBK", "CDSL",
    "CHALET", "CHAMBLFERT", "CHENNPETRO", "CHOICEIN", "CHOLAHLDNG", "CUB", "CLEAN", "COHANCE",
    "CAMS", "CONCORDBIO", "CRAFTSMAN", "CREDITACC", "CROMPTON", "CYIENT", "DCMSHRIRAM", "DOMS",
    "DALBHARAT", "DATAPATTNS", "DEEPAKFERT", "DEEPAKNTR", "DELHIVERY", "DEVYANI", "LALPATHLAB", "EIDPARRY",
    "EIHOTEL", "ELECON", "ELGIEQUIP", "EMAMILTD",
]


def _diversify_default_browse(base: list[str], extra_count: int) -> list[str]:
    """Tops up the market-cap-ranked POPULAR_SYMBOLS with more stocks chosen
    for sector spread. Ranking by market cap alone makes the default browse
    view ~23% Financial Services and leaves sectors like Textiles, Realty,
    Power and Telecom with 1-3 names each even though the full dataset has
    plenty more - round-robin across sectors here so no single sector
    dominates what a user sees before they search."""
    seen = set(base)
    by_sector: dict[str, list[str]] = {}
    for s in ALL_STOCKS:
        sector = s.get("sector")
        symbol = s["symbol"]
        if not sector or symbol in seen:
            continue
        by_sector.setdefault(sector, []).append(symbol)

    extra: list[str] = []
    sector_names = list(by_sector.keys())
    i = 0
    while len(extra) < extra_count and sector_names:
        sector = sector_names[i % len(sector_names)]
        bucket = by_sector[sector]
        if bucket:
            extra.append(bucket.pop(0))
            i += 1
        else:
            sector_names.remove(sector)
            if not sector_names:
                break
            i %= len(sector_names)

    return base + extra


DEFAULT_BROWSE_SYMBOLS = _diversify_default_browse(POPULAR_SYMBOLS, 200)


def get_stock_meta(symbol: str) -> dict:
    symbol = symbol.strip().upper()
    return _BY_SYMBOL.get(symbol, {"symbol": symbol, "name": symbol, "sector": None})


def _popular(limit: int) -> list[dict]:
    return [get_stock_meta(sym) for sym in DEFAULT_BROWSE_SYMBOLS[:limit]]


def search_stocks(query: str, limit: int = 500) -> list[dict]:
    q = query.strip().upper()
    if not q:
        return _popular(limit)

    ranked: list[tuple[int, str, dict]] = []
    for s in ALL_STOCKS:
        symbol = s["symbol"]
        name = s["name"].upper()
        if symbol == q:
            rank = 0
        elif symbol.startswith(q):
            rank = 1
        elif q in symbol:
            rank = 2
        elif name.startswith(q):
            rank = 3
        elif q in name:
            rank = 4
        else:
            continue
        ranked.append((rank, symbol, s))

    ranked.sort(key=lambda t: (t[0], t[1]))
    return [s for _, _, s in ranked[:limit]]
