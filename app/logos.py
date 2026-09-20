"""Company logo lookup for the mobile app's stock avatars.

Clearbit's free public logo API (logo.clearbit.com) - the old default for
this exact use case - was sunset on Dec 1, 2025 (confirmed by DNS lookup: the
subdomain no longer resolves at all). Its recommended migration path,
logo.dev, is free (500k requests/month) but requires a registered
"publishable" token, so this proxies through the backend rather than
embedding a token in the mobile bundle.

Two lookup tiers, tried in order:
1. _DOMAINS - a small curated symbol->domain map for names ambiguous enough
   that a plain text search could plausibly mismatch (e.g. very short or
   generic names). Domain lookup is exact, so this is the higher-confidence
   path where we've bothered to verify it by hand.
2. logo.dev's "display from name" mode (img.logo.dev/name/{company name}),
   which resolves via their brand-search index - tested against real NSE
   names (including smaller/generic ones like "Page Industries Ltd") and it
   matched correctly even with the "Ltd" suffix left in, so this covers the
   other ~1400 symbols without needing them hand-mapped. fallback=404 keeps
   logo.dev from returning its own generic monogram on a miss, so our 404
   still reaches the mobile app's onError and its themed colored-monogram
   fallback - same as tier 1.

Both tiers 404 (not error) when LOGO_DEV_API_KEY isn't configured, or when
neither lookup finds anything - the fail-soft pattern used throughout this
app (see also the Reddit/YouTube social sources).
"""

import os
import urllib.parse

LOGO_DEV_API_KEY = os.environ.get("LOGO_DEV_API_KEY")

_DOMAINS = {
    "RELIANCE": "ril.com", "TCS": "tcs.com", "HDFCBANK": "hdfcbank.com",
    "ICICIBANK": "icicibank.com", "INFY": "infosys.com", "HINDUNILVR": "hul.co.in",
    "ITC": "itcportal.com", "SBIN": "sbi.co.in", "BHARTIARTL": "airtel.in",
    "KOTAKBANK": "kotak.com", "AXISBANK": "axisbank.com", "BAJFINANCE": "bajajfinserv.in",
    "BAJAJFINSV": "bajajfinserv.in", "ASIANPAINT": "asianpaints.com",
    "MARUTI": "marutisuzuki.com", "M&M": "mahindra.com", "TITAN": "titancompany.in",
    "SUNPHARMA": "sunpharma.com", "ULTRACEMCO": "ultratechcement.com", "WIPRO": "wipro.com",
    "HCLTECH": "hcltech.com", "TECHM": "techmahindra.com", "NESTLEIND": "nestle.in",
    "POWERGRID": "powergrid.in", "NTPC": "ntpc.co.in", "ONGC": "ongcindia.com",
    "COALINDIA": "coalindia.in", "TATASTEEL": "tatasteel.com", "TATAMOTORS": "tatamotors.com",
    "TATACONSUM": "tataconsumer.com", "JSWSTEEL": "jsw.in", "ADANIENT": "adanienterprises.com",
    "ADANIPORTS": "adaniports.com", "ADANIGREEN": "adanigreenenergy.com",
    "ADANIPOWER": "adanipower.com", "GRASIM": "grasim.com", "DRREDDY": "drreddys.com",
    "CIPLA": "cipla.com", "DIVISLAB": "divislabs.com", "EICHERMOT": "eichermotors.com",
    "HEROMOTOCO": "heromotocorp.com", "BAJAJ-AUTO": "bajajauto.com", "BRITANNIA": "britannia.co.in",
    "DABUR": "dabur.com", "GODREJCP": "godrejcp.com", "MARICO": "marico.com",
    "COLPAL": "colgatepalmolive.co.in", "PIDILITIND": "pidilite.com", "DMART": "dmartindia.com",
    "TRENT": "trentlimited.com", "INDIGO": "goindigo.in", "IRCTC": "irctc.co.in",
    "ZOMATO": "zomato.com", "ETERNAL": "zomato.com", "NYKAA": "nykaa.com", "PAYTM": "paytm.com",
    "POLICYBZR": "policybazaar.com", "NAUKRI": "naukri.com", "SWIGGY": "swiggy.com",
    "LT": "larsentoubro.com", "DLF": "dlf.in", "HAVELLS": "havells.com", "VOLTAS": "voltas.com",
    "SIEMENS": "siemens.co.in", "ABB": "abb.com", "BOSCHLTD": "bosch.in", "MRF": "mrftyres.com",
    "APOLLOHOSP": "apollohospitals.com", "MAXHEALTH": "maxhealthcare.in",
    "FORTIS": "fortishealthcare.com", "HDFCLIFE": "hdfclife.com", "SBILIFE": "sbilife.co.in",
    "ICICIGI": "icicilombard.com", "MUTHOOTFIN": "muthootfinance.com", "PNB": "pnbindia.in",
    "BANKBARODA": "bankofbaroda.in", "CANBK": "canarabank.com", "IDFCFIRSTB": "idfcfirstbank.com",
    "FEDERALBNK": "federalbank.co.in", "INDUSINDBK": "indusind.com", "YESBANK": "yesbank.in",
    "VEDL": "vedantalimited.com", "HINDALCO": "hindalco.com", "HINDZINC": "hzlindia.com",
    "AMBUJACEM": "ambujacement.com", "SHREECEM": "shreecement.com", "ACC": "acclimited.com",
    "GAIL": "gailonline.com", "IOC": "iocl.com", "BPCL": "bharatpetroleum.in",
    "HINDPETRO": "hindustanpetroleum.com", "LICI": "licindia.in", "GODREJPROP": "godrejproperties.com",
    "OBEROIRLTY": "oberoirealty.com", "PRESTIGE": "prestigeconstructions.com",
    "LODHA": "lodhagroup.com", "MPHASIS": "mphasis.com", "PERSISTENT": "persistent.com",
    "COFORGE": "coforge.com", "LTIM": "ltimindtree.com", "MOTHERSON": "motherson.com",
    "TVSMOTOR": "tvsmotor.com", "ASHOKLEY": "ashokleyland.com", "BATAINDIA": "bata.in",
    "ABFRL": "abfrl.com", "UPL": "upl-ltd.com", "SRF": "srf.com", "PIIND": "piind.com",
    "DEEPAKNTR": "deepaknitrite.com", "BEL": "bel-india.in", "HAL": "hal-india.co.in",
    "BHEL": "bhel.com", "CONCOR": "concorindia.com", "RVNL": "rvnl.org",
}


def get_domain(symbol: str) -> str | None:
    return _DOMAINS.get(symbol.strip().upper())


def get_logo_redirect_url(symbol: str, company_name: str | None) -> str | None:
    """Returns the logo.dev image URL to redirect to, or None if we can't
    look one up at all (no key configured, or no name to search by)."""
    if not LOGO_DEV_API_KEY:
        return None
    domain = get_domain(symbol)
    if domain:
        return f"https://img.logo.dev/{domain}?token={LOGO_DEV_API_KEY}&size=128"
    if company_name:
        encoded = urllib.parse.quote(company_name)
        return f"https://img.logo.dev/name/{encoded}?token={LOGO_DEV_API_KEY}&size=128&fallback=404"
    return None
