"""
src/utils/market_tools.py
LangChain tool for fetching real-time stock data.
Tries Alpha Vantage first, falls back to yfinance.

Usage:
    from src.utils.market_tools import get_stock_data
    result = get_stock_data.invoke("AAPL")
"""

import os
import time
import requests
import yfinance as yf
import re

from functools import wraps
from langchain_core.tools import tool
from dotenv import load_dotenv
from src.utils.logger import get_logger

log = get_logger(__name__)

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

_CACHE: dict[str, tuple] = {}
_CACHE_TTL = 1800  # 30 minutes


def _with_cache(fn):
    """Cache non-None return values by ticker for _CACHE_TTL seconds."""
    @wraps(fn)
    def wrapper(ticker: str):
        key = f"{fn.__name__}:{ticker.upper()}"
        if key in _CACHE:
            value, ts = _CACHE[key]
            if time.time() - ts < _CACHE_TTL:
                log.debug("Cache HIT  | %s", key)
                return value
        log.debug("Cache MISS | %s", key)
        result = fn(ticker)
        if result is not None:
            _CACHE[key] = (result, time.time())
        return result
    return wrapper


# ── Alpha Vantage ─────────────────────────────────────────────────────────────

@_with_cache
def _fetch_alpha_vantage(ticker: str) -> dict | None:
    """Fetch stock data from Alpha Vantage. Returns None on any failure."""
    if not ALPHA_VANTAGE_KEY:
        return None
    try:
        # Global quote
        resp = requests.get(ALPHA_VANTAGE_URL, params={
            "function": "GLOBAL_QUOTE",
            "symbol":   ticker,
            "apikey":   ALPHA_VANTAGE_KEY,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "Note" in data:
            log.warning("Alpha Vantage rate limit hit — falling back to yfinance")
            return None

        quote = data.get("Global Quote", {})
        if not quote or not quote.get("05. price"):
            return None

        # Company overview for P/E, market cap
        overview_resp = requests.get(ALPHA_VANTAGE_URL, params={
            "function": "OVERVIEW",
            "symbol":   ticker,
            "apikey":   ALPHA_VANTAGE_KEY,
        }, timeout=10)
        overview = overview_resp.json() if overview_resp.ok else {}

        def _to_float(val) -> float | None:
            try:
                f = float(str(val).rstrip("%"))
                return f if f != 0.0 else None
            except (TypeError, ValueError):
                return None

        raw_pct = quote.get("10. change percent", "0%")
        return {
            "ticker":        ticker.upper(),
            "name":          overview.get("Name") or ticker.upper(),
            "price":         float(quote.get("05. price", 0)),
            "change":        float(quote.get("09. change", 0)),
            "change_pct":    _to_float(raw_pct) or 0.0,
            "volume":        int(quote.get("06. volume", 0)),
            "high":          float(quote.get("03. high", 0)),
            "low":           float(quote.get("04. low", 0)),
            "prev_close":    float(quote.get("08. previous close", 0)),
            "week_52_high":  _to_float(overview.get("52WeekHigh")),
            "week_52_low":   _to_float(overview.get("52WeekLow")),
            "market_cap":    _to_float(overview.get("MarketCapitalization")),
            "pe_ratio":      _to_float(overview.get("PERatio")),
            "dividend_yield":_to_float(overview.get("DividendYield")),
            "sector":        overview.get("Sector", "N/A"),
            "description":   overview.get("Description", "")[:500],
            "source":        "Alpha Vantage",
        }

    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Alpha Vantage error: {e}, falling back to yfinance")
        return None


# ── yFinance ──────────────────────────────────────────────────────────────────

@_with_cache
def _fetch_yfinance(ticker: str) -> dict | None:
    """Fetch stock data from yfinance. Returns None on any failure."""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        # yfinance returns empty dict for invalid tickers
        if not info or info.get("regularMarketPrice") is None:
            return None

        price = (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )

        return {
            "ticker":        ticker.upper(),
            "name":          info.get("shortName") or info.get("longName") or ticker.upper(),
            "price":         float(price or 0),
            "change":        float(info.get("regularMarketChange", 0)),
            "change_pct":    float(info.get("regularMarketChangePercent", 0)),
            "volume":        int(info.get("regularMarketVolume", 0)),
            "high":          float(info.get("regularMarketDayHigh", 0)),
            "low":           float(info.get("regularMarketDayLow", 0)),
            "prev_close":    float(info.get("previousClose", 0)),
            "week_52_high":  info.get("fiftyTwoWeekHigh"),
            "week_52_low":   info.get("fiftyTwoWeekLow"),
            "market_cap":    info.get("marketCap"),
            "pe_ratio":      info.get("trailingPE"),
            "dividend_yield":info.get("dividendYield"),
            "sector":        info.get("sector", "N/A"),
            "description":   (info.get("longBusinessSummary") or "")[:500],
            "source":        "Yahoo Finance",
        }

    except Exception as e:
        print(f"yfinance error for {ticker}: {e}")
        return None

    
def extract_ticker(query: str, llm) -> str | None:
    """Extract stock ticker from natural language query."""

    # 1. Explicit uppercase ticker e.g. AAPL, TSLA
    EXCLUDE = {"I", "A", "AN", "THE", "OR", "AND", "IN", "IS",
               "IT", "BE", "TO", "DO", "SO", "MY", "ME", "WE",
               "US", "BUY", "SELL"}
    matches = re.findall(r'\b([A-Z]{2,5})\b', query)
    for match in matches:
        if match not in EXCLUDE:
            return match

    # 2. LLM extracts company name
    response = llm.invoke(
        f"Extract only the company name from this query. "
        f"Return just the company name, nothing else. "
        f"If no company is mentioned return NONE.\n\nQuery: {query}"
    ).content.strip()

    if not response or response.upper() == "NONE":
        return None

    # 3. yfinance looks up the ticker
    try:
        results = yf.Search(response, max_results=1)
        if results.quotes:
            return results.quotes[0].get("symbol")
    except Exception:
        pass

    return None

# ── LangChain Tool ────────────────────────────────────────────────────────────

@tool
def get_stock_data(ticker: str) -> str:
    """
    Fetch real-time stock data for a given ticker symbol.
    Returns price, volume, P/E ratio, 52-week range, market cap and more.
    Use this when the user asks about a specific stock or company.

    Args:
        ticker: Stock ticker symbol e.g. AAPL, TSLA, MSFT
    """
    ticker = ticker.strip().upper()

    # Try Alpha Vantage first, fall back to yfinance
    data = _fetch_alpha_vantage(ticker) or _fetch_yfinance(ticker)

    if not data:
        return (
            f"Could not retrieve data for ticker '{ticker}'. "
            "Please check the ticker symbol and try again."
        )

    # Format as readable string for the LLM
    lines = [
        f"Stock Data for {data['ticker']} (Source: {data['source']})",
        f"Price:          ${data['price']:.2f}",
        f"Change:         {data['change']:+.2f} ({data['change_pct']:+.2f}%)",
        f"Day High/Low:   ${data['high']:.2f} / ${data['low']:.2f}",
        f"Prev Close:     ${data['prev_close']:.2f}",
        f"Volume:         {data['volume']:,}",
        f"52-Week High:   ${data['week_52_high']:.2f}" if data['week_52_high'] else "52-Week High:   N/A",
        f"52-Week Low:    ${data['week_52_low']:.2f}"  if data['week_52_low']  else "52-Week Low:    N/A",
        f"Market Cap:     {data['market_cap']}",
        f"P/E Ratio:      {data['pe_ratio']}",
        f"Dividend Yield: {data['dividend_yield']}",
        f"Sector:         {data['sector']}",
    ]

    if data["description"]:
        lines.append(f"About:          {data['description']}...")

    return "\n".join(lines)


@tool
def get_stock_news(ticker: str) -> str:
    """
    Fetch recent news headlines for a given stock ticker.
    Use this when the user asks about recent news or events for a company.

    Args:
        ticker: Stock ticker symbol e.g. AAPL, TSLA, MSFT
    """
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news[:5]

        if not news:
            return f"No recent news found for {ticker}."

        lines = [f"Recent News for {ticker}:"]
        for i, item in enumerate(news, 1):
            # yfinance v0.2.40+ nests content inside "content" key
            content   = item.get("content", item)
            title     = (
                content.get("title")
                or item.get("title")
                or "No title"
            )
            publisher = (
                content.get("provider", {}).get("displayName")
                or content.get("publisher")
                or item.get("publisher")
                or "Unknown"
            )
            lines.append(f"{i}. {title} — {publisher}")

        return "\n".join(lines)

    except Exception as e:
        return f"Could not fetch news for {ticker}: {e}"

# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(get_stock_data.invoke("AAPL"))
    print()
    print(get_stock_news.invoke("AAPL"))
