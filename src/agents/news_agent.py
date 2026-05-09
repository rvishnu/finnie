"""
src/agents/news_agent.py
News Synthesizer Agent — fetches headlines for one or more tickers,
deduplicates, tags each with a sentiment signal, and produces a plain-English
market briefing via RAG + LLM.

Usage:
    from src.agents.news_agent import NewsSynthesizerAgent
    agent = NewsSynthesizerAgent()

    result = agent.run("What's the latest news on AAPL and NVDA?")
    print(result["answer"])
    for h in result["headlines"]:
        print(h["ticker"], h["sentiment"], h["title"])
"""

import re
import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import load_llm
from src.rag.retriever import get_retriever
from src.utils.logger import get_logger

log = get_logger(__name__)

MAX_HEADLINES_PER_TICKER = 5
MAX_TICKERS              = 5

# Simple keyword-based sentiment — avoids extra LLM calls
BULLISH_WORDS = {
    "surge", "jump", "rise", "rises", "beat", "beats", "gain", "gains",
    "profit", "growth", "record", "upgrade", "upgraded", "rally", "rallies",
    "soar", "soars", "strong", "wins", "positive", "expand", "expands",
    "high", "boom", "outperform", "exceed", "exceeds", "bullish",
}
BEARISH_WORDS = {
    "drop", "drops", "fall", "falls", "decline", "declines", "miss", "misses",
    "loss", "losses", "cut", "cuts", "downgrade", "downgraded", "crash", "crashes",
    "concern", "concerns", "risk", "risks", "weak", "warns", "warning",
    "negative", "shrink", "shrinks", "low", "slump", "slumps", "bearish",
    "layoff", "layoffs", "lawsuit", "fraud", "recall",
}

PROMPT = ChatPromptTemplate.from_template("""
You are Finnie, a friendly financial education assistant.
Synthesize the news headlines below into a clear, beginner-friendly market briefing.
Explain what the news means for investors and why it matters.
Group related themes together if multiple tickers are involved.
Always end with a disclaimer that this is educational, not financial advice.

Tickers Covered: {tickers}

Recent Headlines:
{headlines}

Financial Knowledge (from knowledge base):
{context}

User Question: {query}

Market Briefing:
""")


class NewsSynthesizerAgent:

    def __init__(self):
        self.retriever = get_retriever()
        self.llm       = load_llm()

    # ── Ticker extraction ─────────────────────────────────────────────────────

    def _extract_tickers(self, query: str) -> list[str]:
        """
        Pull all ticker symbols from the query.
        Tries regex first, then LLM for company name resolution.
        """
        EXCLUDE = {
            "I", "A", "AN", "THE", "OR", "AND", "IN", "IS", "IT",
            "BE", "TO", "DO", "SO", "MY", "ME", "WE", "US",
            "BUY", "SELL", "FOR", "ON", "AT", "IF",
        }
        tickers = []

        # Explicit uppercase ticker symbols
        matches = re.findall(r'\b([A-Z]{1,5})\b', query)
        for m in matches:
            if m not in EXCLUDE and m not in tickers:
                tickers.append(m)

        if tickers:
            return tickers[:MAX_TICKERS]

        # LLM fallback: resolve company names to tickers
        response = self.llm.invoke(
            "Extract all stock ticker symbols from this text. "
            "Convert company names to tickers (Apple→AAPL, Microsoft→MSFT, Nvidia→NVDA, etc.). "
            "Return ONLY comma-separated ticker symbols, nothing else. "
            "If none found, return NONE.\n\nText: " + query
        ).content.strip()

        if response.upper() == "NONE" or not response:
            return []

        for part in response.split(","):
            t = part.strip().upper()
            if t and t != "NONE" and t not in tickers:
                tickers.append(t)

        return tickers[:MAX_TICKERS]

    # ── News fetching ─────────────────────────────────────────────────────────

    def _fetch_raw_news(self, ticker: str) -> list[dict]:
        """Fetch structured headlines from yfinance for a single ticker."""
        try:
            news = yf.Ticker(ticker).news[:MAX_HEADLINES_PER_TICKER]
        except Exception:
            return []

        results = []
        for item in news:
            content   = item.get("content", item)
            title     = (
                content.get("title")
                or item.get("title")
                or ""
            )
            publisher = (
                content.get("provider", {}).get("displayName")
                or content.get("publisher")
                or item.get("publisher")
                or "Unknown"
            )
            if title:
                results.append({
                    "ticker":    ticker.upper(),
                    "title":     title,
                    "publisher": publisher,
                })
        return results

    def _deduplicate(self, headlines: list[dict]) -> list[dict]:
        """Remove headlines with duplicate titles across tickers."""
        seen   = set()
        unique = []
        for h in headlines:
            key = h["title"].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(h)
        return unique

    # ── Sentiment ─────────────────────────────────────────────────────────────

    def _sentiment(self, title: str) -> str:
        """Keyword-based sentiment: bullish / bearish / neutral."""
        words   = re.findall(r'\b\w+\b', title.lower())
        bullish = sum(1 for w in words if w in BULLISH_WORDS)
        bearish = sum(1 for w in words if w in BEARISH_WORDS)
        if bullish > bearish:
            return "bullish"
        if bearish > bullish:
            return "bearish"
        return "neutral"

    def _tag_headlines(self, headlines: list[dict]) -> list[dict]:
        """Add sentiment field to each headline dict."""
        for h in headlines:
            h["sentiment"] = self._sentiment(h["title"])
        return headlines

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_headlines_for_llm(self, headlines: list[dict]) -> str:
        lines = []
        for i, h in enumerate(headlines, 1):
            lines.append(
                f"{i}. [{h['ticker']}] {h['title']} — {h['publisher']} ({h['sentiment']})"
            )
        return "\n".join(lines) if lines else "No headlines available."

    def _get_rag_context(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Synthesize financial news for the tickers mentioned in the query.

        Args:
            query: Natural language question about financial news.

        Returns:
            {
                "answer":    str,
                "headlines": [{"ticker", "title", "publisher", "sentiment"}, ...],
                "tickers":   list[str],
                "error":     str | None,
            }
        """
        if not query or not query.strip():
            return {
                "answer":    "Please ask about a company or ticker, e.g. 'What's the latest news on AAPL?'",
                "headlines": [],
                "tickers":   [],
                "error":     "no_input",
            }

        log.info("NewsSynthesizerAgent | query=%r", query[:60])
        tickers = self._extract_tickers(query)

        if not tickers:
            return {
                "answer":    "I couldn't identify any stock tickers in your question. Please mention a ticker symbol (e.g. AAPL) or company name.",
                "headlines": [],
                "tickers":   [],
                "error":     "no_tickers",
            }

        # Fetch, deduplicate, tag
        raw = []
        for ticker in tickers:
            raw.extend(self._fetch_raw_news(ticker))

        headlines = self._tag_headlines(self._deduplicate(raw))

        if not headlines:
            return {
                "answer":    f"No recent news found for {', '.join(tickers)}. They may be invalid tickers or news is temporarily unavailable.",
                "headlines": [],
                "tickers":   tickers,
                "error":     "no_news",
            }

        rag_query   = query
        context     = self._get_rag_context(rag_query)
        headlines_str = self._format_headlines_for_llm(headlines)

        chain  = PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({
            "tickers":   ", ".join(tickers),
            "headlines": headlines_str,
            "context":   context,
            "query":     query,
        })

        log.info("NewsSynthesizerAgent | done | tickers=%s | headlines=%d", tickers, len(headlines))
        return {
            "answer":    answer,
            "headlines": headlines,
            "tickers":   tickers,
            "error":     None,
        }


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = NewsSynthesizerAgent()

    queries = [
        "What's the latest news on AAPL and NVDA?",
        "Any news about Tesla?",
        "What's happening with Microsoft and Google stocks?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        result = agent.run(q)
        print(f"Tickers : {result['tickers']}")
        print(f"Headlines ({len(result['headlines'])}):")
        for h in result["headlines"]:
            print(f"  [{h['sentiment']:8s}] [{h['ticker']}] {h['title']}")
        print(f"Briefing:\n{result['answer']}")
        print("-" * 60)
