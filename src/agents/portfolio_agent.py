"""
src/agents/portfolio_agent.py
Portfolio Analysis Agent — takes user holdings, fetches live prices,
calculates metrics, and generates a plain-English analysis via LLM.

Usage:
    from src.agents.portfolio_agent import PortfolioAnalysisAgent
    agent = PortfolioAnalysisAgent()

    # Structured input
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5, "BND": 20})

    # Natural language input
    result = agent.run(query="I have 10 Apple shares, 5 Microsoft, and 20 BND")

    print(result["answer"])
    print(result["metrics"])
"""

import yfinance as yf
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import load_llm
from src.rag.retriever import get_retriever
from src.utils.logger import get_logger

log = get_logger(__name__)

class _Holding(BaseModel):
    ticker: str   = Field(description="Stock ticker symbol, e.g. AAPL. Convert company names to tickers.")
    shares: float = Field(description="Number of shares held")

class _PortfolioParams(BaseModel):
    holdings: list[_Holding] = Field(description="All holdings extracted from the text")


PROMPT = ChatPromptTemplate.from_template("""
You are Finnie, a friendly financial education assistant.
Analyze the portfolio metrics below and explain them in plain English.
Be encouraging but honest about any concentration risks.
Always end with a disclaimer that this is educational, not financial advice.

Portfolio Metrics:
{metrics}

Financial Knowledge (from knowledge base):
{context}

User's risk tolerance: {risk_profile}

Provide a clear analysis covering:
1. Overall portfolio health
2. Diversification assessment
3. Sector concentration risks (if any)
4. Whether the allocation matches the user's risk profile
5. One educational takeaway about portfolio construction

Analysis:
""")


class PortfolioAnalysisAgent:

    def __init__(self):
        self.retriever        = get_retriever()
        self.llm              = load_llm()
        self.chain            = PROMPT | self.llm | StrOutputParser()
        self.portfolio_parser = self.llm.with_structured_output(_PortfolioParams)

    def _parse_portfolio_from_text(self, text: str) -> dict[str, float]:
        """Extract holdings from natural language using structured output."""
        parsed = self.portfolio_parser.invoke(
            f"Extract all stock holdings from this text:\n\n{text}"
        )
        return {h.ticker.upper(): h.shares for h in parsed.holdings}

    def _fetch_holding(self, ticker: str, shares: float) -> dict | None:
        """Fetch live price and metadata for a single holding."""
        try:
            stock = yf.Ticker(ticker)
            info  = stock.info

            price = (
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or info.get("previousClose")
            )

            if not price:
                return None

            return {
                "ticker":         ticker.upper(),
                "shares":         shares,
                "price":          float(price),
                "position_value": float(price) * shares,
                "sector":         info.get("sector", "Unknown"),
                "asset_type":     _classify_asset(ticker, info),
                "pe_ratio":       info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "name":           info.get("shortName", ticker),
            }
        except Exception:
            return None

    def _calculate_metrics(self, holdings: list[dict]) -> dict:
        """Compute portfolio-level metrics from individual holdings."""
        total_value = sum(h["position_value"] for h in holdings)

        if total_value == 0:
            return {}

        for h in holdings:
            h["allocation_pct"] = round(h["position_value"] / total_value * 100, 2)

        # Sector breakdown
        sector_totals: dict[str, float] = {}
        for h in holdings:
            sector = h["sector"]
            sector_totals[sector] = sector_totals.get(sector, 0) + h["position_value"]
        sector_pct = {
            s: round(v / total_value * 100, 2)
            for s, v in sector_totals.items()
        }

        # Asset type breakdown
        asset_totals: dict[str, float] = {}
        for h in holdings:
            atype = h["asset_type"]
            asset_totals[atype] = asset_totals.get(atype, 0) + h["position_value"]
        asset_pct = {
            a: round(v / total_value * 100, 2)
            for a, v in asset_totals.items()
        }

        score = _diversification_score(holdings, sector_pct)

        return {
            "total_value":           round(total_value, 2),
            "holdings":              holdings,
            "sector_pct":            sector_pct,
            "asset_pct":             asset_pct,
            "num_positions":         len(holdings),
            "diversification_score": score,
        }

    def _format_metrics_for_llm(self, metrics: dict) -> str:
        """Convert metrics dict to a readable string for the prompt."""
        lines = [
            f"Total Portfolio Value: ${metrics['total_value']:,.2f}",
            f"Number of Positions:   {metrics['num_positions']}",
            f"Diversification Score: {metrics['diversification_score']}/10",
            "",
            "Holdings:",
        ]
        for h in metrics["holdings"]:
            lines.append(
                f"  {h['ticker']} ({h['name']}): "
                f"{h['shares']} shares @ ${h['price']:.2f} = "
                f"${h['position_value']:,.2f} ({h['allocation_pct']}%) "
                f"[{h['sector']}]"
            )
        lines.append("")
        lines.append("Sector Breakdown:")
        for sector, pct in metrics["sector_pct"].items():
            lines.append(f"  {sector}: {pct}%")
        lines.append("")
        lines.append("Asset Type Breakdown:")
        for atype, pct in metrics["asset_pct"].items():
            lines.append(f"  {atype}: {pct}%")
        return "\n".join(lines)

    def _get_rag_context(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def run(
        self,
        query: str = "",
        portfolio: dict[str, float] | None = None,
        risk_profile: str = "moderate",
    ) -> dict:
        """
        Analyze a portfolio.

        Args:
            query:        Natural language description of the portfolio (used if
                          portfolio dict is not provided, and for RAG context).
            portfolio:    Dict of {ticker: shares}, e.g. {"AAPL": 10, "MSFT": 5}
            risk_profile: "conservative" | "moderate" | "aggressive"

        Returns:
            {
                "answer":  str,
                "metrics": dict,
                "failed":  [tickers that couldn't be fetched]
            }
        """
        log.info("PortfolioAnalysisAgent | positions=%d | risk=%s", len(portfolio or {}), risk_profile)
        if portfolio is None:
            if not query:
                return {"answer": "Please provide a portfolio to analyze.", "metrics": {}, "failed": []}
            portfolio = self._parse_portfolio_from_text(query)

        if not portfolio:
            return {
                "answer": (
                    "I couldn't parse any holdings from your input. "
                    "Try a format like: 'AAPL: 10 shares, MSFT: 5 shares, BND: 20 shares'"
                ),
                "metrics": {},
                "failed":  [],
            }

        # Fetch live data for all holdings
        holdings = []
        failed   = []
        for ticker, shares in portfolio.items():
            data = self._fetch_holding(ticker.upper(), shares)
            if data:
                holdings.append(data)
            else:
                failed.append(ticker.upper())

        if not holdings:
            return {
                "answer": "Could not fetch data for any of the provided tickers. Please check your ticker symbols.",
                "metrics": {},
                "failed":  failed,
            }

        metrics     = self._calculate_metrics(holdings)
        metrics_str = self._format_metrics_for_llm(metrics)
        rag_query   = query or "portfolio diversification asset allocation risk"
        context     = self._get_rag_context(rag_query)

        answer = self.chain.invoke({
            "metrics":      metrics_str,
            "context":      context,
            "risk_profile": risk_profile,
        })

        log.info("PortfolioAnalysisAgent | done | value=$%.0f | score=%s | failed=%s",
                 metrics.get("total_value", 0),
                 metrics.get("diversification_score", "n/a"),
                 failed or "none")
        return {
            "answer":  answer,
            "metrics": metrics,
            "failed":  failed,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_asset(ticker: str, info: dict) -> str:
    """Classify a holding as Stock, ETF, Bond ETF, or Crypto."""
    quote_type = info.get("quoteType", "").upper()
    if quote_type == "ETF":
        name = (info.get("shortName") or ticker).upper()
        if any(w in name for w in ("BOND", "TREASURY", "FIXED", "AGG", "BND", "TLT", "LQD")):
            return "Bond ETF"
        return "ETF"
    if quote_type == "CRYPTOCURRENCY":
        return "Crypto"
    return "Stock"


def _diversification_score(holdings: list[dict], sector_pct: dict[str, float]) -> int:
    """Simple diversification score out of 10. Deducts for concentration risks."""
    score = 10

    if len(holdings) < 3:
        score -= 4
    elif len(holdings) < 5:
        score -= 2

    for h in holdings:
        if h["allocation_pct"] > 40:
            score -= 3
            break
        elif h["allocation_pct"] > 25:
            score -= 1
            break

    for pct in sector_pct.values():
        if pct > 60:
            score -= 3
            break
        elif pct > 40:
            score -= 1
            break

    has_bonds = any(h["asset_type"] == "Bond ETF" for h in holdings)
    if not has_bonds:
        score -= 1

    return max(0, min(10, score))


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = PortfolioAnalysisAgent()

    result = agent.run(
        portfolio={"AAPL": 10, "MSFT": 5, "BND": 20, "NVDA": 3, "RKLB": 300, "NOW":100,
                   "QQQ": 50},
        risk_profile="aggressive",
    )

    print(f"\nDiversification Score: {result['metrics'].get('diversification_score')}/10")
    print(f"Total Value: ${result['metrics'].get('total_value', 0):,.2f}")
    if result["failed"]:
        print(f"Failed tickers: {result['failed']}")
    print(f"\nAnalysis:\n{result['answer']}")
