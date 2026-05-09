"""
src/mcp_server/server.py
Finnie MCP Server — exposes all 6 finance agents as tools for Claude Desktop.

Each tool is one of Finnie's agents. Claude Desktop calls them directly;
no LangGraph router needed here since Claude IS the reasoner.

Add to ~/Library/Application Support/Claude/claude_desktop_config.json:
    {
      "mcpServers": {
        "finnie": {
          "command": "uv",
          "args": ["run", "python", "src/mcp_server/server.py"],
          "cwd": "/Users/vishnu/PycharmProjects/IK/python/finnie"
        }
      }
    }

Then restart Claude Desktop — Finnie's tools appear in the tool picker.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.fastmcp import FastMCP

from src.agents.goal_agent      import GoalPlanningAgent
from src.agents.market_agent    import MarketAnalysisAgent
from src.agents.news_agent      import NewsSynthesizerAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.qa_agent        import FinanceQAAgent
from src.agents.tax_agent       import TaxEducationAgent

mcp = FastMCP("Finnie Finance Assistant")


# ── Lazy agent singletons (loaded once on first call) ─────────────────────────

_agents: dict = {}

def _get(name: str):
    if name not in _agents:
        _agents[name] = {
            "portfolio": PortfolioAnalysisAgent,
            "goal":      GoalPlanningAgent,
            "market":    MarketAnalysisAgent,
            "news":      NewsSynthesizerAgent,
            "tax":       TaxEducationAgent,
            "qa":        FinanceQAAgent,
        }[name]()
    return _agents[name]


# ── Holdings parser (shared by analyze_portfolio) ─────────────────────────────

def _parse_holdings(text: str) -> dict[str, int]:
    holdings: dict[str, int] = {}
    for m in re.finditer(r'\b([A-Z]{1,5})\s*:\s*(\d+(?:\.\d+)?)\b', text.upper()):
        holdings[m.group(1)] = int(float(m.group(2)))
    if holdings:
        return holdings
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\s+([A-Z]{1,5})\b', text.upper()):
        holdings[m.group(2)] = int(float(m.group(1)))
    return holdings


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_portfolio(holdings: str, risk_profile: str = "moderate") -> str:
    """
    Analyze a stock portfolio — diversification score, sector allocation,
    asset mix, and personalized recommendations.

    Args:
        holdings:     Holdings in any of these formats:
                        "AAPL: 10, MSFT: 5, BND: 20"
                        "10 AAPL, 5 MSFT, 20 BND"
        risk_profile: conservative | moderate | aggressive  (default: moderate)
    """
    parsed = _parse_holdings(holdings)
    if not parsed:
        return "Could not parse holdings. Use format: AAPL: 10, MSFT: 5, BND: 20"

    result  = _get("portfolio").run(portfolio=parsed, risk_profile=risk_profile)
    metrics = result.get("metrics", {})
    answer  = result.get("answer", "")
    failed  = result.get("failed", [])

    if not metrics:
        return answer

    lines = [
        f"Portfolio Value:       ${metrics.get('total_value', 0):,.2f}",
        f"Positions:             {metrics.get('num_positions', 0)}",
        f"Diversification Score: {metrics.get('diversification_score', 0)} / 10",
        f"Sector Allocation:     {metrics.get('sector_pct', {})}",
        f"Asset Mix:             {metrics.get('asset_pct', {})}",
    ]
    if failed:
        lines.append(f"Failed tickers (no data): {', '.join(failed)}")
    lines += ["", answer]
    return "\n".join(lines)


@mcp.tool()
def plan_financial_goal(
    goal_amount: float,
    time_horizon_years: float,
    current_savings: float = 0.0,
    risk_profile: str = "moderate",
) -> str:
    """
    Calculate how much to save monthly to reach a savings or retirement goal.
    Shows projections with and without investment growth.

    Args:
        goal_amount:         Target amount in dollars (e.g. 2000000 for $2M)
        time_horizon_years:  Years until the goal (e.g. 20)
        current_savings:     Amount already saved, default 0
        risk_profile:        conservative (4%) | moderate (7%) | aggressive (10%)
    """
    result  = _get("goal").run(
        goal_amount=goal_amount,
        time_horizon_years=time_horizon_years,
        current_savings=current_savings,
        risk_profile=risk_profile,
    )
    metrics = result.get("metrics", {})
    answer  = result.get("answer", "")

    if not metrics:
        return answer

    lines = [
        f"Goal:                    ${metrics.get('goal_amount', 0):,.0f}",
        f"Time Horizon:            {metrics.get('time_horizon_years', 0)} years",
        f"Current Savings:         ${metrics.get('current_savings', 0):,.0f}",
        f"Gap:                     ${metrics.get('gap', 0):,.0f}",
        f"Monthly (cash, no growth): ${metrics.get('monthly_no_growth', 0):,.2f}",
        f"Monthly (invested at {metrics.get('annual_return_pct', 7)}%): ${metrics.get('monthly_with_growth', 0):,.2f}",
        f"Projected Value (if invested): ${metrics.get('projected_value', 0):,.0f}",
        "",
        answer,
    ]
    return "\n".join(lines)


@mcp.tool()
def get_stock_data(ticker: str) -> str:
    """
    Get real-time stock price, P/E ratio, market cap, 52-week range,
    and a plain-English company analysis.

    Args:
        ticker: Stock symbol, e.g. AAPL, TSLA, NVDA, SPY
    """
    result = _get("market").run(f"Tell me about {ticker} stock")
    return result.get("answer", f"Could not fetch data for {ticker}.")


@mcp.tool()
def get_financial_news(query: str) -> str:
    """
    Fetch and summarize recent financial news for one or more stock tickers.
    Each headline is tagged bullish / bearish / neutral.

    Args:
        query: Natural language query mentioning tickers,
               e.g. "latest news on NVDA and MSFT"
    """
    result    = _get("news").run(query)
    headlines = result.get("headlines", [])
    answer    = result.get("answer", "")
    error     = result.get("error")

    if error:
        return answer

    header = "Headlines:\n" + "\n".join(
        f"  [{h['sentiment']:8s}] [{h['ticker']}] {h['title']}"
        for h in headlines[:8]
    )
    return header + "\n\n" + answer


@mcp.tool()
def get_tax_education(query: str) -> str:
    """
    Explain US tax concepts related to investing.
    Handles: capital gains tax, IRA / Roth IRA / 401k / HSA contribution limits,
    tax-loss harvesting, and general tax questions.

    Args:
        query: Tax question, e.g.
               "I sold AAPL after 8 months with a $5,000 gain — 22% bracket"
               "How much can I contribute to my Roth IRA?"
               "I have a $4,000 loss this year, can I harvest it?"
    """
    result   = _get("tax").run(query)
    metrics  = result.get("metrics", {})
    answer   = result.get("answer", "")
    scenario = result.get("scenario", "")

    if not metrics:
        return answer

    if scenario == "capital_gains":
        header = (
            f"Gain: ${metrics.get('gain', 0):,.2f}  |  "
            f"Type: {metrics.get('holding_type', '')}  |  "
            f"Rate: {metrics.get('tax_rate_pct', 0)}%  |  "
            f"Est. tax: ${metrics.get('estimated_tax', 0):,.2f}  |  "
            f"Net gain: ${metrics.get('net_gain', 0):,.2f}"
        )
        return header + "\n\n" + answer

    if scenario == "tax_loss":
        header = (
            f"Loss: ${metrics.get('total_loss', 0):,.2f}  |  "
            f"Deductible this year: ${metrics.get('deductible_this_year', 0):,.2f}  |  "
            f"Carryforward: ${metrics.get('carryforward_to_next', 0):,.2f}  |  "
            f"Tax saving: ${metrics.get('estimated_tax_saving', 0):,.2f}"
        )
        return header + "\n\n" + answer

    return answer


@mcp.tool()
def answer_finance_question(query: str) -> str:
    """
    Answer a general financial education question using a curated knowledge base.
    Use for: what is X, how does Y work, ETFs vs mutual funds, compound interest,
    diversification, dollar-cost averaging, Sharpe ratio, bond basics, etc.

    Args:
        query: Any general finance or investing question
    """
    result = _get("qa").run(query)
    return result.get("answer", "")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
