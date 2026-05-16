import re
from typing import Annotated

from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId

from src.utils.logger import get_logger
from src.agents.qa_agent        import FinanceQAAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.market_agent    import MarketAnalysisAgent
from src.agents.goal_agent      import GoalPlanningAgent
from src.agents.news_agent      import NewsSynthesizerAgent
from src.agents.tax_agent       import TaxEducationAgent
from src.utils.market_tools import _fetch_alpha_vantage, _fetch_yfinance

log = get_logger(__name__)


# ── Lazy agent singletons ─────────────────────────────────────────────────────

_agent_cache: dict = {}


def _load() -> dict:
    """Load all agents once and cache them for the process lifetime."""
    global _agent_cache
    if not _agent_cache:
        _agent_cache = {
            "qa":        FinanceQAAgent(),
            "portfolio": PortfolioAnalysisAgent(),
            "market":    MarketAnalysisAgent(),
            "goal":      GoalPlanningAgent(),
            "news":      NewsSynthesizerAgent(),
            "tax":       TaxEducationAgent(),
        }
    return _agent_cache


# ── Tools — each agent exposed as a callable the LLM can choose ──────────────
#
# The LLM sees the docstring and parameter names to decide when and how to
# call each tool.  `state` is injected by LangGraph — the LLM never sees it.

@tool
def answer_finance_question(
    query: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Answer a general financial education question.
    Use for: what is X, how does Y work, difference between A and B,
    compound interest, index funds, diversification, ETFs, bonds, etc.
    Also knows about all investment concepts covered by the other tools, so can answer general questions
    Knows what are the investment types (stocks, ETFs, bonds)
    """
    log.info("Tool | answer_finance_question | query=%r", query[:60])
    result = _load()["qa"].run(query)
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    if citations:
        sources = "\n".join(
            f"- {c['title']}" + (f": {c['url']}" if c.get("url") else "")
            for c in citations
        )
        out = f"{answer}\n\nSources:\n{sources}"
        log.info("Tool | answer_finance_question | done | citations=%d answer_len=%d", len(citations), len(out))
        return out
    log.info("Tool | answer_finance_question | done | answer_len=%d", len(answer))
    return answer


@tool
def analyze_portfolio(
    query: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Analyze a user's stock portfolio.
    Use if for Goal Analysis if user mentions their portfolio to find the total value,
    which feeds into the goal planning tool.
    Use when the user mentions their holdings, wants to know allocation,
    diversification score, sector breakdown, or portfolio health.
    Examples: "I have 10 AAPL and 5 MSFT", "analyze my portfolio".

    """
    log.info("Tool | analyze_portfolio | query=%r", query[:60])
    result = _load()["portfolio"].run(
        query=query,
        risk_profile=state.get("risk_profile", "moderate"),
    )
    metrics = result.get("metrics", {})
    answer  = result.get("answer", "")

    failed = result.get("failed", [])
    if metrics:
        total_value = metrics.get("total_value", 0)
        holdings_lines = "".join(
            f"  {h['ticker']}: {h['shares']} shares @ USD {h['price']:,.2f}"
            f" = USD {h['position_value']:,.2f} ({h['allocation_pct']:.1f}%)"
            f" | Sector: {h.get('sector','N/A')}"
            f" | P/E: {h.get('pe_ratio','N/A')}"
            f" | Div Yield: {'{:.2f}%'.format(h['dividend_yield']*100) if h.get('dividend_yield') else 'N/A'}"
            f"\n"
            for h in metrics.get("holdings", [])
        )
        summary = (
            f"Portfolio Total Value: USD {total_value:,.2f}\n"
            f"Positions: {metrics.get('num_positions', 0)}\n"
            f"Diversification Score: {metrics.get('diversification_score', 0)}/10\n"
            f"Individual Positions (live prices):\n{holdings_lines}"
            f"Sectors: {metrics.get('sector_pct', {})}\n\n"
        )
        if failed:
            summary += (
                f"Could not fetch data for: {', '.join(failed)}. "
                f"These ticker symbols may be invalid or misspelled — "
                f"please tell the user to double-check them.\n\n"
            )
        # Persist the resolved tickers back to state so follow-up turns ("what's the news on my stocks?")
        # use the correct ticker symbols, even when the user typed company names like "Google" or "NVIDIA".
        resolved_holdings = {h["ticker"]: h["shares"] for h in metrics.get("holdings", [])}
        log.info("Tool | analyze_portfolio | done | value=%.2f positions=%d failed=%s",
                 total_value, metrics.get("num_positions", 0), failed or "none")
        return Command(update={
            "messages":         [ToolMessage(content=summary + answer, tool_call_id=tool_call_id)],
            "portfolio_value":  total_value,
            "portfolio_holdings": resolved_holdings or state.get("portfolio_holdings"),
        })

    log.info("Tool | analyze_portfolio | done | no metrics | answer_len=%d", len(answer))
    return Command(update={
        "messages": [ToolMessage(content=answer, tool_call_id=tool_call_id)],
    })


@tool
def get_market_data(
    query: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Get real-time stock price, market data, and analysis for a company.
    Use when the user asks about a specific stock's price, performance,
    Use this to find current price for the user's holdings to connect portfolio → goal analysis.
    P/E ratio, market cap, or wants company analysis.
    Examples: "How is AAPL doing?", "Tell me about Tesla stock".
    """
    log.info("Tool | get_market_data | query=%r", query[:60])
    result = _load()["market"].run(query)
    ticker = result.get("ticker")
    answer = result.get("answer", "")

    if ticker:
        raw = _fetch_alpha_vantage(ticker) or _fetch_yfinance(ticker)
        if raw:
            price  = raw.get("price", 0)
            hi52   = raw.get("week_52_high")
            lo52   = raw.get("week_52_low")
            pe     = raw.get("pe_ratio")
            div    = raw.get("dividend_yield")
            cap    = raw.get("market_cap")

            pct_from_hi = f" ({abs((price-hi52)/hi52*100):.1f}% below 52w high)" if hi52 and hi52 > 0 else ""
            cap_str = ("USD {:.2f}T".format(cap/1e12) if cap and cap>=1e12 else
                       "USD {:.2f}B".format(cap/1e9)  if cap and cap>=1e9  else
                       "USD {:.2f}M".format(cap/1e6)  if cap and cap>=1e6  else "N/A")
            div_str = f"{div*100:.2f}%" if div else "N/A"

            structured = (
                f"=== Market Data: {ticker} ===\n"
                f"Current Price:  USD {price:,.2f}\n"
                f"52-Week High:   {'USD {:,.2f}'.format(hi52) + pct_from_hi if hi52 else 'N/A'}\n"
                f"52-Week Low:    {'USD {:,.2f}'.format(lo52) if lo52 else 'N/A'}\n"
                f"P/E Ratio:      {pe if pe else 'N/A'}\n"
                f"Dividend Yield: {div_str}\n"
                f"Market Cap:     {cap_str}\n"
                f"Sector:         {raw.get('sector', 'N/A')}\n\n"
            )
            log.info("Tool | get_market_data | done | ticker=%s price=%.2f", ticker, price)
            return structured + answer

    log.info("Tool | get_market_data | done | answer_len=%d", len(answer))
    return answer


@tool
def plan_financial_goal(
    query: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Plan a savings or retirement goal — calculate monthly contributions needed,
    projected value, and investment growth impact.
    ALSO handles withdrawal / decumulation: how much to withdraw from a nest egg,
    how long savings will last, or sustainable monthly income from a portfolio.
    Use when the user mentions saving for retirement, a house, education,
    or any financial target with an amount and timeline.
    ALSO use when the user asks: "how much can I withdraw", "how long will my money last",
    "retirement income", or mentions a large nest egg and asks about monthly spending.
    Examples: "I want $2M in 20 years", "How much to save for a house in 5 years",
    "With $3M how much can I withdraw monthly?", "Will $2M last 30 years at $8k/month?".
    Also use other tools to find the value of the user's current portfolio.
    """
    log.info("Tool | plan_financial_goal | query=%r", query[:60])

    goal_amount         = state.get("goal_amount")
    time_horizon_years  = state.get("time_horizon_years")
    current_savings     = state.get("current_savings")
    portfolio_value     = state.get("portfolio_value") or 0.0
    annual_contribution = state.get("annual_contribution")
    risk_profile        = state.get("risk_profile", "moderate")

    # Fill gaps from the current message using the goal agent's LLM parser.
    # Only overwrite existing state values for risk/contribution (user may update them);
    # never overwrite an established goal/timeline.
    agent = _load()["goal"]
    withdrawal_mode   = False
    withdrawal_amount = None
    if query:
        parsed = agent._parse_goal_from_text(query)
        withdrawal_mode   = parsed.get("withdrawal_mode", False)
        withdrawal_amount = parsed.get("withdrawal_amount")

        if withdrawal_mode:
            # In decumulation mode, the user's "$X" is the nest egg, not a savings goal.
            nest_egg_parsed = parsed.get("nest_egg") or parsed.get("current_savings")
            if current_savings is None and nest_egg_parsed:
                current_savings = nest_egg_parsed
            if not time_horizon_years:
                time_horizon_years = parsed.get("time_horizon_years")
            if parsed.get("risk_profile"):
                risk_profile = parsed["risk_profile"]
        else:
            if not goal_amount:
                goal_amount = parsed.get("goal_amount")
            if not time_horizon_years:
                time_horizon_years = parsed.get("time_horizon_years")
            if current_savings is None and parsed.get("current_savings") is not None:
                current_savings = parsed["current_savings"]
            if parsed.get("risk_profile"):
                risk_profile = parsed["risk_profile"]
            if annual_contribution is None:
                annual_contribution = parsed.get("annual_contribution")

    # ── Withdrawal / decumulation path ────────────────────────────────────────
    if withdrawal_mode:
        nest_egg = (current_savings or 0.0) + portfolio_value
        log.info("Tool | plan_financial_goal | withdrawal mode | nest_egg=%.2f horizon=%s monthly_wd=%s",
                 nest_egg, time_horizon_years, withdrawal_amount)

        def _wd_respond(content: str) -> Command:
            updates: dict = {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
            if current_savings is not None:
                updates["current_savings"] = current_savings
            if time_horizon_years:
                updates["time_horizon_years"] = time_horizon_years
            if risk_profile:
                updates["risk_profile"] = risk_profile
            return Command(update=updates)

        if nest_egg <= 0:
            return _wd_respond(
                "To calculate your withdrawal plan I need to know your current savings or portfolio value. "
                "How much do you have saved or invested?"
            )

        result  = agent.run_withdrawal(
            nest_egg=nest_egg,
            risk_profile=risk_profile,
            time_horizon_years=time_horizon_years,
            monthly_withdrawal=withdrawal_amount,
            query=query,
        )
        metrics = result.get("metrics", {})
        answer  = result.get("answer", "")

        summary_lines = [f"Nest egg: USD {nest_egg:,.0f}"]
        if metrics.get("monthly_withdrawal"):
            summary_lines.append(f"Monthly withdrawal: USD {metrics['monthly_withdrawal']:,.2f}")
        if metrics.get("duration_years"):
            summary_lines.append(f"Duration: {metrics['duration_years']} years")
        if metrics.get("rule_of_4pct_monthly"):
            summary_lines.append(f"4% Rule benchmark: USD {metrics['rule_of_4pct_monthly']:,.2f}/month")
        log.info("Tool | plan_financial_goal | withdrawal done | mode=%s", metrics.get("mode"))
        return _wd_respond("\n".join(summary_lines) + "\n\n" + answer)

    # Infer current_savings from a portfolio result injected into the query context.
    if current_savings is None and "Portfolio Value:" in query:
        m = re.search(r"Portfolio Value:\s*\$?([\d,]+(?:\.\d+)?)", query)
        if m:
            current_savings = float(m.group(1).replace(",", ""))
            log.info("Tool | plan_financial_goal | inferred current_savings=%.2f from portfolio output",
                     current_savings)

    def _respond(content: str) -> Command:
        """Return a Command that writes resolved params to state and the answer to messages."""
        updates: dict = {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
        if goal_amount:                    updates["goal_amount"]          = goal_amount
        if time_horizon_years:             updates["time_horizon_years"]   = time_horizon_years
        if current_savings is not None:    updates["current_savings"]      = current_savings
        if risk_profile:                   updates["risk_profile"]         = risk_profile
        if annual_contribution:            updates["annual_contribution"]  = annual_contribution
        return Command(update=updates)

    missing = []
    if not goal_amount:
        missing.append("your savings target (e.g. '$500,000' or '$1 million')")
    if not time_horizon_years:
        missing.append("your timeline (e.g. 'in 20 years' or 'by age 65')")
    if missing:
        log.info("Tool | plan_financial_goal | missing=%s", missing)
        return _respond("To build your savings plan I need: " + " and ".join(missing) + ". Could you share those?")

    # Combine cash savings with portfolio value so goal planning uses the real starting balance.
    # current_savings stays as-is in state (cash only); effective_savings is the working total.
    if current_savings is None and portfolio_value == 0.0:
        log.info("Tool | plan_financial_goal | awaiting current_savings")
        contrib_note = f", investing USD {annual_contribution/12:,.0f}/month" if annual_contribution else ""
        return _respond(
            f"Got it — aiming for USD {goal_amount:,.0f} in {time_horizon_years:.0f} years"
            f"{contrib_note}. "
            "How much have you already saved toward this goal? "
            "(Just reply with the amount, or say '0' if you're starting from scratch.)"
        )

    effective_savings = portfolio_value + (current_savings or 0.0)
    if portfolio_value > 0 and current_savings:
        log.info("Tool | plan_financial_goal | effective_savings=%.2f (portfolio=%.2f + cash=%.2f)",
                 effective_savings, portfolio_value, current_savings)

    result = agent.run(
        query=query,
        goal_amount=goal_amount,
        time_horizon_years=time_horizon_years,
        current_savings=effective_savings,
        risk_profile=risk_profile,
        annual_contribution=annual_contribution,
    )
    metrics = result.get("metrics", {})
    answer  = result.get("answer", "")

    if metrics:
        summary = (
            f"Current savings: ${metrics.get('current_savings', 0):,.0f}\n"
            f"Time horizon: {metrics.get('time_horizon_years', 0)} years\n"
            f"Assumed annual return: {metrics.get('annual_return_pct', 7)}%\n"
        )
        if "goal_amount" in metrics:
            summary += (
                f"Goal: ${metrics.get('goal_amount', 0):,.0f}\n"
                f"Monthly needed (cash only): ${metrics.get('monthly_no_growth', 0):,.2f}\n"
                f"Monthly needed (invested):  ${metrics.get('monthly_with_growth', 0):,.2f}\n"
                f"Projected value (if contributions invested): ${metrics.get('projected_value', 0):,.2f}\n\n"
            )
            log.info("Tool | plan_financial_goal | done | goal=%.0f monthly_growth=%.2f",
                     metrics.get("goal_amount", 0), metrics.get("monthly_with_growth", 0))
        else:
            summary += (
                f"Annual contribution: ${metrics.get('annual_contribution', 0):,.2f}\n"
                f"Monthly contribution: ${metrics.get('monthly_contribution', 0):,.2f}\n"
                f"Projected value (invested): ${metrics.get('projected_value', 0):,.2f}\n"
                f"Projected value (cash only): ${metrics.get('projected_cash', 0):,.2f}\n\n"
            )
            log.info("Tool | plan_financial_goal | projection done | projected=%.0f",
                     metrics.get("projected_value", 0))
        return _respond(summary + answer)

    log.info("Tool | plan_financial_goal | done | no metrics | answer_len=%d", len(answer))
    return _respond(answer)


@tool
def get_financial_news(
    query: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Fetch and synthesize recent financial news for one or more stock tickers.
    Use when the user asks about recent news, headlines, or what is happening
    with a specific company or set of companies.
    Examples: "What's the news on NVDA?", "Latest on AAPL and MSFT".
    """
    log.info("Tool | get_financial_news | query=%r", query[:60])
    result = _load()["news"].run(query)
    headlines = result.get("headlines", [])
    answer    = result.get("answer", "")

    if headlines:
        summary = "Headlines:\n" + "\n".join(
            f"  [{h['sentiment']:8s}] [{h['ticker']}] {h['title']}"
            for h in headlines[:8]
        ) + "\n\n"
        log.info("Tool | get_financial_news | done | headlines=%d", len(headlines))
        return summary + answer
    log.info("Tool | get_financial_news | done | no headlines | answer_len=%d", len(answer))
    return answer


@tool
def get_tax_education(
    query: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Explain tax concepts related to investing.
    Use for: capital gains tax (short-term vs long-term), IRA/Roth IRA,
    401k, HSA contribution limits, tax-loss harvesting.
    Examples: "I sold stock after 8 months with a $5k gain",
              "How much can I put in my Roth IRA?",
              "What is tax-loss harvesting?".
    """
    log.info("Tool | get_tax_education | query=%r", query[:60])
    result = _load()["tax"].run(query)
    metrics = result.get("metrics", {})
    answer  = result.get("answer", "")

    if metrics:
        scenario = result.get("scenario", "")
        if scenario == "capital_gains":
            per_stock = metrics.get("per_stock", [])
            if per_stock:
                stock_lines = "\n".join(
                    f"  {s['ticker']}: {s['shares']:.0f} shares × "
                    f"(USD {s['current_price']:,.2f} − USD {s['purchase_price']:,.2f}) "
                    f"= USD {s['gain']:,.2f}"
                    for s in per_stock
                )
                summary = (
                    f"Per-stock gains (Python-computed):\n{stock_lines}\n"
                    f"Total Gain: USD {metrics.get('gain', 0):,.2f} | "
                    f"Type: {metrics.get('holding_type', '')} | "
                    f"Tax rate: {metrics.get('tax_rate_pct', 0)}% | "
                    f"Estimated tax: USD {metrics.get('estimated_tax', 0):,.2f} | "
                    f"Net gain: USD {metrics.get('net_gain', 0):,.2f}\n\n"
                )
            else:
                summary = (
                    f"Gain: USD {metrics.get('gain', 0):,.2f} | "
                    f"Type: {metrics.get('holding_type', '')} | "
                    f"Tax rate: {metrics.get('tax_rate_pct', 0)}% | "
                    f"Estimated tax: USD {metrics.get('estimated_tax', 0):,.2f} | "
                    f"Net gain: USD {metrics.get('net_gain', 0):,.2f}\n\n"
                )
            log.info("Tool | get_tax_education | done | scenario=capital_gains tax=%.2f positions=%d",
                     metrics.get("estimated_tax", 0), len(per_stock))
            return summary + answer
        if scenario == "tax_loss_harvesting":
            summary = (
                f"Loss: ${metrics.get('total_loss', 0):,.2f} | "
                f"Deductible this year: ${metrics.get('deductible_this_year', 0):,.2f} | "
                f"Tax saving: ${metrics.get('estimated_tax_saving', 0):,.2f}\n\n"
            )
            log.info("Tool | get_tax_education | done | scenario=tax_loss_harvesting saving=%.2f",
                     metrics.get("estimated_tax_saving", 0))
            return summary + answer
    log.info("Tool | get_tax_education | done | scenario=general | answer_len=%d", len(answer))
    return answer


# All tools in one list — bound to the LLM and registered with ToolNode
TOOLS = [
    answer_finance_question,
    analyze_portfolio,
    get_market_data,
    plan_financial_goal,
    get_financial_news,
    get_tax_education,
]
