"""
src/workflow/graph.py
ReAct (Reasoning + Acting) workflow using LangGraph.

Pattern:
    1. Code calls LLM with the user query + available tools
    2. LLM reasons and returns a tool call (or a final answer)
    3. Code executes the chosen tool and appends the result to messages
    4. Code calls LLM again — LLM sees the result and reasons further
    5. Repeat until LLM returns a final answer (no tool call)

This means the LLM can call MULTIPLE agents in one turn:
    "What's the news on my portfolio stocks?"
    → LLM calls analyze_portfolio  (gets tickers)
    → LLM calls get_financial_news (gets news for those tickers)
    → LLM writes one combined answer

Memory: MemorySaver persists the full state (messages, goal, savings,
risk profile) per thread_id across conversation turns.

Usage:
    from src.workflow.graph import invoke

    r = invoke("I want $2M in 20 years. I have $100K.", thread_id="u1")
    r = invoke("I'm aggressive with risk.",              thread_id="u1")
    r = invoke("Actually I have $200K saved.",           thread_id="u1")
    r = invoke("What about 401k and NVDA news?",         thread_id="u1")
    print(r["answer"])
"""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.workflow.router import extract_params
from src.core.llm import load_llm
from src.agents.qa_agent        import FinanceQAAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.market_agent    import MarketAnalysisAgent
from src.agents.goal_agent      import GoalPlanningAgent
from src.agents.news_agent      import NewsSynthesizerAgent
from src.agents.tax_agent       import TaxEducationAgent


# ── State ─────────────────────────────────────────────────────────────────────

class FinnieState(TypedDict):
    # Full conversation history — LangGraph appends each message automatically
    messages:           Annotated[list, add_messages]

    # Persisted user context — restored by MemorySaver on every turn
    risk_profile:       str
    goal_amount:        float | None
    time_horizon_years: float | None
    current_savings:    float


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
    """
    result = _load()["qa"].run(query)
    return result.get("answer", "")


@tool
def analyze_portfolio(
    query: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Analyze a user's stock portfolio.
    Use when the user mentions their holdings, wants to know allocation,
    diversification score, sector breakdown, or portfolio health.
    Examples: "I have 10 AAPL and 5 MSFT", "analyze my portfolio".
    """
    result = _load()["portfolio"].run(
        query=query,
        risk_profile=state.get("risk_profile", "moderate"),
    )
    metrics = result.get("metrics", {})
    answer  = result.get("answer", "")

    if metrics:
        summary = (
            f"Portfolio Value: ${metrics.get('total_value', 0):,.2f}\n"
            f"Positions: {metrics.get('num_positions', 0)}\n"
            f"Diversification Score: {metrics.get('diversification_score', 0)}/10\n"
            f"Sectors: {metrics.get('sector_pct', {})}\n\n"
        )
        return summary + answer
    return answer


@tool
def get_market_data(
    query: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Get real-time stock price, market data, and analysis for a company.
    Use when the user asks about a specific stock's price, performance,
    P/E ratio, market cap, or wants company analysis.
    Examples: "How is AAPL doing?", "Tell me about Tesla stock".
    """
    result = _load()["market"].run(query)
    return result.get("answer", "")


@tool
def plan_financial_goal(
    query: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Plan a savings or retirement goal — calculate monthly contributions needed,
    projected value, and investment growth impact.
    Use when the user mentions saving for retirement, a house, education,
    or any financial target with an amount and timeline.
    Examples: "I want $2M in 20 years", "How much to save for a house in 5 years".
    """
    result = _load()["goal"].run(
        query=query,
        goal_amount=state.get("goal_amount"),
        time_horizon_years=state.get("time_horizon_years"),
        current_savings=state.get("current_savings", 0.0),
        risk_profile=state.get("risk_profile", "moderate"),
    )
    metrics = result.get("metrics", {})
    answer  = result.get("answer", "")

    if metrics:
        summary = (
            f"Goal: ${metrics.get('goal_amount', 0):,.0f} "
            f"in {metrics.get('time_horizon_years', 0)} years\n"
            f"Current savings: ${metrics.get('current_savings', 0):,.0f}\n"
            f"Monthly needed (cash): ${metrics.get('monthly_no_growth', 0):,.2f}\n"
            f"Monthly needed (invested at {metrics.get('annual_return_pct', 7)}%): "
            f"${metrics.get('monthly_with_growth', 0):,.2f}\n\n"
        )
        return summary + answer
    return answer


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
    result = _load()["news"].run(query)
    headlines = result.get("headlines", [])
    answer    = result.get("answer", "")

    if headlines:
        summary = "Headlines:\n" + "\n".join(
            f"  [{h['sentiment']:8s}] [{h['ticker']}] {h['title']}"
            for h in headlines[:8]
        ) + "\n\n"
        return summary + answer
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
    result = _load()["tax"].run(query)
    metrics = result.get("metrics", {})
    answer  = result.get("answer", "")

    if metrics:
        scenario = result.get("scenario", "")
        if scenario == "capital_gains":
            summary = (
                f"Gain: ${metrics.get('gain', 0):,.2f} | "
                f"Type: {metrics.get('holding_type', '')} | "
                f"Tax rate: {metrics.get('tax_rate_pct', 0)}% | "
                f"Estimated tax: ${metrics.get('estimated_tax', 0):,.2f} | "
                f"Net gain: ${metrics.get('net_gain', 0):,.2f}\n\n"
            )
            return summary + answer
        if scenario == "tax_loss_harvesting":
            summary = (
                f"Loss: ${metrics.get('total_loss', 0):,.2f} | "
                f"Deductible this year: ${metrics.get('deductible_this_year', 0):,.2f} | "
                f"Tax saving: ${metrics.get('estimated_tax_saving', 0):,.2f}\n\n"
            )
            return summary + answer
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


# ── System prompt ─────────────────────────────────────────────────────────────

def _system_prompt(state: FinnieState) -> str:
    """Build a dynamic system prompt that includes the user's current context."""
    goal    = f"${state.get('goal_amount', 0):,.0f}"          if state.get("goal_amount")        else "not set"
    horizon = f"{state.get('time_horizon_years', 0)} years"   if state.get("time_horizon_years") else "not set"
    savings = f"${state.get('current_savings', 0.0):,.0f}"

    return f"""You are Finnie, a friendly financial education assistant.

You have access to specialized tools. Reason step by step:
  1. Understand what the user is asking
  2. Decide which tool(s) to call — you can call MULTIPLE tools if needed
  3. Call the first tool and read the result
  4. If more information is needed, call another tool
  5. When you have everything, write a clear, beginner-friendly final answer

Known user context (use automatically — do not ask the user to repeat):
  - Risk profile:    {state.get("risk_profile", "moderate")}
  - Savings goal:    {goal}
  - Time horizon:    {horizon}
  - Current savings: {savings}

━━━ OUT-OF-SCOPE GUARDRAIL ━━━
Some topics are completely outside your expertise. For these, do NOT call any tool.
Instead, reply ONLY with the exact message below (fill in [topic]):

  "I'm Finnie, a financial education assistant. I'm not able to help with [topic].
   I can help you with: stock portfolio analysis, retirement and savings goal planning,
   real-time market data, financial news, tax education (capital gains, IRA, 401k, HSA),
   and general investing questions. What would you like to explore?"

Topics that are OUT OF SCOPE (always use the guardrail message):
  - Traffic, weather, sports, cooking, travel, health conditions, relationships
  - Car insurance, home insurance, health insurance, life insurance premiums
  - Starting or valuing a business, business loans, business strategy
  - Legal advice, lawsuits, contracts, immigration
  - Real estate prices, mortgage rates, property buying advice
  - Cryptocurrency price predictions or trading signals
  - Any topic unrelated to personal finance and investing education

Topics that ARE in scope (use your tools):
  - Stock prices, ETFs, bonds, index funds, dividends
  - Portfolio analysis and diversification
  - Retirement planning, savings goals, compound interest
  - Capital gains tax, IRA/Roth IRA, 401k, HSA accounts
  - Financial news for specific tickers
  - General financial education (what is X, how does Y work)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Guidelines for in-scope answers:
  - Explain financial concepts in plain English, avoid jargon
  - Be encouraging but honest about risks
  - Always end with: "This is for educational purposes only and is not financial advice."
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def param_extractor_node(state: FinnieState) -> dict:
    """
    Runs once at the start of each turn.
    Extracts financial parameters from the latest message and merges them
    into state — handles mid-conversation updates like "actually I have $200K".
    """
    # Last human message is always the most recent query
    last_human = next(
        (m for m in reversed(state.get("messages", []))
         if hasattr(m, "type") and m.type == "human"),
        None,
    )
    if not last_human:
        return {}

    new = extract_params(str(last_human.content))
    if not new:
        return {}

    updates: dict = {}
    if "current_savings"    in new: updates["current_savings"]    = new["current_savings"]
    if "goal_amount"        in new: updates["goal_amount"]         = new["goal_amount"]
    if "time_horizon_years" in new: updates["time_horizon_years"]  = new["time_horizon_years"]
    if "risk_profile"       in new: updates["risk_profile"]        = new["risk_profile"]
    return updates


def agent_node(state: FinnieState) -> dict:
    """
    Core ReAct node — LLM reasons about the query and decides what to do:
      - Returns a tool call → ToolNode executes it, loop continues
      - Returns a plain message → conversation turn is complete
    """
    llm_with_tools = load_llm().bind_tools(TOOLS)

    messages = [SystemMessage(content=_system_prompt(state))] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the Finnie ReAct graph.

    Flow per turn:
        START → param_extractor → agent ⟵──────────────┐
                                     ↓ tool_call?        │
                                  tool_node ─────────────┘
                                     ↓ no tool_call
                                    END
    """
    builder = StateGraph(FinnieState)

    builder.add_node("param_extractor", param_extractor_node)
    builder.add_node("agent",           agent_node)
    builder.add_node("tools",           ToolNode(TOOLS))

    # Entry: always extract params first, then let LLM reason
    builder.add_edge(START,             "param_extractor")
    builder.add_edge("param_extractor", "agent")

    # ReAct loop: if LLM returned a tool call → execute → reason again
    #             if LLM returned a final answer → done
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


# ── Module-level singleton ────────────────────────────────────────────────────

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

def invoke(query: str, thread_id: str = "default") -> dict:
    """
    Run one conversational turn through the Finnie ReAct workflow.

    Args:
        query:     The user's message.
        thread_id: Session ID — all turns with the same ID share memory.

    Returns:
        {
            "answer":    str,   final answer from the LLM
            "messages":  list,  full updated message history
        }
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = _get_graph().invoke(
        {
            "messages":          [HumanMessage(content=query)],
            "risk_profile":      "moderate",
            "current_savings":   0.0,
            "goal_amount":       None,
            "time_horizon_years": None,
        },
        config=config,
    )

    # Last message is always the LLM's final answer
    last = result["messages"][-1]
    return {
        "answer":   str(last.content),
        "messages": result["messages"],
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tid = "demo"

    turns = [
        "I want to retire in 20 years. I have $100K saved. I want $2 million.",
        "I'm pretty aggressive, I can take risk.",
        "Actually I have $200K saved, not $100K.",
        "Can I use my 401k for this? And what's the latest news on NVDA?",
        "I want to start a business and sell it for $2M.",
    ]

    for q in turns:
        print(f"\nUser  : {q}")
        r = invoke(q, thread_id=tid)
        print(f"Finnie: {r['answer'][:1000]}...")
        print("-" * 60)
