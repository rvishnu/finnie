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

import re
from typing import Annotated

from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages

from src.core.llm import load_llm
from src.utils.logger import get_logger
from src.workflow.state import FinnieState
from src.workflow.tools import TOOLS
from src.workflow.prompts import _system_prompt, _synth_prompt

log = get_logger(__name__)


# ── Nodes ─────────────────────────────────────────────────────────────────────

# "AAPL: 100" or "MSFT - 200"
_TICKER_RE = re.compile(r'\b([A-Z]{2,5})\s*[:\-]\s*(\d+(?:\.\d+)?)\b')
# "100 AAPL" or "1000 QQQ" — number-first format
_TICKER_RE_REVERSE = re.compile(r'\b(\d+(?:\.\d+)?)\s+([A-Z]{2,5})\b')


def param_extractor_node(state: FinnieState) -> dict:
    """
    Runs once at the start of each turn.
    Extracts portfolio holdings and age from the latest message into state.
    """
    last_human = next(
        (m for m in reversed(state.get("messages", []))
         if hasattr(m, "type") and m.type == "human"),
        None,
    )
    if not last_human:
        return {}

    text = str(last_human.content)
    updates: dict = {}

    # Extract portfolio holdings — try "TICKER: N" format first, then "N TICKER" format
    holdings = {m.group(1): float(m.group(2)) for m in _TICKER_RE.finditer(text)}
    if len(holdings) < 2:
        holdings = {m.group(2): float(m.group(1)) for m in _TICKER_RE_REVERSE.finditer(text)}
    if len(holdings) >= 2:          # require at least 2 tickers to avoid false positives
        updates["portfolio_holdings"] = holdings
        log.info("ParamExtractor | portfolio=%s", list(holdings.keys()))

    age_m = re.search(r'\b(?:i am|i\'m|im)\s+(\d{1,3})\s*(?:years?\s*old)?\b', text.lower())
    if age_m:
        candidate = int(age_m.group(1))
        if 18 <= candidate <= 100:
            updates["age"] = candidate
            log.info("ParamExtractor | age=%d", candidate)

    if updates:
        log.info("ParamExtractor | extracted=%s", {k: v for k, v in updates.items() if k != "portfolio_holdings"})
    return updates


def agent_node(state: FinnieState) -> dict:
    """
    Core ReAct node — LLM reasons about the query and decides what to do:
      - Returns a tool call → ToolNode executes it, loop continues
      - Returns a plain message → conversation turn is complete
    """
    llm_with_tools = load_llm().bind_tools(TOOLS)

    # Keep only the last 20 messages to avoid unbounded context growth.
    # The system prompt always carries the key user context (goal, savings, risk),
    # so trimming old turns doesn't lose critical state.
    recent = trim_messages(
        state["messages"],
        max_tokens=20,
        token_counter=len,      # count by message count, not tokens
        strategy="last",
        start_on="human",       # never start mid-tool-call
        include_system=False,
    )
    messages = [SystemMessage(content=_system_prompt(state))] + recent
    log.debug("LLM | sending %d messages (trimmed from %d)", len(messages), len(state["messages"]) + 1)
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        log.info("LLM | → tool_calls: %s", [tc["name"] for tc in response.tool_calls])
    else:
        log.info("LLM | → final answer (%d chars)", len(str(response.content)))

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

    return builder.compile(checkpointer=_MEMORY)


# ── Shared memory — both graphs use the same MemorySaver so switching
#    between ReAct and fan-out mid-conversation never loses state ────────────

_MEMORY = MemorySaver()
_graph  = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Smart parallel fan-out graph ──────────────────────────────────────────────
#
# Unlike the ReAct graph (LLM picks tools one at a time, sequentially),
# this graph first routes to only the RELEVANT agents, then runs them in
# parallel:
#
#   START → param_extractor → smart_fanout → tools → synth → END
#
# smart_fanout_node: asks LLM which subset of tools is needed → AIMessage
# ToolNode:          runs only those tools in parallel → ToolMessages
# synth_node:        LLM synthesizes all results into one answer

# One-line descriptions used by the router — kept short so the routing call
# is cheap and the LLM focuses on intent rather than implementation detail.
_TOOL_DESCRIPTIONS = {
    "answer_finance_question": "General financial education: what is X, how does Y work, compound interest, ETFs, bonds, index funds",
    "analyze_portfolio":       "Analyse the user's portfolio holdings: allocation, diversification score, sector breakdown",
    "get_market_data":         "Real-time price, P/E ratio, market cap and analysis for specific stock tickers",
    "plan_financial_goal":     "Savings / retirement goal planning: can I reach $X in Y years, monthly contributions needed",
    "get_financial_news":      "Recent news headlines for specific stocks, tickers, or all portfolio holdings",
    "get_tax_education":       "Tax on investments: capital gains, IRA/Roth IRA, 401k, HSA, tax-loss harvesting",
}


class _ToolSelection(BaseModel):
    tools: list[str]


_PORTFOLIO_REF_PHRASES = (
    "my holding", "my portfolio", "my stock", "my top",
    "all the stock", "all stocks", "all my stock",
    "those stocks", "these stocks", "the stocks",
    "news of", "news on", "news about",
    "summary of", "summarize",
)


def _enrich_query(state: FinnieState, query: str) -> str:
    """Inject remembered portfolio tickers when the user refers to their holdings without listing them."""
    holdings = state.get("portfolio_holdings") or {}
    if not holdings or _TICKER_RE.search(query):
        return query
    if not any(p in query.lower() for p in _PORTFOLIO_REF_PHRASES):
        return query
    ticker_list = ", ".join(f"{t}: {int(s)}" for t, s in holdings.items())
    return f"Portfolio: {ticker_list}\n\nQuestion: {query}"


def _build_ctx_note(state: FinnieState) -> str:
    """Summarize established conversation context for the routing prompt."""
    parts = []
    if state.get("goal_amount"):
        parts.append(f"goal ${state['goal_amount']:,.0f}")
    if state.get("time_horizon_years"):
        parts.append(f"timeline {state['time_horizon_years']:.0f} yr")
    if state.get("current_savings") is not None:
        parts.append(f"savings ${state['current_savings']:,.0f}")
    if state.get("risk_profile") and state.get("risk_profile") != "moderate":
        parts.append(f"risk {state['risk_profile']}")
    if state.get("portfolio_holdings"):
        parts.append(f"portfolio {list(state['portfolio_holdings'].keys())}")
    return f"\nConversation context already established: {', '.join(parts)}." if parts else ""


def _select_tools(query: str, ctx_note: str) -> list[str]:
    """Ask the LLM (structured output) to pick 2-3 relevant tools for this query."""
    tool_list = "\n".join(f"- {name}: {desc}" for name, desc in _TOOL_DESCRIPTIONS.items())
    routing_prompt = (
        f"Select the tools needed to give a complete, well-rounded answer to this query.\n\n"
        f"Available tools:\n{tool_list}\n\n"
        f"Query: {query}{ctx_note}\n\n"
        "Rules (apply the FIRST matching rule and stop — do not stack rules):\n"
        "- News, headlines, or recent events for a specific stock or ticker → get_financial_news + get_market_data\n"
        "- News for the user's portfolio stocks / all stocks / how news affects selling decision → get_financial_news + analyze_portfolio\n"
        "- General advice, tips, or education → answer_finance_question + plan_financial_goal (if goal in context, else + get_tax_education)\n"
        "- Any 'explain', 'what is', 'how does', 'what are' question (NOT news/prices) → answer_finance_question + get_tax_education\n"
        "- Context has goal_amount + timeline and message adds savings/contribution/risk → plan_financial_goal + get_tax_education\n"
        "- Portfolio questions (holdings, sectors, allocation, P/E, rate sensitivity) → analyze_portfolio + get_market_data\n"
        "- Retirement / savings goal questions → plan_financial_goal + get_tax_education\n"
        "- 'Is my allocation right for my age?' → analyze_portfolio + answer_finance_question\n"
        "- Rate hike / interest rate vulnerability → analyze_portfolio + answer_finance_question\n"
        "- Stock news or market events → get_financial_news + get_market_data\n"
        "- Tax questions (selling, gains, IRA, 401k) → get_tax_education + answer_finance_question\n"
        "- Portfolio + tax (selling stocks, capital gains on holdings) → analyze_portfolio + get_tax_education + answer_finance_question\n"
        "- Portfolio + retirement goal → analyze_portfolio + plan_financial_goal + get_tax_education\n"
        "- 52-week high, dividends, P/E for a specific stock → get_market_data + answer_finance_question\n"
        "Select 2 tools for simple single-domain queries. Select 3 tools when the query clearly spans "
        "multiple domains (e.g. portfolio + tax, portfolio + goal, news + education). Never select more than 3.\n"
    )
    valid_names = {t.name for t in TOOLS}
    selection = load_llm().with_structured_output(_ToolSelection).invoke([HumanMessage(content=routing_prompt)])
    selected = [name for name in selection.tools if name in valid_names]
    return selected[:3] or ["answer_finance_question"]


_MARKET_INTENT = {"get_market_data", "get_financial_news", "analyze_portfolio"}


def _apply_goal_override(state: FinnieState, selected: list[str]) -> list[str]:
    """Pull in plan_financial_goal for follow-ups when an active goal is in state and no market tool was picked."""
    if not (state.get("goal_amount") or state.get("time_horizon_years")):
        return selected
    if "plan_financial_goal" in selected or any(t in selected for t in _MARKET_INTENT):
        return selected
    log.info("SmartFanOut | injected plan_financial_goal for active goal context")
    return ["plan_financial_goal"] + [s for s in selected if s != "get_financial_news"][:1]


def _apply_portfolio_override(selected: list[str], holdings: dict) -> list[str]:
    """Ensure get_market_data is always paired with analyze_portfolio.

    The routing LLM sometimes picks answer_finance_question as the companion,
    which skips live price data. Force the correct pairing whenever holdings exist.
    """
    if "analyze_portfolio" in selected and "get_market_data" not in selected and holdings:
        log.info("SmartFanOut | injected get_market_data alongside analyze_portfolio")
        return ["analyze_portfolio", "get_market_data"]
    return selected


def _build_tool_calls(query: str, selected: list[str], holdings: dict) -> list[dict]:
    """Build tool_calls, expanding get_market_data to per-ticker calls for comparison/portfolio queries."""
    q = query.lower()
    is_comparison        = any(t in q for t in ["p/e", "pe ratio", "price-to-earnings", "compare", "versus", "vs"])
    is_portfolio_analysis = "analyze_portfolio" in selected and bool(holdings)

    if (is_comparison or is_portfolio_analysis) and "get_market_data" in selected and holdings:
        # Cap at 5 tickers to keep parallel calls manageable; sort by value (shares) descending
        top_n = sorted(holdings.items(), key=lambda x: x[1], reverse=True)[:5]
        if is_comparison:
            per_ticker = [
                {"name": "get_market_data", "args": {"query": f"P/E ratio and valuation for {t}"},
                 "id": f"call_md_{t}", "type": "tool_call"}
                for t, _ in top_n
            ] + [{"name": "get_market_data", "args": {"query": "S&P 500 SPY average P/E ratio valuation"},
                  "id": "call_md_SPY", "type": "tool_call"}]
        else:
            per_ticker = [
                {"name": "get_market_data", "args": {"query": f"current price and analysis for {t}"},
                 "id": f"call_md_{t}", "type": "tool_call"}
                for t, _ in top_n
            ]
        return [
            {"name": name, "args": {"query": query}, "id": f"call_{name}", "type": "tool_call"}
            for name in selected if name != "get_market_data"
        ] + per_ticker

    return [
        {"name": name, "args": {"query": query}, "id": f"call_{name}", "type": "tool_call"}
        for name in selected
    ]


def smart_fanout_node(state: FinnieState) -> dict:
    """Route to the relevant tools and emit parallel tool_calls for ToolNode to execute."""
    last_human = next(
        (m for m in reversed(state.get("messages", []))
         if hasattr(m, "type") and m.type == "human"),
        None,
    )
    query    = str(last_human.content) if last_human else ""
    holdings = state.get("portfolio_holdings") or {}

    query    = _enrich_query(state, query)
    ctx_note = _build_ctx_note(state)
    log.info("SmartFanOut | routing query=%r ctx=%s", query[:120], ctx_note[:80])

    selected = _select_tools(query, ctx_note)
    selected = _apply_goal_override(state, selected)
    selected = _apply_portfolio_override(selected, holdings)
    log.info("SmartFanOut | selected=%s | query=%r", selected, query[:60])

    return {"messages": [AIMessage(content="", tool_calls=_build_tool_calls(query, selected, holdings))]}


def synth_node(state: FinnieState) -> dict:
    """Synthesize all parallel tool results into one final answer."""
    recent = trim_messages(
        state["messages"],
        max_tokens=30,
        token_counter=len,
        strategy="last",
        start_on="human",
        include_system=False,
    )
    messages = [SystemMessage(content=_synth_prompt(state))] + recent
    log.debug("Synth | sending %d messages", len(messages))
    response = load_llm().invoke(messages)
    log.info("Synth | answer_len=%d", len(str(response.content)))
    return {"messages": [response]}


def build_all_graph():
    """Build and compile the smart parallel fan-out graph."""
    builder = StateGraph(FinnieState)

    builder.add_node("param_extractor", param_extractor_node)
    builder.add_node("smart_fanout",    smart_fanout_node)
    builder.add_node("tools",           ToolNode(TOOLS))
    builder.add_node("synth",           synth_node)

    builder.add_edge(START,             "param_extractor")
    builder.add_edge("param_extractor", "smart_fanout")
    builder.add_edge("smart_fanout",    "tools")
    builder.add_edge("tools",           "synth")
    builder.add_edge("synth",           END)

    return builder.compile(checkpointer=_MEMORY)


_all_graph = None


def _get_all_graph():
    global _all_graph
    if _all_graph is None:
        _all_graph = build_all_graph()
    return _all_graph


# ── Public API ────────────────────────────────────────────────────────────────

# Default values injected only on the first turn of a thread.
# Subsequent turns must NOT include these or LangGraph's LastValue channel
# will overwrite persisted checkpoint values (e.g. reset "aggressive" → "moderate").
_FIRST_TURN_DEFAULTS = {
    "risk_profile":        "moderate",
    "current_savings":     None,
    "goal_amount":         None,
    "time_horizon_years":  None,
    "annual_contribution": None,
    "portfolio_holdings":  None,
    "portfolio_value":     None,
    "age":                 None,
}


def invoke(query: str, thread_id: str = "default", _initial: dict | None = None) -> dict:
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
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
    log.info("Invoke | thread=%s | query=%r", thread_id[:8], query[:80])

    if _initial is not None:
        initial = _initial
    else:
        initial = {"messages": [HumanMessage(content=query)]}
        if not _get_graph().checkpointer.get(config):
            initial.update(_FIRST_TURN_DEFAULTS)

    try:
        result = _get_graph().invoke(initial, config=config)
    except GraphRecursionError:
        log.warning("Invoke | thread=%s | recursion limit hit — returning graceful error", thread_id[:8])
        return {
            "answer": (
                "I hit a complexity limit while researching your question. "
                "Please try rephrasing or breaking it into smaller parts — for example, "
                "ask about each stock separately."
            ),
            "messages": initial.get("messages", []),
        }

    # Last message is always the LLM's final answer
    last = result["messages"][-1]
    n_messages = len(result["messages"])
    log.info("Invoke | thread=%s | messages=%d | answer_len=%d",
             thread_id[:8], n_messages, len(str(last.content)))
    return {
        "answer":   str(last.content),
        "messages": result["messages"],
    }


def invoke_all(query: str, thread_id: str = "default", _initial: dict | None = None) -> dict:
    """
    Run one conversational turn through the smart parallel fan-out workflow.
    The LLM first selects only the relevant agents, then runs them in parallel
    and synthesizes the results into one answer.

    Args:
        query:     The user's message.
        thread_id: Session ID — all turns with the same ID share memory.

    Returns:
        {
            "answer":    str,   final synthesized answer
            "messages":  list,  full updated message history
        }
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    log.info("InvokeAll | thread=%s | query=%r", thread_id[:8], query[:80])

    if _initial is not None:
        initial = _initial
    else:
        initial = {"messages": [HumanMessage(content=query)]}
        if not _get_all_graph().checkpointer.get(config):
            initial.update(_FIRST_TURN_DEFAULTS)

    try:
        result = _get_all_graph().invoke(initial, config=config)
    except GraphRecursionError:
        log.warning("InvokeAll | thread=%s | recursion limit hit", thread_id[:8])
        return {
            "answer": (
                "I hit a complexity limit while researching your question. "
                "Please try rephrasing or breaking it into smaller parts."
            ),
            "messages":    initial.get("messages", []),
            "agents_used": [],
        }
    last = result["messages"][-1]

    # Collect tools called in THIS turn only (after the last HumanMessage)
    import re as _re
    msgs = result["messages"]
    last_human_idx = next(
        (i for i in range(len(msgs) - 1, -1, -1)
         if hasattr(msgs[i], "type") and msgs[i].type == "human"),
        0,
    )
    agents_used = [
        m.name for m in msgs[last_human_idx:]
        if isinstance(m, ToolMessage) and hasattr(m, "name")
    ]

    # Escape bare $ so Streamlit/MathJax doesn't treat currency as LaTeX math
    answer = _re.sub(r"(?<!\\)\$", r"\\$", str(last.content))

    log.info("InvokeAll | thread=%s | agents=%s | answer_len=%d",
             thread_id[:8], agents_used, len(answer))
    return {
        "answer":      answer,
        "messages":    result["messages"],
        "agents_used": agents_used,
    }


# ── Queries that need sequential reasoning (tool B uses tool A's output) ──────
#
# For these, the ReAct graph is used so the LLM can chain tool calls.
# Everything else goes through the faster parallel fan-out graph.

_SEQUENTIAL_QUERIES = (
    "p/e", "pe ratio", "price-to-earnings",
    "compare", "versus", "vs ",
    "rate hike", "rate sensitive", "interest rate", "vulnerable",
    "52-week", "52 week",
    "dividend",
    "at a loss", "trading at a loss", "tax benefit", "tax loss",
    "allocation", "too aggressive", "too conservative", "for my age",
    "which of my", "most vulnerable", "most exposed",
    "falls", "drops", "impact on my portfolio",
    "shares of", "retire after selling", "selling my shares",
    "if i sell", "when i sell", "want to sell", "going to sell",
    "sell all", "sell my", "planning to sell",
    "how much tax", "tax will i", "tax do i", "tax on selling",
    # Withdrawal / decumulation — need sequential tool chaining
    "withdraw", "withdrawal", "drawdown", "draw down",
    "how long will", "how long would", "how long can",
    "live off", "live on my", "retirement income",
    "how much can i take", "how much can i spend",
    "recalculate", "calculate with",
    # Retirement + portfolio goal — analyze_portfolio must run first to feed portfolio_value
    # into plan_financial_goal; parallel fanout cannot do this ordering
    "retire with", "retire in", "want to retire",
    "save for retirement", "retirement goal",
)


def chat(query: str, thread_id: str = "default") -> dict:
    """
    Single entry point for a conversational turn.

    Routes to the ReAct graph when the query requires sequential tool use
    (e.g. one tool's output feeds the next), or to the parallel fan-out
    graph for everything else — never both.

    Returns:
        {
            "answer":      str,   final answer, $ signs escaped for Streamlit
            "messages":    list,  full message history
            "agents_used": list,  tool names called this turn
        }
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial: dict = {"messages": [HumanMessage(content=query)]}
    if not _MEMORY.get(config):
        initial.update(_FIRST_TURN_DEFAULTS)

    if any(k in query.lower() for k in _SEQUENTIAL_QUERIES):
        result = invoke(query, thread_id=thread_id, _initial=initial)
        msgs = result["messages"]
        last_human_idx = next(
            (i for i in range(len(msgs) - 1, -1, -1)
             if hasattr(msgs[i], "type") and msgs[i].type == "human"),
            0,
        )
        agents_used = [
            m.name for m in msgs[last_human_idx:]
            if isinstance(m, ToolMessage) and hasattr(m, "name")
        ]
        answer = re.sub(r"(?<!\\)\$", r"\\$", result["answer"])
        return {**result, "answer": answer, "agents_used": agents_used}

    return invoke_all(query, thread_id=thread_id, _initial=initial)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tid = "demo"

    turns = [
       """I have 1000 AAPL, 500 MSFT, 300 GOOGL, 200 TSLA, and 800 NVDA stocks.
I want to retire in 20 years with $2 million and I currently have $100K saved.
Add the value of my portfolio to my savings and tell me if I'm on track to reach my goal. 
Also, what's the news on these stocks?,
I'm aggressive with risk.
can I actually reach my $2M goal, and can you explain how compound interest works?
"""
    ]

    for q in turns:
        print(f"\nUser  : {q}")
        r = invoke(q, thread_id=tid)
        print(f"Finnie: {r['answer'][:10000]}...")
        print("-" * 60)
