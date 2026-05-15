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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages

from src.core.llm import load_llm
from src.utils.logger import get_logger
from src.workflow.state import FinnieState
from src.workflow.tools import TOOLS
from src.workflow.prompts import _system_prompt

log = get_logger(__name__)


# ── Nodes ─────────────────────────────────────────────────────────────────────

_TICKER_RE = re.compile(r'\b([A-Z]{1,5})\s*[:\-]\s*(\d+(?:\.\d+)?)\b')


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

    # Extract portfolio holdings: patterns like "AAPL: 100" or "MSFT - 200"
    holdings = {m.group(1): float(m.group(2)) for m in _TICKER_RE.finditer(text)}
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
    "get_financial_news":      "Recent news headlines for specific stocks or tickers",
    "get_tax_education":       "Tax on investments: capital gains, IRA/Roth IRA, 401k, HSA, tax-loss harvesting",
}


class _ToolSelection(BaseModel):
    tools: list[str]


def smart_fanout_node(state: FinnieState) -> dict:
    """
    Use the LLM to select only the relevant agents for this query,
    then emit a single AIMessage with those tool_calls so ToolNode
    runs them in parallel.
    """
    last_human = next(
        (m for m in reversed(state.get("messages", []))
         if hasattr(m, "type") and m.type == "human"),
        None,
    )
    query    = str(last_human.content) if last_human else ""
    holdings = state.get("portfolio_holdings") or {}

    # If the query references "my holdings/portfolio" without listing tickers,
    # inject the remembered portfolio so tools can work with it.
    query_lower = query.lower()
    refs_portfolio = any(p in query_lower for p in ["my holding", "my portfolio", "my stock", "my top"])
    if refs_portfolio and holdings and not _TICKER_RE.search(query):
        ticker_list = ", ".join(f"{t}: {int(s)}" for t, s in holdings.items())
        query = f"Portfolio: {ticker_list}\n\nQuestion: {query}"

    # Build state context so the routing LLM understands ongoing conversations
    ctx_parts = []
    if state.get("goal_amount"):
        ctx_parts.append(f"goal ${state['goal_amount']:,.0f}")
    if state.get("time_horizon_years"):
        ctx_parts.append(f"timeline {state['time_horizon_years']:.0f} yr")
    if state.get("current_savings") is not None:
        ctx_parts.append(f"savings ${state['current_savings']:,.0f}")
    if state.get("risk_profile") and state.get("risk_profile") != "moderate":
        ctx_parts.append(f"risk {state['risk_profile']}")
    if state.get("portfolio_holdings"):
        ctx_parts.append(f"portfolio {list(state['portfolio_holdings'].keys())}")
    ctx_note = (
        f"\nConversation context already established: {', '.join(ctx_parts)}."
        if ctx_parts else ""
    )

    log.info("SmartFanOut | routing query=%r ctx=%s", query[:120], ctx_note[:80])

    tool_list = "\n".join(
        f"- {name}: {desc}" for name, desc in _TOOL_DESCRIPTIONS.items()
    )
    routing_prompt = (
        f"Select the tools needed to give a complete, well-rounded answer to this query.\n\n"
        f"Available tools:\n{tool_list}\n\n"
        f"Query: {query}{ctx_note}\n\n"
        "Rules (apply the FIRST matching rule and stop — do not stack rules):\n"
        "- News, headlines, or recent events for a specific stock or ticker → get_financial_news + get_market_data\n"
        "- General advice, tips, or education ('give me advice', 'tips for investing', 'best practices') → answer_finance_question + plan_financial_goal (only if goal is in context, else answer_finance_question + get_tax_education)\n"
        "- Any 'explain', 'what is', 'how does', 'what are' question (NOT about news or prices) → answer_finance_question + get_tax_education\n"
        "- If conversation context shows goal_amount + timeline are already known and this message "
        "  provides savings, contribution, or risk info → plan_financial_goal + get_tax_education\n"
        "- Portfolio questions (holdings, sectors, allocation, dividends, P/E, rate sensitivity) → analyze_portfolio + get_market_data\n"
        "- Retirement / savings goal questions → plan_financial_goal + get_tax_education\n"
        "- 'Is my allocation right for my age?' → analyze_portfolio + answer_finance_question\n"
        "- Rate hike / interest rate vulnerability → analyze_portfolio + answer_finance_question\n"
        "- Stock news or market events → get_financial_news + get_market_data\n"
        "- Tax questions (selling, gains, IRA, 401k) → get_tax_education + answer_finance_question\n"
        "- 52-week high, dividends, P/E for a specific stock → get_market_data + answer_finance_question\n"
        "- ALWAYS select exactly 2 tools. Never more than 2 unless the query explicitly mentions multiple stocks that each need individual price lookups.\n"
    )

    structured_llm = load_llm().with_structured_output(_ToolSelection)
    selection = structured_llm.invoke([HumanMessage(content=routing_prompt)])

    valid_names = {t.name for t in TOOLS}
    selected = [name for name in selection.tools if name in valid_names]
    if not selected:
        selected = ["answer_finance_question"]  # safe fallback

    # Hard override: if an active goal is in state, pull in plan_financial_goal for
    # follow-up messages like "I have savings of $100k" that don't mention the goal explicitly.
    # Skip when the LLM already picked a market-intent tool — the user is asking about
    # prices/news/portfolio, not updating their goal.
    _MARKET_INTENT = {"get_market_data", "get_financial_news", "analyze_portfolio"}
    if (state.get("goal_amount") or state.get("time_horizon_years")) and \
            "plan_financial_goal" not in selected and \
            not any(t in selected for t in _MARKET_INTENT):
        selected = ["plan_financial_goal"] + [s for s in selected if s != "get_financial_news"][:1]
        log.info("SmartFanOut | injected plan_financial_goal for active goal context")

    log.info("SmartFanOut | selected=%s | query=%r", selected, query[:60])

    # Replace the single get_market_data call with one call per ticker when:
    #   a) P/E / comparison query  → top 3 tickers + SPY benchmark
    #   b) Portfolio analysis query → top 3 tickers (detailed price + analysis)
    is_comparison     = any(t in query_lower for t in ["p/e", "pe ratio", "price-to-earnings", "compare", "versus", "vs"])
    is_portfolio_analysis = "analyze_portfolio" in selected and bool(holdings)

    if (is_comparison or is_portfolio_analysis) and "get_market_data" in selected and holdings:
        top3 = sorted(holdings.items(), key=lambda x: x[1], reverse=True)[:3]
        if is_comparison:
            per_ticker_calls = [
                {"name": "get_market_data",
                 "args": {"query": f"P/E ratio and valuation for {ticker}"},
                 "id": f"call_md_{ticker}", "type": "tool_call"}
                for ticker, _ in top3
            ]
            per_ticker_calls.append(
                {"name": "get_market_data",
                 "args": {"query": "S&P 500 SPY average P/E ratio valuation"},
                 "id": "call_md_SPY", "type": "tool_call"}
            )
        else:
            per_ticker_calls = [
                {"name": "get_market_data",
                 "args": {"query": f"current price and analysis for {ticker}"},
                 "id": f"call_md_{ticker}", "type": "tool_call"}
                for ticker, _ in top3
            ]
        tool_calls = [
            {"name": name, "args": {"query": query}, "id": f"call_{name}", "type": "tool_call"}
            for name in selected if name != "get_market_data"
        ] + per_ticker_calls
    else:
        tool_calls = [
            {"name": name, "args": {"query": query}, "id": f"call_{name}", "type": "tool_call"}
            for name in selected
        ]

    return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}


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
    messages = [SystemMessage(content=_system_prompt(state))] + recent
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
    log.info("Invoke | thread=%s | query=%r", thread_id[:8], query[:80])

    # Only pass defaults on the first turn — subsequent invokes must NOT pass
    # these fields or LangGraph's LastValue channel will overwrite the
    # persisted checkpoint values (e.g. resetting "aggressive" → "moderate").
    initial: dict = {"messages": [HumanMessage(content=query)]}
    if not _get_graph().checkpointer.get(config):
        initial.update({
            "risk_profile":       "moderate",
            "current_savings":    None,
            "goal_amount":        None,
            "time_horizon_years": None,
            "annual_contribution": None,
            "portfolio_holdings": None,
            "age":                None,
        })

    result = _get_graph().invoke(initial, config=config)

    # Last message is always the LLM's final answer
    last = result["messages"][-1]
    n_messages = len(result["messages"])
    log.info("Invoke | thread=%s | messages=%d | answer_len=%d",
             thread_id[:8], n_messages, len(str(last.content)))
    return {
        "answer":   str(last.content),
        "messages": result["messages"],
    }


def invoke_all(query: str, thread_id: str = "default") -> dict:
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
    config = {"configurable": {"thread_id": thread_id}}
    log.info("InvokeAll | thread=%s | query=%r", thread_id[:8], query[:80])

    initial: dict = {"messages": [HumanMessage(content=query)]}
    if not _get_all_graph().checkpointer.get(config):
        initial.update({
            "risk_profile":       "moderate",
            "current_savings":    None,
            "goal_amount":        None,
            "time_horizon_years": None,
            "annual_contribution": None,
            "portfolio_holdings": None,
            "age":                None,
        })

    result = _get_all_graph().invoke(initial, config=config)
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


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tid = "demo"

    turns = [
       """I have 10 AAPL, 5 MSFT, 3 GOOGL, 2 TSLA, and 8 NVDA stocks.
I want to retire in 20 years with $2 million and I currently have $100K saved.
I'm aggressive with risk.
can I actually reach my $2M goal, and can you explain how compound interest works?
"""
    ]

    for q in turns:
        print(f"\nUser  : {q}")
        r = invoke(q, thread_id=tid)
        print(f"Finnie: {r['answer'][:10000]}...")
        print("-" * 60)
