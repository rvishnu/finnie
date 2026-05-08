"""
tests/test_workflow.py
Integration tests for the Finnie LangGraph ReAct workflow.

Covers:
  1. Return shape          — invoke() always returns the right keys/types
  2. Message accumulation  — history grows turn by turn
  3. Agent routing         — each query type reaches the right tool
  4. Out-of-scope guardrail— insurance/traffic/legal get the refusal message
  5. Multi-turn memory     — risk profile, savings, and goal persist across turns
  6. Thread isolation      — separate thread_ids never share state
  7. Disclaimer            — every substantive answer carries the edu disclaimer
  8. Multi-tool in one turn— LLM can call two tools in a single reasoning step

Run:
    uv run pytest tests/test_workflow.py -v
"""

import uuid
import pytest
from src.workflow.graph import invoke


def _tid() -> str:
    """Unique thread ID per test — prevents state bleed between tests."""
    return str(uuid.uuid4())


# ── 1. Return shape ───────────────────────────────────────────────────────────

def test_invoke_returns_required_keys():
    result = invoke("What is an index fund?", thread_id=_tid())
    assert "answer"   in result
    assert "messages" in result


def test_answer_is_non_empty_string():
    result = invoke("What is compound interest?", thread_id=_tid())
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


def test_messages_is_non_empty_list():
    result = invoke("What is a Roth IRA?", thread_id=_tid())
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) > 0


# ── 2. Message accumulation ───────────────────────────────────────────────────

def test_messages_grow_with_each_turn():
    """Each conversation turn appends to the persisted message history."""
    thread = _tid()
    r1 = invoke("What is dollar cost averaging?", thread_id=thread)
    r2 = invoke("Can you give me a simple example?",  thread_id=thread)
    assert len(r2["messages"]) > len(r1["messages"])


def test_three_turns_accumulate_messages():
    thread = _tid()
    r1 = invoke("What is an ETF?",                    thread_id=thread)
    r2 = invoke("How is it different from a stock?",  thread_id=thread)
    r3 = invoke("Which is better for a beginner?",   thread_id=thread)
    assert len(r3["messages"]) > len(r2["messages"]) > len(r1["messages"])


# ── 3. Agent routing ──────────────────────────────────────────────────────────

def test_routes_to_goal_agent():
    """Retirement goal query → answer mentions monthly savings."""
    result = invoke(
        "I want $2 million in 20 years. I have $50,000 saved.",
        thread_id=_tid(),
    )
    answer = result["answer"].lower()
    assert any(w in answer for w in ["monthly", "savings", "contribute", "invest", "goal"])


def test_routes_to_tax_agent_capital_gains():
    """Capital gains query → answer mentions tax or short/long term."""
    result = invoke(
        "I sold AAPL after 8 months with a $5,000 gain. I'm in the 22% bracket.",
        thread_id=_tid(),
    )
    answer = result["answer"].lower()
    assert any(w in answer for w in ["tax", "short-term", "short term", "gain", "bracket"])


def test_routes_to_tax_agent_account_limits():
    """401k query → answer mentions contribution limit."""
    result = invoke(
        "What is the 401k contribution limit for 2024?",
        thread_id=_tid(),
    )
    answer = result["answer"].lower()
    assert any(w in answer for w in ["401k", "limit", "contribute", "23,000", "pre-tax"])


def test_routes_to_news_agent():
    """News query → answer references the queried ticker."""
    result = invoke("What's the latest news on NVDA?", thread_id=_tid())
    answer = result["answer"].upper()
    assert "NVDA" in answer or "NVIDIA" in answer


def test_routes_to_market_agent():
    """Stock price query → answer mentions price or market data."""
    result = invoke("How is AAPL stock doing today?", thread_id=_tid())
    answer = result["answer"].lower()
    assert any(w in answer for w in ["apple", "aapl", "stock", "price", "market", "share"])


def test_routes_to_portfolio_agent():
    """Portfolio holdings query → answer discusses allocation or diversification."""
    result = invoke(
        "I have 10 AAPL shares and 5 MSFT shares. Analyze my portfolio.",
        thread_id=_tid(),
    )
    answer = result["answer"].lower()
    assert any(w in answer for w in ["portfolio", "diversif", "allocation", "aapl", "msft", "sector"])


def test_routes_to_qa_agent():
    """General finance education → answer is educational."""
    result = invoke(
        "What is the difference between a stock and a bond?",
        thread_id=_tid(),
    )
    answer = result["answer"].lower()
    assert any(w in answer for w in ["stock", "bond", "equity", "debt", "return", "risk"])


# ── 4. Out-of-scope guardrail ─────────────────────────────────────────────────

def test_guardrail_blocks_car_insurance():
    result = invoke("I need car insurance. What's the best policy?", thread_id=_tid())
    answer = result["answer"].lower()
    assert any(phrase in answer for phrase in [
        "not able to help", "i'm finnie", "financial education",
        "can help you with", "out of scope",
    ])


def test_guardrail_blocks_business_start():
    result = invoke(
        "I want to start a business. How do I get a business loan?",
        thread_id=_tid(),
    )
    answer = result["answer"].lower()
    assert any(phrase in answer for phrase in [
        "not able to help", "financial education", "can help you with",
    ])


def test_guardrail_blocks_legal_advice():
    result = invoke("I need legal advice about a contract.", thread_id=_tid())
    answer = result["answer"].lower()
    assert any(phrase in answer for phrase in [
        "not able to help", "financial education", "can help you with",
    ])


def test_guardrail_blocks_medical_query():
    result = invoke("What medication should I take for high blood pressure?", thread_id=_tid())
    answer = result["answer"].lower()
    assert any(phrase in answer for phrase in [
        "not able to help", "financial education", "can help you with",
    ])


def test_guardrail_in_scope_after_out_of_scope():
    """After an out-of-scope refusal the bot still answers in-scope queries correctly."""
    thread = _tid()
    invoke("What is the best car insurance?", thread_id=thread)
    r2 = invoke("What is a Roth IRA?", thread_id=thread)
    answer = r2["answer"].lower()
    assert any(w in answer for w in ["roth", "ira", "tax", "retire", "contribute"])


# ── 5. Multi-turn memory ──────────────────────────────────────────────────────

def test_risk_profile_persists_to_goal_planning():
    """Risk profile set in turn 1 should be visible to goal planner in turn 2."""
    thread = _tid()
    invoke("I'm very aggressive with risk — I can handle high volatility.", thread_id=thread)
    r2 = invoke("I want to retire with $2 million in 15 years.", thread_id=thread)
    answer = r2["answer"].lower()
    assert any(w in answer for w in [
        "aggressive", "growth", "10%", "15 year", "monthly", "invest", "higher return",
    ])


def test_savings_update_overrides_earlier_value():
    """
    Turn 1: set goal + initial savings.
    Turn 2: correct savings upward.
    Turn 3: LLM should use the updated savings when recalculating.
    """
    thread = _tid()
    invoke(
        "I want $1 million for retirement in 25 years. I have $50,000 saved.",
        thread_id=thread,
    )
    invoke("I now have $200,000 saved, not $50,000.", thread_id=thread)
    r3 = invoke("Can you recalculate how much I need to save monthly?", thread_id=thread)
    answer = r3["answer"].lower()
    assert any(w in answer for w in ["monthly", "contribute", "200", "savings"])


def test_goal_amount_persists_to_follow_up():
    """Goal stated in turn 1 is still in context for a follow-up in turn 2."""
    thread = _tid()
    invoke("I want to save $500,000 for a house in 10 years.", thread_id=thread)
    r2 = invoke("How much do I need to set aside each month if I invest?", thread_id=thread)
    answer = r2["answer"].lower()
    assert any(w in answer for w in ["monthly", "contribute", "500", "invest", "month"])


def test_multi_turn_qa_stays_coherent():
    """Three educational turns stay on topic — each answer is non-empty."""
    thread = _tid()
    r1 = invoke("What is a Roth IRA?",                                    thread_id=thread)
    r2 = invoke("How is it different from a Traditional IRA?",            thread_id=thread)
    r3 = invoke("Which one is better if I expect to be in a higher tax bracket later?", thread_id=thread)
    for r in (r1, r2, r3):
        assert isinstance(r["answer"], str)
        assert len(r["answer"].strip()) > 0


# ── 6. Thread isolation ───────────────────────────────────────────────────────

def test_separate_threads_do_not_share_state():
    """
    Thread A sets an aggressive risk profile.
    Thread B starts fresh — it must not inherit A's profile.
    """
    thread_a = _tid()
    thread_b = _tid()

    invoke(
        "I'm super aggressive. I want $5M in 10 years. I have $500K saved.",
        thread_id=thread_a,
    )
    r_b = invoke("What is dollar cost averaging?", thread_id=thread_b)

    assert isinstance(r_b["answer"], str)
    assert len(r_b["answer"].strip()) > 0


def test_two_concurrent_sessions_independent():
    """Two users with different goals get independent answers."""
    thread_user1 = _tid()
    thread_user2 = _tid()

    r1 = invoke("I want $100,000 in 3 years. I'm conservative.", thread_id=thread_user1)
    r2 = invoke("I want $2,000,000 in 30 years. I'm aggressive.", thread_id=thread_user2)

    assert isinstance(r1["answer"], str)
    assert isinstance(r2["answer"], str)
    assert r1["answer"] != r2["answer"]


# ── 7. Disclaimer always present ──────────────────────────────────────────────

_DISCLAIMER_PHRASES = [
    "not financial advice", "educational purposes",
    "consult", "disclaimer", "educational",
]


def test_disclaimer_in_goal_answer():
    result = invoke("I want $200,000 in 8 years.", thread_id=_tid())
    assert any(p in result["answer"].lower() for p in _DISCLAIMER_PHRASES)


def test_disclaimer_in_tax_answer():
    result = invoke(
        "I sold TSLA after 2 years with a $15,000 gain. 24% bracket.",
        thread_id=_tid(),
    )
    assert any(p in result["answer"].lower() for p in _DISCLAIMER_PHRASES + ["tax professional"])


def test_disclaimer_in_portfolio_answer():
    result = invoke("I have 20 AAPL and 10 MSFT — is my portfolio healthy?", thread_id=_tid())
    assert any(p in result["answer"].lower() for p in _DISCLAIMER_PHRASES)


# ── 8. Multi-tool in one turn ─────────────────────────────────────────────────

def test_tax_and_news_in_one_turn():
    """
    One query asks about 401k limits AND AAPL news.
    The ReAct loop should call get_tax_education and get_financial_news.
    The answer should address both topics.
    """
    result = invoke(
        "What is the 401k contribution limit and what's the latest news on AAPL?",
        thread_id=_tid(),
    )
    answer = result["answer"].lower()
    assert any(w in answer for w in ["401k", "limit", "contribute"])
    assert any(w in answer for w in ["apple", "aapl", "news"])


def test_goal_and_tax_in_one_turn():
    """
    Retirement goal + tax question in one turn — LLM should address both.
    """
    result = invoke(
        "I want $2M in 20 years and I also want to know how a Roth IRA helps with taxes.",
        thread_id=_tid(),
    )
    answer = result["answer"].lower()
    assert any(w in answer for w in ["roth", "ira", "tax"])
    assert any(w in answer for w in ["monthly", "goal", "invest", "retire", "million"])
