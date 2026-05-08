"""
tests/test_tax_agent_integration.py
Integration tests for TaxEducationAgent.

Run:
    uv run pytest tests/test_tax_agent_integration.py -v
"""

import pytest
from src.agents.tax_agent import (
    TaxEducationAgent,
    calc_capital_gains,
    calc_tax_loss,
    LONG_TERM_RATE,
    SHORT_TERM_RATE,
    ACCOUNT_LIMITS,
)


@pytest.fixture(scope="module")
def agent():
    return TaxEducationAgent()


# ── Initialization ────────────────────────────────────────────────────────────

def test_agent_loads(agent):
    """Agent initializes without error."""
    assert agent is not None


# ── Return shape ──────────────────────────────────────────────────────────────

def test_run_returns_required_keys(agent):
    """run() always returns answer, metrics, scenario, and error keys."""
    result = agent.run("I sold stock after 14 months with a $5,000 gain. I'm in the 22% bracket.")
    for key in ("answer", "metrics", "scenario", "error"):
        assert key in result, f"Missing key: {key}"


def test_error_is_none_on_valid_query(agent):
    result = agent.run("I sold AAPL after 2 years with a $10,000 gain.")
    assert result["error"] is None


# ── Scenario classification ───────────────────────────────────────────────────

def test_classifies_capital_gains(agent):
    result = agent.run("I sold MSFT after 8 months with a $3,000 gain.")
    assert result["scenario"] == "capital_gains"


def test_classifies_account_limits_roth(agent):
    result = agent.run("How much can I contribute to my Roth IRA this year?")
    assert result["scenario"] == "account_limits"


def test_classifies_account_limits_401k(agent):
    result = agent.run("What is the 401k contribution limit for 2024?")
    assert result["scenario"] == "account_limits"


def test_classifies_account_limits_hsa(agent):
    result = agent.run("What is the HSA contribution limit?")
    assert result["scenario"] == "account_limits"


def test_classifies_tax_loss(agent):
    result = agent.run("I have a $4,000 loss this year. Can I harvest it?")
    assert result["scenario"] == "tax_loss"


# ── Capital gains: unit math (no LLM/network) ─────────────────────────────────

def test_short_term_uses_ordinary_income_rate():
    m = calc_capital_gains(gain=5_000, holding_months=8, bracket="22%")
    assert m["holding_type"] == "short_term"
    assert m["tax_rate_pct"] == 22.0
    assert m["estimated_tax"] == 1_100.0


def test_long_term_uses_preferential_rate():
    m = calc_capital_gains(gain=10_000, holding_months=14, bracket="22%")
    assert m["holding_type"] == "long_term"
    assert m["tax_rate_pct"] == 15.0
    assert m["estimated_tax"] == 1_500.0


def test_long_term_zero_rate_for_low_bracket():
    m = calc_capital_gains(gain=8_000, holding_months=24, bracket="12%")
    assert m["tax_rate_pct"] == 0.0
    assert m["estimated_tax"] == 0.0


def test_long_term_20_pct_for_top_bracket():
    m = calc_capital_gains(gain=50_000, holding_months=18, bracket="37%")
    assert m["tax_rate_pct"] == 20.0
    assert m["estimated_tax"] == 10_000.0


def test_net_gain_equals_gain_minus_tax():
    m = calc_capital_gains(gain=5_000, holding_months=8, bracket="22%")
    assert abs(m["net_gain"] - (m["gain"] - m["estimated_tax"])) < 0.01


def test_exactly_12_months_is_long_term():
    m = calc_capital_gains(gain=5_000, holding_months=12, bracket="22%")
    assert m["holding_type"] == "long_term"


def test_11_months_is_short_term():
    m = calc_capital_gains(gain=5_000, holding_months=11, bracket="22%")
    assert m["holding_type"] == "short_term"


def test_capital_gains_metrics_has_required_keys():
    m = calc_capital_gains(gain=5_000, holding_months=10, bracket="22%")
    for key in ("gain", "holding_type", "holding_period_months",
                "income_bracket", "tax_rate_pct", "estimated_tax", "net_gain"):
        assert key in m, f"Missing key: {key}"


# ── Tax-loss harvesting: unit math (no LLM/network) ──────────────────────────

def test_loss_under_3k_fully_deductible():
    m = calc_tax_loss(loss=2_000, bracket="22%")
    assert m["deductible_this_year"] == 2_000.0
    assert m["carryforward_to_next"] == 0.0


def test_loss_over_3k_caps_at_3k():
    m = calc_tax_loss(loss=5_000, bracket="22%")
    assert m["deductible_this_year"] == 3_000.0
    assert m["carryforward_to_next"] == 2_000.0


def test_tax_saving_equals_deductible_times_rate():
    m = calc_tax_loss(loss=3_000, bracket="22%")
    assert abs(m["estimated_tax_saving"] - 660.0) < 0.01


def test_tax_loss_metrics_has_required_keys():
    m = calc_tax_loss(loss=4_000, bracket="24%")
    for key in ("total_loss", "deductible_this_year",
                "carryforward_to_next", "income_bracket", "estimated_tax_saving"):
        assert key in m, f"Missing key: {key}"


# ── Account limits ────────────────────────────────────────────────────────────

def test_roth_ira_limit_is_correct(agent):
    result = agent.run("What is the Roth IRA contribution limit?")
    assert result["metrics"].get("limit") == ACCOUNT_LIMITS["roth_ira"]["limit"]


def test_401k_limit_is_correct(agent):
    result = agent.run("What is the 401k contribution limit for 2024?")
    assert result["metrics"].get("limit") == ACCOUNT_LIMITS["401k"]["limit"]


def test_hsa_self_limit_is_correct(agent):
    result = agent.run("What is the HSA limit for individual coverage?")
    assert result["metrics"].get("limit_self") == ACCOUNT_LIMITS["hsa"]["limit_self"]


def test_account_metrics_includes_tax_treatment(agent):
    result = agent.run("Tell me about contributing to a Roth IRA.")
    assert "tax_treatment" in result["metrics"]
    assert len(result["metrics"]["tax_treatment"]) > 0


# ── NL parsing: capital gains ─────────────────────────────────────────────────

def test_parses_gain_from_query(agent):
    result = agent.run("I made a $7,500 gain after selling stock held for 6 months. I'm in the 24% bracket.")
    assert result["metrics"].get("gain") == 7_500.0


def test_parses_holding_period_months(agent):
    result = agent.run("I sold after 9 months with a $4,000 gain.")
    assert result["metrics"].get("holding_period_months") == 9


def test_parses_holding_period_years(agent):
    result = agent.run("I sold after 2 years with a $6,000 gain.")
    assert result["metrics"].get("holding_period_months") == 24


def test_parses_income_bracket(agent):
    result = agent.run("I sold TSLA after 10 months with a $5,000 gain. I'm in the 32% bracket.")
    assert result["metrics"].get("income_bracket") == "32%"


def test_k_suffix_amount_parsed(agent):
    result = agent.run("I sold with a $10k gain after 18 months. 22% bracket.")
    assert result["metrics"].get("gain") == 10_000.0


# ── NL parsing: tax-loss ──────────────────────────────────────────────────────

def test_parses_loss_amount(agent):
    result = agent.run("I have a $4,500 loss this year. Can I harvest it? 22% bracket.")
    assert result["metrics"].get("total_loss") == 4_500.0


def test_carryforward_correct_from_nl(agent):
    result = agent.run("I have a $5,000 loss this year. 22% bracket.")
    assert result["metrics"].get("carryforward_to_next") == 2_000.0


# ── Answer quality ────────────────────────────────────────────────────────────

def test_answer_is_non_empty_string(agent):
    result = agent.run("I sold stock after 14 months with a $5,000 gain.")
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


def test_answer_contains_disclaimer(agent):
    result = agent.run("I sold AAPL after 8 months with a $3,000 gain.")
    answer = result["answer"].lower()
    assert any(phrase in answer for phrase in [
        "not tax advice", "not a tax", "consult", "tax professional",
        "educational", "disclaimer",
    ])


def test_capital_gains_answer_mentions_short_or_long_term(agent):
    result = agent.run("I sold after 6 months with a $5,000 gain. 22% bracket.")
    answer = result["answer"].lower()
    assert any(w in answer for w in ["short-term", "short term", "long-term", "long term"])


def test_general_answer_is_non_empty(agent):
    result = agent.run("What is the wash sale rule?")
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


def test_general_metrics_is_empty_dict(agent):
    result = agent.run("What is the wash sale rule?")
    assert result["metrics"] == {}


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_query_returns_error(agent):
    result = agent.run("")
    assert result["error"] == "no_input"
    assert result["metrics"] == {}
    assert len(result["answer"]) > 0


def test_whitespace_only_query_returns_error(agent):
    result = agent.run("   ")
    assert result["error"] == "no_input"


def test_zero_gain_returns_zero_tax():
    m = calc_capital_gains(gain=0, holding_months=6, bracket="22%")
    assert m["estimated_tax"] == 0.0
    assert m["net_gain"] == 0.0


def test_zero_loss_returns_zero_saving():
    m = calc_tax_loss(loss=0, bracket="22%")
    assert m["estimated_tax_saving"] == 0.0
    assert m["carryforward_to_next"] == 0.0
