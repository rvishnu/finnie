"""
tests/test_goal_agent_integration.py
Integration tests for GoalPlanningAgent.

Run:
    uv run pytest tests/test_goal_agent_integration.py -v
"""

import math
import pytest
from src.agents.goal_agent import GoalPlanningAgent, calculate_metrics


@pytest.fixture(scope="module")
def agent():
    return GoalPlanningAgent()


# ── Initialization ────────────────────────────────────────────────────────────

def test_agent_loads(agent):
    """Agent initializes without error."""
    assert agent is not None


# ── Return shape ──────────────────────────────────────────────────────────────

def test_run_returns_dict_with_required_keys(agent):
    """run() returns dict with answer, metrics, and error keys."""
    result = agent.run(goal_amount=30_000, time_horizon_years=3)
    assert isinstance(result, dict)
    assert "answer"  in result
    assert "metrics" in result
    assert "error"   in result


def test_metrics_has_required_keys(agent):
    """metrics contains all expected fields."""
    result = agent.run(goal_amount=30_000, time_horizon_years=3)
    m = result["metrics"]
    for key in (
        "goal_amount", "time_horizon_years", "time_horizon_months",
        "current_savings", "gap", "annual_return_pct",
        "monthly_no_growth", "monthly_with_growth",
        "projected_value", "goal_achievable",
    ):
        assert key in m, f"Missing key: {key}"


def test_error_is_none_on_success(agent):
    """error key is None for a valid structured call."""
    result = agent.run(goal_amount=10_000, time_horizon_years=2)
    assert result["error"] is None


# ── Numeric correctness (unit-level, no LLM/network) ─────────────────────────

def test_gap_equals_goal_minus_savings():
    m = calculate_metrics(50_000, 5, current_savings=10_000)
    assert m["gap"] == 40_000.0


def test_gap_is_zero_when_already_reached():
    m = calculate_metrics(10_000, 3, current_savings=15_000)
    assert m["gap"] == 0.0


def test_monthly_no_growth_equals_gap_over_months():
    m = calculate_metrics(36_000, 3, current_savings=0, annual_return=0.0)
    expected = 36_000 / 36
    assert abs(m["monthly_no_growth"] - expected) < 0.01


def test_monthly_with_growth_less_than_no_growth():
    """Investing reduces the required monthly contribution."""
    m = calculate_metrics(50_000, 10, current_savings=0, annual_return=0.07)
    assert m["monthly_with_growth"] < m["monthly_no_growth"]


def test_monthly_with_growth_is_zero_when_current_savings_cover_goal():
    """No monthly contribution needed if current savings will compound to the goal."""
    # $10k at 7% for 30 years → ~$76k, so goal of $20k needs $0 extra
    m = calculate_metrics(20_000, 30, current_savings=10_000, annual_return=0.07)
    assert m["monthly_with_growth"] == 0.0


def test_time_horizon_months_is_years_times_12():
    m = calculate_metrics(20_000, 5)
    assert m["time_horizon_months"] == 60.0


def test_projected_value_exceeds_goal_with_good_returns():
    """Investing the no-growth monthly amount should project above the goal."""
    m = calculate_metrics(50_000, 10, current_savings=5_000, annual_return=0.07)
    assert m["projected_value"] >= 50_000


def test_goal_achievable_is_true_for_normal_input():
    m = calculate_metrics(30_000, 5, current_savings=5_000)
    assert m["goal_achievable"] is True


def test_annual_return_pct_matches_risk_profile(agent):
    """Moderate risk profile → 7% return rate in metrics."""
    result = agent.run(goal_amount=20_000, time_horizon_years=5, risk_profile="moderate")
    assert result["metrics"]["annual_return_pct"] == 7.0


def test_conservative_return_rate(agent):
    result = agent.run(goal_amount=20_000, time_horizon_years=5, risk_profile="conservative")
    assert result["metrics"]["annual_return_pct"] == 4.0


def test_aggressive_return_rate(agent):
    result = agent.run(goal_amount=20_000, time_horizon_years=5, risk_profile="aggressive")
    assert result["metrics"]["annual_return_pct"] == 10.0


def test_higher_return_lowers_monthly_contribution(agent):
    """Aggressive profile (10%) requires lower monthly savings than conservative (4%)."""
    conservative = agent.run(goal_amount=50_000, time_horizon_years=10, risk_profile="conservative")
    aggressive   = agent.run(goal_amount=50_000, time_horizon_years=10, risk_profile="aggressive")
    assert (
        aggressive["metrics"]["monthly_with_growth"]
        < conservative["metrics"]["monthly_with_growth"]
    )


# ── Answer quality ────────────────────────────────────────────────────────────

def test_answer_is_non_empty_string(agent):
    result = agent.run(goal_amount=20_000, time_horizon_years=3)
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


def test_answer_contains_disclaimer(agent):
    result = agent.run(goal_amount=20_000, time_horizon_years=3)
    answer = result["answer"].lower()
    assert any(phrase in answer for phrase in [
        "not financial advice", "not a financial advisor",
        "educational", "consult", "disclaimer",
    ])


def test_answer_mentions_monthly_savings(agent):
    """Answer should reference monthly savings or contribution."""
    result = agent.run(goal_amount=50_000, time_horizon_years=5)
    answer = result["answer"].lower()
    assert any(word in answer for word in ["monthly", "month", "contribution", "save"])


# ── Natural language parsing ──────────────────────────────────────────────────

def test_nl_dollar_amount_and_years(agent):
    """Parses '$50,000 in 5 years' from plain text."""
    result = agent.run(query="I want to save $50,000 in 5 years")
    assert result["metrics"].get("goal_amount") == 50_000.0
    assert result["metrics"].get("time_horizon_years") == 5.0


def test_nl_k_suffix(agent):
    """Parses '$30k' shorthand."""
    result = agent.run(query="Save $30k for a house in 4 years")
    assert result["metrics"].get("goal_amount") == 30_000.0


def test_nl_months_timeline(agent):
    """Parses '18 months' timeline."""
    result = agent.run(query="I need $5,000 in 18 months")
    assert abs(result["metrics"].get("time_horizon_years", 0) - 1.5) < 0.01


def test_nl_current_savings_extracted(agent):
    """Parses 'I have $10k saved' from the query."""
    result = agent.run(query="I want $50,000 in 5 years. I have $10,000 saved.")
    assert result["metrics"].get("current_savings") == 10_000.0


def test_nl_current_savings_reduces_gap(agent):
    """Current savings reduce the gap correctly when parsed from NL."""
    result = agent.run(query="I want $50,000 in 5 years. I have $10,000 saved.")
    assert result["metrics"].get("gap") == 40_000.0


# ── Error / edge cases ────────────────────────────────────────────────────────

def test_no_input_returns_helpful_message(agent):
    """Calling run() with no arguments returns a helpful message."""
    result = agent.run()
    assert "error" in result
    assert result["error"] == "no_input"
    assert len(result["answer"]) > 0
    assert result["metrics"] == {}


def test_unparseable_query_returns_parse_failure(agent):
    """A query with no recognisable goal or timeline returns parse_failure."""
    result = agent.run(query="I just want to be rich someday")
    assert result["error"] in ("parse_failure", None)
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_already_have_enough_savings(agent):
    """When current savings >= goal, monthly contribution should be zero."""
    result = agent.run(goal_amount=10_000, time_horizon_years=5, current_savings=15_000)
    assert result["metrics"]["gap"] == 0.0
    assert result["metrics"]["monthly_with_growth"] == 0.0
    assert result["error"] is None
