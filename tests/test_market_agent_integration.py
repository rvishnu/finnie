"""
tests/test_market_agent_integration.py
Integration tests for MarketAnalysisAgent.

Run:
    uv run pytest tests/test_market_agent_integration.py -v
"""

import pytest
from src.agents.market_agent import MarketAnalysisAgent


@pytest.fixture(scope="module")
def agent():
    return MarketAnalysisAgent()


def test_agent_loads(agent):
    """Agent initializes without error."""
    assert agent is not None


def test_run_returns_dict(agent):
    """run() returns dict with required keys."""
    result = agent.run("How is Apple stock doing?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "ticker" in result
    assert "source" in result


def test_ticker_extraction_from_symbol(agent):
    """Extracts ticker from explicit symbol."""
    result = agent.run("Tell me about TSLA")
    assert result["ticker"] == "TSLA"


def test_ticker_extraction_from_company_name(agent):
    """Extracts ticker from company name."""
    result = agent.run("How is Microsoft doing?")
    assert result["ticker"] == "MSFT"


def test_answer_is_non_empty(agent):
    """Answer is a non-empty string."""
    result = agent.run("What is Apple's P/E ratio?")
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


def test_invalid_ticker_returns_helpful_message(agent):
    """Unknown ticker returns helpful error message."""
    result = agent.run("Tell me about XYZXYZXYZ stock")
    assert result["answer"] is not None
    assert len(result["answer"]) > 0


def test_no_ticker_in_query(agent):
    """Query with no ticker returns helpful prompt."""
    result = agent.run("How is the market doing?")
    assert result["ticker"] is None
    assert "ticker" in result["answer"].lower() or "symbol" in result["answer"].lower()


def test_source_is_valid(agent):
    """Source is either Yahoo Finance or Alpha Vantage."""
    result = agent.run("Tell me about AAPL")
    assert result["source"] in ("Yahoo Finance", "Alpha Vantage", None)


def test_answer_contains_disclaimer(agent):
    """Answer includes a disclaimer about financial advice."""
    result = agent.run("Should I buy AAPL stock?")
    answer = result["answer"].lower()
    assert any(phrase in answer for phrase in [
        "not financial advice", "not a financial advisor",
        "consult", "disclaimer", "not advice"
    ])
