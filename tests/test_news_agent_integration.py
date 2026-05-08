"""
tests/test_news_agent_integration.py
Integration tests for NewsSynthesizerAgent.

Run:
    uv run pytest tests/test_news_agent_integration.py -v
"""

import pytest
from src.agents.news_agent import (
    NewsSynthesizerAgent,
    BULLISH_WORDS,
    BEARISH_WORDS,
)


@pytest.fixture(scope="module")
def agent():
    return NewsSynthesizerAgent()


# ── Initialization ────────────────────────────────────────────────────────────

def test_agent_loads(agent):
    """Agent initializes without error."""
    assert agent is not None


# ── Return shape ──────────────────────────────────────────────────────────────

def test_run_returns_required_keys(agent):
    """run() always returns answer, headlines, tickers, and error keys."""
    result = agent.run("What's the latest news on AAPL?")
    for key in ("answer", "headlines", "tickers", "error"):
        assert key in result, f"Missing key: {key}"


def test_error_is_none_on_valid_query(agent):
    result = agent.run("What's the news on MSFT?")
    assert result["error"] is None


def test_headlines_is_list(agent):
    result = agent.run("Any news about AAPL?")
    assert isinstance(result["headlines"], list)


def test_tickers_is_list(agent):
    result = agent.run("What's happening with TSLA?")
    assert isinstance(result["tickers"], list)


# ── Ticker extraction ─────────────────────────────────────────────────────────

def test_single_ticker_extracted(agent):
    result = agent.run("Tell me the latest news on AAPL.")
    assert "AAPL" in result["tickers"]


def test_multiple_tickers_extracted(agent):
    result = agent.run("What's the news on AAPL and NVDA?")
    assert "AAPL" in result["tickers"]
    assert "NVDA" in result["tickers"]


def test_tickers_capped_at_max(agent):
    """Should not return more than MAX_TICKERS tickers."""
    result = agent.run("News on AAPL MSFT NVDA TSLA AMZN GOOGL META")
    assert len(result["tickers"]) <= 5


# ── Headline structure ────────────────────────────────────────────────────────

def test_each_headline_has_required_fields(agent):
    result = agent.run("What's the latest on AAPL?")
    for h in result["headlines"]:
        for field in ("ticker", "title", "publisher", "sentiment"):
            assert field in h, f"Headline missing field: {field}"


def test_headline_titles_are_non_empty(agent):
    result = agent.run("What's the news on MSFT?")
    for h in result["headlines"]:
        assert isinstance(h["title"], str)
        assert len(h["title"].strip()) > 0


def test_headline_tickers_are_uppercase(agent):
    result = agent.run("News about AAPL?")
    for h in result["headlines"]:
        assert h["ticker"] == h["ticker"].upper()


def test_no_duplicate_titles(agent):
    """Deduplication ensures no two headlines share the same title."""
    result = agent.run("What's the news on AAPL and MSFT?")
    titles = [h["title"].lower().strip() for h in result["headlines"]]
    assert len(titles) == len(set(titles))


# ── Sentiment tagging ─────────────────────────────────────────────────────────

def test_sentiment_values_are_valid(agent):
    """Every headline sentiment must be bullish, bearish, or neutral."""
    result = agent.run("Latest news on AAPL?")
    valid = {"bullish", "bearish", "neutral"}
    for h in result["headlines"]:
        assert h["sentiment"] in valid, f"Invalid sentiment: {h['sentiment']}"


def test_sentiment_bullish_for_positive_headline(agent):
    """Direct unit test on the _sentiment helper via keyword logic."""
    bullish_title = "Apple stock surges to record high after earnings beat"
    sentiment = agent._sentiment(bullish_title)
    assert sentiment == "bullish"


def test_sentiment_bearish_for_negative_headline(agent):
    bearish_title = "Tesla shares drop on missed delivery targets and loss warning"
    sentiment = agent._sentiment(bearish_title)
    assert sentiment == "bearish"


def test_sentiment_neutral_for_ambiguous_headline(agent):
    neutral_title = "Microsoft announces new product launch date for next quarter"
    sentiment = agent._sentiment(neutral_title)
    assert sentiment == "neutral"


def test_bullish_words_set_is_non_empty():
    assert len(BULLISH_WORDS) > 0


def test_bearish_words_set_is_non_empty():
    assert len(BEARISH_WORDS) > 0


def test_bullish_and_bearish_sets_are_disjoint():
    """No word should appear in both sets."""
    overlap = BULLISH_WORDS & BEARISH_WORDS
    assert len(overlap) == 0, f"Words in both sets: {overlap}"


# ── Answer quality ────────────────────────────────────────────────────────────

def test_answer_is_non_empty_string(agent):
    result = agent.run("What's the latest news on NVDA?")
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


def test_answer_contains_disclaimer(agent):
    result = agent.run("What's the news on AAPL?")
    answer = result["answer"].lower()
    assert any(phrase in answer for phrase in [
        "not financial advice", "not a financial advisor",
        "educational", "consult", "disclaimer",
    ])


def test_multi_ticker_answer_mentions_at_least_one_company(agent):
    """The briefing should reference at least one of the queried tickers."""
    result = agent.run("What's happening with AAPL and MSFT?")
    answer = result["answer"].upper()
    assert "AAPL" in answer or "MSFT" in answer or "APPLE" in answer or "MICROSOFT" in answer


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_query_returns_error(agent):
    result = agent.run("")
    assert result["error"] == "no_input"
    assert result["headlines"] == []
    assert result["tickers"] == []
    assert len(result["answer"]) > 0


def test_whitespace_query_returns_error(agent):
    result = agent.run("   ")
    assert result["error"] == "no_input"


def test_no_ticker_in_query_returns_no_tickers_error(agent):
    result = agent.run("what is happening in the world today")
    assert result["error"] in ("no_tickers", None)


def test_invalid_ticker_returns_no_news_or_graceful(agent):
    """Completely invalid ticker should return no_news error or empty headlines."""
    result = agent.run("News on XYZINVALID999")
    assert result["error"] in ("no_news", None)
    if result["error"] == "no_news":
        assert result["headlines"] == []


def test_answer_still_returned_when_some_tickers_have_no_news(agent):
    """When at least one ticker has news, a full answer should come back."""
    result = agent.run("What's the news on AAPL and XYZINVALID999?")
    if result["error"] is None:
        assert len(result["answer"].strip()) > 0
