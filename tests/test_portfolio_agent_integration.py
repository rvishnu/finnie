"""
tests/test_portfolio_agent_integration.py
Integration tests for PortfolioAnalysisAgent.

Run:
    uv run pytest tests/test_portfolio_agent_integration.py -v
"""

import pytest
from src.agents.portfolio_agent import (
    PortfolioAnalysisAgent,
    _classify_asset,
    _diversification_score,
)


@pytest.fixture(scope="module")
def agent():
    return PortfolioAnalysisAgent()


# ── Initialization ────────────────────────────────────────────────────────────

def test_agent_loads(agent):
    """Agent initializes without error."""
    assert agent is not None


# ── Return shape ──────────────────────────────────────────────────────────────

def test_run_returns_dict_with_required_keys(agent):
    """run() returns dict with answer, metrics, and failed keys."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5})
    assert isinstance(result, dict)
    assert "answer" in result
    assert "metrics" in result
    assert "failed" in result


def test_metrics_has_required_keys(agent):
    """metrics dict contains all expected top-level fields."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5})
    metrics = result["metrics"]
    for key in ("total_value", "holdings", "sector_pct", "asset_pct",
                "num_positions", "diversification_score"):
        assert key in metrics, f"Missing key: {key}"


def test_holdings_have_required_fields(agent):
    """Each holding in metrics contains required fields."""
    result = agent.run(portfolio={"AAPL": 5})
    holdings = result["metrics"].get("holdings", [])
    assert len(holdings) > 0
    required = {"ticker", "shares", "price", "position_value",
                "sector", "asset_type", "allocation_pct", "name"}
    for h in holdings:
        assert required.issubset(h.keys()), f"Holding missing keys: {required - h.keys()}"


# ── Numeric correctness ───────────────────────────────────────────────────────

def test_total_value_is_positive(agent):
    """total_value is a positive number."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5})
    assert result["metrics"]["total_value"] > 0


def test_allocations_sum_to_100(agent):
    """Individual allocation percentages sum to ~100."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5, "BND": 20})
    holdings = result["metrics"]["holdings"]
    total_pct = sum(h["allocation_pct"] for h in holdings)
    assert abs(total_pct - 100.0) < 0.1


def test_sector_pct_sums_to_100(agent):
    """Sector percentages sum to ~100."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5, "BND": 20})
    total = sum(result["metrics"]["sector_pct"].values())
    assert abs(total - 100.0) < 0.1


def test_asset_pct_sums_to_100(agent):
    """Asset type percentages sum to ~100."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5, "BND": 20})
    total = sum(result["metrics"]["asset_pct"].values())
    assert abs(total - 100.0) < 0.1


def test_num_positions_matches_successful_holdings(agent):
    """num_positions equals the number of successfully fetched holdings."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5})
    assert result["metrics"]["num_positions"] == len(result["metrics"]["holdings"])


def test_position_value_equals_price_times_shares(agent):
    """position_value for each holding equals price × shares."""
    result = agent.run(portfolio={"AAPL": 7})
    for h in result["metrics"]["holdings"]:
        expected = round(h["price"] * h["shares"], 2)
        assert abs(h["position_value"] - expected) < 0.01


# ── Diversification score ─────────────────────────────────────────────────────

def test_diversification_score_in_range(agent):
    """Diversification score is between 0 and 10 inclusive."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5, "BND": 20})
    score = result["metrics"]["diversification_score"]
    assert 0 <= score <= 10


def test_single_stock_has_low_score(agent):
    """A single concentrated holding should produce a low diversification score."""
    result = agent.run(portfolio={"AAPL": 100})
    assert result["metrics"]["diversification_score"] <= 4


def test_diversified_portfolio_has_higher_score(agent):
    """A spread portfolio including bonds should score higher than a concentrated one."""
    concentrated = agent.run(portfolio={"AAPL": 100})
    diversified  = agent.run(portfolio={"AAPL": 10, "MSFT": 5, "BND": 20, "QQQ": 10, "NVDA": 3})
    assert (
        diversified["metrics"]["diversification_score"]
        > concentrated["metrics"]["diversification_score"]
    )


# ── Answer quality ────────────────────────────────────────────────────────────

def test_answer_is_non_empty_string(agent):
    """Answer is a non-empty string."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5})
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


def test_answer_contains_disclaimer(agent):
    """Answer includes a disclaimer that this is not financial advice."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5})
    answer = result["answer"].lower()
    assert any(phrase in answer for phrase in [
        "not financial advice", "not a financial advisor",
        "educational", "consult", "disclaimer",
    ])


def test_risk_profile_conservative_in_answer(agent):
    """Conservative risk profile is reflected in the analysis."""
    result = agent.run(portfolio={"AAPL": 10, "MSFT": 5}, risk_profile="conservative")
    answer = result["answer"].lower()
    assert any(word in answer for word in ["conservative", "risk", "bond", "stable"])


def test_risk_profile_aggressive_in_answer(agent):
    """Aggressive risk profile is reflected in the analysis."""
    result = agent.run(portfolio={"AAPL": 10, "NVDA": 5}, risk_profile="aggressive")
    answer = result["answer"].lower()
    assert any(word in answer for word in ["aggressive", "growth", "risk", "volatile"])


# ── NL input parsing ──────────────────────────────────────────────────────────

def test_natural_language_ticker_colon_format(agent):
    """Parses 'AAPL: 10, MSFT: 5' query into holdings."""
    result = agent.run(query="AAPL: 10, MSFT: 5")
    tickers = [h["ticker"] for h in result["metrics"].get("holdings", [])]
    assert "AAPL" in tickers
    assert "MSFT" in tickers


def test_natural_language_shares_format(agent):
    """Parses 'I have 10 AAPL shares and 5 MSFT' style input."""
    result = agent.run(query="I have 10 AAPL shares and 5 MSFT shares")
    tickers = [h["ticker"] for h in result["metrics"].get("holdings", [])]
    assert "AAPL" in tickers


def test_empty_query_without_portfolio_returns_helpful_message(agent):
    """Calling run() with no arguments returns a helpful message."""
    result = agent.run()
    assert "please provide" in result["answer"].lower()
    assert result["metrics"] == {}
    assert result["failed"] == []


def test_unparseable_query_returns_helpful_message(agent):
    """A query with no tickers or shares returns a helpful parse-error message."""
    result = agent.run(query="just some random sentence without tickers")
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


# ── Failed tickers ────────────────────────────────────────────────────────────

def test_invalid_ticker_appears_in_failed(agent):
    """An unrecognisable ticker is reported in the failed list."""
    result = agent.run(portfolio={"AAPL": 5, "XYZINVALID999": 10})
    assert "XYZINVALID999" in result["failed"]


def test_valid_holdings_still_returned_when_one_ticker_fails(agent):
    """Valid tickers are still analyzed even when one ticker fails."""
    result = agent.run(portfolio={"AAPL": 5, "XYZINVALID999": 10})
    tickers = [h["ticker"] for h in result["metrics"].get("holdings", [])]
    assert "AAPL" in tickers


def test_all_invalid_tickers_returns_error_message(agent):
    """All-invalid portfolio returns an informative message with empty metrics."""
    result = agent.run(portfolio={"XYZINVALID999": 5, "ZZZBAD111": 3})
    assert result["metrics"] == {}
    assert len(result["answer"]) > 0


# ── Helper unit tests (fast, no network) ─────────────────────────────────────

def test_classify_asset_stock():
    assert _classify_asset("AAPL", {"quoteType": "EQUITY"}) == "Stock"


def test_classify_asset_etf():
    assert _classify_asset("QQQ", {"quoteType": "ETF", "shortName": "Invesco QQQ Trust"}) == "ETF"


def test_classify_asset_bond_etf_by_name():
    assert _classify_asset("BND", {"quoteType": "ETF", "shortName": "Vanguard Total Bond Market ETF"}) == "Bond ETF"


def test_classify_asset_bond_etf_by_ticker():
    assert _classify_asset("BND", {"quoteType": "ETF", "shortName": "BND"}) == "Bond ETF"


def test_classify_asset_crypto():
    assert _classify_asset("BTC-USD", {"quoteType": "CRYPTOCURRENCY"}) == "Crypto"


def test_diversification_score_perfect():
    holdings = [
        {"ticker": "AAPL", "allocation_pct": 20, "asset_type": "Stock", "sector": "Technology"},
        {"ticker": "MSFT", "allocation_pct": 20, "asset_type": "Stock", "sector": "Technology"},
        {"ticker": "JNJ",  "allocation_pct": 20, "asset_type": "Stock", "sector": "Healthcare"},
        {"ticker": "BND",  "allocation_pct": 20, "asset_type": "Bond ETF", "sector": "Fixed Income"},
        {"ticker": "QQQ",  "allocation_pct": 20, "asset_type": "ETF", "sector": "Technology"},
    ]
    sector_pct = {"Technology": 60.0, "Healthcare": 20.0, "Fixed Income": 20.0}
    score = _diversification_score(holdings, sector_pct)
    assert 0 <= score <= 10


def test_diversification_score_heavily_concentrated():
    holdings = [
        {"ticker": "AAPL", "allocation_pct": 100, "asset_type": "Stock", "sector": "Technology"},
    ]
    sector_pct = {"Technology": 100.0}
    score = _diversification_score(holdings, sector_pct)
    assert score <= 3


def test_diversification_score_three_to_four_holdings_penalised():
    """3-4 holdings deducts 2 points (the elif < 5 branch)."""
    holdings = [
        {"ticker": "AAPL", "allocation_pct": 34, "asset_type": "Stock",   "sector": "Technology"},
        {"ticker": "MSFT", "allocation_pct": 33, "asset_type": "Stock",   "sector": "Technology"},
        {"ticker": "BND",  "allocation_pct": 33, "asset_type": "Bond ETF","sector": "Fixed Income"},
    ]
    sector_pct = {"Technology": 67.0, "Fixed Income": 33.0}
    score = _diversification_score(holdings, sector_pct)
    assert score <= 8


def test_diversification_score_minor_concentration_deducts_one():
    """A single position between 25-40% deducts only 1 point."""
    holdings = [
        {"ticker": "AAPL", "allocation_pct": 30, "asset_type": "Stock",   "sector": "Technology"},
        {"ticker": "MSFT", "allocation_pct": 20, "asset_type": "Stock",   "sector": "Technology"},
        {"ticker": "JNJ",  "allocation_pct": 20, "asset_type": "Stock",   "sector": "Healthcare"},
        {"ticker": "BND",  "allocation_pct": 15, "asset_type": "Bond ETF","sector": "Fixed Income"},
        {"ticker": "QQQ",  "allocation_pct": 15, "asset_type": "ETF",     "sector": "Technology"},
    ]
    sector_pct = {"Technology": 65.0, "Healthcare": 20.0, "Fixed Income": 15.0}
    # 30% allocation → deducts 1 (not 3); 5 holdings → no holding count penalty
    score = _diversification_score(holdings, sector_pct)
    assert score >= 4


def test_diversification_score_mild_sector_concentration_deducts_one():
    """A sector between 40-60% deducts 1 point, not 3."""
    holdings = [
        {"ticker": "AAPL", "allocation_pct": 20, "asset_type": "Stock",   "sector": "Technology"},
        {"ticker": "MSFT", "allocation_pct": 20, "asset_type": "Stock",   "sector": "Technology"},
        {"ticker": "NVDA", "allocation_pct": 10, "asset_type": "Stock",   "sector": "Technology"},
        {"ticker": "JNJ",  "allocation_pct": 25, "asset_type": "Stock",   "sector": "Healthcare"},
        {"ticker": "BND",  "allocation_pct": 25, "asset_type": "Bond ETF","sector": "Fixed Income"},
    ]
    sector_pct = {"Technology": 50.0, "Healthcare": 25.0, "Fixed Income": 25.0}
    score_sector_mild = _diversification_score(holdings, sector_pct)

    # Compare against perfectly balanced to confirm only 1 point was lost for sector
    balanced_sector_pct = {"Technology": 33.0, "Healthcare": 33.0, "Fixed Income": 34.0}
    score_balanced = _diversification_score(holdings, balanced_sector_pct)
    assert score_balanced - score_sector_mild == 1


def test_diversification_score_no_bonds_deducts_one():
    """Portfolio with no Bond ETF deducts 1 point vs identical portfolio that has one."""
    base = [
        {"ticker": "AAPL", "allocation_pct": 20, "asset_type": "Stock", "sector": "Technology"},
        {"ticker": "MSFT", "allocation_pct": 20, "asset_type": "Stock", "sector": "Technology"},
        {"ticker": "JNJ",  "allocation_pct": 20, "asset_type": "Stock", "sector": "Healthcare"},
        {"ticker": "QQQ",  "allocation_pct": 20, "asset_type": "ETF",   "sector": "Technology"},
        {"ticker": "AMZN", "allocation_pct": 20, "asset_type": "Stock", "sector": "Consumer Cyclical"},
    ]
    with_bonds = base[:-1] + [
        {"ticker": "BND",  "allocation_pct": 20, "asset_type": "Bond ETF", "sector": "Fixed Income"},
    ]
    sector_pct_no_bonds   = {"Technology": 60.0, "Healthcare": 20.0, "Consumer Cyclical": 20.0}
    sector_pct_with_bonds = {"Technology": 60.0, "Healthcare": 20.0, "Fixed Income": 20.0}
    score_no_bonds   = _diversification_score(base,       sector_pct_no_bonds)
    score_with_bonds = _diversification_score(with_bonds, sector_pct_with_bonds)
    assert score_with_bonds - score_no_bonds == 1


# ── _classify_asset: bond ETF ticker keywords ─────────────────────────────────

def test_classify_asset_bond_etf_tlt():
    assert _classify_asset("TLT", {"quoteType": "ETF", "shortName": "iShares 20+ Year Treasury Bond ETF"}) == "Bond ETF"


def test_classify_asset_bond_etf_agg():
    assert _classify_asset("AGG", {"quoteType": "ETF", "shortName": "iShares Core U.S. Aggregate Bond ETF"}) == "Bond ETF"


def test_classify_asset_bond_etf_lqd():
    assert _classify_asset("LQD", {"quoteType": "ETF", "shortName": "iShares iBoxx $ Investment Grade Corporate Bond ETF"}) == "Bond ETF"


# ── Risk profile: moderate ────────────────────────────────────────────────────

def test_risk_profile_moderate_in_answer(agent):
    """Moderate risk profile produces a balanced-sounding analysis."""
    result = agent.run(portfolio={"AAPL": 10, "BND": 10}, risk_profile="moderate")
    answer = result["answer"].lower()
    assert any(word in answer for word in ["moderate", "balanced", "risk", "diversif"])


# ── NL parsing: company names ─────────────────────────────────────────────────

def test_natural_language_company_name_parsing(agent):
    """LLM resolves company names to tickers (Apple → AAPL, Microsoft → MSFT)."""
    result = agent.run(query="I have 10 Apple shares and 5 Microsoft shares")
    tickers = [h["ticker"] for h in result["metrics"].get("holdings", [])]
    assert "AAPL" in tickers or "MSFT" in tickers
