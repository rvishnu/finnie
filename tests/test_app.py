"""
tests/test_app.py
Unit tests for src/web_app/app.py helper functions and
integration tests using Streamlit's AppTest framework.

Unit tests are fast — no network, no LLM, no Streamlit runtime.
Integration tests run the full Streamlit script with mocked agents.

Run:
    uv run pytest tests/test_app.py -v
    uv run pytest tests/test_app.py -k "Unit" -v   # unit tests only
    uv run pytest tests/test_app.py -k "App" -v    # integration tests only
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

APP_PATH = str(Path(__file__).parent.parent / "src/web_app/app.py")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — pure helpers, no Streamlit runtime or network calls
# ═══════════════════════════════════════════════════════════════════════════════
# We import app.py with Streamlit mocked so the module-level st.* calls
# become no-ops and the pure helper functions can be tested in isolation.

def _make_st_mock():
    st = MagicMock()
    st.cache_resource = lambda fn: fn          # no-op decorator passthrough
    st.chat_input.return_value = None          # no pending chat message
    st.button.return_value = False             # no button pressed

    # text_input().strip().upper() must return "" so market lookup is skipped
    text_result = MagicMock()
    text_result.strip.return_value.upper.return_value = ""
    st.text_input.return_value = text_result

    # tabs / columns need to unpack correctly
    st.tabs.side_effect   = lambda labels:  [MagicMock() for _ in labels]
    st.columns.side_effect = lambda spec: (
        [MagicMock() for _ in range(spec)]
        if isinstance(spec, int)
        else [MagicMock() for _ in spec]
    )

    # session_state: fresh session — every "key in state" check returns False
    ss = MagicMock()
    ss.__contains__ = MagicMock(return_value=False)
    st.session_state = ss
    return st


@pytest.fixture(scope="function")
def helpers():
    """
    Import src.web_app.app with all external deps mocked.
    Returns the module so unit tests can call its pure helpers directly.
    """
    st_mock = _make_st_mock()
    mocks = {
        "streamlit":                  st_mock,
        "src.workflow.graph":         MagicMock(),
        "src.agents.portfolio_agent": MagicMock(),
        "src.agents.market_agent":    MagicMock(),
        "src.utils.market_tools":     MagicMock(),
    }
    saved = {k: sys.modules.get(k) for k in mocks}
    for k, v in mocks.items():
        sys.modules[k] = v
    sys.modules.pop("src.web_app.app", None)

    import src.web_app.app as app
    yield app

    sys.modules.pop("src.web_app.app", None)
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


# ── _fmt_large ────────────────────────────────────────────────────────────────

class TestFmtLarge:

    def test_trillion(self, helpers):
        assert helpers._fmt_large(2.5e12) == "$2.50T"

    def test_billion(self, helpers):
        assert helpers._fmt_large(3e9) == "$3.00B"

    def test_million(self, helpers):
        assert helpers._fmt_large(1.5e6) == "$1.50M"

    def test_small_number(self, helpers):
        assert helpers._fmt_large(1234) == "$1,234"

    def test_zero(self, helpers):
        assert helpers._fmt_large(0) == "$0"

    def test_none_returns_na(self, helpers):
        assert helpers._fmt_large(None) == "N/A"

    def test_string_na_returns_na(self, helpers):
        assert helpers._fmt_large("N/A") == "N/A"

    def test_numeric_string(self, helpers):
        assert helpers._fmt_large("2000000000") == "$2.00B"


# ── _change_html ──────────────────────────────────────────────────────────────

class TestChangeHtml:

    def test_positive_uses_price_up_class(self, helpers):
        assert "price-up" in helpers._change_html(1.5, 0.5)

    def test_positive_has_up_arrow(self, helpers):
        assert "▲" in helpers._change_html(1.5, 0.5)

    def test_positive_has_plus_sign(self, helpers):
        assert "+" in helpers._change_html(1.5, 0.5)

    def test_negative_uses_price_down_class(self, helpers):
        assert "price-down" in helpers._change_html(-1.5, -0.5)

    def test_negative_has_down_arrow(self, helpers):
        assert "▼" in helpers._change_html(-1.5, -0.5)

    def test_zero_uses_neutral_class(self, helpers):
        assert "price-neutral" in helpers._change_html(0.0, 0.0)

    def test_zero_has_no_arrow(self, helpers):
        html = helpers._change_html(0.0, 0.0)
        assert "▲" not in html
        assert "▼" not in html

    def test_output_is_html_span(self, helpers):
        html = helpers._change_html(1.0, 0.3)
        assert html.startswith("<span") and html.endswith("</span>")


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Streamlit AppTest (full script, mocked agents)
# ═══════════════════════════════════════════════════════════════════════════════

from streamlit.testing.v1 import AppTest

# Shared mock return values so agent calls don't hit the network
_PORTFOLIO_RESULT = {
    "answer":  "Portfolio looks balanced. Educational purposes only.",
    "metrics": {
        "total_value":           15000.0,
        "num_positions":         2,
        "diversification_score": 6,
        "holdings": [
            {"ticker": "AAPL", "name": "Apple Inc.",      "shares": 10,
             "price": 175.0, "position_value": 1750.0, "allocation_pct": 50.0,
             "sector": "Technology", "asset_type": "Stock"},
            {"ticker": "MSFT", "name": "Microsoft Corp.", "shares": 5,
             "price": 350.0, "position_value": 1750.0, "allocation_pct": 50.0,
             "sector": "Technology", "asset_type": "Stock"},
        ],
        "sector_pct": {"Technology": 100.0},
        "asset_pct":  {"Stock": 100.0},
    },
    "failed": [],
}

_MARKET_RESULT = {
    "answer": "AAPL is performing well. Educational purposes only.",
    "ticker": "AAPL",
    "source": "Yahoo Finance",
}

_MARKET_DATA = {
    "ticker": "AAPL", "name": "Apple Inc.", "price": 175.0,
    "change": 1.5, "change_pct": 0.86, "volume": 50_000_000,
    "high": 176.0, "low": 174.0, "prev_close": 173.5,
    "week_52_high": 200.0, "week_52_low": 140.0,
    "market_cap": 2.7e12, "pe_ratio": 28.5, "dividend_yield": 0.005,
    "sector": "Technology", "description": "Apple designs consumer electronics.",
    "source": "Yahoo Finance",
}

_CHAT_RESULT = {
    "answer":   "An index fund tracks a market index. Educational purposes only.",
    "messages": [],
}

# Helpers for the patches used by every integration test
_AGENT_PATCHES = [
    patch("src.web_app.app.chat_invoke",  return_value=_CHAT_RESULT),
    patch("src.web_app.app._portfolio_agent"),
    patch("src.web_app.app._market_agent"),
    patch("src.web_app.app._fetch_stock_info", return_value=_MARKET_DATA),
]


def _run_app() -> AppTest:
    """Run the app with all agent calls mocked and return the AppTest instance."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    with (
        patch("src.web_app.app.chat_invoke",    return_value=_CHAT_RESULT),
        patch("src.web_app.app._portfolio_agent") as mock_pa,
        patch("src.web_app.app._market_agent")    as mock_ma,
        patch("src.web_app.app._fetch_stock_info", return_value=None),
    ):
        mock_pa.return_value.run.return_value = _PORTFOLIO_RESULT
        mock_ma.return_value.run.return_value = _MARKET_RESULT
        at.run()
    return at


class TestAppIntegration:

    # ── Initial load ──────────────────────────────────────────────────────────

    def test_app_loads_without_exception(self):
        at = _run_app()
        assert not at.exception

    def test_session_state_has_required_keys(self):
        at = _run_app()
        for key in ("thread_id", "chat_history", "portfolio_result",
                    "market_data", "market_history", "market_analysis"):
            assert key in at.session_state, f"Missing session key: {key}"

    def test_initial_chat_history_is_empty(self):
        at = _run_app()
        assert at.session_state["chat_history"] == []

    def test_initial_portfolio_result_is_none(self):
        at = _run_app()
        assert at.session_state["portfolio_result"] is None

    def test_initial_market_data_is_none(self):
        at = _run_app()
        assert at.session_state["market_data"] is None

    def test_thread_id_is_non_empty_string(self):
        at = _run_app()
        assert isinstance(at.session_state["thread_id"], str)
        assert len(at.session_state["thread_id"]) > 0

    # ── New Conversation button ───────────────────────────────────────────────

    def test_new_conversation_resets_chat_history(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with (
            patch("src.web_app.app.chat_invoke",  return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            old_thread = at.session_state["thread_id"]
            next(b for b in at.button if "New Conversation" in b.label).click().run()
        assert at.session_state["chat_history"] == []
        assert at.session_state["thread_id"] != old_thread

    # ── Portfolio tab ─────────────────────────────────────────────────────────

    def test_portfolio_empty_holdings_shows_warning(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with (
            patch("src.web_app.app.chat_invoke",  return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            analyze_btn = next(b for b in at.button if "Analyze" in b.label)
            analyze_btn.click().run()
        assert len(at.warning) > 0

    def test_portfolio_invalid_holdings_shows_error(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with (
            patch("src.web_app.app.chat_invoke",  return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent") as mock_pa,
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            mock_pa.return_value.run.return_value = {"answer": "No holdings found.", "metrics": {}, "failed": []}
            at.run()
            at.text_area[0].input("!!! not valid holdings !!!")
            analyze_btn = next(b for b in at.button if "Analyze" in b.label)
            analyze_btn.click().run()
        assert len(at.error) > 0

    def test_portfolio_valid_holdings_stores_result(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with (
            patch("src.web_app.app.chat_invoke",  return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent") as mock_pa,
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            mock_pa.return_value.run.return_value = _PORTFOLIO_RESULT
            at.run()
            at.text_area[0].input("AAPL: 10, MSFT: 5")
            analyze_btn = next(b for b in at.button if "Analyze" in b.label)
            analyze_btn.click().run()
        assert at.session_state["portfolio_result"] is not None

    def test_portfolio_agent_called_with_query(self):
        import streamlit as st
        st.cache_resource.clear()   # evict any real agent cached by a prior test
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with (
            patch("src.workflow.graph.invoke",    return_value=_CHAT_RESULT),
            patch("src.agents.portfolio_agent.PortfolioAnalysisAgent") as mock_cls,
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            mock_cls.return_value.run.return_value = _PORTFOLIO_RESULT
            at.run()
            at.text_area[0].input("AAPL: 10, MSFT: 5")
            analyze_btn = next(b for b in at.button if "Analyze" in b.label)
            analyze_btn.click().run()
            call_kwargs = mock_cls.return_value.run.call_args[1]
        assert call_kwargs["query"] == "AAPL: 10, MSFT: 5"

    # ── Chat tab ──────────────────────────────────────────────────────────────

    def test_chat_calls_workflow_and_appends_history(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with (
            patch("src.workflow.graph.invoke",    return_value=_CHAT_RESULT) as mock_invoke,
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            at.chat_input[0].set_value("What is an index fund?").run()
        mock_invoke.assert_called_once()
        assert len(at.session_state["chat_history"]) == 2   # user + assistant

    def test_chat_history_has_correct_roles(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with (
            patch("src.web_app.app.chat_invoke",  return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            at.chat_input[0].set_value("What is an index fund?").run()
        history = at.session_state["chat_history"]
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
