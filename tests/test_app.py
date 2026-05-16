"""
tests/test_app.py
Tests for src/web_app/app.py.

Unit tests verify pure helper functions in isolation (no Streamlit runtime).
Integration tests run the full Streamlit script via AppTest with all
network/LLM calls mocked out.

Run:
    uv run pytest tests/test_app.py -v
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

APP_PATH = str(Path(__file__).parent.parent / "src/web_app/app.py")

# ─────────────────────────────────────────────────────────────────────────────
# Shared mock payloads
# ─────────────────────────────────────────────────────────────────────────────

_CHAT_RESULT = {
    "answer":      "An index fund tracks a market index.",
    "messages":    [],
    "agents_used": [],
}

_PORTFOLIO_RESULT = {
    "answer": "Portfolio looks balanced.",
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers fixture — imports app with all external deps mocked so pure
# helper functions (_fmt_large, _change_html) can be tested without a
# Streamlit runtime or any network calls.
# ─────────────────────────────────────────────────────────────────────────────

def _make_st_mock():
    st = MagicMock()
    st.cache_resource = lambda fn: fn
    st.cache_data     = lambda **kw: (lambda fn: fn)
    st.chat_input.return_value = None
    st.button.return_value     = False
    text_result = MagicMock()
    text_result.strip.return_value.upper.return_value = ""
    st.text_input.return_value = text_result
    st.tabs.side_effect    = lambda labels: [MagicMock() for _ in labels]
    st.columns.side_effect = lambda spec: (
        [MagicMock() for _ in range(spec)]
        if isinstance(spec, int)
        else [MagicMock() for _ in spec]
    )
    ss = MagicMock()
    ss.__contains__ = MagicMock(return_value=False)
    st.session_state = ss
    return st


@pytest.fixture(scope="function")
def helpers():
    st_mock = _make_st_mock()
    mocks = {
        "streamlit":                  st_mock,
        "streamlit.components.v1":    MagicMock(),
        "src.workflow.graph":         MagicMock(),
        "src.agents.portfolio_agent": MagicMock(),
        "src.agents.market_agent":    MagicMock(),
        "src.utils.market_tools":     MagicMock(),
        "src.utils.logger":           MagicMock(),
        "src.core.llm":               MagicMock(),
        "yfinance":                   MagicMock(),
        "pandas":                     MagicMock(),
        "plotly":                     MagicMock(),
        "plotly.express":             MagicMock(),
        "plotly.graph_objects":       MagicMock(),
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


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — _fmt_large
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — _change_html
# ─────────────────────────────────────────────────────────────────────────────

class TestChangeHtml:

    def test_positive_class(self, helpers):
        assert "price-up" in helpers._change_html(1.5, 0.5)

    def test_positive_arrow(self, helpers):
        assert "▲" in helpers._change_html(1.5, 0.5)

    def test_positive_sign(self, helpers):
        assert "+" in helpers._change_html(1.5, 0.5)

    def test_negative_class(self, helpers):
        assert "price-down" in helpers._change_html(-1.5, -0.5)

    def test_negative_arrow(self, helpers):
        assert "▼" in helpers._change_html(-1.5, -0.5)

    def test_zero_class(self, helpers):
        assert "price-neutral" in helpers._change_html(0.0, 0.0)

    def test_zero_no_arrow(self, helpers):
        html = helpers._change_html(0.0, 0.0)
        assert "▲" not in html and "▼" not in html

    def test_output_is_span(self, helpers):
        html = helpers._change_html(1.0, 0.3)
        assert html.startswith("<span") and html.endswith("</span>")


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — AppTest (full Streamlit script, agents mocked)
# ─────────────────────────────────────────────────────────────────────────────

from streamlit.testing.v1 import AppTest


def _run_app() -> AppTest:
    """Boot the app with all external calls mocked. No user interaction."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    with (
        patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
        patch("src.web_app.app._portfolio_agent") as mock_pa,
        patch("src.web_app.app._market_agent"),
        patch("src.web_app.app._fetch_stock_info", return_value=None),
    ):
        mock_pa.return_value.run.return_value = _PORTFOLIO_RESULT
        at.run()
    return at


class TestAppLoad:
    """App boots cleanly and initialises session state."""

    def test_no_exception_on_load(self):
        assert not _run_app().exception

    def test_required_session_keys_present(self):
        at = _run_app()
        for key in ("thread_id", "chat_history", "portfolio_result",
                    "market_data", "market_history", "market_analysis"):
            assert key in at.session_state, f"Missing session key: {key}"

    def test_chat_history_starts_empty(self):
        assert _run_app().session_state["chat_history"] == []

    def test_portfolio_result_starts_none(self):
        assert _run_app().session_state["portfolio_result"] is None

    def test_market_data_starts_none(self):
        assert _run_app().session_state["market_data"] is None

    def test_thread_id_is_non_empty_string(self):
        tid = _run_app().session_state["thread_id"]
        assert isinstance(tid, str) and len(tid) > 0


class TestNewConversation:
    """New Conversation button resets chat and issues a fresh thread id."""

    def test_resets_chat_history(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            next(b for b in at.button if "New Conversation" in b.label).click().run()
        assert at.session_state["chat_history"] == []

    def test_generates_new_thread_id(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            old_tid = at.session_state["thread_id"]
            next(b for b in at.button if "New Conversation" in b.label).click().run()
        assert at.session_state["thread_id"] != old_tid


class TestPortfolioTab:
    """Portfolio analysis tab — validation and result storage."""

    def _boot(self):
        return AppTest.from_file(APP_PATH, default_timeout=30)

    def test_empty_holdings_shows_warning(self):
        at = self._boot()
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            next(b for b in at.button if "Analyze" in b.label).click().run()
        assert len(at.warning) > 0

    def test_unrecognized_holdings_shows_error(self):
        at = self._boot()
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent") as mock_pa,
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            mock_pa.return_value.run.return_value = {
                "answer": "No holdings found.", "metrics": {}, "failed": []
            }
            at.run()
            at.text_area[0].input("!!! not valid !!!")
            next(b for b in at.button if "Analyze" in b.label).click().run()
        assert len(at.error) > 0

    def test_valid_holdings_stored_in_session(self):
        at = self._boot()
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent") as mock_pa,
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            mock_pa.return_value.run.return_value = _PORTFOLIO_RESULT
            at.run()
            at.text_area[0].input("AAPL: 10, MSFT: 5")
            next(b for b in at.button if "Analyze" in b.label).click().run()
        assert at.session_state["portfolio_result"] is not None

    def test_portfolio_agent_receives_user_query(self):
        import streamlit as st
        st.cache_resource.clear()
        at = self._boot()
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.agents.portfolio_agent.PortfolioAnalysisAgent") as mock_cls,
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            mock_cls.return_value.run.return_value = _PORTFOLIO_RESULT
            at.run()
            at.text_area[0].input("AAPL: 10, MSFT: 5")
            next(b for b in at.button if "Analyze" in b.label).click().run()
            kwargs = mock_cls.return_value.run.call_args[1]
        assert kwargs["query"] == "AAPL: 10, MSFT: 5"


class TestChatTab:
    """Chat tab — graph.invoke is called and history is updated correctly."""

    def _boot(self):
        return AppTest.from_file(APP_PATH, default_timeout=30)

    def test_graph_invoke_is_called(self):
        at = self._boot()
        with (
            patch("src.web_app.app.chat_invoke", return_value=_CHAT_RESULT) as mock_ci,
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            at.chat_input[0].set_value("What is an index fund?").run()
        mock_ci.assert_called_once()

    def test_chat_appends_two_messages(self):
        at = self._boot()
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            at.chat_input[0].set_value("What is an index fund?").run()
        assert len(at.session_state["chat_history"]) == 2

    def test_history_roles_are_user_then_assistant(self):
        at = self._boot()
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            at.chat_input[0].set_value("What is an index fund?").run()
        h = at.session_state["chat_history"]
        assert h[0]["role"] == "user"
        assert h[1]["role"] == "assistant"

    def test_user_message_content_preserved(self):
        at = self._boot()
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            at.chat_input[0].set_value("What is an index fund?").run()
        assert at.session_state["chat_history"][0]["content"] == "What is an index fund?"

    def test_assistant_answer_comes_from_graph(self):
        at = self._boot()
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            at.chat_input[0].set_value("What is an index fund?").run()
        assert _CHAT_RESULT["answer"] in at.session_state["chat_history"][1]["content"]

    def test_second_message_accumulates(self):
        at = self._boot()
        with (
            patch("src.web_app.app.chat_invoke",      return_value=_CHAT_RESULT),
            patch("src.web_app.app._portfolio_agent"),
            patch("src.web_app.app._market_agent"),
            patch("src.web_app.app._fetch_stock_info", return_value=None),
        ):
            at.run()
            at.chat_input[0].set_value("First question").run()
            at.chat_input[0].set_value("Second question").run()
        assert len(at.session_state["chat_history"]) == 4
