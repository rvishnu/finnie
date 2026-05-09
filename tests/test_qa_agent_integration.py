"""
tests/test_qa_agent_integration.py
Integration tests for FinanceQAAgent.

Run:
    uv run pytest tests/test_qa_agent_integration.py -v
"""

import pytest
from src.agents.qa_agent import FinanceQAAgent


@pytest.fixture(scope="module")
def agent():
    return FinanceQAAgent()


def test_agent_loads(agent):
    """Agent initializes without error."""
    assert agent is not None


def test_run_returns_dict(agent):
    """run() returns a dict with answer and citations keys."""
    result = agent.run("What is dollar cost averaging?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "citations" in result


def test_answer_is_non_empty(agent):
    """Answer is a non-empty string."""
    result = agent.run("What is an index fund?")
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


def test_citations_is_list(agent):
    """Citations is always a list."""
    result = agent.run("What is a Roth IRA?")
    assert isinstance(result["citations"], list)


def test_citations_have_required_keys(agent):
    """Each citation has title, url and source keys."""
    result = agent.run("What is the Sharpe ratio?")
    for citation in result["citations"]:
        assert "title" in citation
        assert "url" in citation
        assert "source" in citation


def test_citations_no_duplicates(agent):
    """No duplicate titles in citations."""
    result = agent.run("How does asset allocation work?")
    titles = [c["title"] for c in result["citations"]]
    assert len(titles) == len(set(titles))



def test_investopedia_source_appears(agent):
    """Investopedia source appears for basic finance questions."""
    result = agent.run("What is compound interest?")
    sources = [c["source"] for c in result["citations"]]
    assert "investopedia" in sources


def test_finder_source_appears(agent):
    """FinDER source appears for financial filing questions."""
    result = agent.run("What is the operating income and earnings per share?")
    sources = [c["source"] for c in result["citations"]]
    assert "finder" in sources
