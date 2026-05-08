"""
tests/test_rag_integration.py
Integration tests for FAISS index load and retrieval.

Run:
    uv run pytest tests/test_rag_integration.py -v
"""

import os
import pytest
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = "data/faiss_index"


@pytest.fixture(scope="module")
def vectorstore():
    """Load the FAISS index once for all tests."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return store


def test_index_files_exist():
    """Index files must exist on disk before anything else."""
    assert os.path.exists(os.path.join(INDEX_PATH, "index.faiss")), \
        "index.faiss not found — run loader.py first"
    assert os.path.exists(os.path.join(INDEX_PATH, "index.pkl")), \
        "index.pkl not found — run loader.py first"


def test_index_loads(vectorstore):
    """FAISS index loads without error."""
    assert vectorstore is not None


def test_retrieval_returns_results(vectorstore):
    """A query returns documents."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke("What is dollar cost averaging?")
    assert len(results) == 3


def test_retrieval_documents_have_content(vectorstore):
    """All returned documents have non-empty content."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke("What is a Roth IRA?")
    for doc in results:
        assert doc.page_content.strip() != ""


def test_retrieval_documents_have_metadata(vectorstore):
    """All returned documents have required metadata keys."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke("Explain portfolio diversification")
    for doc in results:
        assert "title" in doc.metadata
        assert "source" in doc.metadata


def test_retrieval_source_is_valid(vectorstore):
    """Source metadata is either investopedia or finder."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke("capital gains tax")
    for doc in results:
        assert doc.metadata["source"] in ("investopedia", "finder")


def test_finder_results_have_answer(vectorstore):
    """FinDER documents must have an answer in metadata."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    results = retriever.invoke("CBOE revenue 2021 2023")
    finder_docs = [d for d in results if d.metadata["source"] == "finder"]
    for doc in finder_docs:
        assert "answer" in doc.metadata
        assert doc.metadata["answer"].strip() != ""


def test_investopedia_query_returns_relevant_result(vectorstore):
    """Investopedia query should return at least one investopedia source."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke("What is an index fund?")
    sources = [d.metadata["source"] for d in results]
    assert "investopedia" in sources


def test_finder_query_returns_relevant_result(vectorstore):
    """Financial filing query should return at least one finder source."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke("operating income total revenue earnings per share")
    sources = [d.metadata["source"] for d in results]
    assert "finder" in sources