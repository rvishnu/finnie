"""
src/agents/market_agent.py
Market Analysis Agent — fetches real-time stock data and provides
plain-English analysis using RAG + yfinance/Alpha Vantage.

Usage:
    from src.agents.market_agent import MarketAnalysisAgent
    agent = MarketAnalysisAgent()
    result = agent.run("How is Apple stock doing?")
    print(result["answer"])
"""

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import load_llm
from src.rag.retriever import get_retriever
from src.utils.market_tools import get_stock_data, get_stock_news, extract_ticker

PROMPT = ChatPromptTemplate.from_template("""
You are Finnie, a financial analysis assistant.
Use the stock data and financial knowledge below to give a clear,
beginner-friendly analysis. Include what the numbers mean in plain English.
Always remind the user this is not financial advice.

Stock Data:
{stock_data}

Recent News:
{news}

Financial Knowledge (from knowledge base):
{context}

User Question:
{question}

Analysis:
""")

# Common ticker mappings for natural language queries
COMPANY_TO_TICKER = {
    "apple":     "AAPL",
    "microsoft": "MSFT",
    "google":    "GOOGL",
    "alphabet":  "GOOGL",
    "amazon":    "AMZN",
    "tesla":     "TSLA",
    "meta":      "META",
    "facebook":  "META",
    "nvidia":    "NVDA",
    "netflix":   "NFLX",
    "intel":     "INTC",
    "amd":       "AMD",
}


class MarketAnalysisAgent:

    def __init__(self):
        self.retriever = get_retriever()
        self.llm       = load_llm()





    def _get_rag_context(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)
    


    def run(self, query: str) -> dict:
        """
        Returns:
            {
                "answer": str,
                "ticker": str | None,
                "source": str   (Alpha Vantage or Yahoo Finance)
            }
        """
        ticker = extract_ticker(query, self.llm)

        if not ticker:
            return {
                "answer": (
                    "I couldn't identify a stock ticker in your question. "
                    "Please mention a ticker symbol (e.g. AAPL) or company name."
                ),
                "ticker": None,
                "source": None,
            }

        stock_data = get_stock_data.invoke(ticker)
        news       = get_stock_news.invoke(ticker)
        context    = self._get_rag_context(query)

        chain = PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({
            "stock_data": stock_data,
            "news":       news,
            "context":    context,
            "question":   query,
        })

        # Extract source from stock_data string
        source = "Yahoo Finance"
        if "Alpha Vantage" in stock_data:
            source = "Alpha Vantage"

        return {
            "answer": answer,
            "ticker": ticker,
            "source": source,
        }


if __name__ == "__main__":
    agent = MarketAnalysisAgent()
    questions = [
        "Should I buy Service Now?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        result = agent.run(q)
        print(f"Ticker: {result['ticker']} | Source: {result['source']}")
        print(f"A: {result['answer']}")
        print("-" * 60)