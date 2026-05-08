"""
src/agents/qa_agent.py
Finance Q&A Agent — answers general financial questions using RAG.
Returns answer + source citations.

Usage:
    from src.agents.qa_agent import FinanceQAAgent
    agent = FinanceQAAgent()
    result = agent.run("What is dollar cost averaging?")
    print(result["answer"])
    print(result["citations"])
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import load_llm
from src.rag.retriever import get_retriever

PROMPT = ChatPromptTemplate.from_template("""
You are Finnie, a friendly financial advisor application.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so honestly.
Do not make up facts. Keep the answer clear and beginner-friendly.

Context:
{context}

Question:
{question}

Answer:
""")


# ── Agent ─────────────────────────────────────────────────────────────────────

class FinanceQAAgent:

    def __init__(self):
        self.retriever = get_retriever()
        self.llm       = load_llm()
        self.chain     = self._build_chain()

    def _build_chain(self):
        return (
            {"context": self.retriever | self._format_docs,
             "question": RunnablePassthrough()}
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_docs(docs) -> str:
        """Combine retrieved chunks into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _get_citations(self, query: str) -> list[dict]:
        """Retrieve source documents and extract citation metadata."""
        docs = self.retriever.invoke(query)
        seen = set()
        citations = []
        for doc in docs:
            title = doc.metadata.get("title", "Unknown")
            url   = doc.metadata.get("url", "")
            source = doc.metadata.get("source", "")
            key = title
            if key not in seen:
                seen.add(key)
                citations.append({
                    "title":  title,
                    "url":    url,
                    "source": source,
                })
        return citations

    def run(self, query: str) -> dict:
        """
        Run the Q&A agent.

        Returns:
            {
                "answer":    str,
                "citations": [{"title": ..., "url": ..., "source": ...}]
            }
        """
        answer    = self.chain.invoke(query)
        citations = self._get_citations(query)
        return {
            "answer":    answer,
            "citations": citations,
        }


# ── Main (quick test) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = FinanceQAAgent()

    questions = [
        "What is the price of bitcoin today??",
      
        ]

    for q in questions:
        print(f"\nQ: {q}")
        result = agent.run(q)
        print(f"A: {result['answer']}")
        print("Sources:")
        for c in result["citations"]:
            if c["url"]:
                print(f"  - {c['title']} ({c['url']})")
            else:
                print(f"  - {c['title']} [{c['source']}]")
        print("-" * 60)