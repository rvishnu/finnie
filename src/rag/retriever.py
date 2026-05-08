"""
src/rag/retriever.py
Loads the FAISS index and returns a retriever for agents to use.
"""

import os
import yaml
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

INDEX_PATH = cfg["rag"]["index_path"]
TOP_K      = cfg["rag"]["top_k"]


_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        _retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return _retriever