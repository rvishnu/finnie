"""
src/core/llm.py
Shared LLM loader — reads model/temperature from config.yaml and returns
the appropriate LangChain chat model (OpenAI or Anthropic).
"""

import os
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

LLM_MODEL   = _cfg["llm"]["model"]
TEMPERATURE = _cfg["llm"]["temperature"]


_llm = None


def load_llm():
    """Return a LangChain chat model configured from config.yaml."""
    global _llm
    if _llm is None:
        if "gpt" in LLM_MODEL or "o1" in LLM_MODEL:
            _llm = ChatOpenAI(
                model=LLM_MODEL,
                temperature=TEMPERATURE,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif "claude" in LLM_MODEL:
            _llm = ChatAnthropic(
                model=LLM_MODEL,
                temperature=TEMPERATURE,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        else:
            raise ValueError(f"Unknown model in config.yaml: {LLM_MODEL}")
    return _llm
