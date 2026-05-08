---
title: Finnie AI Finance Assistant
emoji: 💰
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.57.0
app_file: src/web_app/app.py
python_version: "3.12"
pinned: false
---

# Finnie — AI Finance Assistant

An AI-powered personal finance education assistant built with LangGraph (ReAct pattern), RAG, and a Streamlit UI.

## Features

| Tab | What it does |
|-----|-------------|
| 💬 **Chat** | Conversational assistant — retirement goals, tax questions, portfolio advice, financial news, investing education |
| 📊 **Portfolio** | Live portfolio analysis — sector allocation, asset mix, diversification score, holdings table |
| 📈 **Market** | Real-time stock data — candlestick chart, 52-week gauge, volume, P/E, market cap |

## Architecture

```
User query
    │
    ▼
param_extractor_node       ← extracts savings/goal/risk from plain text
    │
    ▼
agent_node (LLM + tools)   ← ReAct: reasons → picks tool → sees result → reasons again
    │         ▲
    ▼         │
tool_node ────┘            ← executes whichever agent the LLM chose
    │
    ▼
Final answer
```

**Agents / Tools**
- `FinanceQAAgent` — general financial education (RAG + LLM)
- `PortfolioAnalysisAgent` — live yfinance data + diversification metrics
- `MarketAnalysisAgent` — real-time price, P/E, 52-week range
- `GoalPlanningAgent` — PMT-based monthly contribution calculator
- `NewsSynthesizerAgent` — multi-ticker news with sentiment tagging
- `TaxEducationAgent` — capital gains, IRA/401k/HSA limits, tax-loss harvesting

**Memory**: LangGraph `MemorySaver` persists the full state (messages, risk profile, savings goal) across conversation turns per `thread_id`.

## Setup (local)

```bash
# 1. Clone and install
git clone <repo-url>
cd finnie
uv sync          # or: pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# Edit .env and add your keys

# 3. Run
uv run streamlit run src/web_app/app.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Powers the LLM (gpt-4o-mini) and embeddings |
| `ANTHROPIC_API_KEY` | No | Only if switching to a Claude model in config.yaml |
| `ALPHA_VANTAGE_API_KEY` | No | Better market data; falls back to yfinance if absent |

## Disclaimer

Finnie is for **educational purposes only** and does not constitute financial advice.
