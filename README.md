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

An AI-powered personal finance education assistant built with LangGraph (ReAct pattern), RAG (Retrieval-Augmented Generation), and a Streamlit UI. Finnie helps users with retirement planning, portfolio analysis, market research, financial news, tax education, and general finance questions through a conversational interface.

> **Disclaimer:** Finnie is for educational purposes only and does not constitute financial advice.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Setup Instructions](#setup-instructions)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Troubleshooting Guide](#troubleshooting-guide)

---

## Features

| Interface | Feature | Description |
|-----------|---------|-------------|
| 💬 **Chat** | Conversational AI | Multi-turn conversations with persistent memory across session |
| 💬 **Chat** | Retirement Planning | "I want to retire in 25 years with $2M" — calculates monthly contributions |
| 💬 **Chat** | Tax Education | Capital gains, IRA/401k limits, tax-loss harvesting explanations |
| 💬 **Chat** | Financial News | Multi-ticker headlines with bullish/bearish/neutral sentiment tags |
| 💬 **Chat** | Investing Q&A | RAG-powered answers from a curated financial knowledge base |
| 📊 **Portfolio** | Live Portfolio Analysis | Sector allocation, diversification score, asset mix, holdings table |
| 📈 **Market** | Real-Time Stock Data | Candlestick chart, 52-week range, P/E ratio, market cap, volume |
| 🔧 **MCP** | Claude Desktop Tools | 6 tools for use directly inside Claude Code |

---

## Architecture Overview

Finnie is organized into four layers: the user interface layer, the workflow/orchestration layer, the agent layer, and the data layer.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Interface Layer                          │
│         Streamlit Web App              MCP Server (Claude)      │
│   (Chat / Portfolio / Market tabs)   (6 FastMCP tools)          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Orchestration Layer                           │
│                  LangGraph Workflow Graph                        │
│                                                                 │
│  ┌─────────────────┐     ┌──────────────────────────────────┐   │
│  │ param_extractor │────▶│  Decision: Sequential or Parallel │  │
│  │ (tickers, age)  │     └──────────┬───────────────┬────────┘  │
│  └─────────────────┘                │               │           │
│                            ┌────────▼────┐   ┌──────▼────────┐  │
│                            │  ReAct Mode │   │ Fan-Out Mode  │  │
│                            │  agent_node │   │smart_fanout   │  │
│                            │  ↕ tools   │   │parallel tools │  │
│                            │  (loop)    │   │+ synth_node   │  │
│                            └────────────┘   └───────────────┘  │
│                                                                 │
│               FinnieState (MemorySaver — per thread_id)         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                        Agent Layer                              │
│                                                                 │
│   FinanceQAAgent     PortfolioAnalysisAgent  MarketAnalysisAgent│
│   GoalPlanningAgent  NewsSynthesizerAgent    TaxEducationAgent  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                        Data Layer                               │
│                                                                 │
│   FAISS Vector Store          yfinance / Alpha Vantage          │
│   (47 Investopedia articles   (live prices, news, fundamentals) │
│    + FinDER dataset)                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

**ReAct (Reasoning + Acting):** The LLM reasons about what to do, calls one tool, reads the result, and reasons again — looping until it has enough information to answer. Used for queries that require chaining tools (e.g., "compare the P/E ratios of stocks in my portfolio" needs portfolio data first, then market data per ticker).

**Smart Parallel Fan-Out:** For queries that don't require tool chaining, a router LLM selects exactly 2 relevant tools, runs them in parallel, and a synthesis LLM combines the results. This is ~2x faster than sequential ReAct for simple queries.

**Stateful Conversations:** LangGraph's `MemorySaver` persists the `FinnieState` TypedDict (messages, risk profile, savings goal, portfolio holdings, age) across every turn within a session thread. Users can say "I'm 35 targeting $2M" and then follow up with "what if I extend by 5 years" without restating their goal.

**Deterministic Financial Math:** Agents compute financial metrics in pure Python (PMT formula, portfolio weights, tax rates) and only use the LLM to explain the results in natural language. This ensures numeric accuracy.

### Agent Summary

| Agent | Trigger Keywords | Data Source |
|-------|-----------------|-------------|
| `FinanceQAAgent` | General questions, concepts | FAISS RAG |
| `PortfolioAnalysisAgent` | "my portfolio", "holdings", tickers | yfinance + RAG |
| `MarketAnalysisAgent` | Specific ticker, "stock price", "P/E" | Alpha Vantage / yfinance |
| `GoalPlanningAgent` | "retire", "save", "goal", "how much" | Math + RAG |
| `NewsSynthesizerAgent` | "news", "headlines", "latest on" | yfinance news |
| `TaxEducationAgent` | "tax", "capital gains", "IRA", "401k" | Rules + RAG |

---

## Setup Instructions

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager (recommended) or pip
- An OpenAI API key (required)

### 1. Clone and Install

```bash
git clone <repo-url>
cd finnie

# With uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
# Required
OPENAI_API_KEY=sk-...

# Optional — enables richer market data; falls back to yfinance if absent
ALPHA_VANTAGE_API_KEY=your_key_here

# Optional — only needed if switching config.yaml to use Claude models
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Build the RAG Knowledge Base

This step scrapes 47 Investopedia articles and loads the FinDER dataset, then builds a FAISS vector index. Only needs to be done once (or when you want to refresh content).

```bash
uv run python -m src.rag.loader
```

This creates `data/faiss_index/` and caches raw article text in `data/raw/articles.json`. Expect it to take 2–5 minutes on first run due to web scraping rate limits.

### 4. Run the Web App

```bash
uv run streamlit run src/web_app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run as Docker Container

```bash
docker build -t finnie .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-... \
  -e ALPHA_VANTAGE_API_KEY=your_key \
  finnie
```

### 6. Connect to Claude Desktop (MCP)

To use Finnie's tools directly inside Claude Code, add the following to your Claude Desktop config (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "finnie": {
      "command": "uv",
      "args": ["run", "python", "src/mcp_server/server.py"],
      "cwd": "/path/to/finnie"
    }
  }
}
```

Restart Claude Desktop. The 6 Finnie tools will appear in the tool picker.

Alternatively, the project includes a `.mcp.json` file at the root that Claude Code picks up automatically when you open the project directory.

### Configuration File

`config.yaml` controls LLM, embedding, and RAG parameters:

```yaml
llm:
  model: "gpt-4o-mini"       # Change to "claude-3-5-sonnet-20241022" for Claude
  temperature: 0
  max_tokens: 1000

embeddings:
  model: "text-embedding-3-small"

rag:
  chunk_size: 800
  chunk_overlap: 200
  top_k: 4                   # Documents retrieved per query
  index_path: "data/faiss_index"
  raw_cache: "data/raw/articles.json"

market:
  delay_seconds: 2           # Polite delay between web scrapes

app:
  title: "Finnie - AI Finance Assistant"
  port: 8501
```

### Running Tests

```bash
uv run pytest tests/
uv run pytest tests/test_workflow.py -v           # Workflow graph tests
uv run pytest tests/test_portfolio_agent_integration.py -v
```

---

## API Documentation

### MCP Tools (Claude Desktop)

When connected via MCP, Claude has access to these 6 tools:

---

#### `analyze_portfolio`

Analyzes a stock portfolio and returns diversification metrics, sector allocation, and AI-generated commentary.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `holdings` | string | Yes | Portfolio holdings. Accepts "AAPL: 10, MSFT: 5" or "10 AAPL, 5 MSFT" or plain English |
| `risk_profile` | string | No | `conservative`, `moderate` (default), or `aggressive` |

**Example:**
```
analyze_portfolio(holdings="AAPL: 50, MSFT: 30, BND: 40, VTI: 100", risk_profile="moderate")
```

**Returns:** Total portfolio value, per-holding breakdown (price, allocation %, sector, P/E, dividend yield), sector pie data, diversification score (0–10), AI analysis paragraph.

---

#### `plan_financial_goal`

Calculates savings plans for retirement or any financial goal using the PMT (present value of annuity) formula.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `goal_amount` | number | Yes | Target dollar amount (e.g., 2000000 for $2M) |
| `time_horizon_years` | number | Yes | Years until the goal |
| `current_savings` | number | No | Current savings/investments to start from (default: 0) |
| `risk_profile` | string | No | `conservative` (4% return), `moderate` (7%), `aggressive` (10%) |

**Example:**
```
plan_financial_goal(goal_amount=2000000, time_horizon_years=25, current_savings=50000, risk_profile="moderate")
```

**Returns:** Monthly contribution needed, projected final value, assumed annual return rate, AI explanation of the plan.

---

#### `get_stock_data`

Fetches real-time price and fundamental data for a single stock.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ticker` | string | Yes | Stock ticker symbol (e.g., "AAPL", "NVDA") |

**Example:**
```
get_stock_data(ticker="NVDA")
```

**Returns:** Current price, day high/low, 52-week high/low, P/E ratio, dividend yield, market cap, company description, AI analysis.

---

#### `get_financial_news`

Fetches recent news headlines for one or more stocks and returns them with sentiment classification.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Natural language query. Can include ticker symbols. |

**Example:**
```
get_financial_news(query="latest news on NVDA and MSFT")
```

**Returns:** Up to 5 deduplicated headlines per ticker, each tagged bullish/bearish/neutral, plus an AI-written news briefing.

---

#### `get_tax_education`

Answers US investment tax questions including capital gains calculations, retirement account limits, and tax-loss harvesting.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Tax question in plain English |

**Example:**
```
get_tax_education(query="I sold AAPL after 8 months with a $5,000 gain. What do I owe in taxes?")
```

**Returns:** Tax scenario type, computed tax liability or account limits, holding period classification (short/long-term), AI explanation with relevant context.

**Supported Scenarios:**
- Capital gains: short-term (held < 12 months) vs long-term (held ≥ 12 months)
- Account contribution limits: 401k, IRA, Roth IRA, HSA (2024 limits)
- Tax-loss harvesting: deductible amount and carryforward calculation

---

#### `answer_finance_question`

Answers general financial education questions using the RAG knowledge base.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Any finance question |

**Example:**
```
answer_finance_question(query="What is dollar-cost averaging and when should I use it?")
```

**Returns:** AI answer grounded in retrieved knowledge base documents, with citations (title + URL) for each source used.

---

### Python API (Workflow Graph)

For programmatic use, the `FinnieGraph` class in `src/workflow/graph.py` exposes two methods:

```python
from src.workflow.graph import FinnieGraph

graph = FinnieGraph()

# Sequential ReAct — best for chained queries
response = graph.invoke(
    user_message="How much should I save monthly to retire at 65 with $2M?",
    thread_id="session-abc123",
    risk_profile="moderate"
)

# Parallel fan-out — best for simple queries (faster)
response = graph.invoke_all(
    user_message="What's the latest news on Apple?",
    thread_id="session-abc123"
)
```

State persists automatically across calls with the same `thread_id`.

---

## Usage Examples

### Chat Tab — Retirement Planning

**Conversation:**
```
User: I'm 38 years old and want to retire at 65 with $3 million
Finnie: Great goal! With a 27-year runway on a moderate risk profile:
        - Monthly contribution needed: $3,847
        - Starting from $0, assuming 7% annual return
        ...

User: What if I already have $150k saved?
Finnie: With $150,000 already invested, your monthly contribution drops to $2,981
        — your existing savings will grow to ~$873k over 27 years...

User: Switch to aggressive
Finnie: On an aggressive risk profile (10% return), with your $150k head start,
        you'd need only $1,644/month...
```

### Chat Tab — Tax Question

```
User: I sold Tesla stock after 11 months with a $12,000 gain. What do I owe?
Finnie: Since you held for 11 months (under 12), this is a SHORT-TERM capital gain.
        Short-term gains are taxed as ordinary income. If you're in the 22% bracket:
        - Tax owed: $2,640
        Had you waited one more month, the long-term rate (likely 15%) would apply:
        - Tax owed: $1,800 — a $840 difference.
```

### Portfolio Tab

Enter holdings in any of these formats:
- `AAPL: 50, MSFT: 30, BND: 40, VTI: 100`
- `50 shares AAPL, 30 MSFT, 40 BND`
- `I own 50 Apple, 30 Microsoft, 40 bond ETF shares`

Select risk profile (conservative/moderate/aggressive) and click **Analyze**.

**Sample Output:**
```
Total Value: $47,832
Diversification Score: 7.2/10

Holdings:
  AAPL  50 shares  $8,914  18.6%  Technology  P/E 29.4  Div 0.6%
  MSFT  30 shares  $12,480 26.1%  Technology  P/E 34.1  Div 0.8%
  BND   40 shares  $2,940  6.1%   Fixed Income P/E —    Div 3.4%
  VTI  100 shares  $23,498 49.1%  Broad Market P/E 24.8 Div 1.3%

Sector Allocation: Technology 44.7%, Fixed Income 6.1%, Broad Market 49.1%
```

### Market Tab

Enter a ticker (e.g., `NVDA`) and press Enter or click **Get Data**.

**Sample Output:**
```
NVDA — NVIDIA Corporation
Current Price: $875.40
Day Range: $861.20 – $882.10
52-Week Range: $430.16 – $974.00
P/E Ratio: 68.4
Dividend Yield: 0.03%
Market Cap: $2.15T
```

Includes a 30-day candlestick chart with volume.

### MCP in Claude Desktop

After connecting via MCP, ask Claude:

```
"Analyze my portfolio: 100 shares of VTI, 50 AAPL, 20 BND"
"What's my monthly savings target if I want $1.5M in 20 years with $80k already saved?"
"Show me news on NVDA and AMD — any big moves?"
"I harvested a $6,000 tax loss this year. How much can I deduct?"
```

---

## Troubleshooting Guide

### RAG Index Not Found

**Symptom:** `FileNotFoundError: data/faiss_index not found` or agents return empty/irrelevant answers.

**Fix:** Build the index:
```bash
uv run python -m src.rag.loader
```

If scraping fails on specific URLs, the cached `data/raw/articles.json` is used on subsequent runs. Delete it to force a fresh scrape.

---

### OpenAI API Errors

**Symptom:** `AuthenticationError` or `RateLimitError` from OpenAI.

**Fix:**
- Verify `OPENAI_API_KEY` is set correctly in `.env`
- Check usage at [platform.openai.com](https://platform.openai.com)
- `gpt-4o-mini` is cost-efficient; if rate-limited, add `time.sleep()` delays or upgrade your plan

---

### Market Data Missing or Stale

**Symptom:** Portfolio shows $0 or missing prices; market tab shows no data.

**Fix:**
1. yfinance is the fallback — it requires no API key but has rate limits. Wait 30 seconds and retry.
2. Add `ALPHA_VANTAGE_API_KEY` to `.env` for more reliable data (free tier: 25 requests/day).
3. Markets are closed on weekends and holidays — yfinance returns last close price, which is correct behavior.

---

### MCP Server Not Connecting

**Symptom:** Finnie tools don't appear in Claude Desktop.

**Fix:**
1. Verify the path in your `claude_desktop_config.json` or `.mcp.json` is correct
2. Ensure `uv` is on your PATH: `which uv`
3. Test the server manually: `uv run python src/mcp_server/server.py`
4. Check that all dependencies are installed: `uv sync`
5. Restart Claude Desktop after config changes

---

### Streamlit Port Already in Use

**Symptom:** `Address already in use` error on port 8501.

**Fix:**
```bash
# Find what's using the port
lsof -i :8501

# Kill it, or run on a different port
uv run streamlit run src/web_app/app.py --server.port=8502
```

---

### Conversation Memory Not Persisting

**Symptom:** Finnie forgets your goal or portfolio between messages.

**Cause:** Each browser tab gets a fresh `thread_id`. State persists within a tab session but resets on page refresh.

**Fix:** This is expected behavior. Within a single session, context carries forward correctly. If you need persistence across sessions, the `MemorySaver` can be swapped for a `SqliteSaver` — see `src/workflow/graph.py`.

---

### Tax Calculations Seem Wrong

**Symptom:** Tax amounts don't match expectations.

**Notes:**
- Finnie uses 2024 US federal tax rates. State taxes are not included.
- Short-term capital gains use a fixed assumed bracket (22%) unless you specify your income bracket.
- Long-term rates (0%, 15%, 20%) are applied based on stated gain amount.
- For precise tax advice, consult a tax professional.

---

### Slow First Response

**Symptom:** First query takes 10–30 seconds.

**Cause:** The RAG retriever, LLM client, and all 6 agents are lazy-loaded on first use and cached for the process lifetime.

**Fix:** This is by design. Subsequent queries in the same session are significantly faster. Warm-up is unavoidable on cold start.

---

### Docker Container Fails to Start

**Symptom:** Container exits immediately or health check fails.

**Fix:**
```bash
# Check logs
docker logs <container-id>

# Ensure API key is passed
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... finnie

# Health check endpoint
curl http://localhost:8501/_stcore/health
```

Note: The FAISS index must be built before the container starts, or built inside the container. Mount the data directory if you want to persist it:
```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/data:/app/data \
  finnie
```
