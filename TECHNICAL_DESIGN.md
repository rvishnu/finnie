# Finnie — Technical Design Document

## Table of Contents

1. [System Architecture Decisions](#1-system-architecture-decisions)
2. [Agent Communication Protocols](#2-agent-communication-protocols)
3. [RAG Implementation Details](#3-rag-implementation-details)
4. [Performance Considerations](#4-performance-considerations)

---

## 1. System Architecture Decisions

### 1.1 LangGraph as the Orchestration Framework

**Decision:** Use LangGraph's `StateGraph` as the central orchestration engine rather than plain LangChain chains or raw Python.

**Rationale:**

LangGraph models the conversation as a directed graph of nodes connected by conditional edges. This gives three concrete benefits over alternatives:

1. **Built-in state persistence.** `MemorySaver` (or any `BaseCheckpointSaver`) snapshots the full `FinnieState` after every node execution and restores it on the next invocation using the same `thread_id`. There is no application-level session management code.

2. **Tool execution as a first-class loop.** The standard ReAct pattern (reason → act → observe → repeat) maps directly onto a LangGraph cycle: `agent_node → tools → agent_node`. The routing condition (`should_continue`) inspects whether the last message contains `tool_calls` and either loops back or terminates. This is more explicit and debuggable than hidden chain internals.

3. **Parallel execution without extra threading code.** LangGraph's `ToolNode` runs multiple tool calls in parallel using asyncio when more than one tool is invoked in a single LLM response. The parallel fan-out mode exploits this directly.

**Alternative considered:** A simple function-call loop with manual state passing was prototyped first. It worked for single-turn queries but required significant boilerplate to track conversation context, handle tool errors, and support multi-turn follow-up. LangGraph eliminated all of that.

---

### 1.2 Two Execution Modes: ReAct vs. Parallel Fan-Out

**Decision:** Maintain two separate execution paths — sequential ReAct (`invoke`) and smart parallel fan-out (`invoke_all`) — with automatic routing in `chat()`.

**ReAct mode graph:**
```
START → param_extractor → agent_node ⟷ tools → END
```

**Parallel fan-out graph:**
```
START → param_extractor → smart_fanout → tools (parallel) → synth_node → END
```

**Routing heuristic (`chat()`):**

```python
SEQUENTIAL_KEYWORDS = {
    "p/e", "compare", "rate hike", "52-week", "dividend",
    "tax", "allocation", "withdraw", "retirement"
}

def chat(user_message, thread_id, risk_profile):
    if any(kw in user_message.lower() for kw in SEQUENTIAL_KEYWORDS):
        return self.invoke(user_message, thread_id, risk_profile)
    return self.invoke_all(user_message, thread_id, risk_profile)
```

**Why keyword routing instead of always using one mode:**

- ReAct is necessary when Tool B's input depends on Tool A's output (e.g., "analyze the P/E ratios of my holdings" must run `analyze_portfolio` first to know which tickers to query in `get_market_data`).
- Parallel fan-out is ~2x faster for independent queries (news, simple goal planning, Q&A) because both tools run concurrently.
- A pure ReAct approach for everything works but introduces unnecessary latency on simple queries. A pure fan-out approach fails for dependent queries because the routing LLM cannot know intermediate results when planning.

**Why keyword matching instead of LLM routing:**

An LLM call to classify "sequential vs parallel" would add ~500ms of latency to every query. Keyword matching is deterministic, zero-latency, and errs on the side of ReAct (the slower but always-correct mode). False positives (keyword match for a non-sequential query) cost ~300ms of extra latency. False negatives (parallel fan-out for a dependent query) would produce wrong answers. The asymmetry favors the keyword approach.

---

### 1.3 Six Specialized Agents Over One General Agent

**Decision:** Build six domain-specific agent classes rather than one general-purpose agent that calls all APIs.

**Rationale:**

Each agent owns one responsibility and its associated prompts, parsing logic, API calls, and metric computations. This separation provides:

1. **Testability.** Each agent has its own integration test file. Testing `TaxEducationAgent` in isolation doesn't require mocking the portfolio or market APIs.

2. **Prompt specialization.** The system prompt for `PortfolioAnalysisAgent` focuses entirely on diversification interpretation. The system prompt for `TaxEducationAgent` focuses on IRS rules. A single agent with all responsibilities would require a much larger, harder-to-tune system prompt.

3. **Deterministic computation in Python.** Each agent computes financial metrics (PMT formula, portfolio weights, tax rates) in pure Python and passes the computed numbers to the LLM for explanation. This is more reliable than asking the LLM to do arithmetic directly.

4. **Independent scalability.** The market agent and news agent make external HTTP calls. The goal agent and tax agent are pure computation. If rate limits on yfinance become an issue, only the market and news agents need caching or retry logic.

**Trade-off accepted:** Six classes mean six LLM client initializations on cold start. This is mitigated by lazy loading with module-level singletons (see Section 4.1).

---

### 1.4 FinnieState TypedDict as the Single Source of Truth

**Decision:** All inter-node data flows through a single `FinnieState` TypedDict. Nodes read from state and return partial state updates.

```python
class FinnieState(TypedDict):
    messages: Annotated[list, add_messages]
    risk_profile: str
    goal_amount: float | None
    time_horizon_years: float | None
    current_savings: float | None
    annual_contribution: float | None
    portfolio_holdings: dict | None
    portfolio_value: float | None
    age: int | None
```

**Rationale:**

- `messages` uses the `add_messages` reducer, which appends new messages rather than replacing the list. This means nodes only need to return new messages; LangGraph handles merging.
- All financial context fields (`goal_amount`, `portfolio_holdings`, etc.) are optional and default to `None`. Nodes that don't update a field simply omit it from their return dict.
- `MemorySaver` serializes the entire TypedDict to JSON and rehydrates it on the next turn. Using a TypedDict (rather than a dataclass or Pydantic model) keeps serialization trivial.
- Tools that update state (e.g., `analyze_portfolio` setting `portfolio_value`) use LangGraph's `Command` return type, which can both emit a `ToolMessage` and update state fields atomically.

---

### 1.5 FastMCP for Claude Desktop Integration

**Decision:** Expose Finnie's functionality as a FastMCP server rather than a REST API.

**Rationale:**

MCP (Model Context Protocol) is the standard for extending Claude with external tools. `FastMCP` (the Python server library) mirrors the same agent functions already implemented for the workflow graph, exposing them as typed tool definitions that Claude Desktop discovers automatically.

The six MCP tools map 1:1 to the six LangChain tools in `src/workflow/tools.py`. The MCP layer adds only parameter validation and serialization — the agent logic is shared.

**Alternative considered:** A REST API (FastAPI) that the Streamlit app and Claude both call. This would require running a separate HTTP server, managing authentication, and writing OpenAPI specs. FastMCP handles the protocol layer automatically and runs in-process with Claude Desktop via stdio transport.

---

## 2. Agent Communication Protocols

### 2.1 LangGraph Message Protocol

Agents communicate exclusively through the `messages` list in `FinnieState`. Each message is a LangChain `BaseMessage` subtype:

| Message Type | Sender | Purpose |
|-------------|--------|---------|
| `HumanMessage` | User | User query |
| `AIMessage` | LLM | Reasoning or final response |
| `AIMessage` with `tool_calls` | LLM | Request to invoke one or more tools |
| `ToolMessage` | Tool node | Tool result, keyed by `tool_call_id` |
| `SystemMessage` | Graph nodes | Injected context (user state, prompts) |

The conversation flow for a ReAct turn:

```
1. param_extractor_node
   → reads: HumanMessage at end of messages list
   → writes: updates to state (portfolio_holdings, age)
   → does NOT append messages

2. agent_node
   → reads: full messages list + state fields
   → injects: SystemMessage with user context
   → writes: AIMessage (either with tool_calls or final answer)

3. tools (ToolNode)
   → reads: last AIMessage.tool_calls
   → invokes: tool functions in parallel
   → writes: one ToolMessage per tool call

4. agent_node (again, if tool_calls were present)
   → reads: updated messages including ToolMessages
   → writes: AIMessage with final answer
   → returns: no tool_calls → edge routes to END
```

### 2.2 Tool Invocation Protocol

Tools are LangChain `@tool`-decorated functions bound to the LLM via `llm.bind_tools(tools)`. The LLM decides which tool to call and with what arguments; LangGraph's `ToolNode` handles dispatch and error wrapping.

**Tool signature pattern:**

```python
@tool
def analyze_portfolio(query: str, state: Annotated[FinnieState, InjectedState]) -> Command:
    """One-line description used by the routing LLM."""
    ...
    return Command(
        update={"portfolio_value": metrics["total_value"], "portfolio_holdings": holdings},
        messages=[ToolMessage(content=result, tool_call_id=tool_call_id)]
    )
```

Key design points:
- `state: Annotated[FinnieState, InjectedState]` injects the current state without it appearing in the LLM's tool schema. The LLM sees only `query`.
- Returning `Command` allows atomically updating state AND emitting a `ToolMessage`. Without `Command`, a tool can only return a string.
- Tools that don't update state return a plain string; `ToolNode` wraps it in a `ToolMessage` automatically.

### 2.3 Smart Fan-Out Protocol

In parallel fan-out mode, `smart_fanout_node` constructs a synthetic `AIMessage` with multiple `tool_calls` and writes it to the messages list:

```python
# smart_fanout_node
selected_tools = _select_tools(user_message)  # LLM selects 2 tools
tool_calls = [
    {"name": tool_name, "args": {"query": enriched_query}, "id": f"tc_{i}"}
    for i, tool_name in enumerate(selected_tools)
]
return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
```

`ToolNode` sees multiple `tool_calls` in the last `AIMessage` and executes them in parallel. The results arrive as multiple `ToolMessage` objects, which `synth_node` then reads and combines.

**Tool selection protocol (`_select_tools`):**

A lightweight LLM call with a structured prompt selects exactly 2 tools. The prompt lists all tools with one-line descriptions and explicit priority rules to resolve ambiguity. The LLM returns a JSON array of exactly 2 tool names.

```
Priority rules (from prompts.py):
1. News/headlines → get_financial_news + get_market_data
2. General Q&A   → answer_finance_question + plan_financial_goal (or get_tax_education)
3. Portfolio     → analyze_portfolio + get_market_data
4. Tax           → get_tax_education + answer_finance_question
```

**Why exactly 2 tools:** Empirically, 2 tools covers > 95% of queries without over-fetching. Allowing variable N requires the LLM to reason about coverage, which increases prompt complexity and occasionally selects redundant tools.

### 2.4 Parameter Extraction Protocol

`param_extractor_node` runs before the agent on every turn and extracts structured data from raw user text without an LLM call:

**Ticker extraction (regex):**
```python
TICKER_PATTERNS = [
    r'\b([A-Z]{2,5})\s*:\s*(\d+)',      # "AAPL: 100"
    r'(\d+)\s+(?:shares\s+of\s+)?([A-Z]{2,5})',  # "100 shares of AAPL"
]
```

**Age extraction (regex):**
```python
AGE_PATTERN = r"I(?:'m| am) (\d+)(?: years? old)?"
```

If tickers are found, `portfolio_holdings` is updated in state. If age is found, `age` is updated. This pre-populated state is then available to all subsequent nodes without requiring them to re-parse the user message.

**Why regex instead of LLM for extraction:** Extraction runs on every message, including follow-ups like "what if I extend by 5 years?" that contain no new tickers. A regex check costs microseconds; an LLM extraction call costs ~200ms and adds to token usage. The regex patterns cover the formats users actually type.

### 2.5 State Injection into Agent System Prompt

`agent_node` builds a dynamic system prompt from the current state values before calling the LLM:

```python
context_parts = []
if state.get("risk_profile"):
    context_parts.append(f"Risk profile: {state['risk_profile']}")
if state.get("goal_amount"):
    context_parts.append(f"Savings goal: ${state['goal_amount']:,.0f}")
if state.get("portfolio_holdings"):
    holdings_str = ", ".join(f"{k}: {v}" for k, v in state["portfolio_holdings"].items())
    context_parts.append(f"Portfolio: {holdings_str}")
# ... etc

system_prompt = BASE_SYSTEM_PROMPT + "\n\nUser context:\n" + "\n".join(context_parts)
messages_to_send = [SystemMessage(content=system_prompt)] + trimmed_messages
```

This means the LLM always has the user's financial context even on follow-up queries that don't restate it. The system message is not stored in the `messages` list (it's injected transiently) so it doesn't grow the stored state.

---

## 3. RAG Implementation Details

### 3.1 Knowledge Base Composition

The vector store indexes two data sources:

**Source 1: Investopedia Articles (primary)**

47 manually curated URLs covering the full domain of Finnie's agents:

| Category | Articles | Example Topics |
|----------|---------|---------------|
| Finance Q&A | ~15 | DCA, compound interest, diversification, risk, index funds, ETFs vs mutual funds, bonds, liquidity |
| Portfolio | ~12 | Asset allocation, rebalancing, modern portfolio theory, beta, alpha, Sharpe ratio, correlation, expense ratio |
| Market | ~6 | P/E ratio, market cap, EPS, dividend yield, volume, support/resistance, moving averages |
| Goal Planning | ~6 | Emergency fund, retirement planning, rule of 72, net worth, saving vs investing |
| Tax | ~5 | Capital gains tax, 401k, IRA, Roth IRA, tax-loss harvesting |
| Advanced | ~3 | Derivatives, personal finance (50/30/20), insurance, real estate, crypto |

Articles are scraped with BeautifulSoup, extracting text from `<p>` tags. Raw text is cached to `data/raw/articles.json` to avoid re-scraping.

**Source 2: FinDER Dataset (supplementary)**

The `Linq-AI-Research/FinDER` dataset from HuggingFace provides additional financial reference documents. Loaded via the `datasets` library and merged into the same FAISS index.

**Why Investopedia as the primary source:**

Investopedia articles are authoritative, consistently structured, and cover exactly the concepts Finnie needs to explain. The content is stable enough that periodic re-scraping (not real-time) is sufficient. The alternative — using a general-purpose knowledge base — would require more chunks and retrieve less relevant content for finance-specific queries.

### 3.2 Chunking Strategy

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "]
)
```

**Chunk size: 800 characters (~150–200 words)**

- Small enough that a retrieved chunk focuses on one concept.
- Large enough to contain a complete explanation (definition + example).
- At `top_k=4`, the LLM receives ~3,200 characters of context — well within `gpt-4o-mini`'s context window and manageable for synthesis.

**Chunk overlap: 200 characters**

Ensures that sentence boundaries at chunk edges don't lose context. A definition that spans a chunk boundary appears in both chunks, so retrieval doesn't miss it.

**Separator hierarchy:**

`RecursiveCharacterTextSplitter` tries separators in order, preferring paragraph breaks (`\n\n`), then line breaks, then sentence boundaries (`. `), then word boundaries (` `). This preserves semantic coherence within chunks better than fixed-character splitting.

### 3.3 Embedding Model

**Model:** `text-embedding-3-small` (OpenAI)

- 1536-dimensional dense vector
- Cost: $0.02 per million tokens (100x cheaper than `text-embedding-ada-002` at equivalent quality)
- Latency: ~50–100ms per batch
- Dimensionality reduction available (not used — 1536d gives best retrieval quality)

**Why not a local embedding model:**

The FAISS index is built once and reused. The embedding cost for 47 articles is negligible (~$0.001 total). Local models (e.g., `sentence-transformers`) would eliminate the API call but require ~500MB of model weights in memory, which matters in the Docker deployment.

### 3.4 Vector Store: FAISS

**Why FAISS over alternatives (Chroma, Pinecone, Weaviate):**

| Criterion | FAISS | Chroma | Pinecone |
|-----------|-------|--------|----------|
| Hosting | Local file | Local process | Cloud |
| Latency | <5ms | <10ms | 50–200ms |
| Setup | None | None | Account + API key |
| Persistence | Index file | SQLite | Managed |
| Scale | Millions of vectors | Thousands | Billions |

Finnie's knowledge base is small (~5,000 chunks from 47 articles). FAISS flat index search over 5,000 vectors takes under 1ms. There is no operational overhead, no additional service to run, and no network dependency. The index file (`data/faiss_index/`) is ~25MB.

**Index type:** `IndexFlatL2` (exact search, L2 distance). For 5,000 vectors, approximate search (IVF, HNSW) provides no meaningful speedup and risks missing relevant documents.

### 3.5 Retrieval Configuration

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

**top_k = 4:**

- Returns ~3,200 characters of context per query.
- Covers the primary document on the topic plus related context.
- Increasing to 8 yields diminishing returns (retrieved docs become less relevant) and increases LLM input token count.

**Retriever as singleton:**

```python
_retriever = None  # module-level cache

def get_retriever():
    global _retriever
    if _retriever is None:
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return _retriever
```

FAISS index loading from disk takes ~200ms. Loading it on every agent invocation would add noticeable latency. The singleton pattern loads it once per process and shares it across all six agents.

### 3.6 How Agents Use RAG

Each agent queries the retriever with a domain-specific query string, not the raw user message. This produces more relevant retrievals:

```python
# PortfolioAnalysisAgent
rag_query = f"portfolio diversification analysis for {', '.join(holdings.keys())}"

# GoalPlanningAgent
rag_query = f"retirement savings goal {goal_amount} {time_horizon_years} years monthly contribution"

# TaxEducationAgent
rag_query = f"capital gains tax {'long-term' if long_term else 'short-term'} holding period"
```

The retrieved documents are passed to the LLM as context alongside the computed financial metrics:

```
System: You are a financial education assistant.
        [computed metrics]
        [retrieved document chunks]
```

The LLM is instructed to ground its explanation in the retrieved content. For `FinanceQAAgent`, it is instructed to answer using ONLY the provided context and to cite sources.

**Citation extraction:**

```python
docs = retriever.invoke(query)
citations = [
    {"title": doc.metadata.get("title", ""), "url": doc.metadata.get("source", "")}
    for doc in docs
    if doc.metadata.get("source")
]
```

Metadata (`title`, `source` URL) is attached to each `Document` object when articles are scraped and preserved through chunking into the FAISS index.

---

## 4. Performance Considerations

### 4.1 Cold Start and Lazy Loading

**Problem:** Six agents, an LLM client, and a FAISS index all require initialization. Loading everything at startup would add 3–5 seconds before the first request is served.

**Solution:** All expensive resources are initialized lazily on first use and cached as module-level or class-level singletons.

| Resource | Init Cost | Cache Scope | Init Trigger |
|----------|-----------|-------------|-------------|
| FAISS retriever | ~200ms | Module-level singleton | First agent invocation |
| OpenAI LLM client | ~50ms | Per-agent class | First use of that agent |
| Agent instances | ~10ms each | `functools.lru_cache` | First tool call for that agent |
| yfinance data | ~300–800ms | Not cached (always fresh) | Every market/news query |

**Impact:** First query in a session takes 3–8 seconds (sum of all cold starts). Subsequent queries are 1–3 seconds (LLM inference only). This is acceptable for an educational chatbot.

**Future improvement:** Pre-warm all agents on app startup in a background thread to hide cold start latency.

---

### 4.2 LLM Token Usage and Cost Control

**Model choice: `gpt-4o-mini`**

`gpt-4o-mini` is the default for all agent LLM calls. It provides:
- Adequate reasoning quality for financial Q&A and explanation
- 128k context window (more than sufficient for conversation history + RAG chunks)
- Cost: ~$0.15/million input tokens, $0.60/million output tokens
- Latency: ~500–1500ms per call

`gpt-4o` is available via `config.yaml` for higher quality at ~20x the cost.

**Message history trimming:**

Without trimming, message history grows unboundedly and increases input token count per turn. Two strategies are applied:

```python
# agent_node (ReAct mode): keep last 20 messages
trimmed = messages[-20:] if len(messages) > 20 else messages

# synth_node (fan-out mode): keep last 30 messages
trimmed = messages[-30:] if len(messages) > 30 else messages
```

20–30 messages covers ~3–5 turns of a typical multi-tool conversation. Earlier conversation history is preserved in `MemorySaver` but not sent to the LLM — the state fields (`goal_amount`, `portfolio_holdings`, etc.) carry the relevant structured data forward.

**RAG context budget:**

At `top_k=4` with 800-char chunks, RAG adds ~800 tokens per query. This is a fixed cost regardless of conversation length.

**Tool description tokens:**

The routing LLM for fan-out receives tool descriptions of ~10 words each × 6 tools = ~60 tokens. These are intentionally kept minimal to reduce the cost of the routing call.

---

### 4.3 External API Rate Limiting

**yfinance:**
- No API key required; rate limits are implicit
- Hitting limits causes `JSONDecodeError` or empty responses
- Mitigation: `market.delay_seconds: 2` in `config.yaml` applies a polite delay between scraping calls during RAG index builds
- For market/news queries in production, yfinance is called once per ticker per user request — not looped — so rate limits are rarely hit in normal usage

**Alpha Vantage (optional):**
- Free tier: 25 requests/day, 5 requests/minute
- Used only for `MarketAnalysisAgent` (`get_global_quote` and `OVERVIEW` endpoints)
- Falls back to yfinance automatically if Alpha Vantage returns an error or rate limit response:
  ```python
  try:
      data = alpha_vantage_client.get_quote(ticker)
  except Exception:
      data = yfinance_fallback(ticker)
  ```

**OpenAI:**
- Rate limits depend on account tier
- All LLM calls are sequential per request (no parallel LLM calls)
- Retry logic is handled by the `openai` SDK's built-in exponential backoff

---

### 4.4 Parallel Execution in Fan-Out Mode

When `smart_fanout_node` selects 2 tools, LangGraph's `ToolNode` runs them in parallel using Python's `asyncio.gather` internally. The wall-clock time for the fan-out mode is roughly:

```
t_total ≈ t_routing + max(t_tool1, t_tool2) + t_synthesis
```

Compared to sequential ReAct:

```
t_total ≈ t_agent + t_tool1 + t_agent + t_tool2 + t_agent
```

For typical queries:
- Routing call: ~500ms
- Tool execution (each): ~1–2s (yfinance + LLM)
- Synthesis: ~800ms
- **Fan-out total: ~2.5–3.5s**

- ReAct agent call: ~800ms × 3 calls = 2.4s
- Tool execution: ~1–2s × 2 sequential tools = 2–4s
- **ReAct total: ~4.4–6.4s**

Fan-out mode is ~40–50% faster for simple two-tool queries. For queries requiring tool chaining, ReAct is the only correct option regardless of speed.

---

### 4.5 Portfolio Metrics: Python vs. LLM Computation

All numerical financial computations happen in Python, not in the LLM:

**Portfolio weights:**
```python
total_value = sum(price * shares for ticker, (price, shares) in holdings_data.items())
allocation = {ticker: (price * shares) / total_value for ticker, (price, shares) in holdings_data.items()}
```

**Diversification score (0–10):**
```python
def _diversification_score(holdings_data, sector_allocation):
    score = 10.0
    # Penalize concentration: largest single position > 40%
    max_alloc = max(allocation.values())
    if max_alloc > 0.4:
        score -= (max_alloc - 0.4) * 10
    # Penalize lack of bonds/fixed income
    has_bonds = any("Fixed Income" in sec for sec in sector_allocation)
    if not has_bonds:
        score -= 1.5
    # Penalize fewer than 3 positions
    if len(holdings_data) < 3:
        score -= 2.0
    return max(0, round(score, 1))
```

**Goal planning PMT formula:**
```python
r = annual_return_rate / 12  # monthly rate
n = time_horizon_years * 12  # total months
FV_current = current_savings * (1 + r) ** n
remaining = goal_amount - FV_current
monthly_needed = remaining * r / ((1 + r) ** n - 1)
```

**Why not use the LLM for math:** LLMs are unreliable for multi-step arithmetic. A compounding interest calculation over 30 years with 7% annual return and monthly contributions involves enough floating-point operations that LLM hallucination is a real risk. Python computes it deterministically; the LLM only explains the result in natural language.

---

### 4.6 Streamlit State Management

The Streamlit app maintains conversation state in `st.session_state`:

```python
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
```

**Thread ID per session:** Each browser tab gets a unique `thread_id` on load. The LangGraph `MemorySaver` uses this as the checkpoint key, so conversations are isolated between tabs.

**Message display:** The `messages` list in session state is a display cache only — it mirrors what's stored in `MemorySaver` but is read from `st.session_state` for rendering (avoids calling LangGraph state reads on every re-render).

**Streamlit re-runs:** Every user interaction triggers a full script re-run. The `FinnieGraph` instance is cached with `@st.cache_resource` to avoid re-initializing the LangGraph graph and LLM clients on every re-run:

```python
@st.cache_resource
def get_graph():
    return FinnieGraph()
```

---

### 4.7 Memory Scaling and MemorySaver Limitations

`MemorySaver` stores checkpoints in memory (Python dict). This has two implications:

1. **Process restart resets all sessions.** All conversation history is lost when the Streamlit process restarts. This is acceptable for an educational tool with no authentication, but would need to change for a production deployment.

2. **Memory grows with session count.** Each active `thread_id` holds the full message history. For a single-server deployment with light usage, this is not a concern. At scale, a `SqliteSaver` or `PostgresSaver` would provide durability and allow the process to restart without losing sessions:

```python
# Production swap: replace MemorySaver with SqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("finnie_sessions.db")
graph = workflow.compile(checkpointer=checkpointer)
```

No other code changes are needed — LangGraph's checkpointer interface is consistent across implementations.
