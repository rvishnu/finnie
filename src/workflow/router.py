"""
src/workflow/router.py
Query classifier and parameter extractor for the LangGraph workflow.
Pure functions — no graph or agent dependencies, fully unit-testable.
"""

import re
from langchain_core.messages import BaseMessage
from src.core.llm import load_llm


# ── Routing keyword rules ─────────────────────────────────────────────────────

ROUTING_RULES: dict[str, list[str]] = {
    "portfolio": [
        "portfolio", "my holdings", "my stocks", "my shares",
        "i own", "i hold", "allocation", "diversif", "rebalance",
    ],
    "market": [
        "stock price", "how is", "how's", "trading at", "pe ratio",
        "market cap", "earnings", "52 week", "share price", "analyst",
    ],
    "goal": [
        "save", "saving", "retire", "retirement", "reach", "target",
        "plan for", "years from now", "how much do i need",
        "monthly contribution", "i want to have", "i want to reach",
    ],
    "news": [
        "news", "headline", "latest", "what's happening",
        "what happened", "recent", "announced", "announce",
    ],
    "tax": [
        "tax", "capital gains", "ira", "401k", "roth", "hsa",
        "deduct", "i sold", "harvest", "write off", "contribution limit",
    ],
    "qa": [
        "what is", "explain", "how does", "tell me about",
        "define", "difference between", "what are",
    ],
}

# Phrases that are clearly outside the six-agent scope
OUT_OF_SCOPE_SIGNALS: list[str] = [
    "start a business", "business plan", "business valuation",
    "home insurance", "health insurance", "life insurance",
    "car insurance", "auto insurance", "mortgage rate",
    "legal advice", "lawyer", "attorney", "medical",
]


# ── Public: classifier ────────────────────────────────────────────────────────

def classify_query(
    query: str,
    messages: list[BaseMessage] | None = None,
    state: dict | None = None,
) -> tuple[str, float]:
    """
    Classify a user query into an agent name with a confidence score.

    Returns:
        (agent_name, confidence)
        agent_name ∈ {portfolio, market, goal, news, tax, qa, fallback}
        confidence ∈ [0.0, 1.0]
    """
    q = query.lower()

    # Out-of-scope check — fast exit before keyword scoring
    if any(signal in q for signal in OUT_OF_SCOPE_SIGNALS):
        return "fallback", 0.3

    # Score each agent by keyword hits
    scores: dict[str, int] = {agent: 0 for agent in ROUTING_RULES}
    for agent, keywords in ROUTING_RULES.items():
        for kw in keywords:
            if kw in q:
                scores[agent] += 1

    best_agent = max(scores, key=lambda a: scores[a])
    best_score = scores[best_agent]

    # Two or more keyword hits → high confidence, no LLM needed
    if best_score >= 2:
        return best_agent, 0.9

    # One keyword hit → LLM confirms or overrides
    if best_score == 1:
        llm_agent = _llm_classify(query, messages)
        confidence = 0.8 if llm_agent == best_agent else 0.7
        return llm_agent, confidence

    # No keyword hit → full LLM classification
    llm_agent = _llm_classify(query, messages)
    return llm_agent, 0.5


def _llm_classify(
    query: str,
    messages: list[BaseMessage] | None = None,
) -> str:
    """LLM fallback classifier. Returns one of the 7 agent names."""
    history = ""
    if messages:
        recent  = messages[-4:]  # last 4 messages for follow-up context
        history = "\n".join(
            f"{'User' if m.type == 'human' else 'Assistant'}: {str(m.content)[:200]}"
            for m in recent
        )

    prompt = (
        "Classify the user query into exactly one category:\n"
        "  portfolio  — the user's own stock holdings or portfolio analysis\n"
        "  market     — stock prices, market data, company analysis\n"
        "  goal       — savings goals, retirement planning, financial targets\n"
        "  news       — recent financial news or headlines\n"
        "  tax        — taxes, capital gains, IRA/401k/HSA accounts\n"
        "  qa         — general financial education questions\n"
        "  fallback   — out of scope or completely unclear\n\n"
        + (f"Recent conversation:\n{history}\n\n" if history else "")
        + f"Query: {query}\n\n"
        "Return ONLY the category name, nothing else."
    )

    llm      = load_llm()
    response = llm.invoke(prompt).content.strip().lower()
    valid    = {"portfolio", "market", "goal", "news", "tax", "qa", "fallback"}
    return response if response in valid else "qa"


# ── Public: parameter extractor ───────────────────────────────────────────────

def extract_params(query: str) -> dict:
    """
    Extract financial parameters from a query.
    Returns only the fields that were explicitly found — never overwrites
    with None.

    Detects:
        current_savings    — "I have $200K", "actually I have $200K"
        goal_amount        — "want $2M", "reach $2 million", "target $500k"
        time_horizon_years — "in 20 years", "in 18 months"
        risk_profile       — "aggressive", "conservative", "moderate"
    """
    params: dict = {}

    # ── current_savings ───────────────────────────────────────────────────────
    # Requires ownership language so "Apple is worth $200B" doesn't match
    saved_match = re.search(
        r'(?:i\s+(?:now\s+)?have|already\s+saved?|current\s+savings?\s+(?:is|of)?'
        r'|actually\s+have|updated?\s+to|now\s+have|savings?\s+of)\s+'
        r'\$?\s*([\d,]+(?:\.\d+)?)\s*([kKmM]?)',
        query, re.IGNORECASE,
    )
    if saved_match:
        params["current_savings"] = _parse_amount(
            saved_match.group(1), saved_match.group(2)
        )

    # ── goal_amount ───────────────────────────────────────────────────────────
    goal_match = re.search(
        r'(?:want|reach|target|need|goal\s+of|retire\s+with|sell\s+for|worth|save\s+up\s+to|have)\s+'
        r'\$?\s*([\d,]+(?:\.\d+)?)\s*([kKmM]?)',
        query, re.IGNORECASE,
    )
    if goal_match:
        params["goal_amount"] = _parse_amount(
            goal_match.group(1), goal_match.group(2)
        )

    # ── time_horizon_years ────────────────────────────────────────────────────
    year_m  = re.search(r'\b(\d+(?:\.\d+)?)\s*years?\b',  query, re.IGNORECASE)
    month_m = re.search(r'\b(\d+(?:\.\d+)?)\s*months?\b', query, re.IGNORECASE)
    if year_m:
        params["time_horizon_years"] = float(year_m.group(1))
    elif month_m:
        params["time_horizon_years"] = round(float(month_m.group(1)) / 12, 4)

    # ── risk_profile ──────────────────────────────────────────────────────────
    q = query.lower()
    if any(w in q for w in ("conservative", "safe", "low risk", "cautious", "play it safe")):
        params["risk_profile"] = "conservative"
    elif any(w in q for w in ("aggressive", "high risk", "risky", "growth", "i can take risk")):
        params["risk_profile"] = "aggressive"
    elif any(w in q for w in ("moderate", "balanced", "medium risk")):
        params["risk_profile"] = "moderate"

    return params


def merge_params(existing: dict, new_params: dict) -> dict:
    """
    Merge newly extracted params into the existing state dict.
    New values always win — handles mid-conversation updates like
    "actually I have $200K" overriding an earlier $100K.
    Only keys present in new_params are touched.
    """
    updated = dict(existing)
    updated.update(new_params)
    return updated


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_amount(digits: str, suffix: str) -> float:
    val = float(digits.replace(",", ""))
    s   = suffix.lower()
    if s == "k": val *= 1_000
    elif s == "m": val *= 1_000_000
    return val
