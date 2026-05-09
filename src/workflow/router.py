"""
src/workflow/router.py
Parameter extractor for the LangGraph workflow.
Routing is handled by the LLM via ReAct tool selection in graph.py.
Pure functions — no graph or agent dependencies, fully unit-testable.
"""

import re


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
