"""
src/agents/tax_agent.py
Tax Education Agent — classifies the user's tax question, calculates
relevant metrics (capital gains tax, account limits), and generates a
plain-English explanation via LLM + RAG.

Supported scenarios:
    1. Capital gains  — "I sold AAPL after 8 months with a $5,000 gain"
    2. Account limits — "How much can I contribute to my Roth IRA?"
    3. Tax-loss harvesting — "I have a $3,000 loss. Can I use it?"
    4. General tax question — answered via RAG + LLM

Usage:
    from src.agents.tax_agent import TaxEducationAgent
    agent = TaxEducationAgent()

    result = agent.run("I sold stock after 14 months with a $8,000 gain. I'm in the 22% bracket.")
    print(result["answer"])
    print(result["metrics"])
"""

import re
import yfinance as yf
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import load_llm
from src.rag.retriever import get_retriever
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Tax constants (2024 US, single filer) ─────────────────────────────────────

# Long-term capital gains rate by ordinary income bracket label
LONG_TERM_RATE: dict[str, float] = {
    "10%":  0.00,
    "12%":  0.00,
    "22%":  0.15,
    "24%":  0.15,
    "32%":  0.15,
    "35%":  0.20,
    "37%":  0.20,
}

# Short-term gains are taxed as ordinary income — just use the bracket rate
SHORT_TERM_RATE: dict[str, float] = {
    "10%":  0.10,
    "12%":  0.12,
    "22%":  0.22,
    "24%":  0.24,
    "32%":  0.32,
    "35%":  0.35,
    "37%":  0.37,
}

# 2024 contribution limits
ACCOUNT_LIMITS: dict[str, dict] = {
    "401k": {
        "limit":        23_000,
        "catch_up":     30_500,   # age >= 50
        "tax_treatment": "Pre-tax (Traditional) or after-tax (Roth 401k)",
    },
    "ira": {
        "limit":        7_000,
        "catch_up":     8_000,
        "tax_treatment": "Traditional: pre-tax deductible; Roth: after-tax, tax-free growth",
    },
    "roth_ira": {
        "limit":        7_000,
        "catch_up":     8_000,
        "tax_treatment": "After-tax contributions; qualified withdrawals are tax-free",
    },
    "hsa": {
        "limit_self":   4_150,
        "limit_family": 8_300,
        "catch_up":     1_000,    # age >= 55
        "tax_treatment": "Triple tax advantage: pre-tax in, tax-free growth, tax-free out (medical)",
    },
}

class _StockPosition(BaseModel):
    ticker:         str          = Field(description="Stock ticker symbol, e.g. AAPL")
    shares:         float        = Field(description="Number of shares held")
    purchase_price: float        = Field(description="Price paid per share when purchased")
    current_price:  float | None = Field(None, description="Current or selling price per share if mentioned in context")


class _TaxQuery(BaseModel):
    scenario:       Literal["capital_gains", "account_limits", "tax_loss", "general"] = Field(
                        description="Tax scenario: capital_gains for sale/gain/profit, "
                                    "account_limits for IRA/401k/HSA contributions, "
                                    "tax_loss for harvesting losses, general otherwise")
    positions:      list[_StockPosition] | None = Field(None, description=(
        "Extract each individual stock position when the user lists multiple stocks with shares and prices. "
        "Populate current_price if a selling price or current market price is mentioned for that ticker. "
        "Do NOT compute gains — just extract the raw numbers; Python will do the arithmetic."
    ))
    gain_or_loss:   float | None = Field(None, description=(
        "Total dollar gain or loss ONLY if stated explicitly as a single dollar amount. "
        "Leave null if individual stock positions are provided — gains will be computed from positions."
    ))
    holding_months: int   | None = Field(None, description=(
        "Holding period in months (12 months = 1 year). "
        "Return null if not mentioned — the caller will resolve from year references."
    ))
    bracket:        str          = Field("22%", description="Income tax bracket e.g. '22%'. Default 22% if not mentioned")
    account_type:   str   | None = Field(None, description="Account type: 401k, roth_ira, ira, or hsa")


PROMPT = ChatPromptTemplate.from_template("""
You are Finnie, a friendly financial education assistant specialising in US taxes.
Explain the tax situation below in plain English. Be accurate, beginner-friendly,
and encouraging. Always end with a disclaimer that this is educational, not tax advice,
and that the user should consult a qualified tax professional for their specific situation.

Tax Scenario: {scenario}

Calculated Metrics:
{metrics}

Financial Knowledge (from knowledge base):
{context}

Provide a clear explanation covering:
1. What tax rule applies and why
2. What the numbers mean in plain English
3. One actionable tip to improve their tax outcome
4. The disclaimer

Explanation:
""")


def _fetch_current_price(ticker: str) -> float | None:
    """Fetch live market price from yfinance for a single ticker."""
    try:
        info = yf.Ticker(ticker).info
        return (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )
    except Exception:
        return None


class TaxEducationAgent:

    def __init__(self):
        self.retriever    = get_retriever()
        self.llm          = load_llm()
        self.chain        = PROMPT | self.llm | StrOutputParser()
        self.query_parser = self.llm.with_structured_output(_TaxQuery)

    # ── Calculators ───────────────────────────────────────────────────────────

    def _calc_capital_gains(self, q: _TaxQuery, holding_months: int | None = None) -> dict:
        months       = holding_months if holding_months is not None else q.holding_months
        is_long_term = months is not None and months >= 12
        holding_type = "long_term" if is_long_term else "short_term"
        tax_rate     = (LONG_TERM_RATE if is_long_term else SHORT_TERM_RATE).get(q.bracket, 0.15)

        # Compute gain in Python from per-stock positions (avoids LLM arithmetic errors).
        # Fall back to the stated gain_or_loss only when no positions are available.
        per_stock: list[dict] = []
        if q.positions:
            total_gain = 0.0
            for pos in q.positions:
                current_price = pos.current_price
                if current_price is None:
                    current_price = _fetch_current_price(pos.ticker)
                    if current_price:
                        log.info("TaxEducationAgent | fetched live price %s=%.2f", pos.ticker, current_price)
                if current_price is not None:
                    stock_gain = (current_price - pos.purchase_price) * pos.shares
                    per_stock.append({
                        "ticker":         pos.ticker,
                        "shares":         pos.shares,
                        "purchase_price": pos.purchase_price,
                        "current_price":  current_price,
                        "gain":           round(stock_gain, 2),
                    })
                    total_gain += stock_gain
            gain = total_gain if per_stock else (q.gain_or_loss or 0.0)
        else:
            gain = q.gain_or_loss or 0.0

        estimated_tax = round(gain * tax_rate, 2)
        result = {
            "scenario":              "capital_gains",
            "gain":                  round(gain, 2),
            "holding_period_months": months,
            "holding_type":          holding_type,
            "income_bracket":        q.bracket,
            "tax_rate_pct":          round(tax_rate * 100, 1),
            "estimated_tax":         estimated_tax,
            "net_gain":              round(gain - estimated_tax, 2),
        }
        if per_stock:
            result["per_stock"] = per_stock
        return result

    def _calc_account_limits(self, q: _TaxQuery) -> dict:
        account = (q.account_type or "ira").lower().replace(" ", "_")
        info    = ACCOUNT_LIMITS.get(account, ACCOUNT_LIMITS["ira"])
        return {
            "scenario":     "account_limits",
            "account_type": account.upper().replace("_", " "),
            **info,
        }

    def _calc_tax_loss(self, q: _TaxQuery) -> dict:
        loss                 = abs(q.gain_or_loss or 0.0)
        deductible_this_year = min(loss, 3_000)
        carryforward         = max(loss - 3_000, 0.0)
        rate                 = SHORT_TERM_RATE.get(q.bracket, 0.22)
        return {
            "scenario":             "tax_loss_harvesting",
            "total_loss":           round(loss, 2),
            "deductible_this_year": round(deductible_this_year, 2),
            "carryforward_to_next": round(carryforward, 2),
            "income_bracket":       q.bracket,
            "estimated_tax_saving": round(deductible_this_year * rate, 2),
        }

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_metrics(self, metrics: dict) -> str:
        lines = []
        currency_keys = {"tax", "gain", "loss", "saving", "net", "deductible", "carryforward", "price"}
        for k, v in metrics.items():
            if k in ("scenario", "per_stock"):
                continue
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                lines.append(
                    f"  {label}: USD {v:,.2f}"
                    if any(ck in k for ck in currency_keys)
                    else f"  {label}: {v}"
                )
            else:
                lines.append(f"  {label}: {v}")
        if metrics.get("per_stock"):
            lines.append("\n  Per-stock breakdown (Python-computed):")
            for s in metrics["per_stock"]:
                lines.append(
                    f"    {s['ticker']}: {s['shares']:.0f} shares × "
                    f"(USD {s['current_price']:,.2f} − USD {s['purchase_price']:,.2f}) "
                    f"= USD {s['gain']:,.2f}"
                )
        return "\n".join(lines)

    def _get_rag_context(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Answer a tax-related question.

        Args:
            query: Natural language tax question.

        Returns:
            {
                "answer":   str,
                "metrics":  dict  (empty for general questions),
                "scenario": str,
                "error":    str | None,
            }
        """
        if not query or not query.strip():
            return {
                "answer":   "Please ask a tax question. For example: 'I sold stock after 14 months with a $8,000 gain — how much tax do I owe?'",
                "metrics":  {},
                "scenario": "none",
                "error":    "no_input",
            }

        dated_query = f"[Today's date: {date.today().isoformat()}]\n{query}"
        parsed   = self.query_parser.invoke(dated_query)
        scenario = parsed.scenario
        log.info("TaxEducationAgent | scenario=%s | query=%r", scenario, query[:60])

        # ── Deterministic holding-period override ─────────────────────────────
        # LLMs are unreliable at date arithmetic. Apply these rules in order:
        # 1. Explicit "long term" phrase → definitely ≥ 12 months
        # 2. Purchase year in text (e.g. "in 2023") → compute months from Jan of that year
        # This overrides the LLM-parsed holding_months so long-term is never misclassified.
        holding_months = parsed.holding_months
        q_lower = query.lower()
        _LONG_TERM_PHRASES = ("long term", "long-term", "more than a year", "over a year",
                              "several years", "many years", "years ago")
        if any(p in q_lower for p in _LONG_TERM_PHRASES):
            holding_months = 999
            log.info("TaxEducationAgent | holding_months=999 (long-term phrase detected)")
        elif holding_months is None or holding_months < 12:
            year_m = re.search(r'\b(20\d{2})\b', query)
            if year_m:
                purchase_year = int(year_m.group(1))
                today = date.today()
                computed = (today.year - purchase_year) * 12 + today.month
                if computed > (holding_months or 0):
                    holding_months = computed
                    log.info("TaxEducationAgent | holding_months=%d (from year %d)", holding_months, purchase_year)
        # ─────────────────────────────────────────────────────────────────────

        if scenario == "capital_gains":
            if parsed.gain_or_loss is None and not parsed.positions:
                return {
                    "answer": (
                        "To calculate your capital gains tax I need two numbers:\n"
                        "1. Your **purchase price per share** (what you paid when you bought)\n"
                        "2. Your **selling price per share** (or I can look up the current price)\n\n"
                        "For example: 'I bought NVDA at $50 per share and am selling at the current price.' "
                        "Once I have that, I can calculate your exact gain and tax."
                    ),
                    "metrics":  {},
                    "scenario": "capital_gains",
                    "error":    "missing_gain",
                }
            metrics = self._calc_capital_gains(parsed, holding_months=holding_months)
        elif scenario == "account_limits":
            metrics = self._calc_account_limits(parsed)
        elif scenario == "tax_loss":
            metrics = self._calc_tax_loss(parsed)
        else:
            metrics = {}

        metrics_str = self._format_metrics(metrics) if metrics else "No specific metrics calculated — general question."
        context     = self._get_rag_context(query)

        answer = self.chain.invoke({
            "scenario": scenario.replace("_", " ").title(),
            "metrics":  metrics_str,
            "context":  context,
        })

        return {
            "answer":   answer,
            "metrics":  metrics,
            "scenario": scenario,
            "error":    None,
        }


# ── Helpers (module-level, for unit tests) ────────────────────────────────────

def calc_capital_gains(gain: float, holding_months: int, bracket: str = "22%") -> dict:
    is_long_term  = holding_months >= 12
    holding_type  = "long_term" if is_long_term else "short_term"
    rate_table    = LONG_TERM_RATE if is_long_term else SHORT_TERM_RATE
    tax_rate      = rate_table.get(bracket, 0.15)
    estimated_tax = round(gain * tax_rate, 2)
    return {
        "scenario":              "capital_gains",
        "gain":                  round(gain, 2),
        "holding_period_months": holding_months,
        "holding_type":          holding_type,
        "income_bracket":        bracket,
        "tax_rate_pct":          round(tax_rate * 100, 1),
        "estimated_tax":         estimated_tax,
        "net_gain":              round(gain - estimated_tax, 2),
    }


def calc_tax_loss(loss: float, bracket: str = "22%") -> dict:
    deductible   = min(loss, 3_000)
    carryforward = max(loss - 3_000, 0.0)
    rate         = SHORT_TERM_RATE.get(bracket, 0.22)
    return {
        "scenario":             "tax_loss_harvesting",
        "total_loss":           round(loss, 2),
        "deductible_this_year": round(deductible, 2),
        "carryforward_to_next": round(carryforward, 2),
        "income_bracket":       bracket,
        "estimated_tax_saving": round(deductible * rate, 2),
    }


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = TaxEducationAgent()

    cases = [
        "I sold AAPL after 8 months with a $5,000 gain. I'm in the 22% bracket.",
        "I sold MSFT after 2 years with a $10,000 gain. I'm in the 24% bracket.",
        "How much can I contribute to my Roth IRA in 2024?",
        "I have a $4,500 loss this year. Can I use tax-loss harvesting?",
        "What is the difference between a Traditional and Roth IRA?",
    ]

    for q in cases:
        print(f"\nQ: {q}")
        result = agent.run(q)
        print(f"Scenario : {result['scenario']}")
        if result["metrics"]:
            print(f"Metrics  : {result['metrics']}")
        print(f"Answer   :\n{result['answer']}")
        print("-" * 60)
