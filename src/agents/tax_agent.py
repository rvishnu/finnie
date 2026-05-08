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
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import load_llm
from src.rag.retriever import get_retriever

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

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


class TaxEducationAgent:

    def __init__(self):
        self.retriever = get_retriever()
        self.llm       = load_llm()

    # ── Query classification ───────────────────────────────────────────────────

    def _classify(self, query: str) -> str:
        """Return one of: capital_gains | account_limits | tax_loss | general."""
        q = query.lower()
        if any(w in q for w in ("sold", "sell", "gain", "profit", "capital gain")):
            return "capital_gains"
        if any(w in q for w in ("roth", "ira", "401k", "hsa", "contribute", "contribution", "limit")):
            return "account_limits"
        if any(w in q for w in ("loss", "harvest", "tax-loss", "write off", "write-off")):
            return "tax_loss"
        return "general"

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_dollar(self, text: str) -> float | None:
        """Extract the first dollar amount from text (supports $5k, $5,000)."""
        m = re.search(
            r'\$\s*([\d,]+(?:\.\d+)?)\s*([kKmM]?)',
            text,
        )
        if not m:
            return None
        val = float(m.group(1).replace(",", ""))
        suf = m.group(2).lower()
        if suf == "k":
            val *= 1_000
        elif suf == "m":
            val *= 1_000_000
        return val

    def _parse_holding_months(self, text: str) -> int | None:
        """Extract holding period in months from text."""
        year_m  = re.search(r'\b(\d+(?:\.\d+)?)\s*years?\b',  text, re.IGNORECASE)
        month_m = re.search(r'\b(\d+(?:\.\d+)?)\s*months?\b', text, re.IGNORECASE)
        if year_m:
            return int(float(year_m.group(1)) * 12)
        if month_m:
            return int(float(month_m.group(1)))
        return None

    def _parse_bracket(self, text: str) -> str:
        """Extract income tax bracket string like '22%' from text."""
        m = re.search(r'\b(10|12|22|24|32|35|37)\s*%', text)
        return f"{m.group(1)}%" if m else "22%"  # default to 22% if not mentioned

    def _parse_account_type(self, text: str) -> str:
        """Detect which account type is being asked about."""
        q = text.lower()
        if "roth ira" in q or "roth" in q:
            return "roth_ira"
        if "hsa" in q:
            return "hsa"
        if "401k" in q or "401(k)" in q:
            return "401k"
        if "ira" in q:
            return "ira"
        return "ira"

    # ── Calculators ───────────────────────────────────────────────────────────

    def _calc_capital_gains(self, query: str) -> dict:
        gain             = self._parse_dollar(query) or 0.0
        holding_months   = self._parse_holding_months(query)
        bracket          = self._parse_bracket(query)
        is_long_term     = holding_months is not None and holding_months >= 12
        holding_type     = "long_term" if is_long_term else "short_term"
        tax_rate         = (LONG_TERM_RATE if is_long_term else SHORT_TERM_RATE).get(bracket, 0.15)
        estimated_tax    = round(gain * tax_rate, 2)

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

    def _calc_account_limits(self, query: str) -> dict:
        account   = self._parse_account_type(query)
        info      = ACCOUNT_LIMITS[account]
        return {
            "scenario":      "account_limits",
            "account_type":  account.upper().replace("_", " "),
            **info,
        }

    def _calc_tax_loss(self, query: str) -> dict:
        loss = self._parse_dollar(query) or 0.0
        # US allows up to $3,000 of net losses to offset ordinary income per year
        deductible_this_year   = min(loss, 3_000)
        carryforward           = max(loss - 3_000, 0.0)
        bracket                = self._parse_bracket(query)
        rate                   = SHORT_TERM_RATE.get(bracket, 0.22)
        tax_saving_this_year   = round(deductible_this_year * rate, 2)

        return {
            "scenario":               "tax_loss_harvesting",
            "total_loss":             round(loss, 2),
            "deductible_this_year":   round(deductible_this_year, 2),
            "carryforward_to_next":   round(carryforward, 2),
            "income_bracket":         bracket,
            "estimated_tax_saving":   tax_saving_this_year,
        }

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_metrics(self, metrics: dict) -> str:
        lines = []
        for k, v in metrics.items():
            if k == "scenario":
                continue
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                lines.append(f"  {label}: ${v:,.2f}" if "tax" in k or "gain" in k or "loss" in k or "saving" in k or "net" in k or "deductible" in k or "carryforward" in k else f"  {label}: {v}")
            else:
                lines.append(f"  {label}: {v}")
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

        scenario = self._classify(query)

        if scenario == "capital_gains":
            metrics = self._calc_capital_gains(query)
        elif scenario == "account_limits":
            metrics = self._calc_account_limits(query)
        elif scenario == "tax_loss":
            metrics = self._calc_tax_loss(query)
        else:
            metrics = {}

        metrics_str = self._format_metrics(metrics) if metrics else "No specific metrics calculated — general question."
        context     = self._get_rag_context(query)

        chain  = PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({
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
