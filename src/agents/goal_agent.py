"""
src/agents/goal_agent.py
Financial Goal Planning Agent — parses a user's savings goal, calculates
required monthly contributions, and generates a plain-English plan via LLM.

Usage:
    from src.agents.goal_agent import GoalPlanningAgent
    agent = GoalPlanningAgent()

    # Structured input
    result = agent.run(
        goal_amount=50000,
        time_horizon_years=5,
        current_savings=10000,
        risk_profile="moderate",
    )

    # Natural language input
    result = agent.run(query="I want to save $50k for a house in 5 years. I have $10k saved.")

    print(result["answer"])
    print(result["metrics"])
"""

import re

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import load_llm
from src.rag.retriever import get_retriever
from src.utils.logger import get_logger

log = get_logger(__name__)


def _fv(pv: float, monthly_rate: float, months: float, monthly_pmt: float) -> float:
    """FV = PV*(1+r)^n + PMT*((1+r)^n - 1)/r; degrades to simple sum when rate=0."""
    if monthly_rate > 0 and months > 0:
        g = (1 + monthly_rate) ** months
        return pv * g + monthly_pmt * (g - 1) / monthly_rate
    return pv + monthly_pmt * months


class _GoalParams(BaseModel):
    goal_amount:        float | None = Field(None, description=(
        "Target savings amount in dollars. "
        "Users may phrase it many ways: 'save $30k', 'need 50,000 dollars', 'want to have 100k', "
        "'retire with 3M', 'Need 5 Million', '30,000 bucks', '100K USD', etc. "
        "Return null if not explicitly stated."
    ))
    time_horizon_years: float | None = Field(None, description=(
        "Number of YEARS FROM NOW until the goal. "
        "If the user says 'I am X years old and want to retire at age Y', compute Y - X. "
        "If the user says 'in Z years', use Z. "
        "If the user says 'by year YYYY', 'in YYYY', or 'until YYYY', compute YYYY - 2026. "
        "Never use the retirement age itself as the horizon. "
        "Return null only if no age or timeline is mentioned at all."
    ))
    current_savings:    float | None = Field(None, description=(
        "Amount already saved or invested in dollars. "
        "Do NOT use salary, income, or annual earnings as current savings. "
        "Return null (not 0) if no saved/invested amount is mentioned."
    ))
    current_age:        float | None = Field(None, description=(
        "User's current age if mentioned, else null. "
        "Fill this whenever both current age and retirement age are present so the horizon can be verified."
    ))
    retirement_age:     float | None = Field(None, description=(
        "Target retirement age if mentioned, else null. "
        "Fill this whenever both current age and retirement age are present so the horizon can be verified."
    ))
    risk_profile:       str   | None = Field(None, description=(
        "User's risk tolerance. Map to one of: 'conservative', 'moderate', 'aggressive'. "
        "Words like 'aggressive', 'high risk', 'risky' → 'aggressive'; "
        "'moderate', 'balanced', 'medium' → 'moderate'; "
        "'conservative', 'safe', 'low risk' → 'conservative'. "
        "Return null if not mentioned."
    ))
    annual_contribution: float | None = Field(None, description=(
        "Total annual savings or investment contribution in dollars. "
        "Sum ALL sources explicitly mentioned (personal + employer match + any other). "
        "Do NOT include goal_amount, current_savings, or target amounts as contributions. "
        "If given as monthly, multiply by 12. "
        "Return null if no ongoing contribution amount is mentioned."
    ))


# Annual return assumptions by risk profile
RETURN_RATES = {
    "conservative": 0.04,
    "moderate":     0.07,
    "aggressive":   0.10,
}

PROJECTION_PROMPT = ChatPromptTemplate.from_template("""
You are Finnie, a friendly financial education assistant.
The user wants to know how their ongoing contributions will grow over time.
Be encouraging and honest. Always end with a disclaimer that this is educational, not financial advice.
Use currency like "USD 50,000" instead of "$50,000" so the answer renders correctly in Markdown.

Projection Details:
{metrics}

Financial Knowledge (from knowledge base):
{context}

User's risk tolerance: {risk_profile}
Assumed annual return: {annual_return_pct}%

Provide a clear explanation covering:
1. How much they are projected to have at the end of the period (invested vs. cash only)
2. Whether their current contribution rate looks healthy for long-term goals
3. How the assumed {annual_return_pct}% annual return drives the outcome
4. One practical tip (e.g. increasing contributions over time, employer match maximization)
5. A brief note on the power of starting early / compounding

Plan:
""")

PROMPT = ChatPromptTemplate.from_template("""
You are Finnie, a friendly financial education assistant.
Help the user understand their savings goal and how to achieve it.
If any information is missing, ask the user for it in a friendly way.
Be encouraging and honest. Always end with a disclaimer that this is educational, not financial advice.
Use currency like "USD 50,000" instead of "$50,000" so the answer renders correctly in Markdown.
Users can also say the currency in various ways, e.g. "50k dollars", "30,000 bucks", "100K USD", "3M", "3 Million",etc.

Goal Metrics:
{metrics}

Financial Knowledge (from knowledge base):
{context}

User's risk tolerance: {risk_profile}
Assumed annual return: {annual_return_pct}%

IMPORTANT — use ONLY the numbers from "Goal Metrics" above. Do NOT compute your own figures.
Key metric meanings:
  • "Monthly Savings (no growth)"   = cash needed per month with zero investment return
  • "Monthly Savings (with growth)" = how much to invest per month to hit the goal exactly
  • "Projected Value"               = the ending balance if the user makes those monthly_with_growth contributions AND earns the assumed return — it is NOT what the current savings alone will become.

If "User's Stated Contribution Scenario" appears in the metrics:
  • START your answer with a direct YES or NO based on "Will they reach the goal?"
  • Show "Projected Value (with stated)" as what they will actually have
  • Compare it to the goal and explain the surplus or shortfall
  • Then mention what the minimum required contribution is for reference

If "Monthly Savings (with growth)" > 0, the user MUST make those contributions. Never say they don't need to save more unless that number is truly 0 or negative.

Provide a clear plan covering:
1. Direct answer: will their stated contribution reach the goal? (if stated)
2. The required monthly savings — quote BOTH numbers from the metrics (no-growth and with-growth)
3. How the assumed {annual_return_pct}% annual return affects the outcome
4. One practical tip for staying on track
5. A note on what happens if they invest vs. keep the money in cash

Plan:
""")


class GoalPlanningAgent:

    def __init__(self):
        self.retriever        = get_retriever()
        self.llm              = load_llm()
        self.chain            = PROMPT | self.llm | StrOutputParser()
        self.projection_chain = PROJECTION_PROMPT | self.llm | StrOutputParser()
        self.goal_parser      = self.llm.with_structured_output(_GoalParams)

    def _parse_goal_from_text(self, text: str) -> dict:
        parsed = self.goal_parser.invoke(f"Extract financial goal details from this text: {text}")

        # Verify/correct horizon using age arithmetic when both ages are present
        horizon = parsed.time_horizon_years
        if parsed.current_age and parsed.retirement_age:
            age_based = parsed.retirement_age - parsed.current_age
            if age_based > 0 and (horizon is None or abs(horizon - age_based) > 1):
                log.info(
                    "GoalParser | correcting horizon from %.0f to %.0f using age arithmetic",
                    horizon or 0, age_based,
                )
                horizon = age_based

        return {
            "goal_amount":        parsed.goal_amount,
            "time_horizon_years": horizon,
            "current_savings":    parsed.current_savings,
            "risk_profile":       parsed.risk_profile,
            "annual_contribution": parsed.annual_contribution,
        }

    @staticmethod
    def _calculate_metrics(
        goal_amount: float,
        time_horizon_years: float,
        current_savings: float,
        annual_return: float,
        annual_contribution: float | None = None,
    ) -> dict:
        months       = time_horizon_years * 12
        monthly_rate = annual_return / 12
        gap          = max(goal_amount - current_savings, 0.0)

        monthly_no_growth = gap / months if months > 0 else gap

        if monthly_rate > 0 and months > 0:
            g = (1 + monthly_rate) ** months
            monthly_with_growth = max(goal_amount - current_savings * g, 0.0) * monthly_rate / (g - 1)
        else:
            monthly_with_growth = monthly_no_growth

        projected_value = _fv(current_savings, monthly_rate, months, monthly_with_growth)

        result = {
            "goal_amount":         round(goal_amount, 2),
            "time_horizon_years":  round(time_horizon_years, 2),
            "time_horizon_months": round(months, 1),
            "current_savings":     round(current_savings, 2),
            "gap":                 round(gap, 2),
            "annual_return_pct":   round(annual_return * 100, 1),
            "monthly_no_growth":   round(monthly_no_growth, 2),
            "monthly_with_growth": round(monthly_with_growth, 2),
            "projected_value":     round(projected_value, 2),
            "goal_achievable":     monthly_with_growth >= 0,
        }

        if annual_contribution and annual_contribution > 0:
            monthly_stated   = annual_contribution / 12
            projected_stated = _fv(current_savings, monthly_rate, months, monthly_stated)
            result["stated_annual_contribution"] = round(annual_contribution, 2)
            result["projected_with_stated"]      = round(projected_stated, 2)
            result["on_track"]                   = projected_stated >= goal_amount

        return result

    @staticmethod
    def _format_metrics_for_llm(metrics: dict) -> str:
        lines = [
            f"Goal Amount:                   USD {metrics['goal_amount']:,.2f}",
            f"Time Horizon:                  {metrics['time_horizon_years']} years ({metrics['time_horizon_months']} months)",
            f"Current Savings:               USD {metrics['current_savings']:,.2f}",
            f"Gap to Goal:                   USD {metrics['gap']:,.2f}",
            f"Assumed Annual Return:          {metrics['annual_return_pct']}%",
            f"Monthly Savings (no growth):   USD {metrics['monthly_no_growth']:,.2f}",
            f"Monthly Savings (with growth): USD {metrics['monthly_with_growth']:,.2f}",
            f"Projected Value (saving minimum + investing): USD {metrics['projected_value']:,.2f}",
        ]
        if "stated_annual_contribution" in metrics:
            verdict = "YES — goal will be exceeded" if metrics["on_track"] else "NO — goal will not be reached"
            lines += [
                "",
                "--- User's Stated Contribution Scenario ---",
                f"Stated Annual Contribution:    USD {metrics['stated_annual_contribution']:,.2f}",
                f"Projected Value (with stated): USD {metrics['projected_with_stated']:,.2f}",
                f"Will they reach the goal?      {verdict}",
            ]
        return "\n".join(lines)

    @staticmethod
    def _calculate_projection_metrics(
        annual_contribution: float,
        time_horizon_years: float,
        current_savings: float,
        annual_return: float,
    ) -> dict:
        months               = time_horizon_years * 12
        monthly_rate         = annual_return / 12
        monthly_contribution = annual_contribution / 12

        projected_value = _fv(current_savings, monthly_rate, months, monthly_contribution)
        projected_cash  = current_savings + monthly_contribution * months

        return {
            "annual_contribution":  round(annual_contribution, 2),
            "monthly_contribution": round(monthly_contribution, 2),
            "time_horizon_years":   round(time_horizon_years, 2),
            "time_horizon_months":  round(months, 1),
            "current_savings":      round(current_savings, 2),
            "annual_return_pct":    round(annual_return * 100, 1),
            "projected_value":      round(projected_value, 2),
            "projected_cash":       round(projected_cash, 2),
        }

    @staticmethod
    def _format_projection_metrics_for_llm(metrics: dict) -> str:
        return "\n".join([
            f"Annual Contribution:           USD {metrics['annual_contribution']:,.2f}",
            f"Monthly Contribution:          USD {metrics['monthly_contribution']:,.2f}",
            f"Time Horizon:                  {metrics['time_horizon_years']} years ({metrics['time_horizon_months']} months)",
            f"Current Savings:               USD {metrics['current_savings']:,.2f}",
            f"Assumed Annual Return:          {metrics['annual_return_pct']}%",
            f"Projected Value (if invested): USD {metrics['projected_value']:,.2f}",
            f"Projected Value (cash only):   USD {metrics['projected_cash']:,.2f}",
        ])

    @staticmethod
    def _escape_markdown_currency(text: str) -> str:
        return re.sub(r"(?<!\\)\$", r"\\$", text)

    def _get_rag_context(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def run(
        self,
        query: str = "",
        goal_amount: float | None = None,
        time_horizon_years: float | None = None,
        current_savings: float | None = None,
        risk_profile: str | None = None,
        annual_contribution: float | None = None,
    ) -> dict:
        """
        Plan a financial goal.

        Args:
            query:               Natural language description of the goal.
            goal_amount:         Target savings amount in dollars.
            time_horizon_years:  Years until the goal.
            current_savings:     Amount already saved (default 0).
            risk_profile:        "conservative" | "moderate" | "aggressive"

        Returns:
            {
                "answer":  str,
                "metrics": dict,
                "error":   str | None
            }
        """
        if query:
            parsed = self._parse_goal_from_text(query)
            goal_amount         = goal_amount        or parsed["goal_amount"]
            time_horizon_years  = time_horizon_years or parsed["time_horizon_years"]
            if current_savings is None:
                current_savings = parsed["current_savings"]
            if risk_profile is None:
                risk_profile = parsed.get("risk_profile")
            # Caller-provided annual_contribution (from state) wins; fall back to query parse
            if annual_contribution is None:
                annual_contribution = parsed.get("annual_contribution")
        elif goal_amount is None or time_horizon_years is None:
            return {
                "answer":  "Please describe your financial goal so I can help you plan for it.",
                "metrics": {},
                "error":   "no_input",
            }

        risk_profile = (risk_profile or "moderate").lower()
        if current_savings is None:
            current_savings = 0.0

        log.info("GoalPlanningAgent | goal=%s | contrib=%s/yr | horizon=%s yr | risk=%s",
                 f"${goal_amount:,.0f}" if goal_amount else "projection",
                 f"${annual_contribution:,.0f}" if annual_contribution else "none",
                 time_horizon_years or "?",
                 risk_profile)
        annual_return = RETURN_RATES.get(risk_profile, RETURN_RATES["moderate"])

        # Projection mode: user asks "how much will I have?" without a target goal
        if not goal_amount and annual_contribution and time_horizon_years:
            metrics     = self._calculate_projection_metrics(annual_contribution, time_horizon_years, current_savings, annual_return)
            metrics_str = self._format_projection_metrics_for_llm(metrics)
            context     = self._get_rag_context(query or f"saving {annual_contribution} per year for {time_horizon_years} years")
            answer = self.projection_chain.invoke({
                "metrics":           metrics_str,
                "context":           context,
                "risk_profile":      risk_profile,
                "annual_return_pct": metrics["annual_return_pct"],
            })
            answer = self._escape_markdown_currency(answer)
            log.info("GoalPlanningAgent | projection done | projected=$%.0f", metrics["projected_value"])
            return {"answer": answer, "metrics": metrics, "error": None}

        if not goal_amount or not time_horizon_years:
            return {
                "answer": (
                    "I couldn't identify your goal amount or timeline. "
                    "Try something like: 'I want to save $30,000 for a car in 3 years.'"
                ),
                "metrics": {},
                "error":   "parse_failure",
            }

        if goal_amount <= 0 or time_horizon_years <= 0:
            return {
                "answer":  "Goal amount and time horizon must both be positive.",
                "metrics": {},
                "error":   "invalid_input",
            }

        metrics     = self._calculate_metrics(goal_amount, time_horizon_years, current_savings, annual_return, annual_contribution)
        metrics_str = self._format_metrics_for_llm(metrics)
        rag_query   = query or f"saving {goal_amount} dollars in {time_horizon_years} years"
        context     = self._get_rag_context(rag_query)

        answer = self.chain.invoke({
            "metrics":           metrics_str,
            "context":           context,
            "risk_profile":      risk_profile,
            "annual_return_pct": metrics["annual_return_pct"],
        })
        answer = self._escape_markdown_currency(answer)

        log.info("GoalPlanningAgent | done | monthly_with_growth=$%.0f | monthly_no_growth=$%.0f",
                 metrics["monthly_with_growth"], metrics["monthly_no_growth"])
        return {
            "answer":  answer,
            "metrics": metrics,
            "error":   None,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def calculate_metrics(
    goal_amount: float,
    time_horizon_years: float,
    current_savings: float = 0.0,
    annual_return: float = 0.07,
) -> dict:
    """Module-level helper — useful for unit tests and external callers."""
    return GoalPlanningAgent._calculate_metrics(goal_amount, time_horizon_years, current_savings, annual_return)


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = GoalPlanningAgent()

    result = agent.run(
        query="I have 600K and want to retire in 20 Years. If I invest $55,000 per annum for the next 20 years, will I be able to save 3 Million"
    )
    m = result["metrics"]
    if "monthly_no_growth" in m:
        print(f"Monthly (no growth):   ${m['monthly_no_growth']:,.2f}")
        print(f"Monthly (with growth): ${m['monthly_with_growth']:,.2f}")
    elif "projected_value" in m:
        print(f"Annual contribution:   ${m['annual_contribution']:,.2f}")
        print(f"Monthly contribution:  ${m['monthly_contribution']:,.2f}")
        print(f"Projected value:       ${m['projected_value']:,.2f}")
        print(f"Cash-only value:       ${m['projected_cash']:,.2f}")
    else:
        print(f"Error: {result.get('error')} — {result.get('answer')}")
    print(f"\nPlan:\n{result['answer']}")
