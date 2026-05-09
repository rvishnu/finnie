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

import yaml
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import load_llm
from src.rag.retriever import get_retriever
from src.utils.logger import get_logger

log = get_logger(__name__)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

class _GoalParams(BaseModel):
    goal_amount:        float | None = Field(None, description="Target savings amount in dollars (e.g. 50000 for $50k)")
    time_horizon_years: float | None = Field(None, description="Years until the goal (e.g. 5.0)")
    current_savings:    float        = Field(0.0,  description="Amount already saved in dollars (0 if not mentioned)")


# Annual return assumptions by risk profile
RETURN_RATES = {
    "conservative": 0.04,
    "moderate":     0.07,
    "aggressive":   0.10,
}

PROMPT = ChatPromptTemplate.from_template("""
You are Finnie, a friendly financial education assistant.
Help the user understand their savings goal and how to achieve it.
If they didn't provide how much money the already have, just ask them what they have.
If any information is missing, ask the user for it in a friendly way.
Be encouraging and honest. Always end with a disclaimer that this is educational, not financial advice.

Goal Metrics:
{metrics}

Financial Knowledge (from knowledge base):
{context}

User's risk tolerance: {risk_profile}
Assumed annual return: {annual_return_pct}%

Provide a clear plan covering:
1. Whether the goal is realistic given the timeline
2. What monthly savings are required (with and without investment growth)
3. How the assumed {annual_return_pct}% annual return affects the outcome
4. One practical tip for staying on track
5. A note on what happens if they invest vs. keep the money in cash

Plan:
""")


class GoalPlanningAgent:

    def __init__(self):
        self.retriever   = get_retriever()
        self.llm         = load_llm()
        self.chain       = PROMPT | self.llm | StrOutputParser()
        self.goal_parser = self.llm.with_structured_output(_GoalParams)

    def _parse_goal_from_text(self, text: str) -> dict:
        """Extract goal_amount, time_horizon_years, and current_savings from natural language."""
        parsed = self.goal_parser.invoke(
            f"Extract the financial goal details from this text:\n\n{text}"
        )
        return {
            "goal_amount":        parsed.goal_amount,
            "time_horizon_years": parsed.time_horizon_years,
            "current_savings":    parsed.current_savings or 0.0,
        }

    def _calculate_metrics(
        self,
        goal_amount: float,
        time_horizon_years: float,
        current_savings: float,
        annual_return: float,
    ) -> dict:
        """Compute savings plan metrics using standard annuity formulas."""
        months       = time_horizon_years * 12
        monthly_rate = annual_return / 12
        gap          = max(goal_amount - current_savings, 0.0)

        # Monthly savings needed with NO investment growth (simple cash savings)
        monthly_no_growth = gap / months if months > 0 else gap

        # Monthly savings needed WITH compound growth
        # FV = PV*(1+r)^n + PMT*((1+r)^n - 1)/r  →  solve for PMT
        if monthly_rate > 0 and months > 0:
            growth_factor = (1 + monthly_rate) ** months
            fv_of_current = current_savings * growth_factor
            remaining_gap = goal_amount - fv_of_current
            monthly_with_growth = (
                max(remaining_gap, 0.0) * monthly_rate / (growth_factor - 1)
            )
        else:
            monthly_with_growth = monthly_no_growth

        # Projected final value if the user saves monthly_no_growth AND invests it
        if monthly_rate > 0 and months > 0:
            growth_factor   = (1 + monthly_rate) ** months
            projected_value = (
                current_savings * growth_factor
                + monthly_no_growth * (growth_factor - 1) / monthly_rate
            )
        else:
            projected_value = current_savings + monthly_no_growth * months

        return {
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

    def _format_metrics_for_llm(self, metrics: dict) -> str:
        return "\n".join([
            f"Goal Amount:                   ${metrics['goal_amount']:,.2f}",
            f"Time Horizon:                  {metrics['time_horizon_years']} years ({metrics['time_horizon_months']} months)",
            f"Current Savings:               ${metrics['current_savings']:,.2f}",
            f"Gap to Goal:                   ${metrics['gap']:,.2f}",
            f"Assumed Annual Return:          {metrics['annual_return_pct']}%",
            f"Monthly Savings (no growth):   ${metrics['monthly_no_growth']:,.2f}",
            f"Monthly Savings (with growth): ${metrics['monthly_with_growth']:,.2f}",
            f"Projected Value (if invested): ${metrics['projected_value']:,.2f}",
        ])

    def _get_rag_context(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def run(
        self,
        query: str = "",
        goal_amount: float | None = None,
        time_horizon_years: float | None = None,
        current_savings: float = 0.0,
        risk_profile: str = "moderate",
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
        log.info("GoalPlanningAgent | goal=%s | horizon=%s yr | risk=%s",
                 f"${goal_amount:,.0f}" if goal_amount else "unknown",
                 time_horizon_years or "?",
                 risk_profile)
        annual_return = RETURN_RATES.get(risk_profile, RETURN_RATES["moderate"])

        if goal_amount is None or time_horizon_years is None:
            if not query:
                return {
                    "answer":  "Please describe your financial goal so I can help you plan for it.",
                    "metrics": {},
                    "error":   "no_input",
                }
            parsed = self._parse_goal_from_text(query)
            goal_amount        = goal_amount        or parsed["goal_amount"]
            time_horizon_years = time_horizon_years or parsed["time_horizon_years"]
            current_savings    = current_savings    or parsed["current_savings"]

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

        metrics     = self._calculate_metrics(goal_amount, time_horizon_years, current_savings, annual_return)
        metrics_str = self._format_metrics_for_llm(metrics)
        rag_query   = query or f"saving {goal_amount} dollars in {time_horizon_years} years"
        context     = self._get_rag_context(rag_query)

        answer = self.chain.invoke({
            "metrics":           metrics_str,
            "context":           context,
            "risk_profile":      risk_profile,
            "annual_return_pct": metrics["annual_return_pct"],
        })

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
    agent = object.__new__(GoalPlanningAgent)  # skip __init__ (no LLM/retriever needed)
    return agent._calculate_metrics(goal_amount, time_horizon_years, current_savings, annual_return)


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = GoalPlanningAgent()

    result = agent.run(
        goal_amount=50_000,
        time_horizon_years=5,
        current_savings=10_000,
        risk_profile="moderate",
    )
    print(f"Monthly (no growth):   ${result['metrics']['monthly_no_growth']:,.2f}")
    print(f"Monthly (with growth): ${result['metrics']['monthly_with_growth']:,.2f}")
    print(f"\nPlan:\n{result['answer']}")

    print("\n--- NL input ---")
    result2 = agent.run(query="I want to save $20k for a vacation in 2 years. I have $3,000 saved already.")
    print(f"Metrics: {result2['metrics']}")
    print(f"\nPlan:\n{result2['answer']}")
