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
import math

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
    withdrawal_mode: bool = Field(False, description=(
        "True when the user asks how much they can WITHDRAW from existing savings, "
        "how long their nest egg will last, or asks about retirement income drawdown. "
        "Key signals: 'withdraw', 'how long will it last', 'drawdown', 'take out each month', "
        "'live off my savings', 'retirement income', 'how much can I spend'. "
        "False for accumulation questions (saving toward a goal amount)."
    ))
    withdrawal_amount: float | None = Field(None, description=(
        "Monthly withdrawal amount in dollars if the user specifies how much they want to withdraw. "
        "Example: 'withdraw $10,000/month' → 10000. Return null if not mentioned."
    ))
    nest_egg: float | None = Field(None, description=(
        "The starting savings balance for withdrawal/decumulation planning. "
        "Use when the user says 'I have $X' or 'starting with $X' in a withdrawal context. "
        "This is money they already HAVE, not a goal to save toward. "
        "Return null if not in withdrawal mode."
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


WITHDRAWAL_PROMPT = ChatPromptTemplate.from_template("""
You are Finnie, a friendly financial education assistant.
The user wants to know how much they can withdraw from their retirement nest egg, or how long it will last.

Withdrawal Metrics:
{metrics}

Financial Knowledge (from knowledge base):
{context}

User's risk tolerance: {risk_profile}
Assumed annual return: {annual_return_pct}%

IMPORTANT — use ONLY the numbers from "Withdrawal Metrics" above. Do NOT recompute or estimate.
Use "USD X,XXX" format for all currency amounts. Never use "$X,XXX".

Provide a clear answer covering:
1. Direct answer: how much they can withdraw per month (or how long the money lasts)
2. The 4% rule as a benchmark — quote the safe monthly amount from the metrics as a reference
3. How the assumed {annual_return_pct}% annual return affects sustainability
4. Sequence-of-returns risk: why a bad early year can cut the timeline shorter than expected
5. Tax note: withdrawals from traditional 401k/IRA are ordinary taxable income; Roth withdrawals are tax-free
6. One practical tip (e.g. flexible spending, keeping 1-2 years in cash, delaying Social Security)

Always end with: "This is for educational purposes only and is not financial advice."

Answer:
""")


class GoalPlanningAgent:

    def __init__(self):
        self.retriever         = get_retriever()
        self.llm               = load_llm()
        self.chain             = PROMPT | self.llm | StrOutputParser()
        self.projection_chain  = PROJECTION_PROMPT | self.llm | StrOutputParser()
        self.withdrawal_chain  = WITHDRAWAL_PROMPT | self.llm | StrOutputParser()
        self.goal_parser       = self.llm.with_structured_output(_GoalParams)

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
            "goal_amount":         parsed.goal_amount,
            "time_horizon_years":  horizon,
            "current_savings":     parsed.current_savings,
            "risk_profile":        parsed.risk_profile,
            "annual_contribution": parsed.annual_contribution,
            "withdrawal_mode":     parsed.withdrawal_mode,
            "withdrawal_amount":   parsed.withdrawal_amount,
            "nest_egg":            parsed.nest_egg,
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
    def _calculate_withdrawal_metrics(
        nest_egg: float,
        annual_return: float,
        time_horizon_years: float | None = None,
        monthly_withdrawal: float | None = None,
    ) -> dict:
        monthly_rate = annual_return / 12
        safe_monthly = round(nest_egg * 0.04 / 12, 2)

        if time_horizon_years and not monthly_withdrawal:
            # Given nest egg + duration → compute sustainable monthly withdrawal
            months = time_horizon_years * 12
            if monthly_rate > 0:
                w = nest_egg * monthly_rate / (1 - (1 + monthly_rate) ** -months)
            else:
                w = nest_egg / months
            return {
                "nest_egg":            round(nest_egg, 2),
                "time_horizon_years":  round(time_horizon_years, 2),
                "monthly_withdrawal":  round(w, 2),
                "annual_withdrawal":   round(w * 12, 2),
                "annual_return_pct":   round(annual_return * 100, 1),
                "rule_of_4pct_monthly": safe_monthly,
                "mode":                "withdrawal_by_duration",
            }

        if monthly_withdrawal and not time_horizon_years:
            # Given nest egg + desired withdrawal → compute how long it lasts
            if monthly_rate > 0 and monthly_withdrawal > nest_egg * monthly_rate:
                months = -math.log(1 - nest_egg * monthly_rate / monthly_withdrawal) / math.log(1 + monthly_rate)
                years = round(months / 12, 1)
                lasts_forever = False
            elif monthly_rate > 0 and monthly_withdrawal <= nest_egg * monthly_rate:
                years = None
                lasts_forever = True
            else:
                years = round(nest_egg / monthly_withdrawal / 12, 1)
                lasts_forever = False
            return {
                "nest_egg":            round(nest_egg, 2),
                "monthly_withdrawal":  round(monthly_withdrawal, 2),
                "annual_withdrawal":   round(monthly_withdrawal * 12, 2),
                "duration_years":      years,
                "lasts_forever":       lasts_forever,
                "annual_return_pct":   round(annual_return * 100, 1),
                "rule_of_4pct_monthly": safe_monthly,
                "mode":                "withdrawal_duration",
            }

        if time_horizon_years and monthly_withdrawal:
            # Both given — check if the requested withdrawal is sustainable
            months = time_horizon_years * 12
            if monthly_rate > 0:
                max_w = nest_egg * monthly_rate / (1 - (1 + monthly_rate) ** -months)
            else:
                max_w = nest_egg / months
            return {
                "nest_egg":                  round(nest_egg, 2),
                "time_horizon_years":        round(time_horizon_years, 2),
                "monthly_withdrawal":        round(monthly_withdrawal, 2),
                "max_sustainable_monthly":   round(max_w, 2),
                "sustainable_for_horizon":   monthly_withdrawal <= max_w,
                "annual_return_pct":         round(annual_return * 100, 1),
                "rule_of_4pct_monthly":      safe_monthly,
                "mode":                      "withdrawal_check",
            }

        # Default overview: 4% rule + multiple durations
        overview: dict = {
            "nest_egg":              round(nest_egg, 2),
            "annual_return_pct":     round(annual_return * 100, 1),
            "rule_of_4pct_monthly":  safe_monthly,
            "rule_of_4pct_annual":   round(nest_egg * 0.04, 2),
        }
        for years in [20, 25, 30, 35]:
            months = years * 12
            if monthly_rate > 0:
                w = nest_egg * monthly_rate / (1 - (1 + monthly_rate) ** -months)
            else:
                w = nest_egg / months
            overview[f"monthly_{years}yr"] = round(w, 2)
        overview["mode"] = "withdrawal_overview"
        return overview

    @staticmethod
    def _format_withdrawal_metrics_for_llm(metrics: dict) -> str:
        mode = metrics.get("mode", "withdrawal_overview")
        lines = [f"Nest Egg:              USD {metrics['nest_egg']:,.2f}",
                 f"Assumed Annual Return: {metrics['annual_return_pct']}%",
                 f"4% Rule (safe rate):   USD {metrics['rule_of_4pct_monthly']:,.2f}/month"]
        if mode == "withdrawal_by_duration":
            lines += [
                f"Time Horizon:          {metrics['time_horizon_years']} years",
                f"Monthly Withdrawal:    USD {metrics['monthly_withdrawal']:,.2f}",
                f"Annual Withdrawal:     USD {metrics['annual_withdrawal']:,.2f}",
            ]
        elif mode == "withdrawal_duration":
            dur = "indefinitely (growth exceeds withdrawals)" if metrics.get("lasts_forever") else f"{metrics['duration_years']} years"
            lines += [
                f"Requested Monthly:     USD {metrics['monthly_withdrawal']:,.2f}",
                f"Annual Withdrawal:     USD {metrics['annual_withdrawal']:,.2f}",
                f"Duration:             {dur}",
            ]
        elif mode == "withdrawal_check":
            verdict = "YES — sustainable" if metrics["sustainable_for_horizon"] else "NO — too high"
            lines += [
                f"Time Horizon:          {metrics['time_horizon_years']} years",
                f"Requested Monthly:     USD {metrics['monthly_withdrawal']:,.2f}",
                f"Max Sustainable:       USD {metrics['max_sustainable_monthly']:,.2f}/month",
                f"Sustainable?           {verdict}",
            ]
        else:  # overview
            lines += [
                f"4% Rule Annual:        USD {metrics['rule_of_4pct_annual']:,.2f}",
                f"If withdrawn over 20 years: USD {metrics.get('monthly_20yr', 0):,.2f}/month",
                f"If withdrawn over 25 years: USD {metrics.get('monthly_25yr', 0):,.2f}/month",
                f"If withdrawn over 30 years: USD {metrics.get('monthly_30yr', 0):,.2f}/month",
                f"If withdrawn over 35 years: USD {metrics.get('monthly_35yr', 0):,.2f}/month",
            ]
        return "\n".join(lines)

    @staticmethod
    def _escape_markdown_currency(text: str) -> str:
        return re.sub(r"(?<!\\)\$", r"\\$", text)

    def _get_rag_context(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def run_withdrawal(
        self,
        nest_egg: float,
        risk_profile: str = "moderate",
        time_horizon_years: float | None = None,
        monthly_withdrawal: float | None = None,
        query: str = "",
    ) -> dict:
        """
        Decumulation / withdrawal planning.

        Returns:
            {"answer": str, "metrics": dict, "error": None}
        """
        annual_return = RETURN_RATES.get(risk_profile, RETURN_RATES["moderate"])
        log.info("GoalPlanningAgent.run_withdrawal | nest_egg=$%.0f | horizon=%s | monthly_wd=%s",
                 nest_egg, time_horizon_years or "?", f"${monthly_withdrawal:,.0f}" if monthly_withdrawal else "?")

        metrics     = self._calculate_withdrawal_metrics(nest_egg, annual_return, time_horizon_years, monthly_withdrawal)
        metrics_str = self._format_withdrawal_metrics_for_llm(metrics)
        context     = self._get_rag_context(query or f"retirement withdrawal nest egg {nest_egg}")

        answer = self.withdrawal_chain.invoke({
            "metrics":           metrics_str,
            "context":           context,
            "risk_profile":      risk_profile,
            "annual_return_pct": metrics["annual_return_pct"],
        })
        answer = self._escape_markdown_currency(answer)
        log.info("GoalPlanningAgent.run_withdrawal | done | mode=%s", metrics.get("mode"))
        return {"answer": answer, "metrics": metrics, "error": None}

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
