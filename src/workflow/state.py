from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class FinnieState(TypedDict):
    # Full conversation history — LangGraph appends each message automatically
    messages:             Annotated[list, add_messages]

    # Persisted user context — restored by MemorySaver on every turn
    risk_profile:         str
    goal_amount:          float | None
    time_horizon_years:   float | None
    current_savings:      float | None   # None = not yet provided by user
    annual_contribution:  float | None   # yearly contribution ($2k/mo → 24000/yr)
    portfolio_holdings:   dict | None    # {"AAPL": 100, "MSFT": 200, ...}
    age:                  int | None     # user's current age
