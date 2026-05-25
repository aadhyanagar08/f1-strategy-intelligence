from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from state import AgentState, StrategyRecommendation
from agents.planner import planner_agent
from agents.retrieval import retrieval_agent
from agents.data_agent import data_agent
from agents.synthesis import synthesis_agent


def _route_from_planner(state: AgentState) -> list[Send]:
    """Fan-out from planner to retrieval and/or data agents in parallel."""
    route_to = state.get("route_to", ["retrieval", "data"])
    sends: list[Send] = []

    if "retrieval" in route_to:
        sends.append(Send("retrieval_agent", state))
    if "data" in route_to:
        sends.append(Send("data_agent", state))

    # If planner somehow returned neither, skip straight to synthesis
    if not sends:
        sends.append(Send("synthesis_agent", state))

    return sends


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_agent)
    builder.add_node("retrieval_agent", retrieval_agent)
    builder.add_node("data_agent", data_agent)
    builder.add_node("synthesis_agent", synthesis_agent)

    builder.add_edge(START, "planner")

    # Conditional fan-out: planner → one or both specialist agents (parallel via Send)
    builder.add_conditional_edges(
        "planner",
        _route_from_planner,
        ["retrieval_agent", "data_agent", "synthesis_agent"],
    )

    # Fan-in: both specialist agents converge on synthesis (LangGraph waits for all
    # in-flight branches before executing a node with multiple incoming edges)
    builder.add_edge("retrieval_agent", "synthesis_agent")
    builder.add_edge("data_agent", "synthesis_agent")
    builder.add_edge("synthesis_agent", END)

    return builder.compile()


graph = build_graph()


def run_pipeline(query: str) -> StrategyRecommendation | None:
    initial_state: AgentState = {
        "query": query,
        "sub_tasks": [],
        "retrieval_results": [],
        "data_results": [],
        "synthesis_output": None,
        "route_to": [],
        "error": None,
    }

    final_state = graph.invoke(initial_state)
    return final_state.get("synthesis_output")
