import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState, SubTask

_SYSTEM = """You are an F1 strategy planning agent. Decompose the user's strategy question into sub-tasks and decide which specialist agents to invoke.

Return ONLY valid JSON in this exact shape:
{
  "sub_tasks": [
    {
      "task_id": "1",
      "task_type": "retrieval|data|both",
      "description": "what this sub-task covers",
      "query": "exact query to pass to the agent"
    }
  ],
  "route_to": ["retrieval", "data"]
}

Rules:
- Use "retrieval" for FIA regulations, technical/sporting codes, rule interpretations
- Use "data" for race results, lap times, telemetry, championship standings, tyre data
- Use "both" (task_type) and include both strings in route_to for questions needing regulation context AND race data
- route_to must be a non-empty list containing "retrieval", "data", or both"""


def planner_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    response = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Strategy question: {state['query']}"),
    ])

    try:
        parsed = json.loads(response.content)
        sub_tasks = [SubTask(**t) for t in parsed.get("sub_tasks", [])]
        route_to = parsed.get("route_to", ["retrieval", "data"])
        if not route_to:
            route_to = ["retrieval", "data"]
    except Exception:
        sub_tasks = [
            SubTask(
                task_id="1",
                task_type="both",
                description=state["query"],
                query=state["query"],
            )
        ]
        route_to = ["retrieval", "data"]

    return {"sub_tasks": sub_tasks, "route_to": route_to}
