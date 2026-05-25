import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState, StrategyRecommendation

_SYSTEM = """You are a world-class F1 strategy analyst. Respond with ONLY this exact format and nothing else:

RECOMMENDATION: <one paragraph describing the optimal strategy>
EVIDENCE: <point 1> | <point 2> | <point 3>
CITATIONS: <source 1> | <source 2>
CONFIDENCE: <decimal number between 0.0 and 1.0>
REASONING: <one sentence explaining your confidence level>
ALTERNATIVES: <alternative 1> | <alternative 2>

Do not include any other text, preamble, or explanation outside this format."""

_DATA_TRUNCATE = 1500


def _parse_response(text: str) -> StrategyRecommendation:
    """Parse the pipe-delimited label format into a StrategyRecommendation."""

    def _extract(label: str) -> str:
        pattern = rf"^{label}:\s*(.+?)(?=\n[A-Z]+:|$)"
        m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def _split_pipe(value: str) -> list:
        return [v.strip() for v in value.split("|") if v.strip()]

    recommendation = _extract("RECOMMENDATION")
    evidence_raw = _extract("EVIDENCE")
    citations_raw = _extract("CITATIONS")
    confidence_raw = _extract("CONFIDENCE")
    reasoning = _extract("REASONING")
    alternatives_raw = _extract("ALTERNATIVES")

    # Parse confidence — default to 0.5 if the model produced garbage
    try:
        confidence = float(re.search(r"[0-9]*\.?[0-9]+", confidence_raw).group())
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = 0.5

    return StrategyRecommendation(
        recommendation=recommendation or "See reasoning.",
        supporting_evidence=_split_pipe(evidence_raw) or ["No evidence extracted."],
        citations=_split_pipe(citations_raw),
        confidence_score=confidence,
        reasoning=reasoning or "Derived from available regulation and race data.",
        alternative_strategies=_split_pipe(alternatives_raw),
    )


def synthesis_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=800,
    )

    reg_blocks = "\n\n".join(
        f"[REG-{i + 1}] {r.source} (score={r.score:.3f})\n{r.content}"
        for i, r in enumerate(state.get("retrieval_results", []))
    ) or "No regulation context retrieved."

    data_blocks = "\n\n".join(
        f"[DATA-{i + 1}] source={d.source} endpoint={d.endpoint}\n"
        + json.dumps(d.data, default=str)[:_DATA_TRUNCATE]
        for i, d in enumerate(state.get("data_results", []))
    ) or "No race data retrieved."

    user_msg = (
        f"Query: {state['query']}\n\n"
        f"=== FIA REGULATION CONTEXT ===\n{reg_blocks}\n\n"
        f"=== RACE DATA ===\n{data_blocks}\n\n"
        "Produce a comprehensive F1 strategy recommendation using the exact format specified."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        raw_text = response.content
        output = _parse_response(raw_text)
    except Exception as exc:
        output = StrategyRecommendation(
            recommendation=f"LLM call failed: {exc}",
            supporting_evidence=["Pipeline error — check Ollama connectivity."],
            citations=[],
            confidence_score=0.0,
            reasoning=str(exc),
            alternative_strategies=[],
        )

    return {"synthesis_output": output}
