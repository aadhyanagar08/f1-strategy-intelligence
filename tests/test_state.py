"""Unit tests for Pydantic state models — no LLM or network required."""

import pytest
from state import (
    SubTask,
    RetrievalResult,
    StrategyRecommendation,
)


def test_sub_task_model():
    t = SubTask(task_id="1", task_type="retrieval", description="Check DRS rules", query="DRS activation zones")
    assert t.task_type == "retrieval"


def test_retrieval_result_defaults():
    r = RetrievalResult(content="Article 14.3...", source="2024_sporting_regulations.pdf")
    assert r.page is None
    assert r.score == 0.0


def test_strategy_recommendation_confidence_bounds():
    with pytest.raises(Exception):
        StrategyRecommendation(
            recommendation="Two-stop",
            supporting_evidence=[],
            citations=[],
            confidence_score=1.5,  # out of range
            reasoning="test",
        )


def test_strategy_recommendation_valid():
    rec = StrategyRecommendation(
        recommendation="Undercut on lap 28",
        supporting_evidence=[
            "FastF1: lap delta of 0.4s after lap 28 suggests tyre deg.",
            "Jolpica: 2024 Montreal average pit loss 22s.",
        ],
        citations=["FastF1", "Jolpica"],
        confidence_score=0.82,
        reasoning="High tyre degradation justifies early stop.",
        alternative_strategies=["Overcut on lap 32"],
    )
    assert rec.confidence_score == 0.82
    assert len(rec.supporting_evidence) == 2
