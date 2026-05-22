import operator
from typing import Annotated, Any, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class SubTask(BaseModel):
    task_id: str
    task_type: str  # "retrieval" | "data" | "both"
    description: str
    query: str


class RetrievalResult(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: float = 0.0


class DataResult(BaseModel):
    source: str  # "jolpica" | "fastf1"
    endpoint: str
    data: dict


class StrategyRecommendation(BaseModel):
    recommendation: str
    supporting_evidence: List[str]
    citations: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    alternative_strategies: List[str] = Field(default_factory=list)


class AgentState(TypedDict):
    query: str
    sub_tasks: List[SubTask]
    retrieval_results: Annotated[List[RetrievalResult], operator.add]
    data_results: Annotated[List[DataResult], operator.add]
    synthesis_output: Optional[StrategyRecommendation]
    route_to: List[str]
    error: Optional[str]
