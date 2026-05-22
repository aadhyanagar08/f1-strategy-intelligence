import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from graph import graph  # noqa: E402  (must be after load_dotenv)
from state import StrategyRecommendation  # noqa: E402

app = FastAPI(
    title="F1 Strategy Intelligence",
    description="LangGraph-powered multi-agent F1 strategy analysis",
    version="0.1.0",
)


class StrategyRequest(BaseModel):
    query: str


class StrategyResponse(BaseModel):
    query: str
    result: StrategyRecommendation | None = None
    error: str | None = None


@app.post("/strategy", response_model=StrategyResponse)
async def analyze_strategy(req: StrategyRequest) -> StrategyResponse:
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    initial_state = {
        "query": req.query,
        "sub_tasks": [],
        "retrieval_results": [],
        "data_results": [],
        "synthesis_output": None,
        "route_to": [],
        "error": None,
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StrategyResponse(
        query=req.query,
        result=final_state.get("synthesis_output"),
        error=final_state.get("error"),
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
