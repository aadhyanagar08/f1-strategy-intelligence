"""MCP server exposing F1 strategy intelligence as three tools."""

import json
import os

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

mcp = FastMCP("f1-strategy-intelligence")


@mcp.tool()
async def analyze_strategy(query: str) -> str:
    """
    Run the full F1 strategy intelligence pipeline on a natural-language question.

    Invokes the planner, retrieval (RAG over FIA regulations), and data agents,
    then synthesises a Pydantic-validated strategy recommendation with citations
    and a confidence score.
    """
    from graph import graph

    initial_state = {
        "query": query,
        "sub_tasks": [],
        "retrieval_results": [],
        "data_results": [],
        "synthesis_output": None,
        "route_to": [],
        "error": None,
    }

    final_state = await graph.ainvoke(initial_state)
    synthesis = final_state.get("synthesis_output")

    if synthesis is None:
        return "No strategy recommendation could be generated."

    return json.dumps(synthesis.model_dump(), indent=2, default=str)


@mcp.tool()
async def get_race_data(season: int, round: int) -> str:
    """
    Fetch official race results for a specific F1 season and round from the
    Jolpica Ergast API (no API key required).

    Args:
        season: Four-digit year, e.g. 2024.
        round:  Round number within the season, e.g. 5.
    """
    url = f"{JOLPICA_BASE}/{season}/{round}/results.json"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return f"No race data found for season {season}, round {round}."

    return json.dumps(races[0], indent=2, default=str)


@mcp.tool()
async def get_regulation_context(topic: str) -> str:
    """
    Query locally-indexed FIA regulation PDFs using semantic search.

    Returns the top-5 most relevant regulation passages for the given topic.
    Regulations must be ingested via ingest.py before this tool is functional.

    Args:
        topic: Free-text topic, e.g. "tyre compound mandatory pit stop rules".
    """
    try:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_or_create_collection(
            name="fia_regulations", embedding_function=DefaultEmbeddingFunction()
        )

        if col.count() == 0:
            return (
                "Regulation index is empty. "
                "Place FIA PDF files in ./regulations/ and run: python ingest.py"
            )

        raw = col.query(query_texts=[topic], n_results=5)
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]

        passages = []
        for doc, meta in zip(docs, metas):
            src = meta.get("source", "Unknown")
            page = meta.get("page", "?")
            passages.append(f"[{src} p.{page}]\n{doc}")

        return "\n\n---\n\n".join(passages)

    except Exception as exc:
        return f"Error querying regulations: {exc}"


if __name__ == "__main__":
    mcp.run()
