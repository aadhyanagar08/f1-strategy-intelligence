import os
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState, RetrievalResult

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "fia_regulations"
# Fetch a wider candidate pool from ChromaDB; BM25 + RRF then rerank within it
TOP_K_DENSE = 40
TOP_K_FINAL = 5
RRF_K = 60  # standard constant — dampens the impact of top-ranked documents


def _collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
    )


def _hyde(query: str, llm: ChatOpenAI) -> str:
    """HyDE: generate a hypothetical regulation excerpt to bias dense retrieval."""
    resp = llm.invoke([
        SystemMessage(
            content=(
                "You are an FIA technical regulations expert. "
                "Write a 2-3 sentence hypothetical regulation excerpt that would directly answer this question. "
                "Write as if it is an official FIA document."
            )
        ),
        HumanMessage(content=query),
    ])
    return resp.content


def _rrf(dense_ranks: list[int], bm25_ranks: list[int], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion over two ranked lists of candidate indices."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranks):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ranks):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def ingest_pdfs(directory: str) -> None:
    """Load all PDFs from *directory*, chunk them, and upsert into ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    col = _collection()

    pdf_paths = sorted(Path(directory).glob("**/*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found in {directory}")
        return

    for pdf_path in pdf_paths:
        print(f"Ingesting {pdf_path.name} ...", end=" ", flush=True)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        chunks = splitter.split_documents(pages)

        ids, texts, metas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(f"{pdf_path.stem}_c{i}")
            texts.append(chunk.page_content)
            metas.append({
                "source": pdf_path.name,
                "page": chunk.metadata.get("page", 0) + 1,  # PyPDFLoader is 0-indexed
            })

        col.upsert(ids=ids, documents=texts, metadatas=metas)
        print(f"{len(chunks)} chunks indexed")

    print(f"\nDone. Collection '{col.name}' now has {col.count()} total chunks.")


def retrieval_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )
    col = _collection()

    if col.count() == 0:
        return {"retrieval_results": []}

    queries = [
        t.query
        for t in state.get("sub_tasks", [])
        if t.task_type in ("retrieval", "both")
    ] or [state["query"]]

    results: list[RetrievalResult] = []

    for query in queries:
        # Dense retrieval: HyDE-expanded query into ChromaDB (DefaultEmbeddingFunction)
        hyde_doc = _hyde(query, llm)
        n_fetch = min(TOP_K_DENSE, col.count())

        raw = col.query(query_texts=[hyde_doc], n_results=n_fetch)
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]

        if not docs:
            continue

        # Dense ranks: ChromaDB returns results in ascending-distance order, so
        # position 0 is already rank 1 — no transformation needed.
        dense_ranks = list(range(len(docs)))

        # Sparse retrieval: BM25 over the dense candidate pool
        corpus = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_ranks = sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)

        # RRF fusion
        rrf_scores = _rrf(dense_ranks, bm25_ranks)
        top_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:TOP_K_FINAL]

        for idx in top_indices:
            results.append(
                RetrievalResult(
                    content=docs[idx],
                    source=metas[idx].get("source", "FIA Regulations"),
                    page=metas[idx].get("page"),
                    score=rrf_scores[idx],
                )
            )

    return {"retrieval_results": results}
