"""
src/api.py — Phase 6: FastAPI wrapper around the CRAG pipeline.

Builds the CRAG graph ONCE at startup (not per-request — graph construction
loads embedding/reranker models, which is slow and must not repeat per call).
Exposes:
  POST /query   — run a question through the full CRAG pipeline
  GET  /health  — liveness check

Run with uvicorn (see step-by-step instructions after this file).
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.config import config
from src.graph.build_graph import build_crag_graph
from src.graph.nodes import retriever
from src.cache import make_key, get_cached, set_cached
from src.ratelimit import check_rate_limit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crag_api")

MAX_QUERY_LENGTH = 2000  # defensive cap — not from config, just basic input hygiene


# --------------------------------------------------------------------------- #
# App state — the compiled graph lives here, built once at startup
# --------------------------------------------------------------------------- #
class AppState:
    graph = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building CRAG graph...")
    start = time.perf_counter()
    app_state.graph = build_crag_graph()
    logger.info(f"CRAG graph ready in {time.perf_counter() - start:.1f}s.")

    # Warm-up: retrieval only (no LLM call, so no Groq tokens are spent).
    try:
        t = time.perf_counter()
        retriever.reranked_hybrid_search("warm-up query", top_k=1)
        logger.info(f"Retrieval warm-up done in {time.perf_counter() - t:.1f}s.")
    except Exception as e:
        logger.warning(f"Warm-up failed (non-fatal): {e}")

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Industrial Equipment CRAG API",
    description="Corrective RAG workflow over DOE industrial equipment efficiency documents.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: open for now (no frontend built yet — Phase 7). Tighten before deploying (Phase 8).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query cannot be empty or whitespace-only")
         
        return v.strip()
        


class SourceInfo(BaseModel):
    doc_name: str
    page: Optional[int] = None
    score: Optional[float] = None
    is_web: bool


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    web_search_used: bool
    execution_trace: List[str]
    latency_seconds: float
    cached: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: str


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness check. Confirms the graph is built and ready, not just that the server is up."""
    if app_state.graph is None:
        raise HTTPException(status_code=503, detail="Graph not yet initialized.")
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, responses={500: {"model": ErrorResponse}})
def query(req: QueryRequest, request: Request) -> QueryResponse:
    if app_state.graph is None:
        raise HTTPException(status_code=503, detail="Graph not yet initialized.")

    logger.info(f"Query received: {req.query[:100]}...")
    start = time.perf_counter()

    # 1) Cache first: hits cost no tokens and are never rate-limited
    cache_key = make_key(req.query)
    hit = get_cached(cache_key)
    if hit is not None:
        hit["cached"] = True
        hit["latency_seconds"] = round(time.perf_counter() - start, 3)
        logger.info(f"Cache HIT in {hit['latency_seconds']}s")
        return QueryResponse(**hit)

    # 2) Rate limit only real pipeline runs
    client_ip = request.client.host if request.client else "unknown"
    limited = check_rate_limit(client_ip)
    if limited is not None:
        scope, retry_after = limited
        logger.warning(f"Rate limit hit (scope={scope}) for {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached. Please retry shortly." if scope == "ip"
            else "Daily capacity for this demo has been reached. Please try again tomorrow.",
            headers={"Retry-After": str(retry_after)},
        )

    # 3) Run the pipeline
    initial_state: Dict[str, Any] = {
        "query": req.query,
        "documents": [],
        "web_search_required": False,
        "web_search_iterations": 0,
        "generation_iterations": 0,
        "generation": "",
        "grounded": False,
        "execution_trace": [],
    }

    try:
        final_state = app_state.graph.invoke(initial_state)
    except Exception as e:
        latency = time.perf_counter() - start
        logger.error(f"Graph execution failed after {latency:.1f}s: {e}")  # full detail stays in server logs only
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower():
            raise HTTPException(status_code=429, detail="The language model is rate-limited right now. Please retry shortly.")
        raise HTTPException(status_code=500, detail="Pipeline execution failed. See server logs.")

    latency = time.perf_counter() - start

    docs = final_state.get("documents", [])
    sources = [
        SourceInfo(
            doc_name=d.get("doc_name", "Unknown"),
            page=d.get("page") if not str(d.get("doc_name", "")).startswith(("http://", "https://")) else None,
            score=d.get("score") if isinstance(d.get("score"), (int, float)) else None,
            is_web=str(d.get("doc_name", "")).startswith(("http://", "https://")),
        )
        for d in docs
    ]

    response = QueryResponse(
        answer=final_state.get("generation", ""),
        sources=sources,
        web_search_used=final_state.get("web_search_iterations", 0) > 0,
        execution_trace=final_state.get("execution_trace", []),
        latency_seconds=round(latency, 2),
    )

    if response.answer.strip():  # never cache empty answers
        set_cached(cache_key, response.model_dump(), response.web_search_used)

    logger.info(f"Query completed in {latency:.1f}s, web_used={response.web_search_used}")
    return response