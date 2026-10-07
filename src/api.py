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
from src.cache import make_key, get_cached, set_cached, is_cacheable, set_corpus_fingerprint
from src.ratelimit import check_rate_limit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crag_api")

MAX_QUERY_LENGTH = 2000

class AppState:
    graph = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building CRAG graph...")
    start = time.perf_counter()
    app_state.graph = build_crag_graph()
    logger.info(f"CRAG graph ready in {time.perf_counter() - start:.1f}s.")

    try:
        t = time.perf_counter()
        retriever.reranked_hybrid_search("warm-up query", top_k=1)
        logger.info(f"Retrieval warm-up completed in {time.perf_counter() - t:.1f}s.")
    except Exception as e:
        logger.warning(f"Warm-up failed (non-fatal): {e}")

    try:
        fp = set_corpus_fingerprint(retriever.all_chunks)
        logger.info(f"Cache corpus fingerprint: {fp} ({len(retriever.all_chunks)} chunks)")
    except Exception as e:
        logger.warning(f"Corpus fingerprint failed (non-fatal): {e}")

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Industrial Equipment CRAG API",
    description="Corrective RAG workflow with dynamic web modes and document controls.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    skip_reranker: bool = Field(default=False)
    web_search_mode: str = Field(default="default")  # "default", "custom", or "open"
    custom_domains: List[str] = Field(default_factory=list)
    session_id: Optional[str] = Field(default=None)
    bypass_cache: bool = Field(default=False)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("web_search_mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = ["default", "custom", "open"]
        if v not in allowed:
            raise ValueError(f"web_search_mode must be one of {allowed}")
        return v


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
    node_timings: List[Dict[str, Any]] = []

class ErrorResponse(BaseModel):
    error: str
    detail: str


# --------------------------------------------------------------------------- #
# Pipeline Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> Dict[str, str]:
    if app_state.graph is None:
        raise HTTPException(status_code=503, detail="Graph not yet initialized.")
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, responses={500: {"model": ErrorResponse}})
def query(req: QueryRequest, request: Request) -> QueryResponse:
    if app_state.graph is None:
        raise HTTPException(status_code=503, detail="Graph not yet initialized.")

    start = time.perf_counter()

    # 1. Distinct cache key including all operational parameters
    domain_str = ",".join(sorted(req.custom_domains))
    cache_raw = f"{req.query}|{req.skip_reranker}|{req.web_search_mode}|{domain_str}|{req.session_id or ''}"
    cache_key = make_key(cache_raw)

    if not req.bypass_cache:
        hit = get_cached(cache_key)
        if hit is not None:
            hit["cached"] = True
            hit["latency_seconds"] = round(time.perf_counter() - start, 3)
            logger.info(f"Cache HIT in {hit['latency_seconds']}s")
            return QueryResponse(**hit)

    # 2. Rate limiting check
    forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else "unknown")
    limited = check_rate_limit(client_ip)
    if limited is not None:
        scope, retry_after = limited
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached." if scope == "ip" else "Daily capacity reached.",
            headers={"Retry-After": str(retry_after)},
        )

    # 3. Graph execution
    initial_state: Dict[str, Any] = {
        "query": req.query,
        "skip_reranker": req.skip_reranker,
        "web_search_mode": req.web_search_mode,
        "custom_domains": req.custom_domains,
        "session_id": req.session_id,
        "documents": [],
        "web_search_required": False,
        "web_search_iterations": 0,
        "generation_iterations": 0,
        "generation": "",
        "grounded": False,
        "execution_trace": [],
        "node_timings": [],
    }

    try:
        final_state = app_state.graph.invoke(initial_state)
    except Exception as e:
        latency = time.perf_counter() - start
        logger.error(f"Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail="Pipeline execution failed.")

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
        node_timings=final_state.get("node_timings", []),
    )

    if not req.bypass_cache and is_cacheable(response.answer):
        set_cached(cache_key, response.model_dump(), response.web_search_used)

    return response


# --------------------------------------------------------------------------- #
# KB Management Routes
# --------------------------------------------------------------------------- #
@app.get("/kb/documents")
def get_kb_catalog():
    """Returns all base industrial documents and their chunk counts."""
    return {"scope": "base", "documents": retriever.list_documents(scope="base")}


@app.delete("/kb/documents/{doc_name}")
def delete_kb_document(doc_name: str):
    """Surgically deletes a single document by filename without affecting others."""
    retriever.delete_document(doc_name=doc_name, scope="base")
    return {"status": "success", "deleted_document": doc_name}