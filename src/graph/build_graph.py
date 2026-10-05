import functools
import logging
import time

from langgraph.graph import StateGraph, END
from src.graph.state import CRAGState
from src.graph.nodes import (
    retrieve_documents,
    grade_documents,
    rewrite_query,
    web_search,
    generate_answer,
    check_groundedness
)

timing_logger = logging.getLogger("crag_api")


def timed(name: str, fn):
    """Logs wall-clock time per node. Does not touch state or the execution trace."""
    @functools.wraps(fn)
    def wrapper(state):
        t0 = time.perf_counter()
        try:
            return fn(state)
        finally:
            timing_logger.info(f"[timing] node={name} {time.perf_counter() - t0:.2f}s")
    return wrapper


def decide_to_search(state: CRAGState) -> str:
    """Decides if we need to rewrite the query and hit the web."""
    if state["web_search_required"] and state["web_search_iterations"] < 1:
        return "rewrite_query"
    return "generate_answer"

def decide_to_regenerate(state: CRAGState) -> str:
    """Decides if we need to regenerate due to hallucinations, capped at 2 tries."""
    if not state["grounded"] and state["generation_iterations"] < 2:
        return "generate_answer"
    return END

def build_crag_graph():
    """Compiles the full Corrective RAG pipeline."""
    workflow = StateGraph(CRAGState)

    # Add Nodes (each wrapped with timing)
    workflow.add_node("retrieve", timed("retrieve", retrieve_documents))
    workflow.add_node("grade", timed("grade", grade_documents))
    workflow.add_node("rewrite_query", timed("rewrite_query", rewrite_query))
    workflow.add_node("web_search", timed("web_search", web_search))
    workflow.add_node("generate_answer", timed("generate_answer", generate_answer))
    workflow.add_node("check_groundedness", timed("check_groundedness", check_groundedness))

    # Add Edges (Linear paths)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", "check_groundedness")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "grade",
        decide_to_search,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer"
        }
    )

    workflow.add_conditional_edges(
        "check_groundedness",
        decide_to_regenerate,
        {
            "generate_answer": "generate_answer",
            END: END
        }
    )

    return workflow.compile()