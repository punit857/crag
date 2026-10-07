import operator
from typing import Annotated, List, Dict, Any, TypedDict, Optional

class CRAGState(TypedDict):
    """Represents the state of the CRAG LangGraph pipeline."""
    query: str
    
    # --- Dynamic Query Controls ---
    skip_reranker: bool
    web_search_mode: str          # "default", "custom", or "open"
    custom_domains: List[str]     # Used if web_search_mode == "custom"
    session_id: Optional[str]     # Isolates custom user uploads from industrial KB
    
    documents: List[Dict[str, Any]]
    
    # --- Routing & Iteration Tracking ---
    web_search_required: bool
    web_search_iterations: int
    generation_iterations: int
    
    # --- Outputs ---
    generation: str
    grounded: bool
    
    # --- Observability ---
    execution_trace: Annotated[List[str], operator.add]
    node_timings: Annotated[List[Dict[str, Any]], operator.add]