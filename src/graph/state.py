import operator
from typing import Annotated, List, Dict, Any, TypedDict

class CRAGState(TypedDict):
    """Represents the state of the CRAG LangGraph pipeline."""
    query: str
    documents: List[Dict[str, Any]]
    
    # Routing and Iteration Tracking
    web_search_required: bool
    web_search_iterations: int
    generation_iterations: int
    
    # Outputs
    generation: str
    grounded: bool
    
    # Observability
    execution_trace: Annotated[List[str], operator.add]