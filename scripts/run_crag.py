"""
scripts/run_crag.py — Phase 4: End-to-End Pipeline Execution

This script initializes the full Corrective RAG (CRAG) graph, 
submits a user query, and prints the final generated answer, 
the sources used, and the step-by-step execution trace.
"""

import argparse
import sys
from src.graph.build_graph import build_crag_graph

def main():
    parser = argparse.ArgumentParser(description="Run the CRAG pipeline end-to-end.")
    parser.add_argument(
        "query", 
        type=str, 
        nargs="?", 
        help="The technical query to ask the industrial equipment knowledge base."
    )
    args = parser.parse_args()

    # Fallback to prompt if no query is provided via CLI
    query = args.query
    if not query:
        print("No query provided via command line.")
        query = input("Enter your query: ").strip()
        if not query:
            print("Query cannot be empty. Exiting.")
            sys.exit(1)

    print("Building CRAG graph (loading embeddings and models)...")
    try:
        app = build_crag_graph()
    except Exception as e:
        print(f"[ERROR] Failed to build graph: {e}")
        sys.exit(1)

    # Initialize the starting state
    initial_state = {
        "query": query,
        "documents": [],
        "web_search_required": False,
        "web_search_iterations": 0,
        "generation_iterations": 0,
        "generation": "",
        "grounded": False,
        "execution_trace": ["System: Initialized CRAG pipeline"]
    }

    print(f"\nExecuting query: '{query}'")
    print("Routing through LangGraph nodes...\n")
    
    # Execute the graph
    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        print(f"[ERROR] Graph execution failed: {e}")
        sys.exit(1)

    # Output formatting
    print("=" * 60)
    print(" FINAL ANSWER")
    print("=" * 60)
    print(final_state.get("generation", "[No generation produced]"))
    
    print("\n" + "=" * 60)
    print(" EXECUTION TRACE")
    print("=" * 60)
    for step in final_state.get("execution_trace", []):
        print(f" -> {step}")
        
    print("\n" + "=" * 60)
    print(" DOCUMENTS IN FINAL CONTEXT")
    print("=" * 60)
    docs = final_state.get("documents", [])
    if not docs:
        print(" [No documents retrieved]")
    else:
        for idx, doc in enumerate(docs, 1):
            source = doc.get("doc_name", "Unknown Source")
            score = doc.get("score", "N/A")
            if isinstance(score, float):
                score_str = f"{score:.4f}"
            else:
                score_str = str(score)
            print(f" {idx}. {source} (Score/Rank: {score_str})")

if __name__ == "__main__":
    main()