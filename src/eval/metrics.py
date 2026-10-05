import json
import os
from typing import List, Optional, Tuple

from src.retrieval import Retriever


def parse_sources(source_doc: Optional[str], source_page) -> List[Tuple[str, Optional[int]]]:
    """
    Returns a list of (doc_name, page_int_or_None) tuples to match against.

    Handles both formats produced by build_eval_set.py:
      - lookup:    source_doc="39157.pdf", source_page=3            (int)
      - synthesis: source_doc="39157.pdf, mi_cs_ashgrove.pdf",
                    source_page="p.3, p.7"                           (string)

    Also tolerates source_page being None (shouldn't happen for
    lookup/synthesis questions, but guards against it rather than crashing).
    """
    if not source_doc:
        return []

    doc_names = [d.strip() for d in source_doc.split(",")]

    if source_page is None:
        pages = [None] * len(doc_names)
    elif isinstance(source_page, str):
        raw_pages = [p.strip() for p in source_page.split(",")]
        pages = []
        for p in raw_pages:
            cleaned = p.lstrip("pP.").strip()
            pages.append(int(cleaned) if cleaned.isdigit() else None)
    else:
        # plain int — same page applies to the single doc_name
        pages = [source_page]

    # Defensive: if counts don't line up (malformed data), pad with None
    while len(pages) < len(doc_names):
        pages.append(None)

    return list(zip(doc_names, pages))


def is_match(chunk: dict, targets: List[Tuple[str, Optional[int]]]) -> bool:
    """A chunk counts as a hit if it matches ANY of the target (doc, page) pairs."""
    chunk_doc = chunk.get("doc_name", "")
    chunk_page = chunk.get("page", 0)

    for target_doc, target_page in targets:
        if chunk_doc != target_doc:
            continue
        if target_page is None:
            return True  # no page constraint — doc match is enough
        if abs(chunk_page - target_page) <= 1:
            return True
    return False


def evaluate_retrieval():
    with open("data/eval_set_dev.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    eval_qs = [q for q in eval_data if q.get("question_type") in ["lookup", "synthesis"]]

    print("Loading models and building BM25 index...")
    retriever = Retriever()
    variants = ["Dense-only", "Hybrid (BM25+RRF)", "Hybrid + Reranker"]
    metrics = {v: {"hits_at_5": 0, "hits_at_10": 0, "mrr_sum": 0.0} for v in variants}

    total_q = len(eval_qs)
    zero_match_questions = []

    print(f"\nEvaluating {total_q} questions across 3 variants...")

    for i, q in enumerate(eval_qs, 1):
        query = q["question"]
        targets = parse_sources(q.get("source_doc"), q.get("source_page"))

        print(f"[{i}/{total_q}] Testing: {query[:60]}...")

        results_map = {
            "Dense-only": retriever.dense_search(query, top_k=10),
            "Hybrid (BM25+RRF)": retriever.hybrid_search(query, top_k=10),
            "Hybrid + Reranker": retriever.reranked_hybrid_search(query, top_k=10),
        }

        matched_any_variant = False

        for variant_name, retrieved_chunks in results_map.items():
            hit_rank = None

            for rank, chunk in enumerate(retrieved_chunks):
                if is_match(chunk, targets):
                    hit_rank = rank + 1
                    matched_any_variant = True
                    break

            if hit_rank is not None:
                if hit_rank <= 5:
                    metrics[variant_name]["hits_at_5"] += 1
                if hit_rank <= 10:
                    metrics[variant_name]["hits_at_10"] += 1
                metrics[variant_name]["mrr_sum"] += 1.0 / hit_rank

        if not matched_any_variant:
            zero_match_questions.append({
                "id": q.get("id", "unknown"),
                "question": query,
                "targets": [t[0] for t in targets],
            })

    os.makedirs("results", exist_ok=True)
    report_path = "results/retrieval_ablation.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2: Retrieval Baseline and Ablation\n\n")
        f.write("| Variant | Recall@5 | Recall@10 | MRR |\n")
        f.write("|---|---|---|---|\n")

        print("\n=== FINAL RESULTS ===")
        for variant in variants:
            r5 = metrics[variant]["hits_at_5"] / total_q if total_q else 0.0
            r10 = metrics[variant]["hits_at_10"] / total_q if total_q else 0.0
            mrr = metrics[variant]["mrr_sum"] / total_q if total_q else 0.0
            f.write(f"| {variant} | {r5:.4f} | {r10:.4f} | {mrr:.4f} |\n")
            print(f"{variant:20s} - R@5: {r5:.4f}, R@10: {r10:.4f}, MRR: {mrr:.4f}")

        f.write(f"\n**Total Eval Questions Evaluated:** {total_q}\n\n")

        if zero_match_questions:
            f.write("### Zero-Match Questions\n")
            f.write("The following questions had 0 matches across ALL variants in the top 10:\n\n")
            for zq in zero_match_questions:
                f.write(f"- **ID:** {zq['id']}\n")
                f.write(f"  - **Question:** {zq['question']}\n")
                f.write(f"  - **Targets:** {', '.join(zq['targets'])}\n\n")
        else:
            f.write("### Zero-Match Questions\nNo questions failed completely!\n")

    print(f"\nReport written to {report_path}")
    if zero_match_questions:
        print(f"\nWARNING: Found {len(zero_match_questions)} questions with ZERO matches across all variants.")


if __name__ == "__main__":
    evaluate_retrieval()