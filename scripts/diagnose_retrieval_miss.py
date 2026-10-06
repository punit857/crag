"""
Read-only retrieval-miss diagnosis. No LLM calls (embeddings and reranker run locally).
For chosen questions: finds the chunk(s) containing a verbatim fact fragment, then reports
where each retrieval method ranks them. Needs Qdrant up and the api container stopped (RAM).
"""
import json
import re
import sys

from src.graph.nodes import retriever  # global Retriever (loads models, ~20s)

QS_PATH = "data/eval_set_test_v2.json"
TARGETS = {  # question id -> fragment that must appear in the answer-bearing chunk
    "test_000": "between 200 and 400",
    "test_002": "102,600",
    "test_007": "piping diameter",
}

for name in ("all_chunks", "dense_search", "hybrid_search", "reranked_hybrid_search"):
    if not hasattr(retriever, name):
        sys.exit(f"[ERROR] retriever has no '{name}'. Paste src/retrieval.py so I can adapt this.")
has_bm25 = hasattr(retriever, "bm25_search")


def norm(s):
    s = (s or "").lower().replace("\u2011", "-").replace("\u2013", "-")
    return re.sub(r"\s+", " ", s)


def ranks(results, ids):
    return [i + 1 for i, r in enumerate(results) if r["id"] in ids] or "NOT IN TOP 20"


qs = {x["id"]: x for x in json.load(open(QS_PATH, encoding="utf-8"))}
for qid, frag in TARGETS.items():
    query = qs[qid]["question"]
    f = norm(frag)
    targets = [c for c in retriever.all_chunks if f in norm(c["text"])]
    print("=" * 78)
    print(qid, "|", query[:110])
    print(f"fragment {frag!r} found in {len(targets)} of {len(retriever.all_chunks)} chunks")
    for c in targets:
        t = norm(c["text"])
        i = t.find(f)
        print(f"  - {c['doc_name']} p.{c['page']} id={c['id'][:8]} ...{t[max(0, i - 60): i + 140]}...")
    if not targets:
        print("  -> fact is in NO chunk: a chunking/extraction problem, not a retrieval problem")
        continue
    ids = {c["id"] for c in targets}
    print("  dense    rank:", ranks(retriever.dense_search(query, top_k=20), ids))
    if has_bm25:
        print("  bm25     rank:", ranks(retriever.bm25_search(query, top_k=20), ids))
    print("  hybrid   rank (= reranker candidate pool):", ranks(retriever.hybrid_search(query, top_k=20), ids))
    rer = retriever.reranked_hybrid_search(query, top_k=20)
    print("  reranked rank:", ranks(rer, ids))
    for r in rer:
        if r["id"] in ids:
            print(f"     target reranker score: {r['score']:.2f}")
    print("  reranked top-5:", [(r["doc_name"], r["page"], round(r["score"], 2)) for r in rer[:5]])