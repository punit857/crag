"""
Benchmarks the cross-encoder under different thread counts / batch sizes on the SAME 20
candidates, and verifies scores are unchanged (thresholds depend on the score scale).
Makes no LLM calls. Stop uvicorn first (local Qdrant allows one process).
"""
import inspect
import time

from fastembed.rerank.cross_encoder import TextCrossEncoder
from src.graph.nodes import retriever  # existing Retriever (opens local Qdrant)

QUERY = "What flow rate and pressure do the compressed air applications at the Techni-Cast foundry require to operate reliably?"
MODEL = "BAAI/bge-reranker-base"

cands = retriever.hybrid_search(QUERY, top_k=20)
docs = [c["text"] for c in cands]
print(f"{len(docs)} candidates, avg {sum(map(len, docs)) // len(docs)} chars each")

init_params = list(inspect.signature(TextCrossEncoder.__init__).parameters)
rerank_params = list(inspect.signature(TextCrossEncoder.rerank).parameters)
print("TextCrossEncoder.__init__ params:", init_params)
print("TextCrossEncoder.rerank params:  ", rerank_params)

baseline = list(retriever.reranker.rerank(QUERY, docs))  # current production behaviour

thread_opts = [None, 4, 6, 8, 10, 12] if "threads" in init_params else [None]
batch_opts = [None, 8, 32] if "batch_size" in rerank_params else [None]

print("\nthreads | batch | best-of-2 time | max score diff vs baseline")
for th in thread_opts:
    enc = TextCrossEncoder(model_name=MODEL, **({} if th is None else {"threads": th}))
    list(enc.rerank(QUERY, docs[:2]))  # warm-up
    for bs in batch_opts:
        kw = {} if bs is None else {"batch_size": bs}
        times, scores = [], None
        for _ in range(2):
            t = time.perf_counter()
            scores = list(enc.rerank(QUERY, docs, **kw))
            times.append(time.perf_counter() - t)
        diff = max(abs(a - b) for a, b in zip(scores, baseline))
        print(f"{str(th):>7} | {str(bs):>5} | {min(times):6.2f}s | {diff:.6f}")

# Candidate-count effect (default encoder): time only, ranking may change
print("\ncandidates | time (default settings)")
for k in (10, 15, 20):
    t = time.perf_counter()
    list(retriever.reranker.rerank(QUERY, docs[:k]))
    print(f"{k:>10} | {time.perf_counter() - t:.2f}s")