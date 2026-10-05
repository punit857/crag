# Phase 2: Retrieval Baseline and Ablation

| Variant | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| Dense-only | 0.6957 | 0.8696 | 0.4374 |
| Hybrid (BM25+RRF) | 0.7826 | 0.8696 | 0.4987 |
| Hybrid + Reranker | 0.7826 | 0.8696 | 0.5819 |

**Total Eval Questions Evaluated:** 23

### Zero-Match Questions
The following questions had 0 matches across ALL variants in the top 10:

- **ID:** dev_003
  - **Question:** In the context of the industrial assessment center technical guidance, what specific technical parameter or guidance applies to energy efficient technologies?
  - **Targets:** 39157.pdf

