# Phase 5: Final Evaluation (Held-Out Test Set)

**Test set:** `data\eval_set_test_v2.json` (18 questions)

Score cells show `mean (n_valid/n_rows)`. Rows where RAGAS returned NaN are excluded from the mean but are counted in `n_valid`. Failed generations are counted in the Failures column.

| Variant | Recall@5 (±1 page) | Faithfulness | Answer Correctness | Avg Latency (s) | Failures |
|---|---|---|---|---|---|
| naive | 0.9231 (12/13) | 0.5625 (16/18) | 0.5433 (18/18) | 4.70 | 0 |
| hybrid | 1.0000 (13/13) | 0.5556 (15/18) | 0.5175 (18/18) | 13.17 | 0 |
| full_crag | 1.0000 (13/13) | 0.5667 (15/18) | 0.5062 (18/18) | 20.90 | 0 |

**Stricter retrieval metrics (exact page match, top-5):**

| Variant | Exact-page recall | Synthesis: all sources retrieved |
|---|---|---|
| naive | 10/13 | 3/4 |
| hybrid | 12/13 | 3/4 |
| full_crag | 12/13 | 3/4 |

## RAGAS by question type

| Variant | Type | n | Faithfulness | Answer Correctness |
|---|---|---|---|---|
| naive | lookup | 9 | 0.7143 (7/9) | 0.4677 (9/9) |
| naive | synthesis | 4 | 0.7500 (4/4) | 0.6396 (4/4) |
| naive | ambiguous | 2 | 0.5000 (2/2) | 0.5418 (2/2) |
| naive | unanswerable_locally | 3 | 0.0000 (3/3) | 0.6426 (3/3) |
| hybrid | lookup | 9 | 0.6667 (6/9) | 0.4835 (9/9) |
| hybrid | synthesis | 4 | 0.5833 (4/4) | 0.6873 (4/4) |
| hybrid | ambiguous | 2 | 1.0000 (2/2) | 0.5186 (2/2) |
| hybrid | unanswerable_locally | 3 | 0.0000 (3/3) | 0.3926 (3/3) |
| full_crag | lookup | 9 | 0.6667 (6/9) | 0.4248 (9/9) |
| full_crag | synthesis | 4 | 0.6250 (4/4) | 0.7681 (4/4) |
| full_crag | ambiguous | 2 | 1.0000 (2/2) | 0.5186 (2/2) |
| full_crag | unanswerable_locally | 3 | 0.0000 (3/3) | 0.3926 (3/3) |

**Web-fallback trigger rate (full_crag):** 16.7% (3/18 questions)
**Web-fallback trigger accuracy (rows with a defined expectation only; ambiguous rows excluded):** 100.0% (16/16)

## Caveats (read before citing these numbers)

- Tiny sample: 18 questions. One lookup/synthesis question moves recall by several points; differences between hybrid and full_crag are likely within noise.
- The same model family generates and judges (gpt-oss-20b), so self-preference bias may inflate RAGAS scores.
- For `ambiguous`/`unanswerable_locally` rows, ground truth is a meta-explanation, so RAGAS scores there measure refusal-appropriateness, not factual accuracy. Use the by-type table.
- Recall@5 uses a ±1 page window on 2-4 page documents, so it is lenient; the exact-page table is the stricter view.
- full_crag recall is measured on the final post-grading context (only chunks above the high threshold, plus any web results), not on the raw top-5, so it is not directly comparable to naive/hybrid raw retrieval.
- Latency reflects free-tier provider conditions.
- naive/hybrid have no web fallback by design (trigger rate 0%).
- v2 disclosure: an auto-generated v1 test set was found defective and replaced by hand-written v2 questions with machine-verified ground truth. System outputs for three v1 questions were seen before v2 was written.