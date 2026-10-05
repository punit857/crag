# Fallback-Trigger Comparison Report (Phase 3)

**Evaluation Date:** 2026-10-04 02:38:19
**Evaluated Samples:** 66
**Ground Truth Reference:** `data/grader_labels.json` (`heuristic_label` derived via keyword-overlap heuristic)

---

## Executive Summary

| Method | Accuracy vs. Heuristic | Recommended Trigger Setting |
| :--- | :---: | :---: |
| **Reranker-Score Threshold** | **62.12%** | `th_high=-4.2437, th_low=-9.8662` |
| **LLM Grader (`GROQ`)** | **22.73%** | `llm_provider=groq` |

**Winning Method:** `threshold`

> **Important Note on Ground Truth:** The `heuristic_label` in `grader_labels.json` was generated automatically via keyword overlap heuristics, not human annotation. These metrics reflect agreement with a heuristic benchmark, not absolute correctness. Disagreements below should be spot-checked manually.

---

## 1. Reranker-Score Threshold Method

The cross-encoder reranker (`BAAI/bge-reranker-base`) scores each `(query, chunk_text)` pair directly. Thresholds divide scores into `CORRECT` (>= T_high), `AMBIGUOUS` (T_low <= score < T_high), and `INCORRECT` (< T_low).

* **Optimal High Threshold (T_high):** `-4.2437`
* **Optimal Low Threshold (T_low):** `-9.8662`
* **Accuracy:** `0.6212` (62.12%)

### 3x3 Confusion Matrix (Threshold Method)

| Ground Truth \ Predicted | CORRECT | AMBIGUOUS | INCORRECT |
| --- | --- | --- | --- |
| **CORRECT** | 14 | 7 | 1 |
| **AMBIGUOUS** | 2 | 25 | 12 |
| **INCORRECT** | 0 | 3 | 2 |


---

## 2. LLM Grader Method

The LLM (`GROQ` - `openai/gpt-oss-20b`) evaluates each `(query, chunk_text)` pair using a zero-temperature prompt.

* **Accuracy:** `0.2273` (22.73%)

### 3x3 Confusion Matrix (LLM Grader Method)

| Ground Truth \ Predicted | CORRECT | AMBIGUOUS | INCORRECT |
| --- | --- | --- | --- |
| **CORRECT** | 2 | 9 | 11 |
| **AMBIGUOUS** | 1 | 8 | 30 |
| **INCORRECT** | 0 | 0 | 5 |


---

## 3. Disagreement Spot-Check Samples (Top 10)

- **Query:** In the context of the industrial assessment center technical guidance, what energy or operational savings are achieved regarding save energy your?
  - **Chunk Snippet:** Save Energy Now in Your Steam Systems Steam systems account for about 30% of the total energy used in industrial applications for product output. Thes...
  - **Heuristic Label:** `CORRECT`
  - **Threshold Pred:** `CORRECT` | **LLM Grader Pred:** `AMBIGUOUS`

- **Query:** In the context of the industrial assessment center technical guidance, what specific technical parameter or guidance applies to energy efficient technologies?
  - **Chunk Snippet:** 11% to 18% by using energy-efficient technologies and practices. Thus, there is great potential for savings. ITP software tools such as the Pumping Sy...
  - **Heuristic Label:** `CORRECT`
  - **Threshold Pred:** `CORRECT` | **LLM Grader Pred:** `AMBIGUOUS`

- **Query:** In the context of the cement kiln exhaust fan system, what key operating requirement or maintenance practice is specified for because severe vibrationproblem?
  - **Chunk Snippet:** BENEFITS •Saves 175,000 kilowatt-hours (kWh) annually •Increases equipment life •Improves productivity •Saves $6,000 in annual energy costs •Reduces a...
  - **Heuristic Label:** `CORRECT`
  - **Threshold Pred:** `AMBIGUOUS` | **LLM Grader Pred:** `INCORRECT`

- **Query:** In the context of the cement kiln exhaust fan system, what key operating requirement or maintenance practice is specified for because severe vibrationproblem?
  - **Chunk Snippet:** commonly used for the construction of highways, bridges, commercial and industrial complexes, residential homes, and other structures. The Durkee plan...
  - **Heuristic Label:** `AMBIGUOUS`
  - **Threshold Pred:** `AMBIGUOUS` | **LLM Grader Pred:** `INCORRECT`

- **Query:** In the context of the cement kiln exhaust fan system, how does implementing proper technical measures affect efficiency in bestpractices emphasizes plant?
  - **Chunk Snippet:** BestPractices is part of the Office of Industrial Technologies (OIT’s) Industries of the Future strategy, which helps the country’s most energy-intens...
  - **Heuristic Label:** `CORRECT`
  - **Threshold Pred:** `CORRECT` | **LLM Grader Pred:** `INCORRECT`

- **Query:** In the context of the industrial pumping system, what energy or operational savings are achieved regarding energy tips pumping?
  - **Chunk Snippet:** Energy Tips – Pumping Systems Industrial Technologies Program Pumping Systems Tip Sheet #6 • October 2005 Match Pumps to System Requirements An indust...
  - **Heuristic Label:** `CORRECT`
  - **Threshold Pred:** `CORRECT` | **LLM Grader Pred:** `AMBIGUOUS`

- **Query:** In the context of the industrial pumping system, what energy or operational savings are achieved regarding energy tips pumping?
  - **Chunk Snippet:** About DOE’s Industrial Technologies Program The Industrial Technologies Program, through partnerships with industry, government, and non-governmental ...
  - **Heuristic Label:** `AMBIGUOUS`
  - **Threshold Pred:** `AMBIGUOUS` | **LLM Grader Pred:** `INCORRECT`

- **Query:** In the context of the industrial pumping system, what energy or operational savings are achieved regarding energy tips pumping?
  - **Chunk Snippet:** that support a systems approach at the corporate and plant level will be the key to achieving large scale energy efficiency improvement in manufacturi...
  - **Heuristic Label:** `AMBIGUOUS`
  - **Threshold Pred:** `AMBIGUOUS` | **LLM Grader Pred:** `INCORRECT`

- **Query:** In the context of the industrial equipment market assessment, what specific technical parameter or guidance applies to include only facilities?
  - **Chunk Snippet:** Compressed air systems: System Efficiency Improvements Speed Controls Total Specialized systems: Total Total System Improvements 13,248 2,276 15,524 5...
  - **Heuristic Label:** `CORRECT`
  - **Threshold Pred:** `AMBIGUOUS` | **LLM Grader Pred:** `INCORRECT`

- **Query:** In the context of the industrial equipment market assessment, what specific technical parameter or guidance applies to include only facilities?
  - **Chunk Snippet:** U.S. Department of Energy - Energy Efficiency and Renewable Energy Advanced Manufacturing Office A Glimpse at What's Between the Covers of the Market ...
  - **Heuristic Label:** `AMBIGUOUS`
  - **Threshold Pred:** `AMBIGUOUS` | **LLM Grader Pred:** `INCORRECT`

---

## Recommendation & Pipeline Decision

* Recommended `fallback_trigger_method` for `src/config.py`: `threshold`
* Threshold parameters (if using threshold method): `reranker_threshold_high = -4.2437`, `reranker_threshold_low = -9.8662`
* **This report does not auto-edit `src/config.py`.** Apply these values manually after reviewing the disagreement samples above.
