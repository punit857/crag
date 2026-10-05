"""
scripts/evaluate_fallback_triggers.py — Phase 3: Fallback-trigger comparison.

Compares two methods for deciding when the CRAG pipeline should trigger
web-search fallback instead of trusting local retrieval:
  1. Reranker-score threshold (cheap, deterministic)
  2. LLM grader (per-chunk CORRECT/AMBIGUOUS/INCORRECT classification)

Both are evaluated against data/grader_labels.json's heuristic_label field.
NOTE: heuristic_label is itself a keyword-overlap heuristic, not hand-verified
ground truth — these results measure agreement with a heuristic, not absolute
correctness. Spot-check disagreements manually before trusting the winner.
"""

import json
import os
import time
import re
from typing import List, Dict, Tuple, Any

from fastembed.rerank.cross_encoder import TextCrossEncoder
from src.config import config


def get_llm():
    if config.llm_provider.lower() == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=config.groq_model,
            api_key=config.groq_api_key if config.groq_api_key else None,
            temperature=0.0,
        )
    elif config.llm_provider.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            google_api_key=config.gemini_api_key if config.gemini_api_key else None,
            temperature=0.0,
        )
    else:
        raise ValueError(f"Unsupported llm_provider: {config.llm_provider}")


def classify_score(score: float, th_high: float, th_low: float) -> str:
    if score >= th_high:
        return "CORRECT"
    elif score >= th_low:
        return "AMBIGUOUS"
    else:
        return "INCORRECT"


def compute_confusion_matrix(ground_truths: List[str], predictions: List[str]) -> Tuple[Dict[str, Dict[str, int]], float]:
    labels = ["CORRECT", "AMBIGUOUS", "INCORRECT"]
    matrix = {gt: {pred: 0 for pred in labels} for gt in labels}
    correct_count = 0
    total = len(ground_truths)

    for gt, pred in zip(ground_truths, predictions):
        if gt not in labels:
            gt = "AMBIGUOUS"
        if pred not in labels:
            pred = "INCORRECT"
        matrix[gt][pred] += 1
        if gt == pred:
            correct_count += 1

    accuracy = correct_count / total if total > 0 else 0.0
    return matrix, accuracy


def format_confusion_matrix_md(matrix: Dict[str, Dict[str, int]]) -> str:
    labels = ["CORRECT", "AMBIGUOUS", "INCORRECT"]
    md = "| Ground Truth \\ Predicted | CORRECT | AMBIGUOUS | INCORRECT |\n"
    md += "| --- | --- | --- | --- |\n"
    for gt in labels:
        row = f"| **{gt}** |"
        for pred in labels:
            row += f" {matrix[gt][pred]} |"
        md += row + "\n"
    return md


def call_llm_grader_with_retry(llm, query: str, chunk_text: str) -> str:
    prompt = f"""You are an evaluator assessing the relevance of a retrieved document chunk to a user query.

QUERY: {query}
CHUNK TEXT: {chunk_text}

Classify the relevance of the CHUNK TEXT to the QUERY into EXACTLY ONE of these categories:
- CORRECT: The chunk directly contains relevant information to answer or significantly answer the query.
- AMBIGUOUS: The chunk is partially relevant (mentions related systems or concepts), but lacks specific details or is a partial mismatch.
- INCORRECT: The chunk is off-topic or irrelevant.

Respond ONLY with a JSON object in this format:
{{"grade": "CORRECT"}} or {{"grade": "AMBIGUOUS"}} or {{"grade": "INCORRECT"}}"""

    attempts = 0
    max_attempts = config.retry_max_attempts
    backoff = config.retry_backoff_seconds

    while attempts < max_attempts:
        attempts += 1
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            cleaned = content.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            data = json.loads(cleaned)
            grade = data.get("grade", "").upper()
            if grade in ["CORRECT", "AMBIGUOUS", "INCORRECT"]:
                return grade

            match = re.search(r'\b(CORRECT|AMBIGUOUS|INCORRECT)\b', content.upper())
            if match:
                return match.group(1)

            print(f"[LLM Grader Warning] Unparseable grade, attempt {attempts}/{max_attempts}: {content[:100]!r}")

        except Exception as e:
            print(f"[LLM Grader Error] Attempt {attempts}/{max_attempts} failed: {e}")

        if attempts < max_attempts:
            time.sleep(backoff * (2 ** (attempts - 1)))

    print(f"[LLM Grader Error] Exhausted {max_attempts} attempts, defaulting to INCORRECT.")
    return "INCORRECT"


def main():
    labels_path = os.path.join("data", "grader_labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Could not find {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} evaluation records from {labels_path}")
    if not data:
        print("[ERROR] grader_labels.json is empty — nothing to evaluate. Exiting.")
        return

    # Step 1: Reranker Scores
    print("\n--- Step 1: Computing Cross-Encoder Reranker Scores ---")
    reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")

    scores = []
    ground_truths = []

    for item in data:
        q = item["query"]
        chunk = item["chunk_text"]
        gt = item["heuristic_label"]

        score_res = list(reranker.rerank(q, [chunk]))
        score = float(score_res[0])

        scores.append(score)
        ground_truths.append(gt)

    min_s, max_s = min(scores), max(scores)
    print(f"Reranker Scores calculated: Min={min_s:.4f}, Max={max_s:.4f}, Mean={sum(scores)/len(scores):.4f}")

    # NEW CONSTRAINT: Force th_low to leave at least 5% of scores in the INCORRECT bin
    sorted_scores = sorted(scores)
    five_percentile_index = max(1, int(len(sorted_scores) * 0.05))
    min_allowed_th_low = sorted_scores[five_percentile_index]
    
    print(f"Constraint applied: th_low must be > {min_allowed_th_low:.4f} (5th percentile) to prevent collapsing the INCORRECT class.")

    print("\nSweeping threshold combinations...")
    best_acc = -1.0
    best_th_high = 0.0
    best_th_low = 0.0
    best_matrix = None
    best_preds = []

    step = (max_s - min_s) / 40.0 if max_s > min_s else 0.1
    candidate_thresholds = [min_s + i * step for i in range(41)]

    for i in range(len(candidate_thresholds)):
        for j in range(i):
            th_low = candidate_thresholds[j]
            th_high = candidate_thresholds[i]

            # Apply structural constraint
            if th_low <= min_allowed_th_low:
                continue

            preds = [classify_score(s, th_high, th_low) for s in scores]
            matrix, acc = compute_confusion_matrix(ground_truths, preds)

            if acc > best_acc:
                best_acc = acc
                best_th_high = th_high
                best_th_low = th_low
                best_matrix = matrix
                best_preds = preds

    print(f"Best Threshold Pair: High={best_th_high:.4f}, Low={best_th_low:.4f} -> Accuracy={best_acc:.4f} ({best_acc*100:.2f}%)")

    # Step 2: LLM Grader
    print("\n--- Step 2: Evaluating LLM Grader ---")
    if len(data) > config.max_llm_calls_per_eval_run:
        print(f"WARNING: Dataset length ({len(data)}) exceeds max_llm_calls_per_eval_run ({config.max_llm_calls_per_eval_run}). Subsetting dataset.")
        eval_subset = data[:config.max_llm_calls_per_eval_run]
        llm_gt = ground_truths[:config.max_llm_calls_per_eval_run]
    else:
        eval_subset = data
        llm_gt = ground_truths

    llm = get_llm()
    llm_preds = []

    print(f"Running LLM Grader ({config.llm_provider.upper()}) on {len(eval_subset)} samples...")
    for idx, item in enumerate(eval_subset):
        pred = call_llm_grader_with_retry(llm, item["query"], item["chunk_text"])
        llm_preds.append(pred)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_subset):
            print(f"Processed {idx + 1}/{len(eval_subset)} samples...")

    llm_matrix, llm_acc = compute_confusion_matrix(llm_gt, llm_preds)
    print(f"LLM Grader Accuracy: {llm_acc:.4f} ({llm_acc*100:.2f}%)")

    # Step 3: Disagreements for spot-check
    disagreements = []
    for idx, (gt, th_p, llm_p) in enumerate(zip(llm_gt, best_preds[:len(llm_gt)], llm_preds)):
        if th_p != llm_p:
            disagreements.append({
                "index": idx,
                "query": eval_subset[idx]["query"],
                "chunk_snippet": eval_subset[idx]["chunk_text"][:150] + "...",
                "heuristic_gt": gt,
                "threshold_pred": th_p,
                "llm_pred": llm_p,
            })

    # Step 4: Report
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "grader_comparison.md")

    winner = "llm_grader" if llm_acc > best_acc else "threshold"
    model_used = config.groq_model if config.llm_provider.lower() == "groq" else config.gemini_model

    report_content = f"""# Fallback-Trigger Comparison Report (Phase 3)

**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Evaluated Samples:** {len(eval_subset)}
**Ground Truth Reference:** `data/grader_labels.json` (`heuristic_label` derived via keyword-overlap heuristic)

---

## Executive Summary

| Method | Accuracy vs. Heuristic | Recommended Trigger Setting |
| :--- | :---: | :---: |
| **Reranker-Score Threshold** | **{best_acc*100:.2f}%** | `th_high={best_th_high:.4f}, th_low={best_th_low:.4f}` |
| **LLM Grader (`{config.llm_provider.upper()}`)** | **{llm_acc*100:.2f}%** | `llm_provider={config.llm_provider}` |

**Winning Method:** `{winner}`

> **Important Note on Ground Truth:** The `heuristic_label` in `grader_labels.json` was generated automatically via keyword overlap heuristics, not human annotation. These metrics reflect agreement with a heuristic benchmark, not absolute correctness. Disagreements below should be spot-checked manually.

---

## 1. Reranker-Score Threshold Method

The cross-encoder reranker (`BAAI/bge-reranker-base`) scores each `(query, chunk_text)` pair directly. Thresholds divide scores into `CORRECT` (>= T_high), `AMBIGUOUS` (T_low <= score < T_high), and `INCORRECT` (< T_low).

* **Optimal High Threshold (T_high):** `{best_th_high:.4f}`
* **Optimal Low Threshold (T_low):** `{best_th_low:.4f}`
* **Accuracy:** `{best_acc:.4f}` ({best_acc*100:.2f}%)

### 3x3 Confusion Matrix (Threshold Method)

{format_confusion_matrix_md(best_matrix)}

---

## 2. LLM Grader Method

The LLM (`{config.llm_provider.upper()}` - `{model_used}`) evaluates each `(query, chunk_text)` pair using a zero-temperature prompt.

* **Accuracy:** `{llm_acc:.4f}` ({llm_acc*100:.2f}%)

### 3x3 Confusion Matrix (LLM Grader Method)

{format_confusion_matrix_md(llm_matrix)}

---

## 3. Disagreement Spot-Check Samples (Top {min(10, len(disagreements))})

"""
    if disagreements:
        for item in disagreements[:10]:
            report_content += f"""- **Query:** {item['query']}
  - **Chunk Snippet:** {item['chunk_snippet']}
  - **Heuristic Label:** `{item['heuristic_gt']}`
  - **Threshold Pred:** `{item['threshold_pred']}` | **LLM Grader Pred:** `{item['llm_pred']}`

"""
    else:
        report_content += "No disagreements between the two methods.\n\n"

    report_content += f"""---

## Recommendation & Pipeline Decision

* Recommended `fallback_trigger_method` for `src/config.py`: `{winner}`
* Threshold parameters (if using threshold method): `reranker_threshold_high = {best_th_high:.4f}`, `reranker_threshold_low = {best_th_low:.4f}`
* **This report does not auto-edit `src/config.py`.** Apply these values manually after reviewing the disagreement samples above.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nReport written successfully to {report_path}")


if __name__ == "__main__":
    main()