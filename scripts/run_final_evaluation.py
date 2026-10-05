"""
scripts/run_final_evaluation.py — Phase 5: Final evaluation on the held-out test set.

Variants (same generation prompt for all, so only retrieval/correction differs):
  1. naive      — dense-only top-5, direct generation
  2. hybrid     — hybrid+reranked top-5, direct generation
  3. full_crag  — complete CRAG graph (grading, web fallback, groundedness check)

Metrics: Recall@5 (±1 page, Phase 2 logic), exact-page recall, RAGAS faithfulness +
answer_correctness, average latency, web-fallback trigger rate/accuracy (full_crag).

Robustness features:
  - Every result is written to a raw JSON file immediately (crash-safe).
  - --resume continues an interrupted generation run; --from-raw skips generation.
  - RAGAS NaN rows are counted and reported, never silently dropped.
  - Retry backoff time is NOT included in latency.
  - --test-set selects the eval set; v2 runs write to separate *_v2 files so v1
    results are preserved as the documented first pass.

Written against: ragas 0.1.16, langchain-groq 0.1.9, langgraph 0.2.14.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import config

# NOTE: importing src.graph.nodes instantiates ONE global Retriever at import time.
# We reuse that exact instance. Creating a second Retriever() would open the
# local-file Qdrant folder twice, which fails with a lock error.
from src.graph.nodes import generate_answer, get_llm, retriever
from src.graph.build_graph import build_crag_graph
from src.eval.metrics import parse_sources, is_match

TEST_SET_PATH = Path("data") / "eval_set_test.json"  # overridden by --test-set
RESULTS_DIR = Path("results")

VARIANTS = ["naive", "hybrid", "full_crag"]
METRICS = ["faithfulness", "answer_correctness"]
QUESTION_TYPES = ["lookup", "synthesis", "ambiguous", "unanswerable_locally"]

# Rough per-question LLM call counts, used only for the pre-run estimate.
GEN_CALLS_PER_Q = {"naive": 1, "hybrid": 1, "full_crag": 4}
RAGAS_CALLS_PER_ROW = 5  # faithfulness ~2 + answer_correctness ~3

# Free-tier Groq rate limits are per-minute; config's 2s base is too short.
RETRY_BACKOFF = 10.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _json_default(o: Any):
    """Make numpy floats etc. JSON-serializable."""
    try:
        return float(o)
    except Exception:
        return str(o)


def _clean_score(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except Exception:
        return None
    return None if math.isnan(f) else f


def load_test_set() -> List[Dict]:
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"{TEST_SET_PATH} not found.")
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} questions from {TEST_SET_PATH}.")
    return data


def retry_call(fn, *args, max_attempts: Optional[int] = None, backoff: Optional[float] = None, **kwargs):
    """Retry with exponential backoff. Returns (result, seconds_of_the_successful_attempt).
    Backoff sleeps are deliberately excluded from the returned duration."""
    max_attempts = max_attempts or config.retry_max_attempts
    backoff = backoff or RETRY_BACKOFF
    last_err = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            return result, time.perf_counter() - t0
        except Exception as e:
            last_err = e
            print(f"    [WARN] attempt {attempt}/{max_attempts} failed: {str(e)[:200]}")
            if attempt < max_attempts:
                time.sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError(f"Exhausted {max_attempts} attempts. Last error: {last_err}")


def get_judge_llm():
    """LLM used by RAGAS. Same model/provider as the pipeline, but with a larger
    max_tokens so RAGAS's JSON outputs aren't truncated. reasoning_effort='low' is
    REQUIRED for gpt-oss-20b, otherwise it burns the budget on chain-of-thought
    and returns empty content."""
    if config.llm_provider.lower() == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=config.groq_model,
            api_key=config.groq_api_key if config.groq_api_key else None,
            temperature=0.0,
            max_tokens=4096,
            model_kwargs={"reasoning_effort": "low"},
        )
    return get_llm()


# --------------------------------------------------------------------------- #
# Variant runners — each returns a JSON-serializable record
# --------------------------------------------------------------------------- #
def run_one(variant: str, app, query: str) -> Dict[str, Any]:
    try:
        if variant == "naive":
            docs, t1 = retry_call(retriever.dense_search, query, top_k=5)
            res, t2 = retry_call(generate_answer, {"query": query, "documents": docs, "generation_iterations": 0})
            return {"answer": res["generation"], "docs": docs, "latency": t1 + t2, "web_used": False, "error": None}

        if variant == "hybrid":
            docs, t1 = retry_call(retriever.reranked_hybrid_search, query, top_k=5)
            res, t2 = retry_call(generate_answer, {"query": query, "documents": docs, "generation_iterations": 0})
            return {"answer": res["generation"], "docs": docs, "latency": t1 + t2, "web_used": False, "error": None}

        if variant == "full_crag":
            initial_state = {
                "query": query,
                "documents": [],
                "web_search_required": False,
                "web_search_iterations": 0,
                "generation_iterations": 0,
                "generation": "",
                "grounded": False,
                "execution_trace": [],
            }
            final_state, t = retry_call(app.invoke, initial_state)
            return {
                "answer": final_state.get("generation", ""),
                "docs": final_state.get("documents", []),
                "latency": t,
                "web_used": final_state.get("web_search_iterations", 0) > 0,
                "error": None,
            }

        raise ValueError(f"Unknown variant: {variant}")
    except Exception as e:
        return {"answer": "", "docs": [], "latency": None, "web_used": False, "error": str(e)[:500]}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_recall_at_5(questions: List[Dict], results: Dict[str, Dict]) -> Tuple[int, int]:
    """Returns (hits, n_eval) over lookup/synthesis questions, using Phase 2's exact matching
    (±1 page tolerance). A failed generation counts as a miss (honest denominator)."""
    eval_qs = [q for q in questions if q.get("question_type") in ("lookup", "synthesis")]
    hits = 0
    for q in eval_qs:
        targets = parse_sources(q.get("source_doc"), q.get("source_page"))
        rec = results.get(q["id"])
        docs = rec["docs"] if rec and not rec.get("error") else []
        if any(is_match(d, targets) for d in docs[:5]):
            hits += 1
    return hits, len(eval_qs)


def compute_exact_page_recall(questions: List[Dict], results: Dict[str, Dict]) -> Dict[str, Tuple[int, int]]:
    """Stricter than Recall@5: chunk page must equal the target page exactly.
    Returns {'exact': (hits, n), 'all_sources': (hits, n_synthesis)}.
    'all_sources' (synthesis only) requires EVERY source doc to be retrieved, not just one."""
    eval_qs = [q for q in questions if q.get("question_type") in ("lookup", "synthesis")]
    exact_hits, syn_hits, n_syn = 0, 0, 0
    for q in eval_qs:
        targets = parse_sources(q.get("source_doc"), q.get("source_page"))
        rec = results.get(q["id"])
        docs = (rec["docs"] if rec and not rec.get("error") else [])[:5]

        def covered(t):
            return any(d.get("doc_name") == t[0] and (t[1] is None or d.get("page") == t[1]) for d in docs)

        if any(covered(t) for t in targets):
            exact_hits += 1
        if q["question_type"] == "synthesis":
            n_syn += 1
            if targets and all(covered(t) for t in targets):
                syn_hits += 1
    return {"exact": (exact_hits, len(eval_qs)), "all_sources": (syn_hits, n_syn)}


def run_ragas_eval(items: List[Tuple[Dict, Dict]]) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """items: list of (question_dict, result_record). Returns {qid: {metric: score_or_None}}.
    Returns None on total failure (logged loudly). Per-row NaNs become None and are
    counted in the report."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_correctness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    except ImportError as e:
        print(f"[ERROR] RAGAS import failed: {e}")
        return None

    if not items:
        return {}

    qids = []
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for q, rec in items:
        qids.append(q["id"])
        rows["question"].append(q["question"])
        rows["answer"].append(rec["answer"])
        rows["contexts"].append([d.get("text", "") for d in rec["docs"] if d.get("text")] or [""])
        rows["ground_truth"].append(q.get("ground_truth", ""))

    try:
        dataset = Dataset.from_dict(rows)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_correctness],
            llm=LangchainLLMWrapper(get_judge_llm()),
            embeddings=LangchainEmbeddingsWrapper(FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")),
            run_config=RunConfig(timeout=180, max_retries=8, max_wait=60, max_workers=1),
            raise_exceptions=False,
        )
        df = result.to_pandas()
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for i, qid in enumerate(qids):
            out[qid] = {m: (_clean_score(df[m].iloc[i]) if m in df.columns else None) for m in METRICS}

        n_valid = {m: sum(1 for v in out.values() if v[m] is not None) for m in METRICS}
        print(f"[INFO] RAGAS valid rows: {n_valid} of {len(qids)}")
        if all(v == 0 for v in n_valid.values()):
            print("[ERROR] RAGAS returned NaN for EVERY row — judge calls are failing "
                  "(rate limits, truncated JSON, or schema mismatch). Scores not saved.")
            return None
        return out
    except Exception as e:
        print(f"[ERROR] RAGAS evaluation failed: {type(e).__name__}: {e}")
        return None


def mean_valid(vals: List[Optional[float]]) -> Tuple[Optional[float], int]:
    v = [x for x in vals if x is not None]
    return (sum(v) / len(v) if v else None), len(v)


def fmt_score(mean: Optional[float], n_valid: int, n_rows: int) -> str:
    if mean is None:
        return "N/A"
    return f"{mean:.4f} ({n_valid}/{n_rows})"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(questions: List[Dict], raw: Dict, shown: List[str]) -> List[str]:
    n = len(questions)
    qmap = {q["id"]: q for q in questions}
    lines = [
        "# Phase 5: Final Evaluation (Held-Out Test Set)\n",
        f"**Test set:** `{TEST_SET_PATH}` ({n} questions)\n",
        "Score cells show `mean (n_valid/n_rows)`. Rows where RAGAS returned NaN are excluded from "
        "the mean but are counted in `n_valid`. Failed generations are counted in the Failures column.\n",
        "| Variant | Recall@5 (±1 page) | Faithfulness | Answer Correctness | Avg Latency (s) | Failures |",
        "|---|---|---|---|---|---|",
    ]

    per_variant: Dict[str, Dict] = {}
    for v in shown:
        results = raw["results"][v]
        ragas = raw["ragas"].get(v, {})
        ok_ids = [q["id"] for q in questions if results.get(q["id"]) and not results[q["id"]].get("error")]
        n_fail = n - len(ok_ids)

        hits, n_eval = compute_recall_at_5(questions, results)
        lat_vals = [results[i]["latency"] for i in ok_ids if results[i].get("latency") is not None]
        avg_lat = sum(lat_vals) / len(lat_vals) if lat_vals else None

        cells = {}
        for m in METRICS:
            vals = [ragas.get(i, {}).get(m) for i in ok_ids] if ragas else []
            mean, nv = mean_valid(vals)
            cells[m] = fmt_score(mean, nv, len(ok_ids)) if ragas else "N/A"

        per_variant[v] = {"ok_ids": ok_ids, "ragas": ragas}
        recall_str = f"{hits/n_eval:.4f} ({hits}/{n_eval})" if n_eval else "N/A"
        lat_str = f"{avg_lat:.2f}" if avg_lat is not None else "N/A"
        lines.append(f"| {v} | {recall_str} | {cells['faithfulness']} | {cells['answer_correctness']} | {lat_str} | {n_fail} |")

    # --- stricter retrieval metrics ---
    lines.append("\n**Stricter retrieval metrics (exact page match, top-5):**\n")
    lines.append("| Variant | Exact-page recall | Synthesis: all sources retrieved |")
    lines.append("|---|---|---|")
    for v in shown:
        s = compute_exact_page_recall(questions, raw["results"][v])
        e_h, e_n = s["exact"]
        a_h, a_n = s["all_sources"]
        lines.append(f"| {v} | {e_h}/{e_n} | {a_h}/{a_n} |")

    # --- by question type ---
    lines.append("\n## RAGAS by question type\n")
    lines.append("| Variant | Type | n | Faithfulness | Answer Correctness |")
    lines.append("|---|---|---|---|---|")
    for v in shown:
        ragas = per_variant[v]["ragas"]
        if not ragas:
            continue
        for t in QUESTION_TYPES:
            ids = [i for i in per_variant[v]["ok_ids"] if qmap[i].get("question_type") == t]
            if not ids:
                continue
            f_mean, f_n = mean_valid([ragas.get(i, {}).get("faithfulness") for i in ids])
            c_mean, c_n = mean_valid([ragas.get(i, {}).get("answer_correctness") for i in ids])
            lines.append(f"| {v} | {t} | {len(ids)} | {fmt_score(f_mean, f_n, len(ids))} | {fmt_score(c_mean, c_n, len(ids))} |")

    # --- web fallback (full_crag) ---
    if "full_crag" in shown:
        results = raw["results"]["full_crag"]
        ok = [q for q in questions if results.get(q["id"]) and not results[q["id"]].get("error")]
        if ok:
            triggered = [q["id"] for q in ok if results[q["id"]]["web_used"]]
            scored = [q for q in ok if q.get("expected_fallback") is not None]
            correct = [q["id"] for q in scored if results[q["id"]]["web_used"] == bool(q["expected_fallback"])]
            mism = [q["id"] for q in scored if q["id"] not in correct]
            lines.append(f"\n**Web-fallback trigger rate (full_crag):** {100*len(triggered)/len(ok):.1f}% "
                         f"({len(triggered)}/{len(ok)} questions)")
            if scored:
                lines.append(f"**Web-fallback trigger accuracy (rows with a defined expectation only; "
                             f"ambiguous rows excluded):** {100*len(correct)/len(scored):.1f}% "
                             f"({len(correct)}/{len(scored)})")
            if mism:
                detail = ", ".join(
                    f"{i} ({qmap[i].get('question_type')}, expected={qmap[i].get('expected_fallback')}, "
                    f"actual={results[i]['web_used']})" for i in mism)
                lines.append(f"**Trigger mismatches:** {detail}")

    lines.append("\n## Caveats (read before citing these numbers)\n")
    lines.append(f"- Tiny sample: {n} questions. One lookup/synthesis question moves recall by several points; "
                 "differences between hybrid and full_crag are likely within noise.")
    lines.append("- The same model family generates and judges (gpt-oss-20b), so self-preference bias may inflate RAGAS scores.")
    lines.append("- For `ambiguous`/`unanswerable_locally` rows, ground truth is a meta-explanation, so RAGAS scores there "
                 "measure refusal-appropriateness, not factual accuracy. Use the by-type table.")
    lines.append("- Recall@5 uses a ±1 page window on 2-4 page documents, so it is lenient; the exact-page table is the stricter view.")
    lines.append("- full_crag recall is measured on the final post-grading context (only chunks above the high threshold, "
                 "plus any web results), not on the raw top-5, so it is not directly comparable to naive/hybrid raw retrieval.")
    lines.append("- Latency reflects free-tier provider conditions.")
    lines.append("- naive/hybrid have no web fallback by design (trigger rate 0%).")
    lines.append("- v2 disclosure: an auto-generated v1 test set was found defective and replaced by hand-written v2 questions "
                 "with machine-verified ground truth. System outputs for three v1 questions were seen before v2 was written.")
    return lines


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    global TEST_SET_PATH

    parser = argparse.ArgumentParser(description="Phase 5: Final evaluation on held-out test set.")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS metrics")
    parser.add_argument("--max-questions", type=int, default=None, help="First N questions only (smoke test; uses *_smoke output files)")
    parser.add_argument("--dry-run", action="store_true", help="Print call estimate and exit (no API calls)")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run from the raw file")
    parser.add_argument("--from-raw", action="store_true", help="Skip generation; load raw file, run RAGAS (if missing) and write report")
    parser.add_argument("--rerun-ragas", action="store_true", help="Recompute RAGAS even if scores already exist in the raw file")
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS), help="Comma-separated subset of variants to run")
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds to pause between questions (rate-limit pacing)")
    parser.add_argument("--allow-over-cap", action="store_true", help="Run even if the estimate exceeds config.max_llm_calls_per_eval_run")
    parser.add_argument("--test-set", type=str, default=str(TEST_SET_PATH), help="Path to eval set JSON (e.g. data/eval_set_test_v2.json)")
    args = parser.parse_args()

    TEST_SET_PATH = Path(args.test_set)

    run_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in run_variants if v not in VARIANTS]
    if bad:
        sys.exit(f"[ERROR] Unknown variants: {bad}. Valid: {VARIANTS}")

    questions = load_test_set()
    if args.max_questions:
        questions = questions[: args.max_questions]
        print(f"[INFO] Limited to first {len(questions)} questions (--max-questions).")
    n = len(questions)

    suffix = ("_v2" if "v2" in TEST_SET_PATH.stem else "") + ("_smoke" if args.max_questions else "")
    raw_path = RESULTS_DIR / f"final_eval_raw{suffix}.json"
    report_path = RESULTS_DIR / f"final_ablation_table{suffix}.md"

    # ---- estimate ----
    est_gen = 0 if args.from_raw else n * sum(GEN_CALLS_PER_Q[v] for v in run_variants)
    est_ragas = 0 if args.skip_ragas else n * len(run_variants) * RAGAS_CALLS_PER_ROW
    est_total = est_gen + est_ragas
    print(f"[INFO] Estimated LLM calls: ~{est_gen} generation + ~{est_ragas} RAGAS = ~{est_total} total (rough).")
    print(f"[INFO] config.max_llm_calls_per_eval_run = {config.max_llm_calls_per_eval_run}")
    over_cap = est_total > config.max_llm_calls_per_eval_run
    if over_cap:
        print("[WARNING] Estimate exceeds the cap. Use --variants to split across runs, "
              "raise the config value deliberately, or pass --allow-over-cap.")
    if args.dry_run:
        print(f"[INFO] Output files would be: {raw_path} and {report_path}")
        print("[INFO] --dry-run set, exiting without making any API calls.")
        return
    if over_cap and not args.allow_over_cap:
        sys.exit("[ERROR] Refusing to run over the cap without --allow-over-cap.")

    # ---- raw store ----
    RESULTS_DIR.mkdir(exist_ok=True)
    raw: Dict[str, Any] = {"results": {v: {} for v in VARIANTS}, "ragas": {v: {} for v in VARIANTS}}
    if args.resume or args.from_raw:
        if not raw_path.exists():
            sys.exit(f"[ERROR] {raw_path} not found; cannot --resume/--from-raw.")
        loaded = json.loads(raw_path.read_text(encoding="utf-8"))
        for v in VARIANTS:
            raw["results"][v] = loaded.get("results", {}).get(v, {})
            raw["ragas"][v] = loaded.get("ragas", {}).get(v, {})
        print(f"[INFO] Loaded existing raw results from {raw_path}.")
    elif raw_path.exists():
        print(f"[WARNING] {raw_path} exists and will be OVERWRITTEN (use --resume to continue instead).")

    def save_raw():
        raw_path.write_text(json.dumps(raw, indent=2, default=_json_default), encoding="utf-8")

    # ---- generation phase ----
    if not args.from_raw:
        print("\n[INFO] Building CRAG graph and warming up retrieval models (not timed)...")
        app = build_crag_graph() if "full_crag" in run_variants else None
        retriever.dense_search("warm-up query", top_k=1)
        retriever.reranked_hybrid_search("warm-up query", top_k=1)

        for idx, q in enumerate(questions, 1):
            qid, query = q["id"], q["question"]
            print(f"\n[{idx}/{n}] {qid} ({q.get('question_type')}): {query[:70]}...")
            did_work = False
            for v in run_variants:
                existing = raw["results"][v].get(qid)
                if existing and not existing.get("error"):
                    print(f"  {v}: already done, skipping")
                    continue
                rec = run_one(v, app, query)
                raw["results"][v][qid] = rec
                save_raw()
                did_work = True
                if rec["error"]:
                    print(f"  {v}: FAILED — {rec['error'][:150]}")
                else:
                    extra = f", web_used={rec['web_used']}" if v == "full_crag" else ""
                    print(f"  {v}: {rec['latency']:.1f}s, {len(rec['docs'])} docs{extra}")
            if did_work and args.sleep > 0 and idx < n:
                time.sleep(args.sleep)

        # ---- RAGAS phase ----
    if not args.skip_ragas:
        for v in run_variants:
            existing = raw["ragas"].get(v, {})

            def already_scored(qid, existing=existing):
                row = existing.get(qid)
                return bool(row) and any(s is not None for s in row.values())

            items = [(q, raw["results"][v][q["id"]]) for q in questions
                     if raw["results"][v].get(q["id"]) and not raw["results"][v][q["id"]].get("error")
                     and (args.rerun_ragas or not already_scored(q["id"]))]
            if not items:
                print(f"\n[INFO] RAGAS: nothing new to score for '{v}', skipping.")
                continue
            print(f"\n[INFO] Running RAGAS for '{v}' on {len(items)} rows (serial; slow on purpose)...")
            scores = run_ragas_eval(items)
            if scores is not None:
                raw["ragas"][v] = {**({} if args.rerun_ragas else existing), **scores}
                save_raw()
            else:
                print(f"[ERROR] RAGAS failed for '{v}'. Generation results are safe in {raw_path}.")
    else:
        print("\n[INFO] --skip-ragas set, skipping faithfulness/answer_correctness.")

    # ---- report ----
    shown = [v for v in VARIANTS if raw["results"][v]]
    lines = build_report(questions, raw, shown)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[INFO] Report written to {report_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()