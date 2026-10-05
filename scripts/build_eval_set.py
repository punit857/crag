"""
scripts/build_eval_set.py — Builds Phase 0 evaluation data from real PDF content.

Generates:
  - data/eval_set_dev.json   (~65% split)
  - data/eval_set_test.json  (~35% split, held out — never tune against this)
  - data/grader_labels.json  (heuristic-labeled query/chunk pairs for Phase 3)

All questions and ground truths are grounded in actual extracted PDF text,
not fabricated. Grader labels are generated via a keyword-overlap heuristic
and explicitly tagged as such — NOT hand-labeled — per project rules.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PDF_DIR = BASE_DIR / "data" / "raw_pdfs"
DATA_DIR = BASE_DIR / "data"

DEV_SET_PATH = DATA_DIR / "eval_set_dev.json"
TEST_SET_PATH = DATA_DIR / "eval_set_test.json"
GRADER_LABELS_PATH = DATA_DIR / "grader_labels.json"

RNG_SEED = 42
DEV_SPLIT_RATIO = 0.65

SYSTEM_OVERVIEW_MAP = {
    "mi_cs_ashgrove.pdf": "cement kiln exhaust fan system",
    "metal_cs_technicast.pdf": "metal foundry furnace heat recovery system",
    "fujifilm_case_study.pdf": "chilled water plant system",
    "market_assessment_glimpse.pdf": "industrial equipment market assessment",
    "39156.pdf": "industrial assessment center technical guidance",
    "39157.pdf": "industrial assessment center technical guidance",
}

OUT_OF_SCOPE_QUESTIONS = [
    ("What are the standard inspection intervals and blade tip clearance limits for commercial aviation turbofan engines?",
     "Aviation turbofan maintenance and inspection protocols are outside the scope of local DOE industrial guidance."),
    ("How do you calibrate primary coolant loop pressure sensors in a nuclear pressurized water reactor?",
     "Nuclear reactor coolant pressure sensor calibration is not covered in local DOE industrial tip sheets."),
    ("What is the maximum allowable spindle runout for high-precision CNC milling machines under ISO 1940?",
     "CNC machine spindle tolerances and machining precision standards are outside the local equipment corpus."),
    ("What are the EPA OSHA safety compliance mandates for handling anhydrous ammonia refrigerants in cold storage?",
     "Anhydrous ammonia safety and hazardous chemical regulations are outside the scope of local energy efficiency tip sheets."),
    ("How do you diagnose high exhaust gas temperature (EGT) warnings on two-stroke marine propulsion diesel engines?",
     "Marine diesel propulsion thermodynamics and troubleshooting are not covered in local motor and pump guidance."),
    ("What is the recommended solar panel tilt angle and MPPT tracking configuration for commercial rooftop PV arrays?",
     "Photovoltaic solar generation hardware and inverter configuration are outside the industrial equipment guidance corpus."),
    ("What are the transformer oil dissolved gas analysis (DGA) thresholds for detecting thermal faulting in substations?",
     "Electrical substation transformer oil chemical analysis is outside the scope of local equipment efficiency guidance."),
    ("How do you configure PID gains on an automated robotic welding arm to eliminate seam tracking vibration?",
     "Automated robotic arm kinematics and PID tuning for welding are not covered in local DOE tip sheets."),
]

AMBIGUOUS_PROMPTS = [
    "What is the recommended operating pressure for the pump?",
    "How often should the motor be inspected?",
    "What is the standard efficiency rating for the fan?",
    "How much energy does installing a variable frequency drive save?",
    "What is the most common cause of efficiency loss in the system?",
]


def classify_doc_system(doc_name: str) -> str:
    if doc_name in SYSTEM_OVERVIEW_MAP:
        return SYSTEM_OVERVIEW_MAP[doc_name]
    name_lower = doc_name.lower()
    if "pump" in name_lower:
        return "industrial pumping system"
    if "air" in name_lower or "compress" in name_lower:
        return "compressed air system"
    if "steam" in name_lower or "boiler" in name_lower:
        return "steam distribution system"
    if "motor" in name_lower or "drive" in name_lower:
        return "electric motor drive system"
    if "fan" in name_lower or "blower" in name_lower:
        return "industrial fan blower system"
    return "industrial energy system"


def load_corpus() -> List[Dict]:
    pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {RAW_PDF_DIR}. Run fetch_corpus.py first.")

    corpus_pages = []
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            pages = reader.pages
        except Exception as file_err:
            print(f"[WARN] Could not open PDF file {pdf_path.name}: {file_err}")
            continue

        for page_idx, page in enumerate(pages, start=1):
            try:
                raw_text = page.extract_text() or ""
                cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
                if (
                    len(cleaned_text) > 150
                    and cleaned_text.count("...") < 5
                    and "Table of Contents" not in cleaned_text
                ):
                    corpus_pages.append({
                        "doc_name": pdf_path.name,
                        "page_number": page_idx,
                        "text": cleaned_text,
                        "system": classify_doc_system(pdf_path.name),
                    })
            except Exception as page_err:
                print(f"[WARN] Error reading page {page_idx} in {pdf_path.name}: {page_err}")

    print(f"[INFO] Loaded {len(corpus_pages)} valid pages across {len(pdf_files)} documents.")
    return corpus_pages


def extract_page_fact(text: str) -> Optional[Tuple[str, str]]:
    """Extracts a real, verifiable sentence and builds a natural question for it."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]
    candidate_facts = []
    for s in sentences:
        if s.endswith(":") or len(s) > 280:
            continue
        if re.search(
            r"\b(\d+|percent|%|hp|psi|gpm|kwh|vfd|rpm|efficiency|reduce|increase|save|recommend|require)\b",
            s, re.IGNORECASE,
        ):
            candidate_facts.append(s)

    if not candidate_facts:
        return None

    chosen_fact = candidate_facts[0]
    meaningful_words = [
        w for w in re.findall(r"\b[A-Za-z]{4,}\b", chosen_fact)
        if w.lower() not in {
            "this", "that", "with", "from", "have", "were", "been", "their",
            "they", "will", "would", "should", "using", "used",
        }
    ]
    topic_context = (
        " ".join(meaningful_words[:3]).lower() if len(meaningful_words) >= 3 else "equipment operation"
    )

    if re.search(r"\b(reduces?|saved?|saving|decreased?)\b", chosen_fact, re.IGNORECASE):
        question = f"What energy or operational savings are achieved regarding {topic_context}?"
    elif re.search(r"\b(requires?|should|must|recommended?)\b", chosen_fact, re.IGNORECASE):
        question = f"What key operating requirement or maintenance practice is specified for {topic_context}?"
    elif re.search(r"\b(efficiency|performance|improvement)\b", chosen_fact, re.IGNORECASE):
        question = f"How does implementing proper technical measures affect efficiency in {topic_context}?"
    else:
        question = f"What specific technical parameter or guidance applies to {topic_context}?"

    return question, chosen_fact


def build_eval_questions(corpus: List[Dict]) -> List[Dict]:
    qa_list: List[Dict] = []
    valid_pages: List[Tuple[Dict, str, str]] = []
    seen_question_texts: Set[str] = set()

    for page in corpus:
        extracted = extract_page_fact(page["text"])
        if not extracted:
            continue
        question_text, fact = extracted
        if question_text in seen_question_texts:
            continue
        seen_question_texts.add(question_text)
        valid_pages.append((page, question_text, fact))

    print(f"[INFO] Extracted {len(valid_pages)} unique, grounded factual pages for Q&A generation.")
    if len(valid_pages) < 10:
        print("[WARN] Low valid page count — eval set will be smaller than originally targeted. This is fine; do not pad with duplicates.")

    # --- Lookup questions ---
    lookup_target = min(22, len(valid_pages))
    for page_record, question_text, ground_truth in valid_pages[:lookup_target]:
        qa_list.append({
            "question": f"In the context of the {page_record['system']}, {question_text.lower()}",
            "ground_truth": ground_truth,
            "source_doc": page_record["doc_name"],
            "source_page": page_record["page_number"],
            "question_type": "lookup",
            "expected_fallback": False,
        })

    # --- Synthesis questions: distinct documents AND distinct system-label pairs ---
    synth_target = min(14, len(valid_pages))
    seen_pairs: Set[frozenset] = set()
    seen_label_pairs: Set[frozenset] = set()
    synth_added = 0
    rng = random.Random(RNG_SEED)
    indices = list(range(len(valid_pages)))

    attempts = 0
    max_attempts = synth_target * 30
    while synth_added < synth_target and attempts < max_attempts:
        attempts += 1
        if len(indices) < 2:
            break
        i, j = rng.sample(indices, 2)

        p1, q1, gt1 = valid_pages[i]
        p2, q2, gt2 = valid_pages[j]

        if p1["doc_name"] == p2["doc_name"]:
            continue

        pair_key = frozenset({(p1["doc_name"], p1["page_number"]), (p2["doc_name"], p2["page_number"])})
        if pair_key in seen_pairs:
            continue

        # System-label pairs must also be unique, since question text is built
        # from labels, not document identity — two different document pairs
        # sharing the same label pair would otherwise produce identical text.
        label_pair_key = frozenset({p1["system"], p2["system"]})
        if label_pair_key in seen_label_pairs:
            continue

        seen_pairs.add(pair_key)
        seen_label_pairs.add(label_pair_key)

        qa_list.append({
            "question": (
                f"What are the technical efficiency considerations when comparing "
                f"management of {p1['system']} versus {p2['system']}?"
            ),
            "ground_truth": (
                f"For {p1['system']} ({p1['doc_name']}, p.{p1['page_number']}): {gt1} "
                f"For {p2['system']} ({p2['doc_name']}, p.{p2['page_number']}): {gt2}"
            ),
            "source_doc": f"{p1['doc_name']}, {p2['doc_name']}",
            "source_page": f"p.{p1['page_number']}, p.{p2['page_number']}",
            "question_type": "synthesis",
            "expected_fallback": False,
        })
        synth_added += 1

    if synth_added < synth_target:
        print(f"[WARN] Only generated {synth_added}/{synth_target} unique synthesis pairs "
              f"(limited by the number of distinct system-label combinations in this corpus). "
              f"This is acceptable — do not pad with duplicates.")

    # --- Ambiguous questions ---
    for prompt in AMBIGUOUS_PROMPTS:
        qa_list.append({
            "question": prompt,
            "ground_truth": (
                "This query is ambiguous because it does not specify which equipment "
                "or system it refers to."
            ),
            "source_doc": None,
            "source_page": None,
            "question_type": "ambiguous",
            "expected_fallback": True,
        })

    # --- Unanswerable-locally questions ---
    for q, ans in OUT_OF_SCOPE_QUESTIONS:
        qa_list.append({
            "question": q,
            "ground_truth": f"This information is not present in the local DOE equipment tip sheets. {ans}",
            "source_doc": None,
            "source_page": None,
            "question_type": "unanswerable_locally",
            "expected_fallback": True,
        })

    rng.shuffle(qa_list)
    return qa_list


def generate_grader_labels(qa_list: List[Dict], corpus: List[Dict]) -> List[Dict]:
    """
    Generates heuristic-labeled (query, chunk) pairs for Phase 3's
    reranker-threshold vs. LLM-grader comparison. For every grounded
    LOOKUP question (synthesis questions are skipped — their source_doc
    is a comma-separated pair, which this single-target heuristic doesn't
    handle), produces THREE pairs:
      1. positive_target        — the actual correct source chunk
      2. same_system_mismatch   — a different chunk, same equipment system
      3. diff_system_mismatch   — a chunk from a completely different system
    """
    labels: List[Dict] = []
    rng = random.Random(RNG_SEED)

    def term_set(text: str) -> Set[str]:
        return set(re.findall(r"\b\w{4,}\b", text.lower()))

    for qa in qa_list:
        if qa["expected_fallback"] or not qa["source_doc"]:
            continue
        if "," in qa["source_doc"]:
            continue  # synthesis — skip for this single-target heuristic

        combined_terms = term_set(qa["question"]) | term_set(qa["ground_truth"])
        source_doc = qa["source_doc"]
        source_page = qa["source_page"]

        target_page = next(
            (p for p in corpus if p["doc_name"] == source_doc and p["page_number"] == source_page),
            None,
        )
        if not target_page:
            continue

        overlap = len(combined_terms & term_set(target_page["text"]))
        label = "CORRECT" if overlap >= 4 else "AMBIGUOUS"
        labels.append({
            "query": qa["question"],
            "chunk_text": target_page["text"][:300] + "...",
            "heuristic_label": label,
            "match_type": "positive_target",
        })

        qa_system = target_page["system"]

        same_sys_candidates = [
            p for p in corpus
            if p["system"] == qa_system
            and (p["doc_name"] != source_doc or p["page_number"] != source_page)
        ]
        if same_sys_candidates:
            chosen = rng.choice(same_sys_candidates)
            overlap = len(combined_terms & term_set(chosen["text"]))
            sys_label = "AMBIGUOUS" if overlap >= 3 else "INCORRECT"
            labels.append({
                "query": qa["question"],
                "chunk_text": chosen["text"][:300] + "...",
                "heuristic_label": sys_label,
                "match_type": "same_system_mismatch",
            })

        diff_sys_candidates = [p for p in corpus if p["system"] != qa_system]
        if diff_sys_candidates:
            chosen = rng.choice(diff_sys_candidates)
            overlap = len(combined_terms & term_set(chosen["text"]))
            diff_label = "AMBIGUOUS" if overlap >= 5 else "INCORRECT"
            labels.append({
                "query": qa["question"],
                "chunk_text": chosen["text"][:300] + "...",
                "heuristic_label": diff_label,
                "match_type": "diff_system_mismatch",
            })

    label_counts = {
        k: sum(1 for item in labels if item["heuristic_label"] == k)
        for k in ["CORRECT", "AMBIGUOUS", "INCORRECT"]
    }
    print(f"[INFO] Generated {len(labels)} grader labels via keyword-overlap heuristic.")
    print(f"[INFO] Label breakdown: {label_counts}")
    print("[ACTION] Spot-check 10-15 random entries manually to confirm heuristic accuracy "
          "before trusting Phase 3 results — these are heuristic, NOT hand-labeled.")

    return labels


def main():
    print("[INFO] Loading extracted PDF corpus...")
    corpus = load_corpus()
    if not corpus:
        print("[ERROR] Corpus loading failed or corpus is empty. Exiting.")
        return

    print("[INFO] Building evaluation questions...")
    qa_list = build_eval_questions(corpus)
    print(f"[INFO] Generated {len(qa_list)} total questions (after deduplication).")

    question_texts = [qa["question"] for qa in qa_list]
    if len(question_texts) != len(set(question_texts)):
        print("[ERROR] Duplicate question text detected after generation — this should not happen. "
              "Aborting write to avoid shipping a bad eval set.")
        dupes = {q for q in question_texts if question_texts.count(q) > 1}
        print(f"[ERROR] Duplicate questions: {dupes}")
        return

    dev_split_idx = int(len(qa_list) * DEV_SPLIT_RATIO)
    dev_set = qa_list[:dev_split_idx]
    test_set = qa_list[dev_split_idx:]

    for idx, item in enumerate(dev_set):
        item["id"] = f"dev_{idx:03d}"
    for idx, item in enumerate(test_set):
        item["id"] = f"test_{idx:03d}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEV_SET_PATH.write_text(json.dumps(dev_set, indent=2), encoding="utf-8")
    TEST_SET_PATH.write_text(json.dumps(test_set, indent=2), encoding="utf-8")

    print(f"[INFO] Saved {len(dev_set)} items to {DEV_SET_PATH}")
    print(f"[INFO] Saved {len(test_set)} items to {TEST_SET_PATH}")

    type_counts: Dict[str, int] = {}
    for d in dev_set:
        type_counts[d["question_type"]] = type_counts.get(d["question_type"], 0) + 1
    print(f"[INFO] Dev set type distribution: {type_counts}")

    print("[INFO] Generating heuristic grader labels...")
    grader_labels = generate_grader_labels(qa_list, corpus)
    GRADER_LABELS_PATH.write_text(json.dumps(grader_labels, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {len(grader_labels)} grader labels to {GRADER_LABELS_PATH}")

    print("[INFO] Phase 0 eval set generation complete.")


if __name__ == "__main__":
    main()