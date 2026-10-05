"""
Builds data/eval_set_test_v2.json: hand-written questions with machine-verified ground truth.

Rules enforced mechanically:
  - every ground truth has verbatim evidence that must exist in the stated source page
    (evidence may be several short fragments joined by '|'; EVERY fragment must be on that page)
  - source_doc / source_page are DERIVED from the evidence (never typed by hand)
  - every number in a ground truth must appear in that question's evidence text
  - the 5 clean v1 rows are copied unchanged (ambiguous rows: expected_fallback -> None)

Source text: results/source_pages_dump.txt (pdfplumber, same extractor as ingestion).
Note: the PDFs are two-column, so pdfplumber can interleave right-column text into the middle
of left-column sentences. Evidence fragments are therefore kept short and single-line.
"""
import json
import re
import sys
from pathlib import Path

DUMP = Path("results") / "source_pages_dump.txt"
V1 = Path("data") / "eval_set_test.json"
OUT = Path("data") / "eval_set_test_v2.json"
KEEP_FROM_V1 = ["test_001", "test_003", "test_006", "test_013", "test_017"]

# (doc, page, verbatim snippet; use '|' to separate fragments that must all appear on the page)
QS = [
    dict(id="test_000", type="lookup",
         q="What flow rate and pressure do the compressed air applications at the Techni-Cast foundry in Southgate, California require to operate reliably?",
         gt="Between 200 and 400 standard cubic feet per minute (scfm) of compressed air at 90 psig.",
         ev=[("metal_cs_technicast.pdf", 1, "between 200 and 400 standard cubic feet per minute (scfm) of compressed air at 90 pounds per square inch gauge (psig)")]),
    dict(id="test_002", type="lookup",
         q="In the automobile assembly plant example in Compressed Air Tip Sheet #2, what annual savings were calculated from eliminating the inappropriate compressed air uses, before subtracting the electricity costs of the replacement equipment?",
         gt="$102,600 per year.",
         ev=[("compressed_air2.pdf", 2, "= $102,600")]),
    dict(id="test_004", type="lookup",
         q="In Steam Tip Sheet #21's example, what steam flow is needed for a 300-hp steam turbine with a steam rate of 26 lb/hp-hr to replace a fully loaded 300-hp feedwater pump drive motor?",
         gt="7,800 lb/hr (26 lb/hp-hr x 300 hp).",
         ev=[("steam21_rotating_equip.pdf", 1, "Steam Flow = 26 lb/hp-hr x 300 hp"),
             ("steam21_rotating_equip.pdf", 1, "= 7,800 lb/hr")]),
    dict(id="test_007", type="lookup",
         q="In the Market Assessment overview, what range of pump-system energy savings (as a percent of system energy) is listed for increasing piping diameter to reduce friction?",
         gt="5%-20% of system energy.",
         ev=[("market_assessment_glimpse.pdf", 3, "Increase piping diameter to reduce friction. 5%-20%")]),
    dict(id="test_010", type="lookup",
         q="Before the project, what were the size and rated airflow of the baghouse fan system at Ash Grove Cement's Durkee, Oregon plant?",
         gt="A 125-hp centrifugal fan system rated for 30,000 acfm, belt-driven through sheaves.",
         ev=[("mi_cs_ashgrove.pdf", 2, "125-hp centrifugal fan system, rated for 30,000 atmospheric cubic feet per minute (acfm)")]),
    dict(id="test_012", type="lookup",
         q="In the FUJIFILM Hunt Chemicals compressed air case study, what cost reduction did the facility's data show for a 2 PSIG drop in pressure?",
         gt="A 2 PSIG drop in pressure resulted in a 1% reduction in cost.",
         ev=[("fujifilm_case_study.pdf", 2, "2 PSIG|1% reduction")]),
    dict(id="test_014", type="lookup",
         q="In Pumping Systems Tip Sheet #6's example of replacing one oversized 3,500-gpm chilled water pump with a 1,250-gpm pump, what annual energy and cost savings result?",
         gt="About 790,520 kWh per year, or about $39,525 per year at 5 cents per kWh.",
         ev=[("match_pumps_to_system.pdf", 2, "= 790,520 kWh/year"),
             ("match_pumps_to_system.pdf", 2, "At an average energy cost of 5 cents per kWh, annual savings would be about $39,525")]),
    dict(id="test_015", type="lookup",
         q="In the bottling plant case study in Compressed Air Tip Sheet #11, how much lower were the energy costs of the central vacuum system than those of the existing venturi vacuum devices?",
         gt="30% lower.",
         ev=[("compressed_air11.pdf", 2, "energy costs that were 30% lower than that of the venturi devices")]),
    dict(id="test_016", type="lookup",
         q="In Pumping Systems Tip Sheet #4's example, a 300-hp centrifugal pump operates at 55% efficiency but should operate at 78%. What annual energy savings result from restoring design efficiency?",
         gt="415,769 kWh per year (about $20,786 per year at 5 cents per kWh).",
         ev=[("test_pumping_system__pumping_systemts4.pdf", 2, "= 415,769 kWh/year"),
             ("test_pumping_system__pumping_systemts4.pdf", 2, "At an energy cost of 5 cents per kWh, the estimated savings would be $20,786 per year")]),
    dict(id="test_005", type="synthesis",
         q="What share of total industrial energy use do steam systems account for, compared with motor-driven equipment such as pumps, air compressors, and fans?",
         gt="Steam systems account for about 30% of total industrial energy use, versus about 16% for motor-driven equipment (pumps, air compressors, fans).",
         ev=[("39156.pdf", 1, "Steam systems account for about 30% of the total energy used in industrial"),
             ("39157.pdf", 1, "about 16% of all the energy used in U.S. industrial applications")]),
    dict(id="test_008", type="synthesis",
         q="Which project achieved larger annual maintenance savings, the Ash Grove Cement fan ASD retrofit or the Techni-Cast compressed air retrofit, and what were the amounts?",
         gt="Ash Grove saved $10,000 annually in maintenance versus $2,200 annually at Techni-Cast, so Ash Grove's savings were larger.",
         ev=[("mi_cs_ashgrove.pdf", 3, "achieved annual maintenance savings of $10,000"),
             ("metal_cs_technicast.pdf", 4, "$2,200 annually because the new compressor")]),
    dict(id="test_009", type="synthesis",
         q="What share of industrial electricity consumption do DOE's market assessment overview and the full motor-driven systems market assessment paper attribute to motor-driven systems?",
         gt="The overview says all types of motor systems account for 69% of industrial electricity consumption; the full paper says nearly 70%.",
         ev=[("market_assessment_glimpse.pdf", 1, "all types of motor systems account for 69% of all industrial electricity consumption"),
             ("us_industrial_motor_driven.pdf", 1, "nearly 70% of all electricity used in industry is consumed by some type of motor-driven system")]),
    dict(id="test_011", type="synthesis",
         q="To what pressure did FUJIFILM Hunt Chemicals set its compressors, and how did Techni-Cast change its compressor discharge pressure, after their projects?",
         gt="FUJIFILM reduced compressor set pressure to 105 PSIG; Techni-Cast reduced discharge pressure from 120 psig to 100 psig.",
         ev=[("fujifilm_case_study.pdf", 2, "reduced compressor set pressure to 105 PSIG"),
             ("metal_cs_technicast.pdf", 3, "reduced from 120 psig to 100 psig")]),
]


def norm(s: str) -> str:
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "-")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_pages():
    if not DUMP.exists():
        sys.exit(f"[ERROR] {DUMP} missing. Run scripts.dump_source_pages first.")
    text = DUMP.read_text(encoding="utf-8")
    parts = re.split(r"^===== (.+?) \| PAGE (\d+) =====$", text, flags=re.M)
    pages = {}
    for i in range(1, len(parts), 3):
        pages[(parts[i], int(parts[i + 1]))] = norm(parts[i + 2])
    return pages


def main():
    pages = load_pages()
    print(f"[INFO] Loaded {len(pages)} pages from dump (expected 41).")
    errors = []

    v1 = {q["id"]: q for q in json.loads(V1.read_text(encoding="utf-8"))}
    rows = []

    for qid in KEEP_FROM_V1:
        r = dict(v1[qid])
        assert r["question_type"] in ("ambiguous", "unanswerable_locally"), qid
        if r["question_type"] == "ambiguous":
            r["expected_fallback"] = None  # redefined: not scored for web trigger
        rows.append(r)

    for item in QS:
        qid = item["id"]
        ev_text = ""
        pairs = []
        for doc, page, snip in item["ev"]:
            page_text = pages.get((doc, page))
            if page_text is None:
                errors.append(f"{qid}: no such page {doc} p.{page}")
                continue
            frags = [f.strip() for f in norm(snip).split("|") if f.strip()]
            missing = [f for f in frags if f not in page_text]
            if missing:
                errors.append(f"{qid}: evidence NOT FOUND in {doc} p.{page}: {missing!r}")
                continue
            others = [p for (d, p), t in pages.items()
                      if d == doc and p != page and all(f in t for f in frags)]
            if others:
                print(f"[NOTE] {qid}: evidence also appears on {doc} pages {others}")
            ev_text += " " + " ".join(frags)
            if (doc, page) not in pairs:
                pairs.append((doc, page))

        for tok in re.findall(r"\d[\d,\.]*%?", item["gt"]):
            tok = tok.rstrip(",.")
            if tok.lower() not in ev_text:
                errors.append(f"{qid}: number {tok!r} in ground truth not found in evidence")

        if not item["q"].endswith("?"):
            errors.append(f"{qid}: question does not end with '?'")
        if item["type"] == "lookup" and len(pairs) != 1:
            errors.append(f"{qid}: lookup must have exactly one (doc,page), got {pairs}")
        if item["type"] == "synthesis" and len({d for d, _ in pairs}) != 2:
            errors.append(f"{qid}: synthesis must span exactly 2 docs, got {pairs}")
        if errors and any(e.startswith(qid) for e in errors):
            continue

        if item["type"] == "lookup":
            src_doc, src_page = pairs[0][0], pairs[0][1]
        else:
            src_doc = ", ".join(d for d, _ in pairs)
            src_page = ", ".join(f"p.{p}" for _, p in pairs)

        rows.append({
            "question": item["q"],
            "ground_truth": item["gt"],
            "source_doc": src_doc,
            "source_page": src_page,
            "question_type": item["type"],
            "expected_fallback": False,
            "id": qid,
        })

    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    if len(rows) != 18:
        errors.append(f"expected 18 rows, got {len(rows)}")

    if errors:
        print("\n[FAIL] Validation errors (nothing written):")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    rows.sort(key=lambda r: r["id"])
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Wrote {len(rows)} rows to {OUT}. All evidence verified against source pages.")
    for r in rows:
        print(f"  {r['id']} {r['question_type']:<21} {r['source_doc']} | {r['source_page']}")


if __name__ == "__main__":
    main()