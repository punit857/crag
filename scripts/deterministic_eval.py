"""
Deterministic, zero-API-call evaluation of the saved Phase 5 results.
For lookup/synthesis questions it checks, per variant:
  ctx = key numbers from the ground truth appear in the retrieved context
  ans = key numbers appear in the answer
  ref = the answer is the pipeline's standard refusal
Key numbers = numeric tokens in the ground truth (single bare digits ignored; '1%' etc. kept).
This is a proxy: it shows the numbers are present, not that the sentence is right.
"""
import json
import re

QS_PATH = "data/eval_set_test_v2.json"
RAW_PATH = "results/final_eval_raw_v2.json"
VARIANTS = ("naive", "hybrid", "full_crag")
REFUSAL = "i cannot answer this based on the provided documents"


def norm(s):
    s = (s or "").lower().replace("\u2011", "-").replace("\u2013", "-")
    s = re.sub(r"\s+%", "%", s)
    return re.sub(r"\s+", " ", s)


def key_tokens(gt):
    toks = []
    for t in re.findall(r"\d[\d,\.]*%?", gt):
        t = t.rstrip(",.")
        digits = t.replace(",", "").replace(".", "").replace("%", "")
        if len(digits) >= 2 or t.endswith("%"):
            toks.append(t)
    return toks


qs = json.load(open(QS_PATH, encoding="utf-8"))
raw = json.load(open(RAW_PATH, encoding="utf-8"))["results"]


def flags(v, q, toks):
    rec = raw[v].get(q["id"])
    if not rec or rec.get("error"):
        return None
    ctx = norm(" ".join(d.get("text", "") for d in rec["docs"]))
    ans = norm(rec["answer"])
    return {
        "ctx": all(norm(t) in ctx for t in toks),
        "ans": all(norm(t) in ans for t in toks),
        "ref": REFUSAL in ans,
        "web": bool(rec.get("web_used")),
    }


factual = [q for q in qs if q["question_type"] in ("lookup", "synthesis")]
other = [q for q in qs if q["question_type"] not in ("lookup", "synthesis")]

print("FACTUAL QUESTIONS  (Y = yes, . = no)")
print(f"{'id':<9}{'type':<11}{'key numbers':<26}" + "".join(f"{v:<20}" for v in VARIANTS))
tot = {v: dict(n=0, ctx=0, ans=0, ref=0, gen_fail=0, ret_fail=0, odd=0) for v in VARIANTS}
for q in factual:
    toks = key_tokens(q["ground_truth"])
    cells = []
    for v in VARIANTS:
        f = flags(v, q, toks)
        if f is None:
            cells.append("n/a".ljust(20))
            continue
        t = tot[v]
        t["n"] += 1
        t["ctx"] += f["ctx"]
        t["ans"] += f["ans"]
        t["ref"] += f["ref"]
        t["gen_fail"] += int(f["ref"] and f["ctx"])
        t["ret_fail"] += int(f["ref"] and not f["ctx"])
        t["odd"] += int(f["ans"] and not f["ctx"])
        y = lambda b: "Y" if b else "."
        cells.append(f"ctx={y(f['ctx'])} ans={y(f['ans'])} ref={y(f['ref'])}".ljust(20))
    print(f"{q['id']:<9}{q['question_type']:<11}{', '.join(toks)[:24]:<26}" + "".join(cells))

print("\nSUMMARY (factual questions)")
for v in VARIANTS:
    t = tot[v]
    print(f"{v:<10} n={t['n']}  facts_in_context={t['ctx']}  facts_in_answer={t['ans']}  refused={t['ref']}  "
          f"refused_with_facts_in_context={t['gen_fail']}  refused_without={t['ret_fail']}  "
          f"answered_with_facts_not_in_context={t['odd']}")

print("\nREFUSAL-TYPE QUESTIONS (no score, just behaviour)")
for q in other:
    parts = []
    for v in VARIANTS:
        rec = raw[v].get(q["id"])
        if not rec or rec.get("error"):
            parts.append(f"{v}: n/a")
        else:
            parts.append(f"{v}: refused={REFUSAL in norm(rec['answer'])} web={bool(rec.get('web_used'))}")
    print(f"{q['id']} {q['question_type']:<21} " + " | ".join(parts))
    