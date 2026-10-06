"""Prints ground truth, answer snippet and scores per variant for chosen questions. No API calls."""
import json
import sys

IDS = sys.argv[1:] or ["test_002", "test_007", "test_010", "test_012", "test_016"]
qs = {x["id"]: x for x in json.load(open("data/eval_set_test_v2.json", encoding="utf-8"))}
raw = json.load(open("results/final_eval_raw_v2.json", encoding="utf-8"))

for i in IDS:
    print("=" * 70)
    print(i, qs[i]["question_type"])
    print("Q :", qs[i]["question"])
    print("GT:", qs[i]["ground_truth"])
    for v in ("naive", "hybrid", "full_crag"):
        rec = raw["results"][v].get(i, {})
        sc = raw["ragas"].get(v, {}).get(i, {})
        ans = (rec.get("answer") or "").replace("\n", " ")
        print(f"  [{v}] corr={sc.get('answer_correctness')} faith={sc.get('faithfulness')} web={rec.get('web_used')}")
        print(f"     {ans[:300]}")