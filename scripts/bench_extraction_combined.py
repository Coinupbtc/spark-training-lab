#!/usr/bin/env python3
"""Head-to-head: Zwell-35B-v1 (:8890) vs MiaAI v8 (:8889) on the 104 held-out
extraction tasks (never seen by Zwell in training; gold = teacher consensus).

Scores per-field normalized exact match across the 8 schema fields.
"""
import json, re, time, urllib.request
from pathlib import Path
from datasets import Dataset

LAB = Path.home() / "Documents/projects/spark-training-lab"
FIELDS = ["year", "set", "card_name", "card_number", "edition", "holo", "grading_org", "grade"]

rows = [json.loads(l) for l in open(LAB / "datasets/pilot_extraction.jsonl") if l.strip()]
split = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=17)
evalset = split["test"]
print(f"eval tasks: {len(evalset)}")

MODELS = {
    
    "zwell-combined": ("http://127.0.0.1:8890/v1/chat/completions", "zwell-35b-combined"),
}

def norm(v):
    if isinstance(v, str):
        v = v.strip().lower().replace("pokémon", "pokemon")
        return v or None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v

def ask(url, model, prompt):
    body = json.dumps({"model": model, "max_tokens": 500, "temperature": 0.0,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        content = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0)) if m else None

scores = {name: {"fields_ok": 0, "perfect": 0, "fail": 0, "n": 0, "secs": 0.0} for name in MODELS}
for i, row in enumerate(evalset):
    prompt = row["messages"][0]["content"]
    gold = json.loads(row["messages"][1]["content"])
    for name, (url, model) in MODELS.items():
        s = scores[name]
        t0 = time.time()
        try:
            out = ask(url, model, prompt) or {}
        except Exception:
            out = {}
        s["secs"] += time.time() - t0
        if not out:
            s["fail"] += 1; s["n"] += 1; continue
        ok = sum(1 for f in FIELDS if norm(out.get(f)) == norm(gold.get(f)))
        s["fields_ok"] += ok
        s["perfect"] += (ok == len(FIELDS))
        s["n"] += 1
    if (i + 1) % 20 == 0:
        print(f"  {i+1} tasks done", flush=True)

print("\n=== EXTRACTION BENCH (higher is better) ===")
for name, s in scores.items():
    n = s["n"] or 1
    print(f"{name}: field-acc {s['fields_ok']/(n*len(FIELDS))*100:.1f}% | "
          f"perfect {s['perfect']}/{n} ({s['perfect']/n*100:.0f}%) | "
          f"parse-fail {s['fail']} | avg {s['secs']/n:.1f}s/task")
