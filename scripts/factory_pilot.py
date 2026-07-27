#!/usr/bin/env python3
"""Phase-B pilot: distillation data factory, task type A (listing title -> JSON).

Pipeline: real listing titles (set ARB_DATA) -> two teachers (two local OpenAI-compatible teachers) each extract structured JSON -> field-level agreement
grading -> consensus winners written as training JSONL.

Usage: factory_pilot.py [N_TASKS]   (default 20)
Output: datasets/pilot_extraction.jsonl + a grading report on stdout.
"""
import json, random, re, sys, time, urllib.request
import os
from pathlib import Path

LAB = Path.home() / "Documents/projects/spark-training-lab"
ARB = Path(os.environ.get("ARB_DATA", str(Path.home() / "data" / "listings")))
N = 20 if not sys.argv[1:] or sys.argv[1] == "all" else int(sys.argv[1])

TEACHERS = {
    "mimo":   ("http://NODE2_HOST:8100/v1/chat/completions", "xiaomi/mimo-v2.5-pro", 800),
    "qwen35": ("http://127.0.0.1:8889/v1/chat/completions", "qwen35b-miaai", 800),
}

SCHEMA_FIELDS = ["year", "set", "card_name", "card_number", "edition", "holo", "grading_org", "grade"]
PROMPT = """Extract structured data from this trading-card listing title.
Return ONLY a JSON object with exactly these keys:
year (int or null), set (string or null), card_name (string), card_number (string or null),
edition (string or null, e.g. "1st Edition"), holo (true/false/null),
grading_org (string or null, e.g. "PSA"), grade (number or null).

Title: {title}"""


def ask(url, model, max_tok, title):
    payload = {
        "model": model, "max_tokens": max_tok, "temperature": 0.1,
        "messages": [{"role": "user", "content": PROMPT.format(title=title)}],
    }
    # thinking off for both teachers: 35B returns empty content otherwise, and MiMo
    # drops 45s -> 2.3s per task; cross-teacher agreement is the quality gate here
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        content = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0)) if m else None


def norm(v):
    if isinstance(v, str):
        v = v.strip().lower().replace("pokémon", "pokemon")
        return v or None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def agreement(a, b):
    return sum(1 for f in SCHEMA_FIELDS if norm(a.get(f)) == norm(b.get(f)))


titles = []
for f, key in [("courtyard_latest.json", "name"), ("beezie_latest.json", "raw_title")]:
    titles += [it[key] for it in json.load(open(ARB / f)) if it.get(key)]
titles = list(dict.fromkeys(titles))  # dedupe, keep order
random.seed(17)
random.shuffle(titles)
tasks = titles if sys.argv[1:2] == ["all"] else titles[:N]

winners, report = [], {"ok": 0, "disagree": 0, "teacher_fail": 0}
t0 = time.time()
for i, title in enumerate(tasks):
    outs = {}
    for name, (url, model, mt) in TEACHERS.items():
        try:
            outs[name] = ask(url, model, mt, title)
        except Exception as e:
            print(f"[{i}] {name} ERROR: {e}")
    if len([o for o in outs.values() if o]) < 2:
        report["teacher_fail"] += 1
        continue
    agree = agreement(outs["mimo"], outs["qwen35"])
    if agree >= 6:
        gold = {f: outs["mimo"].get(f) for f in SCHEMA_FIELDS}
        winners.append({"messages": [
            {"role": "user", "content": PROMPT.format(title=title)},
            {"role": "assistant", "content": json.dumps(gold)}]})
        report["ok"] += 1
    else:
        report["disagree"] += 1
        diffs = [f for f in SCHEMA_FIELDS if norm(outs['mimo'].get(f)) != norm(outs['qwen35'].get(f))]
        print(f"[{i}] disagreement ({agree}/8) on {diffs}: {title[:70]}")

out = LAB / "datasets/pilot_extraction.jsonl"
with open(out, "w") as f:
    for w in winners:
        f.write(json.dumps(w) + "\n")

print(f"\nFACTORY PILOT: {N} tasks in {time.time()-t0:.0f}s -> {report} -> {len(winners)} pairs at {out}")
