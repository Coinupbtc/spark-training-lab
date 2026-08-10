#!/usr/bin/env python3
"""Factory v2: training pairs with code-computed gold (no LLM grading needed).

Type B — deal math: name/price/fmv -> {discount_pct, under_fmv, margin_usd},
verified against courtyard's own discount field where present.
Type C — clean query: raw beezie title -> clean_query (their hand label is gold).

Output: datasets/v2_deterministic.jsonl
"""
import json, random
from pathlib import Path

LAB = Path.home() / "Documents/projects/spark-training-lab"
ARB = Path.home() / "Documents/projects/pokemon-arb/data"
out = []

items = json.load(open(ARB / "courtyard_latest.json"))
for it in items:
    price, fmv = it.get("price"), it.get("fmv")
    if not price or not fmv or fmv <= 0:
        continue
    disc = round((fmv - price) / fmv * 100, 1)
    gold = {"discount_pct": disc, "under_fmv": price < fmv, "margin_usd": round(fmv - price, 2)}
    # sanity: courtyard's own "N% off" string must agree (±1%) when present
    ds = it.get("discount") or ""
    if ds.endswith("% off"):
        try:
            if abs(float(ds[:-5]) - disc) > 1.0:
                continue
        except ValueError:
            pass
    out.append({"messages": [
        {"role": "user", "content":
         f"A graded card is listed for ${price:.2f} with fair market value ${fmv:.2f}.\n"
         "Return ONLY JSON with keys discount_pct (percent below FMV, 1 decimal), "
         "under_fmv (bool), margin_usd (2 decimals)."},
        {"role": "assistant", "content": json.dumps(gold)}]})

for it in json.load(open(ARB / "beezie_latest.json")):
    raw, clean = it.get("raw_title"), it.get("clean_query")
    if raw and clean:
        out.append({"messages": [
            {"role": "user", "content":
             "Convert this marketplace listing title into a clean price-search query "
             f"(drop year, grade, grader; keep set, card name, number).\nTitle: {raw}\n"
             "Return ONLY the query string."},
            {"role": "assistant", "content": clean}]})

random.seed(17)
random.shuffle(out)
p = LAB / "datasets/v2_deterministic.jsonl"
with open(p, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"v2 deterministic: {len(out)} pairs -> {p}")
