#!/usr/bin/env python3
"""Write a tiny starter JSONL so the lab is non-empty."""
from pathlib import Path
import json

out = Path(__file__).resolve().parents[1] / "datasets" / "sample_train.jsonl"
rows = [
    {
        "messages": [
            {"role": "user", "content": "Summarize this host status in 3 bullets: inference up, 35B on :8889, dual-node fabric linked."},
            {"role": "assistant", "content": "- Agent gateways healthy\n- Default text model on localhost:8889\n- Dual Spark fabric linked"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is LoRA fine-tuning in one sentence?"},
            {"role": "assistant", "content": "LoRA freezes the base model and trains small adapter matrices so you specialize behavior without rewriting all weights."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Should I full-finetune DeepSeek-V4-Flash on two Sparks?"},
            {"role": "assistant", "content": "No — serve it with the DSpark TP=2 recipe; use LoRA only on smaller models for personal data."},
        ]
    },
]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {out} n={len(rows)}")
