#!/usr/bin/env python3
"""Tiny LoRA canary for continuous-distill smoke mode.

Trains Qwen3-0.6B for a few steps on freshly collected Puzzle traces.
Writes adapters/<out>/ + smoke_metrics.json. Never touches live Hermes models.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

LAB = Path.home() / "Documents/projects/spark-training-lab"
DATA = Path(os.environ["DISTILL_SMOKE_DATA"])
OUT = Path(os.environ.get("DISTILL_SMOKE_OUT", LAB / "adapters/distill-smoke"))
BASE = os.environ.get("DISTILL_SMOKE_BASE", "Qwen/Qwen3-0.6B")
STEPS = int(os.environ.get("DISTILL_SMOKE_STEPS", "30"))

assert torch.cuda.is_available(), "CUDA required for smoke LoRA"
assert DATA.is_file(), f"missing data {DATA}"

rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
# Repeat so 30 steps have enough batches on tiny packs
rows = (rows * max(1, 40 // max(len(rows), 1)))[:120]

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
model = get_peft_model(
    model,
    LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
)


def fmt(r):
    text = tok.apply_chat_template(r["messages"], tokenize=False)
    ids = tok(text, truncation=True, max_length=512)
    ids["labels"] = ids["input_ids"].copy()
    return ids


cols = list(rows[0].keys())
ds = Dataset.from_list(rows).map(fmt, remove_columns=cols)

class LossTracker(Trainer):
    losses: list[float] = []

    def log(self, logs, *a, **k):
        if "loss" in logs:
            self.losses.append(float(logs["loss"]))
        super().log(logs, *a, **k)


OUT.mkdir(parents=True, exist_ok=True)
args = TrainingArguments(
    output_dir=str(LAB / "runs" / OUT.name),
    max_steps=STEPS,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    logging_steps=5,
    bf16=True,
    report_to=[],
    save_strategy="no",
)

t0 = time.time()
trainer = LossTracker(
    model=model, args=args, train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tok, padding=True),
)
trainer.train()
model.save_pretrained(OUT)
first = trainer.losses[0] if trainer.losses else None
last = trainer.losses[-1] if trainer.losses else None
metrics = {
    "base": BASE,
    "steps": STEPS,
    "seconds": round(time.time() - t0, 1),
    "loss_first": first,
    "loss_last": last,
    "n_rows_source": len([json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]),
}
(OUT / "smoke_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"SMOKE DISTILL: loss {first} -> {last} in {metrics['seconds']}s | {OUT}", flush=True)
sys.exit(0 if (first is not None and last is not None and last <= first + 0.5) else 1)
