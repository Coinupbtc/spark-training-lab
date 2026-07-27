#!/usr/bin/env python3
"""Phase-A feasibility smoke: prove the LoRA training loop runs on GB10.

Tiny base (Qwen3-0.6B), 30 steps on the lab sample dataset, bf16, rank-16 LoRA.
Pass criteria: runs to completion on CUDA, loss at end < loss at start,
adapter saved to adapters/smoke-qwen06b/. Nothing here touches live services.
"""
import json, time, sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

LAB = Path.home() / "Documents/projects/spark-training-lab"
BASE = "Qwen/Qwen3-0.6B"
OUT = LAB / "adapters/smoke-qwen06b"

assert torch.cuda.is_available(), "CUDA not available on GB10 — feasibility FAIL"
print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")

peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

rows = [json.loads(l) for l in open(LAB / "datasets/sample_train.jsonl") if l.strip()]
# replicate the tiny sample so 30 steps have data
rows = (rows * 40)[:120]

def fmt(r):
    msgs = r.get("messages") or [{"role": "user", "content": r.get("prompt", "")},
                                 {"role": "assistant", "content": r.get("response", r.get("completion", ""))}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    ids = tok(text, truncation=True, max_length=512)
    ids["labels"] = ids["input_ids"].copy()
    return ids

ds = Dataset.from_list(rows).map(fmt, remove_columns=list(rows[0].keys()))

args = TrainingArguments(
    output_dir=str(LAB / "runs/smoke-qwen06b"), max_steps=30, per_device_train_batch_size=4,
    learning_rate=2e-4, logging_steps=5, bf16=True, report_to=[], save_strategy="no",
)

class LossTracker(Trainer):
    losses = []
    def log(self, logs, *a, **k):
        if "loss" in logs: self.losses.append(logs["loss"])
        super().log(logs, *a, **k)

t0 = time.time()
trainer = LossTracker(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True))
trainer.train()
dt = time.time() - t0

model.save_pretrained(OUT)
first, last = trainer.losses[0], trainer.losses[-1]
print(f"\nSMOKE RESULT: 30 steps in {dt:.0f}s | loss {first:.3f} -> {last:.3f} | adapter -> {OUT}")
sys.exit(0 if last < first else 1)
