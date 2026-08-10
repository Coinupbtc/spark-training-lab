#!/usr/bin/env python3
"""Phase-C: QLoRA the Qwen3.6-35B-A3B student on factory consensus pairs.

4-bit NF4 base + rank-32 LoRA on attention projections, gradient checkpointing,
runs on node1 alongside live services (fits in ~35-40G; abort if RAM gets tight).
Output: adapters/student-extraction-v1/ + eval loss before/after.
"""
import json, os, time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorForSeq2Seq, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

LAB = Path.home() / "Documents/projects/spark-training-lab"
# Local merged HF (preferred) or Hub id — continuous-distill sets BASE_MODEL explicitly
BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3.6-35B-A3B")
OUT = LAB / "adapters" / os.environ.get("ADAPTER_NAME", "student-extraction-v1")
# DATA_FILE may be absolute or relative to datasets/
_data = os.environ.get("DATA_FILE", "pilot_extraction.jsonl")
DATA = Path(_data) if _data.startswith("/") else (LAB / "datasets" / _data)
RUN = os.environ.get("ADAPTER_NAME", "student-extraction-v1")
MAX_LEN = int(os.environ.get("DISTILL_MAX_LEN", "512"))
EPOCHS = float(os.environ.get("DISTILL_EPOCHS", "2"))

print(f"loading base {BASE} in 4-bit…", flush=True)
t0 = time.time()
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb_cfg,
                                             torch_dtype=torch.bfloat16, device_map="cuda")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
print(f"base loaded in {time.time()-t0:.0f}s", flush=True)

peft_cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

rows = [json.loads(l) for l in open(DATA) if l.strip()]

def fmt(r):
    text = tok.apply_chat_template(r["messages"], tokenize=False)
    ids = tok(text, truncation=True, max_length=MAX_LEN)
    ids["labels"] = ids["input_ids"].copy()
    return ids

# Drop non-messages columns (id/run_id/teacher) so map stays clean
_cols = list(rows[0].keys()) if rows else ["messages"]
ds = Dataset.from_list(rows).map(fmt, remove_columns=_cols).train_test_split(test_size=0.1, seed=17)
print(f"train {len(ds['train'])} eval {len(ds['test'])}", flush=True)

args = TrainingArguments(
    output_dir=str(LAB / "runs" / RUN),
    num_train_epochs=EPOCHS, per_device_train_batch_size=2, gradient_accumulation_steps=8,
    learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
    logging_steps=5, bf16=True, report_to=[], save_strategy="no",
    eval_strategy="steps", eval_steps=30, per_device_eval_batch_size=2,
)

trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True))

pre = trainer.evaluate()["eval_loss"]
print(f"eval loss BEFORE training: {pre:.4f}", flush=True)
t0 = time.time()
trainer.train()
post = trainer.evaluate()["eval_loss"]
model.save_pretrained(OUT)
print(f"\nPHASE C RESULT: eval loss {pre:.4f} -> {post:.4f} in {(time.time()-t0)/60:.0f}min | adapter -> {OUT}")
