#!/usr/bin/env python3
"""Merge a LoRA adapter into base shards WITHOUT peft (peft 0.19.1 is
incompatible with transformers 5.14's WeightConverter API on the load path).

Math: W_merged = W + (alpha/r) * B @ A, applied per attention projection.
Streams shard-by-shard (~2.6G peak RAM). Usage: manual_merge.py <adapter_dir> <out_dir>
"""
import json, shutil, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

LAB = Path.home() / "Documents/projects/spark-training-lab"
SNAP = next((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots").iterdir())
ADAPTER = Path(sys.argv[1]) if len(sys.argv) > 1 else LAB / "adapters/student-extraction-v1"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else LAB / "models/zwell-35b-extraction-v1"
OUT.mkdir(parents=True, exist_ok=True)

cfg = json.load(open(ADAPTER / "adapter_config.json"))
scale = cfg["lora_alpha"] / cfg["r"]
ad = load_file(ADAPTER / "adapter_model.safetensors")

# adapter key "base_model.model.model.layers.N...q_proj.lora_A.weight" -> base key
deltas = {}
for k in list(ad):
    if k.endswith("lora_A.weight"):
        stem = k[len("base_model.model."):-len(".lora_A.weight")]
        # runtime module tree says model.layers.*; the checkpoint stores
        # model.language_model.layers.* (transformers 5.x renames on load)
        stem = stem.replace("model.layers.", "model.language_model.layers.", 1)
        A, B = ad[k].float(), ad[k.replace("lora_A", "lora_B")].float()
        deltas[stem + ".weight"] = (B @ A) * scale
print(f"adapter: {len(deltas)} target weights, scale={scale}")

applied = 0
for shard in sorted(SNAP.glob("model-*.safetensors")):
    tensors = {}
    with safe_open(shard, framework="pt") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name in deltas:
                t = (t.float() + deltas[name]).to(torch.bfloat16)
                applied += 1
            tensors[name] = t
    save_file(tensors, OUT / shard.name, metadata={"format": "pt"})
    print(f"  {shard.name} done", flush=True)

for aux in SNAP.iterdir():
    if aux.suffix in (".json", ".txt", ".jinja") or aux.name in ("merges.txt",):
        shutil.copy2(aux, OUT / aux.name)

assert applied == len(deltas), f"applied {applied} != {len(deltas)} deltas"
print(f"MERGE DONE: {applied} weights patched -> {OUT}")
