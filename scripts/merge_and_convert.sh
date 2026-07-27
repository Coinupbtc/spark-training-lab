#!/usr/bin/env bash
# Phase-D step 1+2: merge LoRA adapter into the BF16 base, convert to GGUF.
# Run AFTER training completes (needs ~70G RAM free — check free -g first).
# Output: models/student-v1/ (merged HF) -> student-v1-bf16.gguf -> quantize next.
set -euo pipefail
LAB="$HOME/Documents/projects/spark-training-lab"
MERGED="$LAB/models/student-v1"
GGUF_OUT="$LAB/models/student-v1-bf16.gguf"
LLAMA="$HOME/llama.cpp-master"

mkdir -p "$LAB/models"
echo "[1/2] merging adapter into base (CPU, ~70G RAM)…"
"$LAB/.venv/bin/python" - <<'EOF'
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

LAB = Path.home() / "Documents/projects/spark-training-lab"
BASE = "Qwen/Qwen3.6-35B-A3B"
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, LAB / "adapters/student-extraction-v1")
model = model.merge_and_unload()
out = LAB / "models/student-v1"
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
print("merged ->", out)
EOF

echo "[2/2] converting to GGUF…"
"$LAB/.venv/bin/python" "$LLAMA/convert_hf_to_gguf.py" "$MERGED" --outfile "$GGUF_OUT" --outtype bf16
ls -lh "$GGUF_OUT"
echo "DONE. Next: $LLAMA/build/bin/llama-quantize for NVFP4, then bench_v5."
