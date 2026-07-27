# First LoRA plan

## Goal
Teach a mid-size model one measurable skill (example: answer in Adam's short ops style, or format tool calls).

## Data schema (JSONL)
```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Success criteria
- 20-prompt eval set written **before** train
- After LoRA: ≥15/20 preferred vs base (blind or rubric)
- Adapter size < 500MB; load time < 30s

## Commands (placeholder — fill after base model chosen)
```bash
# On spark2 only
cd ~/Documents/projects/spark-training-lab
# python scripts/train_lora.py --base ... --data datasets/train.jsonl --out adapters/exp001
```

## When NOT to train
- You only need a better general model → run DSpark / MiniMax instead
- Dataset < 50 examples of real quality
