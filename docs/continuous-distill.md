# Continuous distill (Option B) — 2026-07-27

**What:** Puzzle teacher → collect traces → optional QLoRA → Telegram **proposal**. Never auto-wires Hermes/`:8889`.

**Where:** `~/Documents/projects/spark-training-lab/scripts/continuous_distill_run.sh`

## Modes

| Mode | Collect | Train | Pauses Puzzle? |
|------|---------|-------|----------------|
| `smoke` | 3 traces | Qwen3-0.6B canary on node1 | No |
| `collect` | full pack | none | No |
| `full` | full pack | 35B QLoRA on spark2 from `models/zwell-35b-combined-v1` | **Yes** |

```bash
DISTILL_MODE=smoke bash ~/Documents/projects/spark-training-lab/scripts/continuous_distill_run.sh
DISTILL_MODE=collect bash ~/Documents/projects/spark-training-lab/scripts/continuous_distill_run.sh
DISTILL_MODE=full bash ~/Documents/projects/spark-training-lab/scripts/continuous_distill_run.sh
```

## Schedule

OS crontab on node1 (**Sun + Tue + Fri 02:15**): **`DISTILL_MODE=full`**.

While you’re asleep it: collect from Puzzle → pause Puzzle → QLoRA on spark2 → restore Puzzle → Telegram proposal. Hermes/`:8889` still never auto-wired.

First full run may rsync ~67G student base to spark2 (one-time). Deep lane is offline for the train window only.

Outputs: `datasets/continuous/`, `adapters/distill-*`, `proposals/*.md`, state `~/.local/state/hermes/continuous-distill/latest.json`.

## Rules

- Failures → `alertbot-send.sh`
- `heavy-job-admit.sh` before heavy work
- Full mode restores Puzzle in an EXIT trap
- Human approve required before any Hermes wire
