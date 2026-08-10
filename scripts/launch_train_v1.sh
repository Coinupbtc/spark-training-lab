#!/usr/bin/env bash
# Detached launcher for the Phase-C QLoRA run: survives harness process reaping,
# logs to runs/train_v1.log, alerts Telegram on finish or failure.
set -uo pipefail
LAB="$HOME/Documents/projects/spark-training-lab"
ALERT="$HOME/.hermes/scripts/alertbot-send.sh"
cd "$LAB"
if .venv/bin/python scripts/train_qlora_35b.py > runs/train_v1.log 2>&1; then
  "$ALERT" "Custom-model build: 35B QLoRA v1 DONE — $(tail -1 runs/train_v1.log | head -c 200)"
else
  "$ALERT" "Custom-model build: 35B QLoRA v1 FAILED (exit $?) — see spark-training-lab/runs/train_v1.log"
fi
