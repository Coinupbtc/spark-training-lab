#!/usr/bin/env bash
# One-command orientation for spark-training-lab
set -euo pipefail
cd "$(dirname "$0")"

echo "==> spark-training-lab"
echo
# Configs + datasets + scripts only; trained LoRA files and metrics stay on the host.
echo "This repo ships datasets, scripts, and adapter configs — not trained LoRA weights, run metrics, or multi-GB base models."
echo
echo "Quick look (no GPU):"
echo "  ls adapters/"
echo "  ls datasets/"
echo "  head -n 3 datasets/*.jsonl 2>/dev/null || true"
echo
echo "Optional Python env for merge/eval helpers:"
if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  python3 -m venv .venv
  ./.venv/bin/pip -q install -U pip
  ./.venv/bin/pip -q install -r requirements.txt
  echo "  .venv ready"
else
  echo "  INSTALL_DEPS=1 ./setup.sh   # creates .venv + installs requirements.txt"
fi
echo
echo "Typical train path (after you place a base model locally):"
echo "  # see scripts/launch_train_v1.sh and docs/"
echo "  ls scripts/"
echo
echo "Merge helpers: scripts/manual_merge.py · scripts/merge_and_convert.sh"
