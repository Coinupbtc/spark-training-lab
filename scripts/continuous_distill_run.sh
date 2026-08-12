#!/usr/bin/env bash
# continuous_distill_run.sh — Option B: Puzzle teacher → collect → (optional) QLoRA → propose.
# NEVER wires Hermes / :8889. Human must approve any later stage.
#
# Modes:
#   smoke  — collect 3 traces + 0.6B LoRA canary on node1 (no Puzzle pause)
#   collect — teacher traces only + Telegram proposal
#   full   — collect + pause Puzzle + QLoRA 35B on spark2 + restore Puzzle + propose
#
# Usage:
#   DISTILL_MODE=smoke bash .../continuous_distill_run.sh
#   DISTILL_MODE=full  bash .../continuous_distill_run.sh
set -euo pipefail

HOME_DIR="${HOME}"
LAB="${HOME_DIR}/Documents/projects/spark-training-lab"
SCRIPTS="${HOME_DIR}/.hermes/scripts"
ALERT="${SCRIPTS}/alertbot-send.sh"
STATE="${HOME_DIR}/.local/state/hermes/continuous-distill"
LOG_DIR="${LAB}/runs/continuous-distill"
NODE2_LANE="${HOME_DIR}/scripts/dgx/node2-deep-lane.sh"
PY_LAB="${LAB}/.venv/bin/python"
COLLECT="${LAB}/scripts/collect_puzzle_traces.py"
TRAIN35="${LAB}/scripts/train_qlora_35b.py"
SMOKE_TRAIN="${LAB}/scripts/smoke_lora.py"
PACK="${LAB}/datasets/distill_task_pack_v1.jsonl"
# Preferred local student base (already merged once on node1)
BASE_LOCAL="${LAB}/models/zwell-35b-combined-v1"
SPARK2="${SPARK2_HOST:-spark2}"

MODE="${DISTILL_MODE:-smoke}"
RUN_ID="${DISTILL_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
LIMIT_SMOKE="${DISTILL_SMOKE_LIMIT:-3}"
LIMIT_FULL="${DISTILL_FULL_LIMIT:-0}"   # 0 = entire pack
MIN_KEEP="${DISTILL_MIN_KEEP:-2}"
PAUSE_PUZZLE="${DISTILL_PAUSE_PUZZLE:-1}"
RESTORE_KEY="${DISTILL_RESTORE_KEY:-puzzle}"

mkdir -p "$STATE" "$LOG_DIR" "${LAB}/datasets/continuous" "${LAB}/adapters" "${LAB}/proposals"
LOCK="${STATE}/run.lock"
exec 200>"$LOCK"
flock -n 200 || { echo "continuous-distill already running"; exit 0; }

LOG="${LOG_DIR}/${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

alert() { [[ -x "$ALERT" ]] && bash "$ALERT" --plain "$*" >/dev/null 2>&1 || true; }

fail() {
  local msg="❌ continuous-distill ${RUN_ID} (${MODE}) FAILED: $*"
  echo "$msg"
  alert "$msg — log: ${LOG}"
  exit 1
}

on_exit() {
  local rc=$?
  # Best-effort Puzzle restore if we paused it
  if [[ "${PUZZLE_PAUSED:-0}" == "1" ]]; then
    echo "=== restore deep lane (${RESTORE_KEY}) ==="
    bash "$NODE2_LANE" start "$RESTORE_KEY" || alert "⚠️ continuous-distill ${RUN_ID}: Puzzle restore FAILED — check node2-deep-lane"
    PUZZLE_PAUSED=0
  fi
  if [[ $rc -ne 0 && "${ALREADY_ALERTED:-0}" != "1" ]]; then
    alert "❌ continuous-distill ${RUN_ID} exited rc=${rc} — see ${LOG}"
  fi
}
trap on_exit EXIT

echo "=== continuous-distill ${RUN_ID} mode=${MODE} $(date -Is) ==="
[[ -x "$PY_LAB" ]] || fail "lab venv missing: $PY_LAB"
[[ -f "$COLLECT" ]] || fail "collect script missing"
[[ -f "$PACK" ]] || fail "task pack missing: $PACK"

# --- admit (node1) before any heavy work ---
if [[ -x "${SCRIPTS}/heavy-job-admit.sh" ]]; then
  bash "${SCRIPTS}/heavy-job-admit.sh" check --min-free-g 10 --max-swap-g 12 --label continuous-distill \
    || fail "heavy-job-admit refused on node1"
fi

OUT_JSONL="${LAB}/datasets/continuous/${RUN_ID}.jsonl"
PROPOSAL="${LAB}/proposals/${RUN_ID}.md"
ADAPTER_NAME="distill-${RUN_ID}"
ADAPTER_DIR="${LAB}/adapters/${ADAPTER_NAME}"

# ---------- COLLECT (always; teacher = Puzzle while up) ----------
collect_limit="$LIMIT_FULL"
[[ "$MODE" == "smoke" ]] && collect_limit="$LIMIT_SMOKE"
[[ "$MODE" == "collect" && "$LIMIT_FULL" -eq 0 ]] && collect_limit=0

# Teacher must answer /v1/models before we burn 24 collect attempts (2026-07-28:
# full cron failed with 24× Connection refused while Puzzle was briefly down).
TEACHER_CHAT_URL="${DISTILL_TEACHER_URL:-http://192.168.100.11:8100/v1/chat/completions}"
TEACHER_MODELS_URL="${TEACHER_CHAT_URL%/chat/completions}/models"
echo "=== teacher preflight ${TEACHER_MODELS_URL} ==="
if ! curl -sf --max-time 8 "$TEACHER_MODELS_URL" >/dev/null 2>&1; then
  fail "Puzzle teacher unreachable (${TEACHER_MODELS_URL}) — start node2-deep-lane before distill"
fi
echo "teacher: UP"

echo "=== collect (limit=${collect_limit}) ==="
export DISTILL_RUN_ID="$RUN_ID"
set +e
"$PY_LAB" "$COLLECT" --pack "$PACK" --out "$OUT_JSONL" --limit "$collect_limit"
collect_rc=$?
set -e
KEPT=$(python3 -c "import json;print(json.load(open('${OUT_JSONL%.jsonl}.report.json'))['kept'])" 2>/dev/null || echo 0)
echo "collect kept=${KEPT} rc=${collect_rc}"
[[ "$KEPT" -ge "$MIN_KEEP" ]] || fail "only ${KEPT} admitted traces (min ${MIN_KEEP})"

EVAL_BEFORE="n/a"
EVAL_AFTER="n/a"
TRAIN_STATUS="skipped"
TRAIN_HOST="none"

# ---------- TRAIN ----------
if [[ "$MODE" == "smoke" ]]; then
  echo "=== smoke train (Qwen3-0.6B canary on node1; does not pause Puzzle) ==="
  # Reuse pack-sized sample: smoke_lora reads sample_train; for canary we point it at our traces
  # by temporarily using a tiny dedicated canary that loads OUT_JSONL
  TRAIN_STATUS="smoke"
  TRAIN_HOST="$(hostname)"
  set +e
  DISTILL_SMOKE_DATA="$OUT_JSONL" DISTILL_SMOKE_OUT="$ADAPTER_DIR" \
    "$PY_LAB" "${LAB}/scripts/smoke_lora_distill.py"
  trc=$?
  set -e
  [[ $trc -eq 0 ]] || fail "smoke train exit ${trc}"
  if [[ -f "${ADAPTER_DIR}/smoke_metrics.json" ]]; then
    EVAL_BEFORE=$(python3 -c "import json;print(json.load(open('${ADAPTER_DIR}/smoke_metrics.json')).get('loss_first','n/a'))")
    EVAL_AFTER=$(python3 -c "import json;print(json.load(open('${ADAPTER_DIR}/smoke_metrics.json')).get('loss_last','n/a'))")
  fi

elif [[ "$MODE" == "full" ]]; then
  echo "=== full QLoRA on spark2 (pauses Puzzle) ==="
  [[ -d "$BASE_LOCAL" ]] || fail "local base missing on node1: $BASE_LOCAL (67G merged HF)"
  [[ "$PAUSE_PUZZLE" == "1" ]] || fail "DISTILL_PAUSE_PUZZLE must be 1 for full mode"

  # Ensure base on spark2 (one-time CX7/rsync if absent)
  if ! ssh -o BatchMode=yes "$SPARK2" "test -f \$HOME/Documents/projects/spark-training-lab/models/zwell-35b-combined-v1/config.json"; then
    echo "=== syncing 67G base to spark2 (first full run) ==="
    alert "⏳ continuous-distill ${RUN_ID}: syncing student base → spark2 (long)"
    ssh -o BatchMode=yes "$SPARK2" "mkdir -p \$HOME/Documents/projects/spark-training-lab/models"
    rsync -a --info=progress2 -e "ssh -o BatchMode=yes" \
      "$BASE_LOCAL/" "${SPARK2}:Documents/projects/spark-training-lab/models/zwell-35b-combined-v1/" \
      || fail "rsync base to spark2 failed"
  fi

  # Sync pack traces + train script env to spark2 lab tree
  rsync -a -e "ssh -o BatchMode=yes" "$OUT_JSONL" \
    "${SPARK2}:Documents/projects/spark-training-lab/datasets/continuous/" \
    || fail "rsync traces failed"
  rsync -a -e "ssh -o BatchMode=yes" "$TRAIN35" \
    "${SPARK2}:Documents/projects/spark-training-lab/scripts/train_qlora_35b.py"

  echo "=== pausing Puzzle deep lane ==="
  alert "⏸ continuous-distill ${RUN_ID}: pausing Puzzle for QLoRA — deep lane briefly offline"
  bash "$NODE2_LANE" stop || fail "could not stop deep lane"
  PUZZLE_PAUSED=1
  # Wait for RAM reclaim
  sleep 20
  ssh -o BatchMode=yes "$SPARK2" 'free -g | head -2'

  # Admit on spark2 after Puzzle down
  ssh -o BatchMode=yes "$SPARK2" \
    "bash \$HOME/.hermes/scripts/heavy-job-admit.sh check --min-free-g 40 --max-swap-g 8 --label continuous-distill-full" \
    || fail "spark2 admit refused after Puzzle stop"

  REMOTE_DATA_NAME="$(basename "$OUT_JSONL")"
  echo "=== train on spark2 adapter=${ADAPTER_NAME} ==="
  set +e
  # shellcheck disable=SC2087
  ssh -o BatchMode=yes "$SPARK2" bash -s <<EOF
set -euo pipefail
cd "\$HOME/Documents/projects/spark-training-lab"
mkdir -p runs
export BASE_MODEL="\$HOME/Documents/projects/spark-training-lab/models/zwell-35b-combined-v1"
export DATA_FILE="\$HOME/Documents/projects/spark-training-lab/datasets/continuous/${REMOTE_DATA_NAME}"
export ADAPTER_NAME="${ADAPTER_NAME}"
export DISTILL_EPOCHS="${DISTILL_EPOCHS:-2}"
export DISTILL_MAX_LEN="${DISTILL_MAX_LEN:-768}"
.venv/bin/python scripts/train_qlora_35b.py 2>&1 | tee "runs/${ADAPTER_NAME}-train.log"
EOF
  trc=$?
  set -e
  [[ $trc -eq 0 ]] || fail "full train exit ${trc}"

  # Pull adapter + last log line metrics
  mkdir -p "$ADAPTER_DIR"
  rsync -a -e "ssh -o BatchMode=yes" \
    "${SPARK2}:Documents/projects/spark-training-lab/adapters/${ADAPTER_NAME}/" \
    "${ADAPTER_DIR}/" || fail "rsync adapter back failed"

  EVAL_LINE=$(ssh -o BatchMode=yes "$SPARK2" \
    "grep 'PHASE C RESULT' \$HOME/Documents/projects/spark-training-lab/runs/${ADAPTER_NAME}-train.log | tail -1" \
    || true)
  if [[ -n "${EVAL_LINE}" ]]; then
    EVAL_BEFORE=$(echo "$EVAL_LINE" | sed -n 's/.*eval loss \([0-9.]*\) ->.*/\1/p')
    EVAL_AFTER=$(echo "$EVAL_LINE" | sed -n 's/.*-> \([0-9.]*\) in.*/\1/p')
  fi
  TRAIN_STATUS="full"
  TRAIN_HOST="$SPARK2"

  echo "=== restoring Puzzle ==="
  bash "$NODE2_LANE" start "$RESTORE_KEY" || fail "Puzzle restore failed"
  PUZZLE_PAUSED=0
  alert "▶️ continuous-distill ${RUN_ID}: Puzzle restore requested"

elif [[ "$MODE" == "collect" ]]; then
  TRAIN_STATUS="collect-only"
else
  fail "unknown DISTILL_MODE=${MODE} (smoke|collect|full)"
fi

# ---------- PROPOSE (never auto-wire) ----------
cat > "$PROPOSAL" <<EOF
# Continuous distill proposal — ${RUN_ID}

- **Mode:** ${MODE}
- **When:** $(date -Is)
- **Teacher:** Puzzle \`192.168.100.11:8100\` (\`Nemotron-75b-Puzzle\`)
- **Traces kept:** ${KEPT} → \`${OUT_JSONL}\`
- **Train:** ${TRAIN_STATUS} on ${TRAIN_HOST}
- **Adapter:** \`${ADAPTER_DIR}\`
- **Eval loss:** ${EVAL_BEFORE} → ${EVAL_AFTER}

## Gate

This run does **not** change Hermes or \`:8889\`. To stage later (manual):

\`\`\`bash
# Inspect adapter
ls -la ${ADAPTER_DIR}
# Optional: merge/GGUF using spark-training-lab recipes, then A/B on a non-Hermes port
\`\`\`

Approve by reply in Telegram, or leave it — next biweekly run will produce a new dated adapter.
EOF

SUMMARY="🧪 Distill proposal ${RUN_ID}
mode=${MODE} kept=${KEPT} train=${TRAIN_STATUS}
loss ${EVAL_BEFORE} → ${EVAL_AFTER}
adapter: ${ADAPTER_DIR}
proposal: ${PROPOSAL}
Hermes/:8889 NOT changed — approve to stage."

echo "$SUMMARY"
alert "$SUMMARY"
ALREADY_ALERTED=1

# State marker for Morning Brief / operators
python3 - <<PY
import json
from pathlib import Path
p = Path("${STATE}/latest.json")
p.write_text(json.dumps({
  "run_id": "${RUN_ID}",
  "mode": "${MODE}",
  "kept": int("${KEPT}"),
  "train": "${TRAIN_STATUS}",
  "adapter": "${ADAPTER_DIR}",
  "proposal": "${PROPOSAL}",
  "eval_before": "${EVAL_BEFORE}",
  "eval_after": "${EVAL_AFTER}",
  "finished": "$(date -Is)",
}, indent=2))
PY

echo "=== DONE ${RUN_ID} ==="
exit 0
