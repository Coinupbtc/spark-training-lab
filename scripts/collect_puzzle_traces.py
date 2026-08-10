#!/usr/bin/env python3
"""Collect teacher traces from Puzzle (or any OpenAI-compatible teacher) into SFT JSONL.

Reads a prompt-only task pack (user messages), calls the teacher, writes
{"messages":[user, assistant], "id", "teacher", "run_id"} rows.

Quality gate: non-empty assistant content, min length, optional JSON parse for
extract-* ids. Does not train and does not touch Hermes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LAB = Path.home() / "Documents/projects/spark-training-lab"
DEFAULT_PACK = LAB / "datasets/distill_task_pack_v1.jsonl"
DEFAULT_TEACHER_URL = os.environ.get(
    "DISTILL_TEACHER_URL", "http://192.168.100.11:8100/v1/chat/completions"
)
DEFAULT_TEACHER_MODEL = os.environ.get("DISTILL_TEACHER_MODEL", "Nemotron-75b-Puzzle")


def ask(url: str, model: str, user_content: str, max_tokens: int, timeout: int) -> str:
    """One chat completion; thinking off when the server supports the Qwen-style flag."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": user_content}],
        # Puzzle/Nemotron may ignore this; harmless on OpenAI-compatible servers
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    msg = data["choices"][0]["message"]
    # Prefer final content; some reasoners put draft in reasoning_content
    content = (msg.get("content") or "").strip()
    if not content:
        content = (msg.get("reasoning_content") or "").strip()
    return content


def admit_row(task_id: str, user: str, assistant: str, min_chars: int) -> str | None:
    """Return None if ok, else a short reject reason."""
    if len(assistant) < min_chars:
        return f"too_short:{len(assistant)}"
    if task_id.startswith("extract-"):
        m = re.search(r"\{.*\}", assistant, re.S)
        if not m:
            return "extract_no_json"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return "extract_bad_json"
        if "card_name" not in obj:
            return "extract_missing_card_name"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all pack rows")
    ap.add_argument("--url", default=DEFAULT_TEACHER_URL)
    ap.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("DISTILL_MAX_TOKENS", "800")))
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--run-id", default=os.environ.get("DISTILL_RUN_ID", time.strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()

    if not args.pack.is_file():
        print(f"FATAL: pack missing: {args.pack}", file=sys.stderr)
        return 2

    tasks = [json.loads(line) for line in args.pack.read_text().splitlines() if line.strip()]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept, rejected = [], []
    t0 = time.time()
    for i, task in enumerate(tasks):
        tid = task.get("id", f"row-{i}")
        user = task["messages"][0]["content"]
        try:
            assistant = ask(args.url, args.model, user, args.max_tokens, args.timeout)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            rejected.append({"id": tid, "reason": f"teacher_error:{exc}"})
            print(f"[{i+1}/{len(tasks)}] {tid} FAIL {exc}", flush=True)
            continue
        reason = admit_row(tid, user, assistant, args.min_chars)
        if reason:
            rejected.append({"id": tid, "reason": reason})
            print(f"[{i+1}/{len(tasks)}] {tid} REJECT {reason}", flush=True)
            continue
        kept.append(
            {
                "id": tid,
                "run_id": args.run_id,
                "teacher": args.model,
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )
        print(f"[{i+1}/{len(tasks)}] {tid} OK ({len(assistant)} chars)", flush=True)

    with args.out.open("w") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "run_id": args.run_id,
        "teacher_url": args.url,
        "teacher_model": args.model,
        "pack": str(args.pack),
        "out": str(args.out),
        "kept": len(kept),
        "rejected": len(rejected),
        "reject_reasons": rejected,
        "seconds": round(time.time() - t0, 1),
    }
    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"kept": report["kept"], "rejected": report["rejected"], "out": str(args.out)}))
    # Soft-fail if zero kept (orchestrator decides hard fail)
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
