#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOT = REPO / "ToT"
EXP = TOT / "experiments"
NAME = "tot_description_411_rerun"
INPUT = TOT / "tot_inputs_train.jsonl"
OUTPUT = EXP / f"{NAME}.jsonl"
SUMMARY = EXP / f"{NAME}_summary.json"
LOG = EXP / f"{NAME}.log"
ERR = EXP / f"{NAME}.err.log"
PROGRESS = EXP / f"{NAME}_progress.json"
STATUS = EXP / f"{NAME}_status.txt"
MODEL = os.environ.get("TOT_MODEL", "gpt-5")
TOTAL = 411
MAX_ATTEMPTS = 10

CURRENT_CHILD: subprocess.Popen | None = None


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()


def write_progress(attempt: int | None = None, child_pid: int | None = None) -> None:
    records = []
    malformed = 0
    if OUTPUT.exists():
        with OUTPUT.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
    latest = {}
    for record in records:
        sid = str(record.get("sample_id", record.get("sample_idx")))
        latest[sid] = record
    records = list(latest.values())
    done = sum(1 for r in records if r.get("final_description"))
    failed = sum(1 for r in records if r.get("error"))
    verified = sum(1 for r in records if r.get("verified"))
    status_counts = Counter(str(r.get("description_status", "unknown")) for r in records)
    payload = {
        "updated_at": now(),
        "attempt": attempt,
        "child_pid": child_pid,
        "total": TOTAL,
        "records_written": len(records),
        "completed_with_description": done,
        "remaining": max(0, TOTAL - done),
        "errors": failed,
        "verified": verified,
        "malformed_lines": malformed,
        "description_status_counts": dict(status_counts),
        "output": str(OUTPUT),
        "summary": str(SUMMARY),
        "log": str(LOG),
    }
    PROGRESS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def handle_signal(signum, frame):
    append(STATUS, f"{now()} wrapper received signal {signum}; stopping child\n")
    child = CURRENT_CHILD
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
    write_progress()
    raise SystemExit(130)


def run_once(attempt: int) -> int:
    global CURRENT_CHILD
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY must be set")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, "-u", str(TOT / "tot_vr.py"),
        "--input", str(INPUT),
        "--output", str(OUTPUT),
        "--summary", str(SUMMARY),
        "--model", MODEL,
        "--workers", "3",
        "--max_output_tokens", "8192",
        "--timeout", "240",
        "--sample_retries", "3",
        "--resume",
        "--verbose",
    ]
    if os.environ.get("TOT_DISABLE_THINKING", "").strip().lower() in {"1", "true", "yes", "on"}:
        cmd.append("--disable_thinking")

    append(LOG, f"\n===== wrapper attempt {attempt} start {now()} =====\n")
    append(STATUS, f"{now()} attempt {attempt} start\n")
    with ERR.open("a", encoding="utf-8") as err_handle:
        child = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=err_handle,
            text=True,
            bufsize=1,
        )
        CURRENT_CHILD = child
        write_progress(attempt=attempt, child_pid=child.pid)
        last_progress = 0.0
        assert child.stdout is not None
        for line in child.stdout:
            append(LOG, line)
            print(line, end="", flush=True)
            now_ts = time.time()
            if now_ts - last_progress >= 15:
                write_progress(attempt=attempt, child_pid=child.pid)
                last_progress = now_ts
        code = child.wait()
    CURRENT_CHILD = None
    append(STATUS, f"{now()} attempt {attempt} exit {code}\n")
    write_progress(attempt=attempt)
    return code


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    for path in (LOG, ERR, STATUS):
        path.touch(exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY must be set")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    append(STATUS, f"{now()} wrapper start, total={TOTAL}, output={OUTPUT}\n")
    write_progress()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            code = run_once(attempt)
        except Exception as exc:
            append(STATUS, f"{now()} attempt {attempt} exception: {exc}\n")
            append(ERR, f"{now()} wrapper exception: {exc}\n")
            code = 1
        if code == 0:
            append(STATUS, f"{now()} wrapper finished successfully\n")
            write_progress(attempt=attempt)
            return 0
        if attempt < MAX_ATTEMPTS:
            append(STATUS, f"{now()} retrying after 60s\n")
            time.sleep(60)
    append(STATUS, f"{now()} wrapper stopped after {MAX_ATTEMPTS} attempts\n")
    write_progress(attempt=MAX_ATTEMPTS)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
