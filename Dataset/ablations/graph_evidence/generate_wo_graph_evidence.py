#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

NO_VULNERABILITY_OUTPUT = "There is no vulnerability."
INSTRUCTION = (
    "You are a security code vulnerability analyzer. Analyze the provided C/C++ "
    "function and explain the vulnerability. Write one coherent paragraph of 3 to 5 "
    "sentences. State what the vulnerability is, why it occurs, how it can be "
    "triggered, and what impact it can have. Do not use parentheses or absolute line "
    "numbers. If there is no vulnerability, output exactly: There is no vulnerability."
)
SAVE_LOCK = Lock()


def extract_code(input_text: str) -> str:
    text = str(input_text or "")
    if "[Code]" in text:
        text = text.split("[Code]", 1)[1]
    if "[CPG Evidence]" in text:
        text = text.split("[CPG Evidence]", 1)[0]
    return text.strip()


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def positive_description_rows(train: List[Dict[str, Any]]) -> List[int]:
    return [
        index for index, row in enumerate(train)
        if str(row.get("Task", "")).strip().lower() == "description"
        and str(row.get("output", "")).strip() != NO_VULNERABILITY_OUTPUT
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def call_model(client: Any, model: str, code: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"Function:\n{code}"},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return str(response.choices[0].message.content or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate w/o Graph Evidence descriptions")
    parser.add_argument("--train_path", default="Dataset/id/expansion/Ultimate_train_expanded.json")
    parser.add_argument("--output_dir", default="Dataset/ablations/graph_evidence")
    parser.add_argument("--output_train", default="Dataset/ablations/graph_evidence/Ultimate_train_wo_graph_evidence.json")
    parser.add_argument("--generated_path", default=None)
    parser.add_argument("--model", default=os.environ.get("TOT_MODEL", "gpt-5"))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    train = load_json(args.train_path)
    if not isinstance(train, list):
        raise ValueError("Training file must contain a JSON list")
    positions = positive_description_rows(train)
    if len(positions) != 411:
        raise ValueError(f"Expected 411 positive Description records, found {len(positions)}")

    samples = []
    for index in positions:
        code = extract_code(train[index].get("input", ""))
        samples.append({
            "index": index,
            "code": code,
        })
    if args.limit is not None:
        samples = samples[: args.limit]

    if args.dry_run:
        print(INSTRUCTION)
        print("\n--- user prompt ---\n")
        print("Function:\n" + samples[0]["code"][:4000])
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = Path(args.generated_path) if args.generated_path else output_dir / "wo_graph_evidence_output.json"

    done: Dict[str, Dict[str, Any]] = {}
    if args.resume and generated_path.exists():
        for record in load_json(str(generated_path)):
            if record.get("description"):
                done[int(record["index"])] = record

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} or use --dry_run to inspect the prompt")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url or None)
    pending = [sample for sample in samples if sample["index"] not in done]
    results = list(done.values())
    started = time.time()

    def run_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            description = call_model(client, args.model, sample["code"])
            error = None
        except Exception as exc:
            description = ""
            error = str(exc)
        return {
            "index": sample["index"],
            "code": sample["code"],
            "description": description,
            "error": error,
        }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_one, sample): sample for sample in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            record = future.result()
            results.append(record)
            if completed % 10 == 0 or completed == len(pending):
                with SAVE_LOCK:
                    write_json(generated_path, sorted(results, key=lambda item: item["index"]))
                print(f"[{completed}/{len(pending)}] {time.time() - started:.0f}s", flush=True)

    write_json(generated_path, sorted(results, key=lambda item: item["index"]))
    failures = [record for record in results if record.get("error")]
    if failures:
        raise RuntimeError(f"{len(failures)} generation calls failed; re-run with --resume")

    by_index = {int(record["index"]): record for record in results if record.get("description")}
    output = [dict(row) for row in train]
    changed = 0
    for index in positions:
        code = extract_code(train[index].get("input", ""))
        record = by_index.get(index)
        if record is None:
            raise ValueError(f"Missing generated description for training record {index}")
        if output[index]["output"] != record["description"]:
            output[index] = dict(output[index])
            output[index]["output"] = record["description"]
            changed += 1

    if len(output) != len(train):
        raise AssertionError("Record count changed")
    for before, after in zip(train, output):
        if before is after:
            continue
        if str(before.get("Task", "")).lower() != "description":
            raise AssertionError("A non-Description record changed")

    output_train = Path(args.output_train)
    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_train.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    manifest = {
        "train_source": args.train_path,
        "generated_descriptions": str(generated_path),
        "output_train": str(output_train),
        "records": len(output),
        "description_positive_records": len(positions),
        "changed_outputs": changed,
        "model": args.model,
        "errors": len(failures),
    }
    write_json(output_dir / "wo_graph_evidence_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
