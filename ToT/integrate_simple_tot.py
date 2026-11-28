#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

NO_VULNERABILITY = "There is no vulnerability."


def normalize_code(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
                if isinstance(item, dict):
                    records.append(item)
        return records
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("data", data.get("records", []))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or JSONL records: {path}")
    return [item for item in data if isinstance(item, dict)]


def description_map(records: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    duplicates: List[str] = []

    for record in records:
        code = record.get("function_code") or record.get("code") or record.get("input")
        description = (
            record.get("final_description")
            or record.get("generated_description")
            or record.get("description")
            or record.get("output")
            or ""
        )
        code = normalize_code(code)
        description = str(description).strip()
        if not code or not description or description == NO_VULNERABILITY:
            continue
        if code in out and out[code] != description:
            duplicates.append(code)
            continue
        out[code] = description

    if duplicates:
        raise ValueError(
            f"Found {len(duplicates)} duplicate code keys with conflicting descriptions"
        )
    return out


def load_train(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("data", data.get("records", []))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def sentence_count(text: str) -> int:
    return len([item for item in re.split(r"(?<=[.!?])\s+", str(text).strip()) if item.strip()])


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Integrate Simple ToT descriptions into a training split")
    parser.add_argument("--descriptions", required=True, help="Simple ToT training JSON or JSONL")
    parser.add_argument("--train", required=True, help="Source training JSON")
    parser.add_argument("--output", required=True, help="New training JSON to write")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON path")
    parser.add_argument("--write", action="store_true", help="Write the output file; otherwise dry run")
    args = parser.parse_args(argv)

    source_path = Path(args.train)
    output_path = Path(args.output)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("--output must differ from --train; the original training file is never modified")

    descriptions = description_map(load_records(args.descriptions))
    train = load_train(args.train)

    desc_positions = [
        index for index, record in enumerate(train)
        if str(record.get("Task", "")).strip().lower() == "description"
        and str(record.get("output", "")).strip() != NO_VULNERABILITY
    ]
    desc_negatives = [
        index for index, record in enumerate(train)
        if str(record.get("Task", "")).strip().lower() == "description"
        and str(record.get("output", "")).strip() == NO_VULNERABILITY
    ]

    if len(desc_positions) != 411:
        raise ValueError(f"Expected 411 positive Description records, found {len(desc_positions)}")
    if len(desc_negatives) != 412:
        raise ValueError(f"Expected 412 negative Description records, found {len(desc_negatives)}")
    if len(descriptions) != 411:
        raise ValueError(f"Expected 411 Simple ToT descriptions, found {len(descriptions)}")

    train_keys = {normalize_code(train[index].get("input", "")) for index in desc_positions}
    missing = sorted(train_keys - set(descriptions))
    extra = sorted(set(descriptions) - train_keys)
    if missing or extra:
        raise ValueError(
            "Description/code mismatch: "
            f"{len(missing)} training positives have no Simple ToT description, "
            f"{len(extra)} Simple ToT descriptions have no training record. "
            "Run the generator on the full 411-sample input."
        )

    output = list(train)
    changed = 0
    for index in desc_positions:
        record = dict(train[index])
        key = normalize_code(record.get("input", ""))
        new_description = descriptions[key]
        if str(record.get("output", "")).strip() != new_description:
            record["output"] = new_description
            changed += 1
        output[index] = record

    for before, after in zip(train, output):
        if before is after:
            continue
        if before == after:
            continue
        before_task = str(before.get("Task", "")).strip().lower()
        before_output = str(before.get("output", "")).strip()
        if before_task != "description" or before_output == NO_VULNERABILITY:
            raise AssertionError("A non-positive-Description record changed")
        if {k: v for k, v in before.items() if k != "output"} != {k: v for k, v in after.items() if k != "output"}:
            raise AssertionError("A positive Description record changed fields other than output")

    before_counts = Counter(str(record.get("Task", "")).strip().lower() for record in train)
    after_counts = Counter(str(record.get("Task", "")).strip().lower() for record in output)
    if before_counts != after_counts:
        raise AssertionError(f"Task counts changed: {before_counts} -> {after_counts}")

    final_descriptions = [
        str(output[index].get("output", "")).strip()
        for index in desc_positions
    ]
    if len(set(final_descriptions)) != 411:
        raise AssertionError("Expected 411 distinct Simple ToT descriptions")

    summary = {
        "descriptions_source": str(Path(args.descriptions)),
        "train_source": str(source_path),
        "output": str(output_path),
        "records": len(output),
        "description_positive_records": len(desc_positions),
        "description_negative_records": len(desc_negatives),
        "changed_outputs": changed,
        "description_word_count": {
            "min": min(len(item.split()) for item in final_descriptions),
            "max": max(len(item.split()) for item in final_descriptions),
        },
        "description_sentence_count_estimate": {
            "min": min(sentence_count(item) for item in final_descriptions),
            "max": max(sentence_count(item) for item in final_descriptions),
        },
        "original_task_counts": dict(before_counts),
        "new_task_counts": dict(after_counts),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.write:
        print("\nDRY RUN ONLY; pass --write to create the new training file.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=1)
        handle.write("\n")

    summary["output_bytes"] = output_path.stat().st_size
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_name(output_path.name + ".manifest.json")
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {output_path}")
    print(f"WROTE {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

