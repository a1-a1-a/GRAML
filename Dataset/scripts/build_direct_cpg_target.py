#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

NO_VULNERABILITY_OUTPUT = "There is no vulnerability."
DIRECT_CPG_INSTRUCTION = (
    "You are a security code vulnerability analyzer. Analyze the following C/C++ "
    "code and output its raw Joern CPG edge records. Use one JSON object per line "
    "with the fields type, src, dst, and source. Preserve the edge type and line "
    "direction. Use this format:\n"
    "[Joern CPG]\n"
    '{"type":"...","src":0,"dst":0,"source":"..."}\n'
    "...\n"
    "Do not write a natural-language vulnerability description."
)


def normalize_code(text: object) -> str:
    return " ".join(str(text or "").split())


def format_joern_records(item: dict) -> str:
    edges = item.get("cpg_edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError(f"No Joern CPG edges for sample {item.get('sample_id')}")
    lines = ["[Joern CPG]"]
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError(f"Invalid Joern CPG edge in sample {item.get('sample_id')}")
        lines.append(json.dumps({
            "type": str(edge["type"]),
            "src": int(edge["src"]),
            "dst": int(edge["dst"]),
            "source": str(edge.get("source", "joern")),
        }, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        default="Dataset/id/expansion/Ultimate_train_expanded.json",
    )
    parser.add_argument(
        "--joern_path",
        default="Dataset/graph_context/cpg_evidence/joern_cpg_evidence.json",
    )
    parser.add_argument(
        "--output_path",
        default=(
            "Dataset/ablations/direct_cpg/"
            "Ultimate_train_direct_cpg.json"
        ),
    )
    args = parser.parse_args()

    with Path(args.train_path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    rows = data["data"] if isinstance(data, dict) and "data" in data else data
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON list or an object with a data list")

    with Path(args.joern_path).open("r", encoding="utf-8-sig") as handle:
        joern_records = json.load(handle)
    by_code = {normalize_code(item["input"]): item for item in joern_records}
    if len(by_code) != len(joern_records):
        raise ValueError("Joern CPG inputs are not unique after normalization")

    converted = 0
    untouched = 0
    seen = set()
    for row in rows:
        if str(row.get("Task", "")).lower() != "description":
            untouched += 1
            continue
        if str(row.get("output", "")).strip() == NO_VULNERABILITY_OUTPUT:
            untouched += 1
            continue
        key = normalize_code(row.get("input"))
        item = by_code.get(key)
        if item is None:
            raise ValueError(f"Missing Joern CPG records for: {row.get('input', '')[:120]!r}")
        if key in seen:
            raise ValueError(f"Duplicate Description input: {row.get('input', '')[:120]!r}")
        seen.add(key)
        row["input"] = str(item["input"])
        row["output"] = format_joern_records(item)
        row["instruction"] = DIRECT_CPG_INSTRUCTION
        row["_target_representation"] = "joern_cpg_edge_records"
        converted += 1

    if converted != 411:
        raise ValueError(f"Expected 411 converted Description records, got {converted}")

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, ensure_ascii=False)
        handle.write("\n")
    print(f"converted={converted} untouched={untouched} output={out}")


if __name__ == "__main__":
    main()
