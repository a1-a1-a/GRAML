#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "graph_context" / "cpg_evidence" / "joern_cpg_evidence.json"
DEFAULT_DESCRIPTIONS = ROOT / "descriptions" / "vulnerability_descriptions.json"


def normalize_code(text: Any) -> str:
    return " ".join(str(text or "").split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Add data-dependence records to CPG evidence")
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--descriptions", default=str(DEFAULT_DESCRIPTIONS))
    args = parser.parse_args()

    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8-sig"))
    descriptions = json.loads(Path(args.descriptions).read_text(encoding="utf-8-sig"))
    if len(evidence) != 411 or len(descriptions) != 411:
        raise ValueError("Expected 411 evidence and 411 canonical Description records")

    by_code = {normalize_code(item["input"]): item for item in evidence}
    if len(by_code) != len(evidence):
        raise ValueError("CPG evidence inputs are not unique after normalization")

    added = 0
    matched = 0
    for record in descriptions:
        item = by_code.get(normalize_code(record.get("code")))
        if item is None:
            raise ValueError(f"Missing CPG evidence for canonical record {record.get('sample_id')}")
        matched += 1
        edges = list(item.get("cpg_edges") or [])
        seen = {(str(edge.get("type")), int(edge["src"]), int(edge["dst"])) for edge in edges}
        data_edges = []
        for relation in record.get("line_relations") or []:
            if relation.get("type") != "DDG":
                continue
            edge = {
                "type": "REACHING_DEF",
                "src": int(relation["src"]),
                "dst": int(relation["dst"]),
                "source": "joern-cpgql:reaching_def",
            }
            key = (edge["type"], edge["src"], edge["dst"])
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)
            data_edges.append(edge)
            added += 1
        item["cpg_edges"] = edges
        item["data_flow_edges"] = [
            edge for edge in edges
            if "REACH" in str(edge.get("type", "")) or str(edge.get("type", "")) == "DDG"
        ]
        item["control_flow_edges"] = [
            edge for edge in edges
            if str(edge.get("type", "")) in {"CONTROL_DEPENDENCE", "CDG"}
        ]

    Path(args.evidence).write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    types = Counter(edge.get("type") for item in evidence for edge in item.get("cpg_edges", []))
    total = sum(types.values())
    print(f"matched={matched} added={added} total_edges={total}")
    print(json.dumps(dict(sorted(types.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
