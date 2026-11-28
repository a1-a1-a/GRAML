#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "descriptions" / "vulnerability_descriptions.json"
DEFAULT_OUT = ROOT / "graph_context" / "critical_lines" / "output"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    crit_rows: List[Dict[str, Any]] = []
    vul_rows: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        critical_lines = list(record.get("critical_lines") or [])
        relations = list(record.get("line_relations") or [])
        vuln_lines = sorted({int(v) for v in (record.get("vulnerability_lines") or [])})
        selected = {int(item["line"]) for item in critical_lines if isinstance(item, dict) and "line" in item}

        crit_rows.append({
            "sample_idx": idx,
            "critical_lines": critical_lines,
            "line_relations": relations,
            "selected_vulnerable_lines": [line for line in vuln_lines if line in selected],
        })
        vul_rows.append({
            "sample_idx": idx,
            "vuln_lines": vuln_lines,
        })

    return crit_rows, vul_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export critical-line artifacts")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    records = load_json(Path(args.source))
    if not isinstance(records, list) or len(records) != 411:
        raise ValueError(f"Expected 411 canonical records, found {len(records) if isinstance(records, list) else 'non-list'}")

    crit_rows, vul_rows = build(records)
    out_dir = Path(args.out_dir)
    dump_json(out_dir / "train_crit.json", crit_rows)
    dump_json(out_dir / "train_vul_lines.json", vul_rows)
    print(f"Wrote {len(crit_rows)} critical-line records to {out_dir}")


if __name__ == "__main__":
    main()
