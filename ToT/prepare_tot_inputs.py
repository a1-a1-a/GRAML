import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_DESCRIPTION_SOURCE = os.path.join(
    "Dataset", "id", "expansion", "Ultimate_train_expanded.json",
)
DEFAULT_CRIT = os.path.join(
    "Dataset", "graph_context", "critical_lines", "output", "train_crit.json",
)
DEFAULT_OUTPUT = os.path.join("ToT", "tot_inputs_train.jsonl")


def extract_code(text: str) -> str:
    if "[Code]" not in text:
        return text.strip()
    start = text.index("[Code]") + len("[Code]")
    code = text[start:]
    if "[CPG Evidence]" in code:
        code = code[: code.index("[CPG Evidence]")]
    return code.strip()


def normalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", code or "").strip()


def signature(code: str) -> str:
    stripped = re.sub(r"/\*.*?\*/", " ", code or "", flags=re.S)
    stripped = re.sub(r"//[^\n]*", " ", stripped)
    brace = stripped.find("{")
    header = stripped[:brace] if brace >= 0 else stripped.splitlines()[0] if stripped else ""
    close = header.rfind(")")
    if close < 0:
        return normalize_code(header)[:200]
    depth = 0
    open_idx = -1
    for idx in range(close, -1, -1):
        if header[idx] == ")":
            depth += 1
        elif header[idx] == "(":
            depth -= 1
            if depth == 0:
                open_idx = idx
                break
    if open_idx < 0:
        return normalize_code(header)[:200]
    before = header[:open_idx].rstrip()
    name_match = re.search(r"([A-Za-z_]\w*)\s*$", before)
    name = name_match.group(1) if name_match else before.split()[-1] if before.split() else ""
    args = normalize_code(header[open_idx : close + 1])
    return f"{name}{args}"


def load_description_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    samples: List[Dict[str, Any]] = []
    for item in data:
        if item.get("Task") != "description":
            continue
        output = str(item.get("output", "")).strip().lower()
        if output in ("no vulnerability", "n/a", "") or "no vulnerability" in output:
            continue
        samples.append(item)
    return samples


def load_critical_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def build_targets(
    description_samples: Sequence[Dict[str, Any]],
    critical_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for row in critical_rows:
        idx = int(row["sample_idx"])
        if idx < 0 or idx >= len(description_samples):
            raise IndexError(
                f"sample_idx {idx} is outside the {len(description_samples)} "
                "positive description samples"
            )
        source = description_samples[idx]
        code = extract_code(str(source.get("input", "")))
        targets.append(
            {
                "sample_id": idx,
                "sample_idx": idx,
                "function_code": code,
                "signature": signature(code),
                "critical_lines": row.get("critical_lines", []),
                "selected_vulnerable_lines": row.get("selected_vulnerable_lines", []),
                "line_relations": row.get("line_relations", []),
                "cve_description": "",
            }
        )
    return targets


def _clean_description(value: Any) -> str:
    text = str(value or "").strip()
    low = text.lower()
    if not text or low in ("n/a", "none", "null") or "no vulnerability" in low:
        return ""
    return text


def _choose_description(candidates: Iterable[str]) -> Optional[str]:
    values = [c for c in candidates if c]
    if not values:
        return None
    values.sort(key=lambda x: (-len(x), x))
    return values[0]


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if line_no <= 3:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}")
                continue
            if isinstance(obj, dict):
                yield obj


def _iter_json_array(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        first = handle.read(1)
    if first == "[":
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        for obj in data:
            if isinstance(obj, dict):
                yield obj
    else:
        yield from _iter_jsonl(path)


def name_arity(code: str) -> str:
    sig = signature(code)
    match = re.match(r"([A-Za-z_]\w*)\((.*)\)$", sig, flags=re.S)
    if not match:
        return ""
    name, args = match.group(1), match.group(2).strip()
    if not args or args == "void":
        return f"{name}/0"
    depth = 0
    arity = 1
    for ch in args:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            arity += 1
    if "..." in args:
        arity += 1
    return f"{name}/{arity}"


def build_cve_index(
    targets: Sequence[Dict[str, Any]],
    cve_jsonl_paths: Sequence[str],
    diversevul_paths: Sequence[str],
) -> Tuple[Dict[str, List[Tuple[int, str]]], Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Any]]:
    target_codes = {normalize_code(t["function_code"]) for t in targets}
    target_sigs = {t["signature"] for t in targets if t["signature"]}
    target_names = {name_arity(t["function_code"]) for t in targets}
    target_names.discard("")

    exact: Dict[str, List[Tuple[int, str]]] = {}
    by_sig: Dict[str, Set[str]] = {}
    by_name: Dict[str, Set[str]] = {}
    stats: Dict[str, Any] = {
        "cve_jsonl_records": 0,
        "diversevul_records": 0,
        "exact_hits": 0,
        "signature_hits": 0,
        "name_arity_hits": 0,
        "sources": [],
    }

    def consider(code: str, description: Any, priority: int, source_name: str) -> None:
        desc = _clean_description(description)
        if not desc or not code:
            return
        normalized = normalize_code(code)
        if not normalized:
            return
        if normalized in target_codes:
            exact.setdefault(normalized, []).append((priority, desc))
            return
        sig = signature(code)
        if sig and sig in target_sigs:
            by_sig.setdefault(sig, set()).add(desc)
            return
        name = name_arity(code)
        if name and name in target_names:
            by_name.setdefault(name, set()).add(desc)

    for path in cve_jsonl_paths:
        count = 0
        for obj in _iter_jsonl(path):
            count += 1
            desc = obj.get("description_label")
            for key in ("func_before", "func_after"):
                consider(str(obj.get(key, "")), desc, 0, "train_jsonl")
        stats["cve_jsonl_records"] += count
        stats["sources"].append({"path": os.path.basename(path), "kind": "train_jsonl", "records": count})

    for path in diversevul_paths:
        count = 0
        for obj in _iter_json_array(path):
            count += 1
            consider(str(obj.get("func", "")), obj.get("message"), 1, "diversevul")
        stats["diversevul_records"] += count
        stats["sources"].append({"path": os.path.basename(path), "kind": "diversevul", "records": count})

    stats["exact_hits"] = len(exact)
    stats["signature_hits"] = len(by_sig)
    stats["name_arity_hits"] = len(by_name)
    return exact, by_sig, by_name, stats


def resolve_cve(
    target: Dict[str, Any],
    exact: Dict[str, List[Tuple[int, str]]],
    by_sig: Dict[str, Set[str]],
    by_name: Dict[str, Set[str]],
) -> None:
    candidates = exact.get(normalize_code(target["function_code"]))
    if candidates:
        candidates.sort(key=lambda item: (item[0], -len(item[1]), item[1]))
        target["cve_description"] = candidates[0][1]
        return

    values = by_sig.get(target["signature"]) or set()
    unique = {normalize_code(v) for v in values}
    if len(unique) == 1:
        target["cve_description"] = _choose_description(values) or ""
        return

    name = name_arity(target["function_code"])
    values = by_name.get(name) or set()
    unique = {normalize_code(v) for v in values}
    if len(unique) == 1:
        target["cve_description"] = _choose_description(values) or ""


def display_path(path: str, root: str) -> str:
    try:
        relative = os.path.relpath(path, root)
    except ValueError:
        return path
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return path
    return relative.replace(os.sep, "/")


def to_record(target: Dict[str, Any]) -> Dict[str, Any]:
    code_lines = target["function_code"].splitlines()
    critical_records: List[str] = []
    critical_lines = list(target.get("critical_lines", []) or [])[:20]
    for value in critical_lines:
        if isinstance(value, dict):
            n = value.get("line")
            text = value.get("code")
            if n is not None and text is not None:
                critical_records.append(f"L{int(n)}: {str(text).strip()}")
            else:
                critical_records.append(str(value))
            continue
        try:
            n = int(value)
        except (TypeError, ValueError):
            critical_records.append(str(value))
            continue
        if 1 <= n <= len(code_lines):
            critical_records.append(f"L{n}: {code_lines[n - 1].strip()}")
        else:
            critical_records.append(f"L{n}: <line unavailable>")

    record = {
        "sample_id": target["sample_id"],
        "sample_idx": target["sample_idx"],
        "function_code": target["function_code"],
        "critical_lines": target.get("critical_lines", []),
        "critical_line_records": "\n".join(critical_records),
        "selected_vulnerable_lines": target.get("selected_vulnerable_lines", []),
        "line_relations": target.get("line_relations", []),
        "cve_description": target.get("cve_description", ""),
    }
    return record


def merge_existing_cve_fields(targets: Sequence[Dict[str, Any]], path: str) -> int:
    by_code: Dict[str, Dict[str, Any]] = {}
    for record in _iter_jsonl(path):
        code = record.get("function_code") or record.get("input") or ""
        key = normalize_code(code)
        if key and record.get("cve_description"):
            by_code[key] = record
    merged = 0
    for target in targets:
        if target.get("cve_description"):
            continue
        match = by_code.get(normalize_code(target.get("function_code", "")))
        if not match:
            continue
        target["cve_description"] = match.get("cve_description", "")
        merged += 1
    return merged


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Prepare ToT-VR input records")
    parser.add_argument("--description_source", default=os.path.join(root, DEFAULT_DESCRIPTION_SOURCE))
    parser.add_argument("--crit", default=os.path.join(root, DEFAULT_CRIT))
    parser.add_argument("--cve_jsonl", action="append", default=[], help="train.jsonl with description_label")
    parser.add_argument("--diversevul", action="append", default=[], help="DiverseVul JSON/JSONL with func and message")
    parser.add_argument("--merge_cve_from", default=None, help="Existing ToT JSONL whose CVE fields are reused by code match")
    parser.add_argument("--output", default=os.path.join(root, DEFAULT_OUTPUT))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict_cve", action="store_true", help="Fail if any sample has no external description")
    args = parser.parse_args(argv)

    description_samples = load_description_samples(args.description_source)
    critical_rows = load_critical_rows(args.crit)
    targets = build_targets(description_samples, critical_rows)
    targets = targets[args.start:]
    if args.limit is not None:
        targets = targets[: args.limit]

    existing_cve = [p for p in args.cve_jsonl if os.path.exists(p)]
    existing_dv = [p for p in args.diversevul if os.path.exists(p)]
    for p in list(args.cve_jsonl) + list(args.diversevul):
        if not os.path.exists(p):
            print(f"WARNING: external description source not found: {p}", file=sys.stderr)

    if existing_cve or existing_dv:
        exact, by_sig, by_name, _ = build_cve_index(targets, existing_cve, existing_dv)
        for target in targets:
            resolve_cve(target, exact, by_sig, by_name)
    if args.merge_cve_from:
        merge_existing_cve_fields(targets, args.merge_cve_from)

    records = [to_record(t) for t in targets]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    missing_ids: List[int] = []
    for target in targets:
        if not target.get("cve_description"):
            missing_ids.append(int(target["sample_idx"]))

    print(f"Wrote {len(records)} records to {display_path(args.output, root)}")
    if args.strict_cve and missing_ids:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
