import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LABEL_DERIVED_KEYS = {
    "critical_lines",
    "Critical Lines",
    "critical line",
    "critical lines",
    "vulnerable_lines",
    "vulnerable line",
    "vulnerable lines",
    "location",
    "Location",
    "cve_description",
    "CVE Description",
    "official_cve_description",
    "patch",
    "commit_message",
}

CPG_EDGE_KEYS = [
    "cpg_edges",
    "edges",
    "line_edges",
    "line_level_edges",
    "dependency_edges",
    "data_flow_edges",
    "control_flow_edges",
]

LINE_KEYS = [
    "graph_context_lines",
    "cpg_context_lines",
    "critical_lines",
    "Critical Lines",
    "suspicious_lines",
    "important_lines",
    "slice_lines",
    "sink_lines",
    "source_lines",
]

RELATION_KEYS = [
    "line_relations",
    "line_relation",
    "relations",
    "dependency_relations",
    "cpg_relations",
    "graph_context",
    "control_flow",
    "data_flow",
    "ast_summary",
]

RISK_PATTERNS = [
    (r"\b(strcpy|strcat|gets|sprintf|vsprintf|scanf|sscanf)\b", 5, "unsafe C library call"),
    (r"\b(memcpy|memmove|memset|strncpy|snprintf|read|recv|fread)\b", 3, "memory/input operation needing bounds validation"),
    (r"\b(copy_from_user|copy_to_user|memdup_user|put_user|get_user)\b", 4, "user-kernel data transfer"),
    (r"\b(malloc|calloc|realloc|kmalloc|kzalloc|new|free|kfree|delete)\b", 3, "allocation or lifetime operation"),
    (r"\b(lock|unlock|mutex|spin_lock|rcu|atomic|refcount)\b", 2, "synchronization or lifetime-sensitive operation"),
    (r"\b(len|size|count|offset|index|idx|nbytes|capacity)\b.*(\+|-|\*|<<|>>)", 2, "size or index arithmetic"),
    (r"(if|while|for)\s*\(.*(len|size|count|offset|index|idx|nbytes).*(<|>|<=|>=|==|!=)", 2, "boundary-related condition"),
    (r"(\*|->|&).*(NULL|nullptr|0)|\b(NULL|nullptr)\b.*(\*|->|&)", 2, "pointer/null-sensitive expression"),
    (r"\b(return|goto)\b.*(-E|NULL|false|FALSE|FAIL|ERR)", 1, "error path"),
]

KEYWORDS = {
    "if",
    "for",
    "while",
    "return",
    "sizeof",
    "struct",
    "static",
    "const",
    "void",
    "int",
    "char",
    "long",
    "short",
    "unsigned",
    "signed",
    "bool",
    "true",
    "false",
    "NULL",
    "nullptr",
}


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as file:
        data = json.loads(file.read().decode("utf-8-sig"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list or a dict with a top-level data field.")
    return data


def write_json(path: str, data: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def normalized_hash(text: Any) -> str:
    normalized = " ".join(str(text or "").split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def get_task(sample: Dict[str, Any]) -> str:
    return str(sample.get("Task", sample.get("task", sample.get("category", sample.get("type", ""))))).strip().lower()


def get_code(sample: Dict[str, Any]) -> str:
    return str(sample.get("input", sample.get("Input", sample.get("code", ""))))


def sample_lookup_keys(sample: Dict[str, Any]) -> List[str]:
    keys = []
    for key in ["index", "idx", "id", "sample_id"]:
        if key in sample and sample[key] is not None:
            keys.append(str(sample[key]))
    return keys


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                line_no = item.get("line", item.get("lineno", item.get("line_no", "")))
                code = item.get("code", item.get("text", item.get("content", "")))
                reason = item.get("reason", item.get("type", item.get("score", "")))
                prefix = f"L{line_no}: " if line_no != "" else ""
                suffix = f" ({reason})" if reason != "" else ""
                lines.append(f"{prefix}{code}{suffix}".strip())
            else:
                lines.append(str(item).strip())
        return "\n".join(line for line in lines if line)
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            rendered = stringify(item)
            if rendered:
                lines.append(f"{key}: {rendered}")
        return "\n".join(lines)
    return str(value).strip()


def load_evidence_maps(path: Optional[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    by_hash: Dict[str, Dict[str, Any]] = {}
    if not path:
        return by_key, by_hash

    with open(path, "rb") as file:
        raw = json.loads(file.read().decode("utf-8-sig"))

    if isinstance(raw, dict) and "data" in raw:
        iterable: Iterable[Any] = raw["data"]
    elif isinstance(raw, dict):
        iterable = []
        for key, value in raw.items():
            record = value if isinstance(value, dict) else {"graph_context": value}
            record = dict(record)
            record.setdefault("id", key)
            iterable.append(record)
    elif isinstance(raw, list):
        iterable = raw
    else:
        raise ValueError(f"{path} must be a JSON list/dict evidence file.")

    for record in iterable:
        if not isinstance(record, dict):
            continue
        for key_name in ["index", "idx", "id", "sample_id"]:
            if key_name in record and record[key_name] is not None:
                by_key[str(record[key_name])] = record
        code = record.get("input", record.get("Input", record.get("code", "")))
        if code:
            by_hash[normalized_hash(code)] = record
    return by_key, by_hash


def load_description_map(path: Optional[str]) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    if not path:
        return descriptions
    for record in read_json(path):
        code = record.get("code", record.get("input", record.get("Input", "")))
        description = record.get(
            "Vulnerable description",
            record.get("vulnerable_description", record.get("description", "")),
        )
        if code and description:
            descriptions[normalized_hash(code)] = str(description).strip()
    return descriptions


def find_external_evidence(
    sample: Dict[str, Any],
    evidence_by_key: Dict[str, Dict[str, Any]],
    evidence_by_hash: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for key in sample_lookup_keys(sample):
        if key in evidence_by_key:
            return evidence_by_key[key]
    return evidence_by_hash.get(normalized_hash(get_code(sample)))


def identifiers(text: str) -> set:
    return {token for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text) if token not in KEYWORDS}


def build_heuristic_evidence(code: str, max_lines: int, max_relations: int) -> Dict[str, Any]:
    numbered_lines = [(index + 1, line.rstrip()) for index, line in enumerate(code.splitlines())]
    scored_lines = []
    for line_no, line in numbered_lines:
        reasons = []
        score = 0
        for pattern, weight, reason in RISK_PATTERNS:
            if re.search(pattern, line):
                score += weight
                reasons.append(reason)
        if score > 0:
            scored_lines.append(
                {
                    "line": line_no,
                    "code": line.strip(),
                    "score": score,
                    "reason": "; ".join(dict.fromkeys(reasons)),
                }
            )

    scored_lines = sorted(scored_lines, key=lambda item: (-item["score"], item["line"]))[:max_lines]
    scored_lines = sorted(scored_lines, key=lambda item: item["line"])

    non_empty_by_line = {line_no: line.strip() for line_no, line in numbered_lines if line.strip()}
    suspicious_line_numbers = {item["line"] for item in scored_lines}
    relations = []
    for item in scored_lines:
        line_no = item["line"]
        current_ids = identifiers(item["code"])
        for neighbor in range(max(1, line_no - 2), min(len(numbered_lines), line_no + 2) + 1):
            if neighbor == line_no or neighbor not in non_empty_by_line:
                continue
            neighbor_ids = identifiers(non_empty_by_line[neighbor])
            shared = sorted(current_ids & neighbor_ids)
            if shared or neighbor in suspicious_line_numbers:
                relation_type = "SHARED_IDENTIFIER" if shared else "NEARBY_RISK_CONTEXT"
                detail = f"shared={','.join(shared[:4])}" if shared else "near another suspicious line"
                relations.append(f"L{neighbor} --{relation_type}--> L{line_no}: {detail}")
            if len(relations) >= max_relations:
                break
        if len(relations) >= max_relations:
            break

    return {
        "evidence_source": "label_free_static_heuristic",
        "suspicious_lines": scored_lines,
        "relations": relations,
    }


def is_valid_code_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("//") and not stripped.startswith("/*") and not stripped.startswith("*")


def line_text_by_number(code: str) -> Dict[int, str]:
    return {index + 1: line.rstrip() for index, line in enumerate(code.splitlines())}


def parse_line_numbers(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(match) for match in re.findall(r"\b(?:L|line\s*)?(\d+)\b", value, flags=re.IGNORECASE)]
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(parse_line_numbers(item))
        return result
    if isinstance(value, dict):
        for key in ["line", "lineno", "line_no", "line_number", "start_line"]:
            if key in value:
                return parse_line_numbers(value[key])
    return []


def parse_cpg_edges(external: Dict[str, Any]) -> List[Dict[str, Any]]:
    edges = []
    for key in CPG_EDGE_KEYS:
        raw_edges = external.get(key)
        if not raw_edges:
            continue
        if isinstance(raw_edges, dict):
            raw_edges = raw_edges.get("edges", raw_edges.get("data", []))
        if isinstance(raw_edges, str):
            for row in raw_edges.splitlines():
                typed = re.search(
                    r"(?:L|line\s*)?(\d+)\s*--\s*([A-Za-z_]+)\s*-->\s*(?:L|line\s*)?(\d+)",
                    row,
                    flags=re.IGNORECASE,
                )
                plain = re.search(
                    r"(?:L|line\s*)?(\d+)\s*[-=]+>\s*(?:L|line\s*)?(\d+)",
                    row,
                    flags=re.IGNORECASE,
                )
                if typed:
                    edges.append({"src": int(typed.group(1)), "dst": int(typed.group(3)), "type": typed.group(2)})
                elif plain:
                    edges.append({"src": int(plain.group(1)), "dst": int(plain.group(2)), "type": key})
            continue
        if not isinstance(raw_edges, list):
            continue
        for edge in raw_edges:
            if isinstance(edge, dict):
                src = edge.get("src", edge.get("source", edge.get("from", edge.get("start"))))
                dst = edge.get("dst", edge.get("target", edge.get("to", edge.get("end"))))
                relation_type = edge.get("type", edge.get("relation", edge.get("edge_type", key)))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                src, dst = edge[0], edge[1]
                relation_type = edge[2] if len(edge) >= 3 else key
            else:
                continue
            src_lines = parse_line_numbers(src)
            dst_lines = parse_line_numbers(dst)
            if src_lines and dst_lines:
                edges.append({"src": src_lines[0], "dst": dst_lines[0], "type": str(relation_type)})
    return edges


def build_cpg_evidence_from_graph(
    code: str,
    external: Dict[str, Any],
    allow_label_evidence: bool,
    max_lines: int,
    max_relations: int,
) -> Optional[Dict[str, Any]]:
    line_map = line_text_by_number(code)
    valid_lines = {line_no for line_no, text in line_map.items() if is_valid_code_line(text)}
    if not valid_lines:
        return None

    edge_seen = set()
    edges = []
    for edge in parse_cpg_edges(external):
        if edge["src"] not in valid_lines or edge["dst"] not in valid_lines or edge["src"] == edge["dst"]:
            continue
        edge["type"] = str(edge["type"]).upper()
        edge_key = (edge["src"], edge["dst"], edge["type"])
        if edge_key in edge_seen:
            continue
        edge_seen.add(edge_key)
        edges.append(edge)
    semantic_edges = [edge for edge in edges if edge["type"] != "AST_CHILD"]
    if semantic_edges:
        edges = semantic_edges
    vulnerable_lines = (
        sorted(set(parse_line_numbers(external.get("vulnerable_lines", external.get("Vulnerable Lines")))))
        if allow_label_evidence
        else []
    )
    vulnerable_lines = [line_no for line_no in vulnerable_lines if line_no in valid_lines]

    precomputed_context: List[Dict[str, Any]] = []
    for key in LINE_KEYS:
        if key in external and (allow_label_evidence or key not in LABEL_DERIVED_KEYS):
            value = external[key]
            if isinstance(value, list):
                for item in value:
                    line_numbers = parse_line_numbers(item)
                    if not line_numbers:
                        continue
                    line_no = line_numbers[0]
                    if line_no in valid_lines:
                        if isinstance(item, dict):
                            precomputed_context.append(
                                {
                                    "line": line_no,
                                    "code": item.get("code", line_map.get(line_no, "")).strip(),
                                    "role": item.get("role", key),
                                }
                            )
                        else:
                            precomputed_context.append(
                                {"line": line_no, "code": line_map.get(line_no, "").strip(), "role": key}
                            )
            else:
                for line_no in parse_line_numbers(value):
                    if line_no in valid_lines:
                        precomputed_context.append(
                            {"line": line_no, "code": line_map.get(line_no, "").strip(), "role": key}
                        )

    if edges and not vulnerable_lines:
        line_scores = Counter()
        edge_weights = {"CONTROL_DEPENDENCE": 3, "CFG_NEXT": 2, "AST_CHILD": 1}
        for edge in edges:
            weight = edge_weights.get(edge["type"], 1)
            line_scores[edge["src"]] += weight
            line_scores[edge["dst"]] += weight
        selected_lines = [
            line_no
            for line_no, _ in sorted(line_scores.items(), key=lambda item: (-item[1], item[0]))[:max_lines]
        ]
        selected_items = [
            {
                "line": line_no,
                "code": line_map.get(line_no, "").strip(),
                "role": "semantic_cpg_context" if any(edge["type"] != "AST_CHILD" for edge in edges) else "ast_cpg_context",
                "score": line_scores[line_no],
            }
            for line_no in sorted(selected_lines)
        ]
        selected = {item["line"] for item in selected_items}
        relation_priority = {"CONTROL_DEPENDENCE": 0, "CFG_NEXT": 1, "AST_CHILD": 2}
        relations = []
        relation_text = []
        for key in RELATION_KEYS:
            rendered = stringify(external.get(key))
            if rendered:
                relation_text.extend([row.strip() for row in rendered.splitlines() if row.strip()])
        for edge in sorted(edges, key=lambda item: (relation_priority.get(item["type"], 9), item["src"], item["dst"])):
            if edge["src"] in selected and edge["dst"] in selected:
                relations.append(f"L{edge['src']} --{edge['type']}--> L{edge['dst']}")
            if len(relations) >= max_relations:
                break
        relations.extend(relation_text)
        return {
            "evidence_source": "joern_cpg_context_lines",
            "suspicious_lines": selected_items,
            "relations": relations[:max_relations],
        }

    if not allow_label_evidence:
        return None
    if not vulnerable_lines or not edges:
        return None

    outgoing_edges: Dict[int, List[int]] = {}
    typed_edges: Dict[Tuple[int, int], List[str]] = {}
    for edge in edges:
        outgoing_edges.setdefault(edge["src"], []).append(edge["dst"])
        typed_edges.setdefault((edge["src"], edge["dst"]), []).append(edge["type"])

    adjacency: Dict[int, set] = {}
    for line_no in valid_lines:
        if outgoing_edges.get(line_no):
            adjacency[line_no] = set(outgoing_edges[line_no])
        else:
            adjacency[line_no] = set(valid_lines - {line_no})

    first_hop = set()
    for vulnerable_line in vulnerable_lines:
        first_hop.update(adjacency.get(vulnerable_line, set()))
        first_hop.update(line_no for line_no, dsts in adjacency.items() if vulnerable_line in dsts)
    first_hop -= set(vulnerable_lines)

    second_scores = Counter()
    for first_line in first_hop:
        for candidate in adjacency.get(first_line, set()):
            if candidate not in first_hop and candidate not in vulnerable_lines:
                second_scores[candidate] += 1

    max_second = max(1, len(line_map) // 5)
    second_items = second_scores.most_common()
    if len(second_items) > max_second:
        second_items = second_items[:max_second]
    second_hop = {line_no for line_no, _ in second_items}

    ordered_lines = []
    for role, line_numbers in [
        ("vulnerable_seed", vulnerable_lines),
        ("first_hop_cpg_neighbor", sorted(first_hop)),
        ("second_hop_ranked_by_cpg_connectivity", sorted(second_hop)),
    ]:
        for line_no in line_numbers:
            if line_no not in {item["line"] for item in ordered_lines}:
                ordered_lines.append({"line": line_no, "code": line_map.get(line_no, "").strip(), "role": role})
    ordered_lines = sorted(ordered_lines, key=lambda item: item["line"])[:max_lines]
    selected = {item["line"] for item in ordered_lines}

    relations = []
    for edge in edges:
        if edge["src"] in selected and edge["dst"] in selected:
            relations.append(f"L{edge['src']} --{edge['type']}--> L{edge['dst']}")
        if len(relations) >= max_relations:
            break

    return {
        "evidence_source": "joern_cpg_mask_pruning_hop_expansion",
        "suspicious_lines": ordered_lines,
        "relations": relations,
    }


def build_random_line_evidence(code: str, max_lines: int, seed: int, sample_index: int) -> Dict[str, Any]:
    rng = random.Random(seed + sample_index)
    lines = [(index + 1, line.strip()) for index, line in enumerate(code.splitlines()) if line.strip()]
    if not lines:
        return {"evidence_source": "random_line_control", "suspicious_lines": [], "relations": []}
    selected = sorted(rng.sample(lines, k=min(max_lines, len(lines))), key=lambda item: item[0])
    return {
        "evidence_source": "random_line_control",
        "suspicious_lines": [
            {"line": line_no, "code": line, "score": 0, "reason": "random control line"}
            for line_no, line in selected
        ],
        "relations": [],
    }


def merge_external_evidence(
    external: Dict[str, Any],
    allow_label_evidence: bool,
    max_lines: int,
    max_relations: int,
) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "evidence_source": external.get("evidence_source", external.get("source", "external_static_or_graph")),
        "suspicious_lines": [],
        "relations": [],
    }

    for key in LINE_KEYS:
        if key in external and (allow_label_evidence or key not in LABEL_DERIVED_KEYS):
            rendered = stringify(external[key])
            if rendered:
                evidence["suspicious_lines"].append(rendered)
    for key in RELATION_KEYS:
        if key in external and (allow_label_evidence or key not in LABEL_DERIVED_KEYS):
            rendered = stringify(external[key])
            if rendered:
                evidence["relations"].append(f"{key}:\n{rendered}")

    if allow_label_evidence:
        for key in LABEL_DERIVED_KEYS:
            if key in external:
                rendered = stringify(external[key])
                if rendered:
                    evidence["relations"].append(f"{key}:\n{rendered}")

    evidence["suspicious_lines"] = evidence["suspicious_lines"][:max_lines]
    evidence["relations"] = evidence["relations"][:max_relations]
    return evidence


def format_evidence_block(evidence: Dict[str, Any]) -> str:
    source = evidence.get("evidence_source", "unknown")
    heading = "[CPG Evidence]" if "cpg" in str(source).lower() or "joern" in str(source).lower() else "[Static Evidence]"
    lines = [heading, f"Evidence source: {source}"]
    suspicious = evidence.get("suspicious_lines") or []
    relations = evidence.get("relations") or []

    lines.append("Suspicious/important lines:")
    if suspicious:
        for item in suspicious:
            if isinstance(item, dict):
                reason = item.get("reason", "")
                role = item.get("role", "")
                suffix_parts = []
                if role:
                    suffix_parts.append(str(role))
                if reason:
                    suffix_parts.append(str(reason))
                suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
                lines.append(f"L{item.get('line', '?')}: {item.get('code', '')}{suffix}")
            else:
                for row in str(item).splitlines():
                    if row.strip():
                        lines.append(row.strip())
    else:
        lines.append("None found by the configured evidence extractor.")

    lines.append("Structural/context relations:")
    if relations:
        for relation in relations:
            for row in str(relation).splitlines():
                if row.strip():
                    lines.append(row.strip())
    else:
        lines.append("None.")

    lines.append("Evidence usage rule: this block is auxiliary, may be incomplete, and is not a ground-truth label.")
    return "\n".join(lines)


def should_include_description(task: str, policy: str) -> bool:
    if policy == "all":
        return True
    if policy == "non_detection":
        return task != "detection"
    return False


def build_input(
    code: str,
    evidence: Optional[Dict[str, Any]],
    description: Optional[str],
    variant: str,
    max_description_chars: int,
) -> str:
    if variant == "code_only":
        return code

    sections = [f"[Code]\n{code.strip()}"]
    if variant in {"cpg_evidence", "evidence", "evidence_desc", "random_lines"} and evidence is not None:
        sections.append(format_evidence_block(evidence))
    if variant in {"evidence_desc", "desc_only"} and description:
        clipped = description[:max_description_chars].strip()
        sections.append(f"[Generated Vulnerability Description]\n{clipped}")
    return "\n\n".join(section for section in sections if section.strip())


def should_inject_evidence(task: str, policy: str) -> bool:
    if policy == "all":
        return True
    if policy == "description_only":
        return task == "description"
    if policy == "non_detection":
        return task != "detection"
    if policy == "detection_only":
        return task == "detection"
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build direct-evidence GRAML datasets by injecting structured static/graph evidence into sample inputs."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--variant", choices=["code_only", "cpg_evidence", "evidence", "evidence_desc", "desc_only", "random_lines"], default="cpg_evidence")
    parser.add_argument("--evidence_path", default=None, help="Optional JSON evidence file keyed by index/idx/id or normalized code.")
    parser.add_argument("--description_path", default=None, help="Optional ToT/GPT description JSON, e.g. ToT_description.json.")
    parser.add_argument("--description_policy", choices=["never", "non_detection", "all"], default="never")
    parser.add_argument("--fallback_evidence", choices=["heuristic", "none"], default="none")
    parser.add_argument("--max_suspicious_lines", type=int, default=8)
    parser.add_argument("--max_relations", type=int, default=12)
    parser.add_argument("--max_description_chars", type=int, default=1200)
    parser.add_argument("--allow_label_evidence_for_detection", action="store_true")
    parser.add_argument(
        "--evidence_task_policy",
        choices=["all", "description_only", "non_detection", "detection_only", "none"],
        default="all",
    )
    parser.add_argument("--preserve_raw_when_no_evidence", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.variant == "cpg_evidence" and not args.evidence_path:
        raise ValueError("--variant cpg_evidence requires --evidence_path with Joern/CPG-derived evidence.")

    data = read_json(args.input_path)
    evidence_by_key, evidence_by_hash = load_evidence_maps(args.evidence_path)
    description_by_hash = load_description_map(args.description_path)

    output = []
    stats = Counter()
    for sample_index, sample in enumerate(data):
        task = get_task(sample)
        code = get_code(sample)
        external = find_external_evidence(sample, evidence_by_key, evidence_by_hash)
        allow_label_evidence = args.allow_label_evidence_for_detection or task != "detection"
        inject_evidence = should_inject_evidence(task, args.evidence_task_policy)

        evidence = None
        if not inject_evidence:
            stats["evidence_skipped_by_task_policy"] += 1
        elif args.variant == "random_lines":
            evidence = build_random_line_evidence(code, args.max_suspicious_lines, args.seed, sample_index)
            stats["evidence_random"] += 1
        elif args.variant in {"cpg_evidence", "evidence", "evidence_desc"}:
            if external:
                evidence = build_cpg_evidence_from_graph(
                    code,
                    external,
                    allow_label_evidence,
                    args.max_suspicious_lines,
                    args.max_relations,
                )
                if evidence:
                    stats["evidence_cpg_graph"] += 1
                elif args.variant == "cpg_evidence":
                    stats["evidence_missing"] += 1
                else:
                    evidence = merge_external_evidence(
                        external,
                        allow_label_evidence,
                        args.max_suspicious_lines,
                        args.max_relations,
                    )
                    stats["evidence_external_text"] += 1 if evidence else 0
            elif args.fallback_evidence == "heuristic":
                evidence = build_heuristic_evidence(code, args.max_suspicious_lines, args.max_relations)
                stats["evidence_heuristic"] += 1
            else:
                stats["evidence_missing"] += 1

        description = None
        if args.variant in {"evidence_desc", "desc_only"} and should_include_description(task, args.description_policy):
            description = description_by_hash.get(normalized_hash(code))
            if description:
                stats["description_matched"] += 1
            else:
                stats["description_missing"] += 1

        new_sample = dict(sample)
        if evidence is None and args.preserve_raw_when_no_evidence:
            new_sample["input"] = code
        else:
            new_sample["input"] = build_input(code, evidence, description, args.variant, args.max_description_chars)
        new_sample["_evidence_variant"] = args.variant
        if evidence:
            new_sample["_evidence_source"] = evidence.get("evidence_source", "")
        output.append(new_sample)
        stats[f"task_{task or 'unknown'}"] += 1

    write_json(args.output_path, output)
    stats["total"] = len(output)
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
