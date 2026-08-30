import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


COMMENT_PREFIXES = ("//", "/*", "*")
CPP_HINT_PATTERNS = [
    r"::",
    r"\b(class|namespace|template|typename|operator)\b",
    r"\b(public|private|protected)\s*:",
    r"\bstd::",
    r"\b[A-Za-z_][A-Za-z0-9_]*\s*&\s*[A-Za-z_][A-Za-z0-9_]*",
    r"<[^>\n]+>\s*&\s*[A-Za-z_][A-Za-z0-9_]*",
    r"\)\s*:",
    r"\)\s*(const|override|final)\s*(?:\{|;)",
]

JOERN_TOKEN_REPLACEMENTS = {
    "OMX_IN": "",
    "OMX_OUT": "",
    "OMX_INOUT": "",
    "IN": "",
    "OUT": "",
    "INOUT": "",
    "TSRMLS_DC": "",
    "TSRMLS_CC": "",
    "TSRMLS_C": "",
    "TSRMLS_D": "",
    "ZEND_FILE_LINE_DC": "",
    "ZEND_FILE_LINE_CC": "",
    "UNSERIALIZE_PARAMETER": "void *unserialize_data",
    "__user": "",
    "__kernel": "",
    "__iomem": "",
    "__force": "",
    "__init": "",
    "__exit": "",
    "__net_init": "",
    "__net_exit": "",
    "__devinit": "",
    "__devexit": "",
    "__ref": "",
    "__read_mostly": "",
    "__must_check": "",
    "__maybe_unused": "",
    "__always_inline": "inline",
    "zend_always_inline": "inline",
}


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as file:
        data = json.loads(file.read().decode("utf-8-sig"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list or a dict with top-level data.")
    return data


def write_json(path: str, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def normalized_hash(text: Any) -> str:
    return hashlib.md5(" ".join(str(text or "").split()).encode("utf-8")).hexdigest()


def get_code(sample: Dict[str, Any]) -> str:
    return str(sample.get("input", sample.get("Input", sample.get("code", ""))))


def get_task(sample: Dict[str, Any]) -> str:
    return str(sample.get("Task", sample.get("task", sample.get("category", sample.get("type", ""))))).strip().lower()


def sample_keys(sample: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    keys = {"sample_id": fallback_id, "input_hash": normalized_hash(get_code(sample))}
    for key in ["index", "idx", "id"]:
        if key in sample and sample[key] is not None:
            keys[key] = sample[key]
    return keys


def is_valid_code_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith(COMMENT_PREFIXES)


def line_map(code: str) -> Dict[int, str]:
    return {index + 1: line.rstrip() for index, line in enumerate(code.splitlines())}


def parse_line_numbers(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(match) for match in re.findall(r"\b(?:L|line\s*)?(\d+)\b", value, flags=re.IGNORECASE)]
    if isinstance(value, list):
        result: List[int] = []
        for item in value:
            result.extend(parse_line_numbers(item))
        return result
    if isinstance(value, dict):
        for key in ["line", "lineno", "line_no", "line_number", "start_line"]:
            if key in value:
                return parse_line_numbers(value[key])
    return []


def parse_location_lines_from_output(sample: Dict[str, Any]) -> List[int]:
    candidates = []
    for key in ["vulnerable_lines", "Vulnerable Lines", "location", "Location", "vul_lines", "lines"]:
        candidates.extend(parse_line_numbers(sample.get(key)))
    if get_task(sample) == "location":
        candidates.extend(parse_line_numbers(sample.get("output", sample.get("Output", ""))))

    code_lines = line_map(get_code(sample))
    output = str(sample.get("output", sample.get("Output", ""))).strip()
    if get_task(sample) == "location" and output and not candidates:
        output_rows = [row.strip() for row in output.splitlines() if row.strip()]
        for line_no, code_line in code_lines.items():
            normalized_code = " ".join(code_line.strip().split())
            for output_row in output_rows:
                if normalized_code and normalized_code == " ".join(output_row.split()):
                    candidates.append(line_no)

    return sorted(set(line_no for line_no in candidates if line_no > 0))


def infer_source_extension(code: str, extension: str) -> str:
    if extension != "auto":
        return extension
    return ".cpp" if any(re.search(pattern, code) for pattern in CPP_HINT_PATTERNS) else ".c"


def sanitize_code_for_joern(code: str) -> str:
    sanitized = code
    for token, replacement in JOERN_TOKEN_REPLACEMENTS.items():
        sanitized = re.sub(rf"\b{re.escape(token)}\b", replacement, sanitized)
    lines = sanitized.splitlines()
    first_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_index is None:
        return sanitized

    first_line = lines[first_index]
    stripped_first = first_line.strip()
    next_index = next((index for index in range(first_index + 1, len(lines)) if lines[index].strip()), None)

    def has_missing_return_type(line: str) -> bool:
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_:~]*\s*\(", line.strip()))

    explicit_match = re.match(r"^(\s*)explicit\s+([A-Za-z_][A-Za-z0-9_:~]*)(.*)$", first_line)
    if explicit_match:
        leading, name, rest = explicit_match.groups()
        flattened_name = name.replace("::", "_").replace("~", "destructor_")
        lines[first_index] = f"{leading}int {flattened_name}{rest}"
    elif re.match(r"^[A-Za-z_][A-Za-z0-9_:~]*$", stripped_first) and next_index is not None:
        if lines[next_index].strip().startswith("("):
            lines[first_index] = first_line.replace(stripped_first, f"int {stripped_first}", 1)
    elif has_missing_return_type(first_line):
        normalized_name = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_:~]*)(.*)$", first_line)
        if normalized_name:
            leading, name, rest = normalized_name.groups()
            flattened_name = name.replace("::", "_").replace("~", "destructor_")
            lines[first_index] = f"{leading}int {flattened_name}{rest}"

    initializer_index = None
    for index in range(first_index + 1, len(lines)):
        stripped = lines[index].lstrip()
        if stripped.startswith(":"):
            initializer_index = index
            break
        if "{" in stripped:
            break
    if initializer_index is not None:
        for index in range(initializer_index, len(lines)):
            if "{" in lines[index]:
                lines[index] = lines[index][lines[index].index("{") :]
                break
            lines[index] = ""

    return "\n".join(lines) + ("\n" if sanitized.endswith("\n") else "")


def safe_filename(sample_number: int, extension: str) -> str:
    return f"sample_{sample_number:06d}{extension}"


def parse_task_filter(task_filter: str) -> Optional[set]:
    if not task_filter or task_filter.strip().lower() == "all":
        return None
    return {task.strip().lower() for task in task_filter.split(",") if task.strip()}


def is_benign_description(sample: Dict[str, Any]) -> bool:
    return get_task(sample) == "description" and str(sample.get("output", sample.get("Output", ""))).strip() in {
        "There is no vulnerability.",
        "No vulnerability.",
        "N/A",
        "",
    }


def load_dataset_samples(
    input_paths: Sequence[str],
    limit: Optional[int],
    task_filter: Optional[set],
    skip_benign_description: bool,
) -> List[Dict[str, Any]]:
    records = []
    for path in input_paths:
        split_name = Path(path).stem
        for local_index, sample in enumerate(read_json(path)):
            if limit is not None and len(records) >= limit:
                return records
            task = get_task(sample)
            if task_filter is not None and task not in task_filter:
                continue
            if skip_benign_description and is_benign_description(sample):
                continue
            code = get_code(sample)
            if not code.strip():
                continue
            sample_number = len(records)
            records.append(
                {
                    "sample_number": sample_number,
                    "split": split_name,
                    "source_path": path,
                    "local_index": local_index,
                    "sample": sample,
                    "code": code,
                }
            )
    return records


def materialize_sources(
    records: List[Dict[str, Any]],
    source_dir: Path,
    extension: str,
) -> Dict[str, Dict[str, Any]]:
    source_dir.mkdir(parents=True, exist_ok=True)
    file_map: Dict[str, Dict[str, Any]] = {}
    for record in records:
        source_extension = infer_source_extension(record["code"], extension)
        file_name = safe_filename(record["sample_number"], source_extension)
        source_path = source_dir / file_name
        source_path.write_text(sanitize_code_for_joern(record["code"]), encoding="utf-8")
        record["generated_source_file"] = str(source_path)
        record["generated_source_name"] = file_name
        record["generated_source_extension"] = source_extension
        file_map[file_name] = record
        file_map[str(source_path.resolve())] = record
    return file_map


def resolve_tool(tool_arg: Optional[str], tool_name: str) -> str:
    if tool_arg:
        return tool_arg
    found = shutil.which(tool_name)
    if not found:
        raise FileNotFoundError(f"Cannot find {tool_name}. Add Joern to PATH or pass --{tool_name.replace('-', '_')}.")
    return found


def run_command(command: List[str], cwd: Optional[Path] = None, allow_fail: bool = False) -> subprocess.CompletedProcess:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result


def run_joern_parse(joern_parse: str, source_dir: Path, work_dir: Path, cpg_path: Path) -> Path:
    result = run_command([joern_parse, str(source_dir), "--output", str(cpg_path)], cwd=work_dir, allow_fail=True)
    if result.returncode == 0 and cpg_path.exists():
        return cpg_path

    print("Retrying joern-parse without --output because this Joern version may not support it.")
    default_cpg = work_dir / "cpg.bin"
    if default_cpg.exists():
        default_cpg.unlink()
    run_command([joern_parse, str(source_dir)], cwd=work_dir)
    if not default_cpg.exists():
        raise FileNotFoundError(f"joern-parse finished but {default_cpg} was not created.")
    if default_cpg != cpg_path:
        shutil.move(str(default_cpg), str(cpg_path))
    return cpg_path


def find_slice_output(prefix: Path) -> Path:
    candidates = [
        prefix,
        prefix.with_suffix(".json"),
        prefix.parent / f"{prefix.name}.json",
        prefix.parent / f"{prefix.name}.data-flow.json",
    ]
    candidates.extend(sorted(prefix.parent.glob(f"{prefix.name}*.json")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find joern-slice JSON output for prefix {prefix}")


def run_joern_slice_data_flow(joern_slice: str, cpg_path: Path, output_prefix: Path) -> Optional[Path]:
    result = run_command(
        [joern_slice, "data-flow", "-o", str(output_prefix), str(cpg_path)],
        cwd=output_prefix.parent,
        allow_fail=True,
    )
    if result.returncode != 0:
        print("WARNING: joern-slice data-flow failed; continuing without data-flow slices.", file=sys.stderr)
        return None
    return find_slice_output(output_prefix)


def run_control_edge_script(joern: str, cpg_path: Path, script_path: Path, output_path: Path) -> Optional[Path]:
    runtime_script = output_path.parent / "_joern_export_line_edges_runtime.sc"
    script_body = script_path.read_text(encoding="utf-8-sig")
    runtime_script.write_text(
        "\n".join(
            [
                f"val cpgFile = {json.dumps(cpg_path.as_posix())}",
                f"val outFile = {json.dumps(output_path.as_posix())}",
                script_body,
            ]
        ),
        encoding="utf-8",
    )
    result = run_command(
        [
            joern,
            "--script",
            str(runtime_script),
        ],
        cwd=output_path.parent,
        allow_fail=True,
    )
    if result.returncode != 0 or not output_path.exists():
        print("WARNING: Joern control-edge script failed; continuing without control-dependence edges.", file=sys.stderr)
        return None
    return output_path


def iter_graph_objects(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, dict):
        if "nodes" in payload and "edges" in payload:
            yield payload
        for value in payload.values():
            yield from iter_graph_objects(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from iter_graph_objects(item)


def node_id(value: Dict[str, Any]) -> Optional[str]:
    for key in ["id", "nodeId", "node_id", "_id"]:
        if key in value:
            return str(value[key])
    return None


def node_line(value: Dict[str, Any]) -> Optional[int]:
    for key in ["lineNumber", "line_number", "line", "lineNo", "startLine"]:
        if key in value and value[key] not in [None, ""]:
            try:
                return int(value[key])
            except ValueError:
                pass
    return None


def node_file(value: Dict[str, Any]) -> str:
    for key in ["parentFile", "file", "filename", "fileName", "sourceFile"]:
        if key in value and value[key]:
            return str(value[key])
    return ""


def edge_endpoint(edge: Dict[str, Any], names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in edge:
            endpoint = edge[name]
            if isinstance(endpoint, dict):
                return node_id(endpoint)
            return str(endpoint)
    return None


def file_lookup_key(path_text: str) -> str:
    if not path_text:
        return ""
    return Path(path_text).name


def record_for_file(file_map: Dict[str, Dict[str, Any]], file_text: str) -> Optional[Dict[str, Any]]:
    if not file_text:
        return None
    return file_map.get(file_text) or file_map.get(file_lookup_key(file_text)) or file_map.get(str(Path(file_text).resolve()))


def parse_data_flow_edges(slice_path: Path, file_map: Dict[str, Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    payload = json.loads(slice_path.read_text(encoding="utf-8"))
    edges_by_sample: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for graph in iter_graph_objects(payload):
        nodes = graph.get("nodes") or []
        raw_edges = graph.get("edges") or []
        node_by_id = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            ident = node_id(node)
            if ident is not None:
                node_by_id[ident] = node

        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            src_id = edge_endpoint(raw_edge, ["src", "source", "from", "inNode", "outNode"])
            dst_id = edge_endpoint(raw_edge, ["dst", "target", "to", "outNode", "inNode"])
            if not src_id or not dst_id or src_id == dst_id:
                continue
            src_node = node_by_id.get(src_id)
            dst_node = node_by_id.get(dst_id)
            if not src_node or not dst_node:
                continue
            src_line = node_line(src_node)
            dst_line = node_line(dst_node)
            if not src_line or not dst_line or src_line == dst_line:
                continue

            src_record = record_for_file(file_map, node_file(src_node))
            dst_record = record_for_file(file_map, node_file(dst_node))
            if not src_record or src_record is not dst_record:
                continue

            edges_by_sample[src_record["sample_number"]].append(
                {
                    "src": src_line,
                    "dst": dst_line,
                    "type": str(raw_edge.get("label", raw_edge.get("type", "DATA_FLOW"))).upper(),
                    "source": "joern-slice:data-flow",
                }
            )
    return edges_by_sample


def parse_control_edges(control_path: Optional[Path], file_map: Dict[str, Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    if control_path is None:
        return defaultdict(list)
    payload = json.loads(control_path.read_text(encoding="utf-8"))
    edges_by_sample: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    if not isinstance(payload, list):
        return edges_by_sample

    for raw_edge in payload:
        if not isinstance(raw_edge, dict):
            continue
        src = raw_edge.get("src", {})
        dst = raw_edge.get("dst", {})
        if not isinstance(src, dict) or not isinstance(dst, dict):
            continue
        src_line = node_line(src)
        dst_line = node_line(dst)
        if not src_line or not dst_line or src_line == dst_line:
            continue
        src_record = record_for_file(file_map, node_file(src))
        dst_record = record_for_file(file_map, node_file(dst))
        if not src_record or src_record is not dst_record:
            continue
        edges_by_sample[src_record["sample_number"]].append(
            {
                "src": src_line,
                "dst": dst_line,
                "type": str(raw_edge.get("type", "CONTROL_DEPENDENCE")).upper(),
                "source": f"joern-cpgql:{str(raw_edge.get('type', 'CONTROL_DEPENDENCE')).lower()}",
            }
        )
    return edges_by_sample


def dedupe_edges(edges: Iterable[Dict[str, Any]], max_edges: int) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for edge in edges:
        key = (int(edge["src"]), int(edge["dst"]), str(edge.get("type", "")))
        if key in seen:
            continue
        seen.add(key)
        result.append({"src": key[0], "dst": key[1], "type": key[2], "source": edge.get("source", "joern")})
        if len(result) >= max_edges:
            break
    return result


def high_degree_context_lines(
    code: str,
    edges: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    lines = line_map(code)
    valid_lines = {line_no for line_no, text in lines.items() if is_valid_code_line(text)}
    degree = Counter()
    for edge in edges:
        if edge["src"] in valid_lines:
            degree[edge["src"]] += 1
        if edge["dst"] in valid_lines:
            degree[edge["dst"]] += 1
    selected = [line_no for line_no, _ in degree.most_common(top_k)]
    return [
        {
            "line": line_no,
            "code": lines.get(line_no, "").strip(),
            "role": "high_degree_cpg_context",
            "graph_score": degree[line_no],
        }
        for line_no in sorted(selected)
    ]


def paper_style_critical_lines(
    code: str,
    edges: List[Dict[str, Any]],
    vulnerable_lines: List[int],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    lines = line_map(code)
    valid_lines = {line_no for line_no, text in lines.items() if is_valid_code_line(text)}
    vulnerable = sorted(set(line_no for line_no in vulnerable_lines if line_no in valid_lines))
    if not vulnerable:
        return [], []

    outgoing = defaultdict(set)
    for edge in edges:
        if edge["src"] in valid_lines and edge["dst"] in valid_lines and edge["src"] != edge["dst"]:
            outgoing[edge["src"]].add(edge["dst"])

    adjacency = {}
    for line_no in valid_lines:
        adjacency[line_no] = set(outgoing[line_no]) if outgoing.get(line_no) else set(valid_lines - {line_no})

    first_hop = set()
    for vulnerable_line in vulnerable:
        first_hop.update(adjacency.get(vulnerable_line, set()))
        first_hop.update(line_no for line_no, dsts in adjacency.items() if vulnerable_line in dsts)
    first_hop -= set(vulnerable)

    second_scores = Counter()
    for first_line in first_hop:
        for candidate in adjacency.get(first_line, set()):
            if candidate not in first_hop and candidate not in vulnerable:
                second_scores[candidate] += 1

    max_second = max(1, len(lines) // 5)
    second_hop = {line_no for line_no, _ in second_scores.most_common(max_second)}

    selected = []
    roles = [
        ("vulnerable_seed", vulnerable),
        ("first_hop_cpg_neighbor", sorted(first_hop)),
        ("second_hop_ranked_by_cpg_connectivity", sorted(second_hop)),
    ]
    selected_lines = set()
    for role, line_numbers in roles:
        for line_no in line_numbers:
            if line_no in selected_lines:
                continue
            selected_lines.add(line_no)
            selected.append(
                {
                    "line": line_no,
                    "code": lines.get(line_no, "").strip(),
                    "role": role,
                    "graph_score": second_scores.get(line_no, 1 if role == "first_hop_cpg_neighbor" else 0),
                }
            )
    selected = sorted(selected, key=lambda item: item["line"])[:top_k]
    selected_set = {item["line"] for item in selected}
    relations = [
        f"L{edge['src']} --{edge['type']}--> L{edge['dst']}"
        for edge in edges
        if edge["src"] in selected_set and edge["dst"] in selected_set
    ]
    return selected, relations


def build_evidence_records(
    records: List[Dict[str, Any]],
    data_edges: Dict[int, List[Dict[str, Any]]],
    control_edges: Dict[int, List[Dict[str, Any]]],
    max_edges_per_sample: int,
    context_top_k: int,
    include_label_lines: bool,
) -> List[Dict[str, Any]]:
    output = []
    for record in records:
        sample_number = record["sample_number"]
        sample = record["sample"]
        code = record["code"]
        edges = dedupe_edges(
            data_edges.get(sample_number, []) + control_edges.get(sample_number, []),
            max_edges=max_edges_per_sample,
        )
        data_only = [edge for edge in edges if "DATA" in edge["type"] or "REACH" in edge["type"]]
        control_only = [edge for edge in edges if "CONTROL" in edge["type"]]

        evidence = {
            **sample_keys(sample, fallback_id=f"{record['split']}:{record['local_index']}"),
            "split": record["split"],
            "local_index": record["local_index"],
            "source_file": record["generated_source_name"],
            "evidence_source": "joern_cpg",
            "input": code,
            "valid_lines": sorted(line_no for line_no, text in line_map(code).items() if is_valid_code_line(text)),
            "cpg_edges": edges,
            "data_flow_edges": data_only,
            "control_flow_edges": control_only,
            "graph_context_lines": high_degree_context_lines(code, edges, top_k=context_top_k),
        }

        vulnerable_lines = parse_location_lines_from_output(sample)
        if include_label_lines and vulnerable_lines:
            critical_lines, relations = paper_style_critical_lines(
                code,
                edges,
                vulnerable_lines=vulnerable_lines,
                top_k=context_top_k,
            )
            evidence["vulnerable_lines"] = vulnerable_lines
            evidence["critical_lines"] = critical_lines
            evidence["line_relations"] = relations

        output.append(evidence)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Joern/CPG-derived line-level evidence for GRAML JSON datasets."
    )
    parser.add_argument("--input_path", action="append", required=True, help="Dataset JSON path. Can be repeated.")
    parser.add_argument("--output_path", required=True, help="Output evidence JSON path.")
    parser.add_argument("--work_dir", default="joern_work", help="Temporary working directory.")
    parser.add_argument("--extension", default="auto", choices=["auto", ".c", ".cpp", ".cc", ".cxx"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task_filter", default="all", help="Comma-separated tasks to include, e.g. description.")
    parser.add_argument("--skip_benign_description", action="store_true")
    parser.add_argument("--context_top_k", type=int, default=12)
    parser.add_argument("--max_edges_per_sample", type=int, default=200)
    parser.add_argument("--include_label_lines", action="store_true", help="Include vulnerable_lines/critical_lines for ToT/location use. Do not inject these into detection test inputs.")
    parser.add_argument("--reuse_cpg", action="store_true", help="Reuse existing cpg.bin and extracted JSON in work_dir if present.")
    parser.add_argument("--enable_joern_slice_data_flow", action="store_true", help="Also run joern-slice data-flow. Disabled by default because it can be very slow on Windows.")
    parser.add_argument("--joern_parse", default=None)
    parser.add_argument("--joern_slice", default=None)
    parser.add_argument("--joern", default=None)
    parser.add_argument("--control_script", default=None)
    args = parser.parse_args()

    joern_parse = resolve_tool(args.joern_parse, "joern-parse")
    joern_slice = resolve_tool(args.joern_slice, "joern-slice")
    joern = resolve_tool(args.joern, "joern")

    input_paths = [str(Path(path).resolve()) for path in args.input_path]
    output_path = str(Path(args.output_path).resolve())
    work_dir = Path(args.work_dir).resolve()
    source_dir = work_dir / "src"
    cpg_path = work_dir / "cpg.bin"
    data_flow_prefix = work_dir / "data_flow_slices"
    control_json = work_dir / "control_edges.json"
    script_path = (
        Path(args.control_script).resolve()
        if args.control_script
        else Path(__file__).with_name("joern_export_control_edges.sc").resolve()
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    records = load_dataset_samples(
        input_paths,
        args.limit,
        task_filter=parse_task_filter(args.task_filter),
        skip_benign_description=args.skip_benign_description,
    )
    if not records:
        raise ValueError("No code samples found.")
    print(f"Loaded {len(records)} code samples.")

    if source_dir.exists() and not args.reuse_cpg:
        shutil.rmtree(source_dir)
    file_map = materialize_sources(records, source_dir, args.extension)
    print(f"Wrote source files to {source_dir}")

    if not args.reuse_cpg or not cpg_path.exists():
        run_joern_parse(joern_parse, source_dir, work_dir, cpg_path)
    else:
        print(f"Reusing existing CPG: {cpg_path}")

    if args.enable_joern_slice_data_flow:
        if args.reuse_cpg:
            try:
                data_flow_json = find_slice_output(data_flow_prefix)
            except FileNotFoundError:
                data_flow_json = run_joern_slice_data_flow(joern_slice, cpg_path, data_flow_prefix)
        else:
            data_flow_json = run_joern_slice_data_flow(joern_slice, cpg_path, data_flow_prefix)
    else:
        print("Skipping joern-slice data-flow; using Joern CPGQL line edges instead.")
        data_flow_json = None

    if args.reuse_cpg and control_json.exists():
        control_path = control_json
    else:
        control_path = run_control_edge_script(joern, cpg_path, script_path, control_json)

    data_edges = parse_data_flow_edges(data_flow_json, file_map) if data_flow_json else defaultdict(list)
    control_edges = parse_control_edges(control_path, file_map)
    evidence = build_evidence_records(
        records,
        data_edges,
        control_edges,
        max_edges_per_sample=args.max_edges_per_sample,
        context_top_k=args.context_top_k,
        include_label_lines=args.include_label_lines,
    )
    write_json(output_path, evidence)

    stats = {
        "samples": len(evidence),
        "samples_with_edges": sum(1 for item in evidence if item["cpg_edges"]),
        "data_edges": sum(len(item["data_flow_edges"]) for item in evidence),
        "control_edges": sum(len(item["control_flow_edges"]) for item in evidence),
        "with_vulnerable_lines": sum(1 for item in evidence if item.get("vulnerable_lines")),
        "output_path": output_path,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
