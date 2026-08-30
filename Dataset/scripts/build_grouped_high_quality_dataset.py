import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SOURCE_YEARS = [str(year) for year in range(2019, 2027)]
MAX_INPUT_CHARS = 8000
MAX_SAMPLES_PER_CVE = 4
MAX_SAMPLES_PER_CVE_LABEL = 2
MAX_NEW_SAMPLES_PER_PROJECT = {"train": 10**9, "valid": 10**9, "test": 10**9}
PROJECT_CAP_OVERRIDES: Dict[str, Dict[str, int]] = {}
BROKEN_FIRST_TOKEN = re.compile(
    r"^(oid\b|atic\b|har\b|onst\b|truct\b|nt\b|f\s*\(|lass\b|ublic\b|rivate\b)"
)


def read_json_array(path: pathlib.Path) -> Tuple[List[Dict[str, Any]], bool]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
        repaired = False
    except json.JSONDecodeError:
        data = json.loads(text.rstrip() + "\n]")
        repaired = True
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data, repaired


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_code(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def code_hash(text: str) -> str:
    return hashlib.sha1(normalize_code(text).encode("utf-8")).hexdigest()


def cve_year(cve: Any) -> str:
    match = re.match(r"^CVE-(\d{4})-", str(cve or ""))
    return match.group(1) if match else ""


def pair_key(item: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(item.get("repo", "")),
        str(item.get("cve", "")),
        str(item.get("commit", "")),
        str(item.get("file", "")),
    )


def task_name(item: Dict[str, Any]) -> str:
    return str(item.get("Task", item.get("task", ""))).strip().lower()


def output_label(item: Dict[str, Any]) -> str:
    return str(item.get("output", item.get("Output", ""))).strip()


def first_line(text: str) -> str:
    lines = (text or "").lstrip().splitlines()
    return lines[0] if lines else ""


def has_broken_first_token(text: str) -> bool:
    return bool(BROKEN_FIRST_TOKEN.match(first_line(text)))


def line_category(item: Dict[str, Any]) -> str:
    before = item.get("before") or ""
    after = item.get("after") or ""
    vulnerable_lines = item.get("vulnerable_lines") or []
    if not vulnerable_lines:
        return "no_vulnerable_lines"
    before_matches = 0
    after_matches = 0
    neither_matches = 0
    for line in vulnerable_lines:
        code = line.get("code", "") if isinstance(line, dict) else ""
        before_matches += int(bool(code) and code in before)
        after_matches += int(bool(code) and code in after)
        neither_matches += int(bool(code) and code not in before and code not in after)
    if before_matches == 0 and after_matches > 0:
        return "only_after_patch_like"
    if before_matches > 0 and after_matches == 0:
        return "only_before_like"
    if before_matches > 0 and after_matches > 0:
        return "both_or_context"
    if neither_matches == len(vulnerable_lines):
        return "neither"
    return "mixed"


def load_base_train(path: pathlib.Path) -> Tuple[List[Dict[str, Any]], Set[str], str]:
    items = read_json(path)
    hashes = {code_hash(str(item.get("input", item.get("Input", "")))) for item in items}
    instructions = [
        str(item.get("instruction", "")).strip()
        for item in items
        if task_name(item) == "detection" and str(item.get("instruction", "")).strip()
    ]
    if not instructions:
        raise ValueError("No detection instruction in base training set")
    return items, hashes, collections.Counter(instructions).most_common(1)[0][0]


def clean_base_detection_split(
    items: List[Dict[str, Any]],
    blocked_hashes: Set[str],
    instruction: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    cleaned = []
    seen_hashes = set()
    removed_blocked = 0
    removed_duplicates = 0
    for item in items:
        if task_name(item) != "detection":
            raise ValueError("Base validation/test contains a non-detection task")
        if output_label(item) not in {"Yes", "No"}:
            raise ValueError("Base validation/test contains an invalid label")
        item_hash = code_hash(str(item.get("input", item.get("Input", ""))))
        if item_hash in blocked_hashes:
            removed_blocked += 1
            continue
        if item_hash in seen_hashes:
            removed_duplicates += 1
            continue
        seen_hashes.add(item_hash)
        normalized = dict(item)
        normalized["instruction"] = instruction
        normalized["Task"] = "detection"
        cleaned.append(normalized)
    return cleaned, {
        "removed_blocked_hashes": removed_blocked,
        "removed_duplicate_hashes": removed_duplicates,
    }


def load_raw_records(raw_dir: pathlib.Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = []
    parse_status = {}
    for path in sorted(raw_dir.glob("*.json")):
        items, repaired = read_json_array(path)
        parse_status[path.name] = {
            "count": len(items),
            "repaired_missing_closing_array": repaired,
        }
        for index, item in enumerate(items):
            copied = dict(item)
            copied["_source_file"] = path.name
            copied["_source_index"] = index
            copied["_year"] = cve_year(item.get("cve"))
            copied["_pair_key"] = pair_key(item)
            copied["_before_hash"] = code_hash(str(item.get("before", "")))
            copied["_after_hash"] = code_hash(str(item.get("after", "")))
            copied["_line_category"] = line_category(item)
            records.append(copied)
    return records, parse_status


def clean_record(record: Dict[str, Any], field: str) -> bool:
    code = str(record.get(field, "") or "")
    required = ["cve", "repo", "commit", "file", "before", "after"]
    return (
        record.get("_year") in SOURCE_YEARS
        and all(str(record.get(key, "")).strip() for key in required)
        and record.get("before") != record.get("after")
        and 80 <= len(code) <= MAX_INPUT_CHARS
        and not has_broken_first_token(code)
    )


def build_candidates(
    records: List[Dict[str, Any]],
    base_split_hashes: Set[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    excluded_cves = {
        record.get("cve")
        for record in records
        if record.get("_before_hash") in base_split_hashes
        or record.get("_after_hash") in base_split_hashes
    }
    raw_candidates = []
    hash_to_cves = collections.defaultdict(set)
    for record in records:
        if record.get("cve") in excluded_cves:
            continue
        for label, field, hash_field in [
            ("Yes", "before", "_before_hash"),
            ("No", "after", "_after_hash"),
        ]:
            if not clean_record(record, field):
                continue
            code_hash_value = record[hash_field]
            hash_to_cves[code_hash_value].add(str(record.get("cve", "")))
            raw_candidates.append(
                {
                    "record": record,
                    "label": label,
                    "field": field,
                    "code_hash": code_hash_value,
                    "code": str(record.get(field, "")),
                }
            )
    ambiguous_hashes = {
        value for value, cves in hash_to_cves.items() if len(cves) > 1
    }
    deduplicated = {}
    for candidate in raw_candidates:
        if candidate["code_hash"] in ambiguous_hashes:
            continue
        record = candidate["record"]
        key = candidate["code_hash"]
        existing = deduplicated.get(key)
        if existing is None or (
            record["_source_file"], record["_source_index"]
        ) < (
            existing["record"]["_source_file"],
            existing["record"]["_source_index"],
        ):
            deduplicated[key] = candidate
    groups: Dict[str, Dict[str, Any]] = {}
    for candidate in deduplicated.values():
        record = candidate["record"]
        cve = str(record["cve"])
        group = groups.setdefault(
            cve,
            {
                "cve": cve,
                "year": record["_year"],
                "repo": str(record["repo"]),
                "pairs": {},
            },
        )
        pair = record["_pair_key"]
        pair_entry = group["pairs"].setdefault(pair, {})
        pair_entry[candidate["label"]] = candidate
    stats = {
        "excluded_cves_matching_base_splits": len(excluded_cves),
        "ambiguous_code_hashes_removed": len(ambiguous_hashes),
        "candidate_sides_after_cleaning": len(raw_candidates),
        "candidate_sides_after_ambiguous_hash_filter": len(deduplicated),
        "candidate_cves": len(groups),
        "candidate_pairs": sum(len(group["pairs"]) for group in groups.values()),
    }
    return groups, stats


def allocate_proportional(
    total: int,
    capacities: Dict[str, int],
) -> Dict[str, int]:
    if total < 0 or total > sum(capacities.values()):
        raise ValueError(f"Cannot allocate {total} from capacities {capacities}")
    capacity_total = sum(capacities.values())
    raw = {
        key: (value * total / capacity_total if capacity_total else 0.0)
        for key, value in capacities.items()
    }
    allocation = {key: min(capacities[key], int(raw[key])) for key in capacities}
    remaining = total - sum(allocation.values())
    order = sorted(
        capacities,
        key=lambda key: (raw[key] - allocation[key], capacities[key], key),
        reverse=True,
    )
    for key in order:
        if remaining <= 0:
            break
        if allocation[key] < capacities[key]:
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Largest-remainder allocation did not converge")
    return allocation


def build_quotas(
    groups: Dict[str, Dict[str, Any]],
    raw_targets: Dict[str, int],
    raw_label_targets: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    pair_capacities = collections.Counter()
    for group in groups.values():
        pair_capacities[group["year"]] += len(group["pairs"])
    year_by_split = {}
    remaining_capacity = dict(pair_capacities)
    for split in ["valid", "test", "train"]:
        year_by_split[split] = allocate_proportional(raw_targets[split], remaining_capacity)
        for year, amount in year_by_split[split].items():
            remaining_capacity[year] -= amount
    quotas = {split: {"Yes": {}, "No": {}} for split in raw_targets}
    for split, year_targets in year_by_split.items():
        yes_target = raw_label_targets[split]["Yes"]
        yes_base = {year: amount // 2 for year, amount in year_targets.items()}
        yes_remaining = yes_target - sum(yes_base.values())
        odd_years = sorted(
            year_targets,
            key=lambda year: (year_targets[year] % 2, year_targets[year], year),
            reverse=True,
        )
        while yes_remaining > 0:
            progress = False
            for year in odd_years:
                if yes_remaining <= 0:
                    break
                if yes_base[year] < year_targets[year]:
                    yes_base[year] += 1
                    yes_remaining -= 1
                    progress = True
            if not progress:
                raise ValueError(f"Could not allocate Yes quota for {split}")
        quotas[split]["Yes"] = yes_base
        quotas[split]["No"] = {
            year: year_targets[year] - yes_base[year] for year in year_targets
        }
    return quotas


def group_capacity(group: Dict[str, Any]) -> Dict[str, int]:
    yes_pairs = sum("Yes" in values for values in group["pairs"].values())
    no_pairs = sum("No" in values for values in group["pairs"].values())
    total_pairs = len(group["pairs"])
    return {
        "Yes": min(MAX_SAMPLES_PER_CVE_LABEL, yes_pairs),
        "No": min(MAX_SAMPLES_PER_CVE_LABEL, no_pairs),
        "total": min(MAX_SAMPLES_PER_CVE, total_pairs),
    }


def assign_groups(
    groups: Dict[str, Dict[str, Any]],
    quotas: Dict[str, Dict[str, Dict[str, int]]],
    seed: int,
) -> Dict[str, Set[str]]:
    assignments = {"train": set(), "valid": set(), "test": set()}
    capacity_by_split_year = {
        split: collections.Counter() for split in assignments
    }
    repo_counts = {
        split: collections.Counter() for split in assignments
    }
    repo_capacity = {
        split: collections.Counter() for split in assignments
    }
    groups_by_year = collections.defaultdict(list)
    for cve, group in groups.items():
        groups_by_year[group["year"]].append((cve, group))
    rng = random.Random(seed)
    for year in SOURCE_YEARS:
        year_groups = list(groups_by_year.get(year, []))
        rng.shuffle(year_groups)
        year_groups.sort(
            key=lambda value: (
                -group_capacity(value[1])["total"],
                value[1]["repo"],
                value[0],
            )
        )
        desired = {
            split: quotas[split]["Yes"].get(year, 0)
            + quotas[split]["No"].get(year, 0)
            for split in assignments
        }
        for cve, group in year_groups:
            capacity = group_capacity(group)["total"]
            eligible_splits = [
                split
                for split in assignments
                if repo_capacity[split][group["repo"]]
                < project_cap(split, group["repo"])
            ]
            if not eligible_splits:
                eligible_splits = list(assignments)
            split = min(
                eligible_splits,
                key=lambda value: (
                    repo_capacity[value][group["repo"]]
                    / max(1, project_cap(value, group["repo"])),
                    capacity_by_split_year[value][year]
                    / max(1, desired[value]),
                    repo_counts[value][group["repo"]],
                    len(assignments[value]),
                    value,
                ),
            )
            assignments[split].add(cve)
            capacity_by_split_year[split][year] += capacity
            repo_capacity[split][group["repo"]] += capacity
            repo_counts[split][group["repo"]] += 1
    return assignments


def select_for_split_year(
    groups: Dict[str, Dict[str, Any]],
    cves: Set[str],
    split: str,
    year: str,
    yes_target: int,
        no_target: int,
        repo_counts: collections.Counter,
) -> Optional[List[Tuple[Dict[str, Any], str]]]:
    group_list = [groups[cve] for cve in cves if groups[cve]["year"] == year]
    selected_pairs = set()
    selected_counts = collections.Counter()
    selected_label_counts = collections.Counter()
    selected: List[Tuple[Dict[str, Any], str]] = []
    remaining = {"Yes": yes_target, "No": no_target}
    while remaining["Yes"] or remaining["No"]:
        available_by_label = {"Yes": [], "No": []}
        for group in group_list:
            cve = group["cve"]
            if repo_counts[group["repo"]] >= project_cap(split, group["repo"]):
                continue
            if selected_counts[cve] >= MAX_SAMPLES_PER_CVE:
                continue
            for pair, values in group["pairs"].items():
                if pair in selected_pairs:
                    continue
                for label in ["Yes", "No"]:
                    if label not in values:
                        continue
                    if selected_label_counts[(cve, label)] >= MAX_SAMPLES_PER_CVE_LABEL:
                        continue
                    if remaining[label] > 0:
                        available_by_label[label].append((group, pair, values[label]))
        available_counts = {
            label: len(values) for label, values in available_by_label.items()
        }
        labels = [label for label in ["Yes", "No"] if remaining[label] > 0]
        if not labels or any(available_counts[label] < remaining[label] for label in labels):
            return None
        selected_label = min(
            labels,
            key=lambda label: (
                available_counts[label] - remaining[label],
                -remaining[label],
                label,
            ),
        )
        other_label = "No" if selected_label == "Yes" else "Yes"
        other_slack = available_counts[other_label] - remaining[other_label]
        options = available_by_label[selected_label]
        chosen = min(
            options,
            key=lambda value: (
                int(
                    other_slack <= 0
                    and other_label in value[2]
                ),
                repo_counts[value[0]["repo"]],
                selected_counts[value[0]["cve"]],
                selected_label_counts[(value[0]["cve"], selected_label)],
                value[0]["repo"],
                value[0]["cve"],
                value[1],
            ),
        )
        group, pair, candidate = chosen
        selected.append((candidate, selected_label))
        selected_pairs.add(pair)
        selected_counts[group["cve"]] += 1
        selected_label_counts[(group["cve"], selected_label)] += 1
        remaining[selected_label] -= 1
        repo_counts[group["repo"]] += 1
    return selected


def build_selection(
    groups: Dict[str, Dict[str, Any]],
    quotas: Dict[str, Dict[str, Dict[str, int]]],
    seed: int,
) -> Optional[Dict[str, Dict[str, List[Tuple[Dict[str, Any], str]]]]]:
    assignments = assign_groups(groups, quotas, seed)
    selection = {split: {"Yes": [], "No": []} for split in assignments}
    repo_counts = {split: collections.Counter() for split in assignments}
    for split in ["train", "valid", "test"]:
        for year in SOURCE_YEARS:
            chosen = select_for_split_year(
                groups,
                assignments[split],
                split,
                year,
                quotas[split]["Yes"].get(year, 0),
                quotas[split]["No"].get(year, 0),
                repo_counts[split],
            )
            if chosen is None:
                return None
            for candidate, label in chosen:
                selection[split][label].append((candidate, label))
    return selection


def make_sft(candidate: Dict[str, Any], instruction: str, label: str) -> Dict[str, str]:
    return {
        "instruction": instruction,
        "input": candidate["code"],
        "output": label,
        "Task": "detection",
    }


def count_labels(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(collections.Counter(output_label(item) for item in items))


def count_by(items: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(collections.Counter(str(item.get(key, "")) for item in items).most_common())


def gini(values: Iterable[int]) -> float:
    ordered = sorted(values)
    if not ordered or sum(ordered) == 0:
        return 0.0
    total = sum(ordered)
    size = len(ordered)
    return sum((2 * index - size - 1) * value for index, value in enumerate(ordered, 1)) / (size * total)


def project_cap(split: str, repo: str) -> int:
    return PROJECT_CAP_OVERRIDES.get(repo, {}).get(split, MAX_NEW_SAMPLES_PER_PROJECT[split])


def main() -> None:
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=pathlib.Path, default=script_dir / "c")
    parser.add_argument("--base_train", type=pathlib.Path, default=repo_dir / "Dataset" / "id" / "Ultimate_train.json")
    parser.add_argument("--base_valid", type=pathlib.Path, default=repo_dir / "Dataset" / "id" / "Ultimate_valid.json")
    parser.add_argument("--base_test", type=pathlib.Path, default=repo_dir / "Dataset" / "id" / "Ultimate_test.json")
    parser.add_argument("--output_dir", type=pathlib.Path, default=script_dir / "processed")
    parser.add_argument("--target_train", type=int, default=21600)
    parser.add_argument("--target_valid", type=int, default=2700)
    parser.add_argument("--target_test", type=int, default=2700)
    parser.add_argument("--attempts", type=int, default=30)
    args = parser.parse_args()

    base_train, base_train_hashes, instruction = load_base_train(args.base_train)
    if len(base_train) > args.target_train:
        raise ValueError("Target training size is smaller than base training size")
    if any(task_name(item) == "detection" and output_label(item) not in {"Yes", "No"} for item in base_train):
        raise ValueError("Base detection labels contain invalid values")
    raw_valid = read_json(args.base_valid)
    raw_test = read_json(args.base_test)
    base_valid, valid_clean_stats = clean_base_detection_split(
        raw_valid,
        base_train_hashes,
        instruction,
    )
    valid_hashes = {
        code_hash(str(item.get("input", item.get("Input", ""))))
        for item in base_valid
    }
    base_test, test_clean_stats = clean_base_detection_split(
        raw_test,
        base_train_hashes | valid_hashes,
        instruction,
    )
    base_test_hashes = {
        code_hash(str(item.get("input", item.get("Input", ""))))
        for item in base_test
    }
    raw_records, parse_status = load_raw_records(args.raw_dir)
    groups, candidate_stats = build_candidates(
        raw_records,
        base_train_hashes | valid_hashes | base_test_hashes,
    )
    raw_targets = {
        "train": args.target_train - len(base_train),
        "valid": args.target_valid - len(base_valid),
        "test": args.target_test - len(base_test),
    }
    target_valid_labels = {
        "Yes": args.target_valid // 2,
        "No": args.target_valid - args.target_valid // 2,
    }
    target_test_labels = {
        "Yes": args.target_test // 2,
        "No": args.target_test - args.target_test // 2,
    }
    base_valid_labels = count_labels(base_valid)
    base_test_labels = count_labels(base_test)
    raw_label_targets = {
        "train": {
            "Yes": raw_targets["train"] // 2,
            "No": raw_targets["train"] - raw_targets["train"] // 2,
        },
        "valid": {
            "Yes": target_valid_labels["Yes"] - base_valid_labels.get("Yes", 0),
            "No": target_valid_labels["No"] - base_valid_labels.get("No", 0),
        },
        "test": {
            "Yes": target_test_labels["Yes"] - base_test_labels.get("Yes", 0),
            "No": target_test_labels["No"] - base_test_labels.get("No", 0),
        },
    }
    quotas = build_quotas(groups, raw_targets, raw_label_targets)
    expected_raw = raw_targets
    if expected_raw != {"train": 3138, "valid": 435, "test": 439}:
        raise ValueError(f"This calibrated build expects raw targets 3138/435/439, got {expected_raw}")
    if raw_label_targets != {
        "train": {"Yes": 1569, "No": 1569},
        "valid": {"Yes": 226, "No": 209},
        "test": {"Yes": 232, "No": 207},
    }:
        raise ValueError(f"Unexpected raw label targets: {raw_label_targets}")
    selection = None
    successful_seed = None
    for seed in range(args.attempts):
        candidate_selection = build_selection(groups, quotas, seed)
        if candidate_selection is not None:
            selection = candidate_selection
            successful_seed = seed
            break
    if selection is None:
        raise RuntimeError(f"Could not find a feasible grouped selection in {args.attempts} attempts")

    train_additions = [
        make_sft(candidate, instruction, label)
        for label in ["Yes", "No"]
        for candidate, _ in selection["train"][label]
    ]
    valid_additions = [
        make_sft(candidate, instruction, label)
        for label in ["Yes", "No"]
        for candidate, _ in selection["valid"][label]
    ]
    test_additions = [
        make_sft(candidate, instruction, label)
        for label in ["Yes", "No"]
        for candidate, _ in selection["test"][label]
    ]
    train_output = base_train + train_additions
    valid_output = base_valid + valid_additions
    test_output = base_test + test_additions
    if len(train_output) != args.target_train or len(valid_output) != args.target_valid or len(test_output) != args.target_test:
        raise AssertionError("Final split sizes are incorrect")
    if count_labels(valid_output) != target_valid_labels or count_labels(test_output) != target_test_labels:
        raise AssertionError(
            f"Validation/test labels are not balanced: valid={count_labels(valid_output)}, "
            f"test={count_labels(test_output)}, expected_valid={target_valid_labels}, "
            f"expected_test={target_test_labels}"
        )

    all_selected = []
    selected_metadata = []
    for split in ["train", "valid", "test"]:
        for label in ["Yes", "No"]:
            for candidate, _ in selection[split][label]:
                record = candidate["record"]
                all_selected.append(candidate)
                selected_metadata.append(
                    {
                        "split": split,
                        "label": label,
                        "year": record["_year"],
                        "cve": record["cve"],
                        "repo": record["repo"],
                        "commit": record["commit"],
                        "file": record["file"],
                        "source_file": record["_source_file"],
                        "source_index": record["_source_index"],
                        "pair_key": list(record["_pair_key"]),
                        "line_category": record["_line_category"],
                        "code_sha1": candidate["code_hash"],
                        "code_chars": len(candidate["code"]),
                    }
                )
    selected_hashes = [candidate["code_hash"] for candidate in all_selected]
    selected_pairs = [tuple(item["pair_key"]) for item in selected_metadata]
    selected_cves_by_split = {
        split: {item["cve"] for item in selected_metadata if item["split"] == split}
        for split in ["train", "valid", "test"]
    }
    if len(selected_hashes) != len(set(selected_hashes)):
        raise AssertionError("Selected raw code hashes are duplicated")
    if len(selected_pairs) != len(set(selected_pairs)):
        raise AssertionError("A repair pair was selected more than once")
    if any(selected_cves_by_split[a] & selected_cves_by_split[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]):
        raise AssertionError("CVE groups overlap across splits")

    train_hashes = {code_hash(item["input"]) for item in train_output}
    valid_hashes = {code_hash(item["input"]) for item in valid_output}
    test_hashes = {code_hash(item["input"]) for item in test_output}
    if train_hashes & valid_hashes or train_hashes & test_hashes or valid_hashes & test_hashes:
        raise AssertionError("Final code hashes overlap across splits")

    valid_path = args.output_dir / "Ultimate_valid_grouped_2019_2026_2700.json"
    test_path = args.output_dir / "Ultimate_test_grouped_2019_2026_2700.json"
    train_path = args.output_dir / "Ultimate_train_grouped_2019_2026_21600.json"
    report_path = args.output_dir / "grouped_high_quality_2019_2026_quality_report.json"
    write_json(train_path, train_output)
    write_json(valid_path, valid_output)
    write_json(test_path, test_output)
    project_stats = {}
    for split in ["train", "valid", "test"]:
        items = [item for item in selected_metadata if item["split"] == split]
        counts = collections.Counter(item["repo"] for item in items)
        project_stats[split] = {
            "unique_projects": len(counts),
            "max_samples_one_project": max(counts.values()) if counts else 0,
            "gini": round(gini(counts.values()), 4),
            "top10": counts.most_common(10),
        }
    report = {
        "design": {
            "base_train": str(args.base_train),
            "base_valid": str(args.base_valid),
            "base_test": str(args.base_test),
            "source_years": SOURCE_YEARS,
            "max_input_chars_for_new_samples": MAX_INPUT_CHARS,
            "max_samples_per_cve": MAX_SAMPLES_PER_CVE,
            "max_samples_per_cve_and_label": MAX_SAMPLES_PER_CVE_LABEL,
            "max_new_samples_per_project": MAX_NEW_SAMPLES_PER_PROJECT,
            "project_cap_overrides": PROJECT_CAP_OVERRIDES,
            "label_source": "before=Yes, after=No; vulnerable_lines used only for audit",
            "split_rule": "CVE-exclusive across train/valid/test; repair pair used once globally",
        },
        "successful_seed": successful_seed,
        "parse_status": parse_status,
        "candidate_stats": candidate_stats,
        "base_cleaning": {
            "original_valid_count": len(raw_valid),
            "clean_valid_count": len(base_valid),
            "valid_labels": base_valid_labels,
            "valid_cleaning": valid_clean_stats,
            "original_test_count": len(raw_test),
            "clean_test_count": len(base_test),
            "test_labels": base_test_labels,
            "test_cleaning": test_clean_stats,
        },
        "raw_targets": raw_targets,
        "raw_label_targets": raw_label_targets,
        "quota_by_split_label_year": quotas,
        "final_counts": {
            "train": len(train_output),
            "valid": len(valid_output),
            "test": len(test_output),
            "ratio": f"{len(train_output)}:{len(valid_output)}:{len(test_output)}",
        },
        "task_counts": {
            "train": dict(collections.Counter(task_name(item) for item in train_output)),
            "valid": dict(collections.Counter(task_name(item) for item in valid_output)),
            "test": dict(collections.Counter(task_name(item) for item in test_output)),
        },
        "label_counts": {
            "train": count_labels(train_output),
            "valid": count_labels(valid_output),
            "test": count_labels(test_output),
        },
        "selected_new_counts": {
            split: dict(collections.Counter(item["label"] for item in selected_metadata if item["split"] == split))
            for split in ["train", "valid", "test"]
        },
        "selected_new_by_year": {
            split: dict(collections.Counter(item["year"] for item in selected_metadata if item["split"] == split))
            for split in ["train", "valid", "test"]
        },
        "selected_new_line_categories": {
            split: dict(collections.Counter(item["line_category"] for item in selected_metadata if item["split"] == split))
            for split in ["train", "valid", "test"]
        },
        "project_stats": project_stats,
        "integrity": {
            "selected_raw_code_hash_duplicates": len(selected_hashes) - len(set(selected_hashes)),
            "selected_repair_pair_duplicates": len(selected_pairs) - len(set(selected_pairs)),
            "selected_cve_overlap_train_valid": len(selected_cves_by_split["train"] & selected_cves_by_split["valid"]),
            "selected_cve_overlap_train_test": len(selected_cves_by_split["train"] & selected_cves_by_split["test"]),
            "selected_cve_overlap_valid_test": len(selected_cves_by_split["valid"] & selected_cves_by_split["test"]),
            "final_hash_overlap_train_valid": len(train_hashes & valid_hashes),
            "final_hash_overlap_train_test": len(train_hashes & test_hashes),
            "final_hash_overlap_valid_test": len(valid_hashes & test_hashes),
        },
        "outputs": {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
            "report": str(report_path),
        },
        "selected_metadata": selected_metadata,
    }
    write_json(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "selected_metadata"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
