#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tot_vr import Branch, ToTVRGenerator

PROMPT_VERSION = "simple-tot-v1.0"
NO_VULNERABILITY = "There is no vulnerability."

DESCRIPTION_INSTRUCTION = (
    "You are a security code vulnerability analyzer. "
    "Your task is to analyze the following code and determine whether it contains vulnerabilities. "
    "Write the final vulnerability explanation as one coherent paragraph in professional security-analysis style "
    "that clearly states what the vulnerability is, why it occurs as the root cause, how it can be triggered, "
    "how the internal logic or data or control flow leads to the flaw, and what impact an attacker may achieve. "
    "The output must be a single paragraph of 3 to 5 sentences, must not contain any parentheses, "
    "and may quote small relevant portions of the source code without including line numbers or bracket references. "
    "The writing must be fluent, concise, and technically professional. "
    "If you determine that there is no vulnerability, only output \"There is no vulnerability.\""
)


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def code_key(text: Any) -> str:
    return normalize_text(text)


def extract_code(text: Any) -> str:
    value = str(text or "")
    match = re.search(
        r"\[Code\]\s*\n(.*?)(?:\n\n\[CPG Evidence\]|\Z)",
        value,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return value.strip()


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
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


def load_samples(path: str) -> List[Dict[str, Any]]:
    raw = load_json_or_jsonl(path)
    samples: List[Dict[str, Any]] = []
    seen_ids: Dict[str, int] = {}

    for index, item in enumerate(raw):
        task = str(item.get("Task", item.get("task", ""))).strip().lower()
        output = str(item.get("output", item.get("Output", ""))).strip()
        if task and task != "description":
            continue
        if output == NO_VULNERABILITY:
            continue

        code = (
            item.get("function_code")
            or item.get("code")
            or item.get("input")
            or ""
        )
        code = extract_code(code)
        if not code.strip():
            continue

        sample_id = item.get("sample_id", item.get("sample_idx", index))
        sample_key = str(sample_id)
        if sample_key in seen_ids:
            seen_ids[sample_key] += 1
            sample_key = f"{sample_key}#{seen_ids[sample_key]}"
        else:
            seen_ids[sample_key] = 1

        samples.append(
            {
                "sample_id": sample_id,
                "sample_key": sample_key,
                "sample_idx": item.get("sample_idx", index),
                "function_code": code,
                "cve_description": str(item.get("cve_description") or ""),
            }
        )
    return samples


class SimpleToTGenerator(ToTVRGenerator):

    def __init__(
        self,
        *args: Any,
        synthesis_mode: str = "consensus",
        use_cve_description: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if synthesis_mode not in {"consensus", "judge"}:
            raise ValueError("synthesis_mode must be 'consensus' or 'judge'")
        self.synthesis_mode = synthesis_mode
        self.use_cve_description = use_cve_description

    @staticmethod
    def _format_branches_for_prompt(branches: Sequence[Branch]) -> str:
        if not branches:
            return "(none)"
        return "\n\n".join(branch.describe(include_status=False) for branch in branches)

    def _judge_prompt(self, function_code: str, branches: Sequence[Branch]) -> str:
        return (
            "You are selecting among candidate vulnerability explanations produced by a tree-of-thought search. "
            "Use only the source code below. Rank the candidates by how well the source code supports their "
            "vulnerability type, trigger, causal chain, and impact. Do not use any external CVE description.\n\n"
            f"Source Code:\n{function_code}\n\n"
            f"Candidate Explanations:\n{self._format_branches_for_prompt(branches)}\n\n"
            "Return a JSON object with this schema:\n"
            '{"selected_branch_id": "Branch1", "ranking": [{"id": "Branch1", "score": 0.0, "reason": "..."}], '
            '"reason": "..."}\n'
            "Scores must be between 0 and 1. Use the same branch ids as the input."
        )

    def _select_branch(self, function_code: str, branches: Sequence[Branch]) -> Tuple[Optional[Branch], str]:
        data, raw = self._call_json(
            self._judge_prompt(function_code, branches),
            stage="simple ToT branch selection",
        )
        selected_id = ""
        if isinstance(data, dict):
            selected_id = str(data.get("selected_branch_id") or data.get("selected_branch") or "").strip()
        selected = next((branch for branch in branches if branch.branch_id == selected_id), None)
        if selected is None and isinstance(data, dict):
            for row in data.get("ranking", []) or []:
                if not isinstance(row, dict):
                    continue
                candidate_id = str(row.get("id") or row.get("branch_id") or "").strip()
                selected = next((branch for branch in branches if branch.branch_id == candidate_id), None)
                if selected is not None:
                    break
        if selected is None and branches:
            selected = max(branches, key=lambda branch: len(branch.causal_explanation or ""))
        return selected, raw

    def _synthesis_prompt(
        self,
        function_code: str,
        branches: Sequence[Branch],
        selected: Optional[Branch],
        cve_description: str,
    ) -> str:
        if selected is not None:
            candidate_block = selected.describe(include_status=False)
            mode_instruction = (
                "Use the selected candidate explanation below as the main hypothesis, but correct it when the "
                "source code clearly contradicts it. Do not invent facts that are absent from the source code "
                "or the selected explanation."
            )
        else:
            candidate_block = self._format_branches_for_prompt(branches)
            mode_instruction = (
                "Choose the candidate explanation that is best supported by the source code and synthesize a "
                "single final explanation. If candidates conflict, keep the most plausible mechanism. Do not "
                "invent facts that are absent from the source code or the candidate explanations."
            )

        cve_block = ""
        if self.use_cve_description and cve_description.strip():
            cve_block = (
                "\nExternal vulnerability description for identity reference only. Use it only to keep the "
                "vulnerability identity consistent; all code-level statements must remain supported by the "
                "source code:\n"
                f"{cve_description.strip()}\n"
            )

        return (
            DESCRIPTION_INSTRUCTION
            + "\n\nYou are given a C/C++ function and candidate vulnerability explanations produced by a "
            "tree-of-thought search.\n\n"
            + mode_instruction
            + "\nReport only vulnerability facts; omit branch labels, candidate selection, verification scores, and CVE wording. "
            "Do not write absolute line numbers such as L12 or line 17. Return only the final vulnerability "
            "explanation as one paragraph.\n\n"
            f"Source Code:\n{function_code}\n\n"
            f"Candidate Explanations:\n{candidate_block}\n"
            f"{cve_block}"
        )

    @staticmethod
    def _clean_description(text: str) -> str:
        value = str(text or "").strip()
        value = re.sub(r"^```(?:text|markdown)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
        value = re.sub(
            r"^(?:final\s+description|description|answer)\s*:\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = value.strip().strip('"').strip("'").strip()
        value = re.sub(r"\s*\n+\s*", " ", value).strip()
        return value

    def generate_simple_description(
        self,
        function_code: str,
        cve_description: str = "",
        sample_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        branches, raw_initialization, expansion_info = self.step_1_branch_initialization(
            function_code,
            "(none)",
        )
        reflection_error = ""
        try:
            retained, pruned, raw_reflection, reflection_info = self.step_2_branch_reflection_and_pruning(
                function_code,
                branches,
            )
        except Exception as exc:
            reflection_error = str(exc)
            retained, pruned = list(branches), []
            raw_reflection, reflection_info = "", {"fallback": True, "error": reflection_error}

        if not retained:
            retained = list(branches)
        if not retained:
            raise ValueError("Simple ToT produced no branch after initialization and reflection")

        selected: Optional[Branch] = None
        selection_raw = ""
        selection_error = ""
        if self.synthesis_mode == "judge":
            try:
                selected, selection_raw = self._select_branch(function_code, retained)
            except Exception as exc:
                selection_error = str(exc)
                selected = None

        prompt = self._synthesis_prompt(function_code, retained, selected, cve_description)
        description = self._clean_description(self._call_llm(prompt))
        description = self._strip_line_number_annotations(description)
        if not description:
            raise ValueError("The model returned an empty description")

        status = f"simple_tot_{self.synthesis_mode}"
        if description.strip() == NO_VULNERABILITY:
            status = "no_vulnerability"

        return {
            "sample_id": sample_id,
            "synthesis_mode": self.synthesis_mode,
            "prompt_version": PROMPT_VERSION,
            "model": self.model,
            "uses_graph_evidence": False,
            "uses_cve_description": bool(self.use_cve_description and cve_description.strip()),
            "num_branches_initial": len(branches),
            "num_branches_retained": len(retained),
            "num_branches_pruned": len(pruned),
            "selected_branch_id": selected.branch_id if selected is not None else None,
            "branches_initial": [branch.to_dict() for branch in branches],
            "branches_retained": [branch.to_dict() for branch in retained],
            "branches_pruned": [branch.to_dict() for branch in pruned],
            "final_description": description,
            "generated_description": description,
            "description_status": status,
            "reflection_error": reflection_error,
            "selection_error": selection_error,
            "elapsed_seconds": round(time.time() - start, 3),
            "trace": {
                "step_1_initialization_raw": raw_initialization,
                "step_1_expansion": expansion_info,
                "step_2_reflection_raw": raw_reflection,
                "step_2_reflection": reflection_info,
                "branch_selection_raw": selection_raw,
            },
        }


def read_completed(path: Path) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_key = str(record.get("sample_key", record.get("sample_id")))
            completed[sample_key] = record
    return completed


def sentence_count(text: str) -> int:
    return len([item for item in re.split(r"(?<=[.!?])\s+", str(text).strip()) if item.strip()])


def summarize(records: Sequence[Dict[str, Any]], total_inputs: int) -> Dict[str, Any]:
    valid = [r for r in records if r.get("final_description") and not r.get("error")]
    errors = len(records) - len(valid)
    descs = [str(r["final_description"]) for r in valid]
    word_counts = [len(item.split()) for item in descs]
    sent_counts = [sentence_count(item) for item in descs]
    branch_counts = [int(r.get("num_branches_initial", 0)) for r in valid]
    status_counts: Dict[str, int] = {}
    for record in records:
        status = str(record.get("description_status") or ("error" if record.get("error") else "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    def stats(values: List[int]) -> Dict[str, Any]:
        if not values:
            return {"min": 0, "median": 0, "mean": 0, "max": 0}
        return {
            "min": min(values),
            "median": statistics.median(values),
            "mean": round(statistics.mean(values), 3),
            "max": max(values),
        }

    return {
        "prompt_version": PROMPT_VERSION,
        "total_inputs": total_inputs,
        "records_written": len(records),
        "completed": len(valid),
        "errors": errors,
        "no_vulnerability": sum(1 for item in descs if item.strip() == NO_VULNERABILITY),
        "word_count": stats(word_counts),
        "sentence_count_estimate": stats(sent_counts),
        "branch_count": stats(branch_counts),
        "description_status_counts": status_counts,
        "uses_graph_evidence": False,
    }


def build_generator(args: argparse.Namespace) -> SimpleToTGenerator:
    extra_body: Optional[Dict[str, Any]] = None
    if args.disable_thinking:
        extra_body = {"thinking": {"type": "disabled"}}
    api_key = os.environ.get(args.api_key_env)
    return SimpleToTGenerator(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url or None,
        min_branches=args.min_branches,
        max_branches=args.max_branches,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
        extra_body=extra_body,
        verbose=args.verbose,
        synthesis_mode=args.synthesis_mode,
        use_cve_description=args.use_cve_description,
    )


def run_one(args: argparse.Namespace, sample: Dict[str, Any], local: threading.local) -> Dict[str, Any]:
    try:
        generator = getattr(local, "generator", None)
        if generator is None:
            generator = build_generator(args)
            local.generator = generator
        result = generator.generate_simple_description(
            function_code=sample["function_code"],
            cve_description=sample.get("cve_description", ""),
            sample_id=sample["sample_id"],
        )
        result["sample_key"] = sample["sample_key"]
        result["sample_idx"] = sample.get("sample_idx")
        result["error"] = None
        return result
    except Exception as exc:
        return {
            "sample_id": sample["sample_id"],
            "sample_key": sample["sample_key"],
            "sample_idx": sample.get("sample_idx"),
            "synthesis_mode": args.synthesis_mode,
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "uses_graph_evidence": False,
            "uses_cve_description": bool(args.use_cve_description),
            "final_description": "",
            "generated_description": "",
            "description_status": "error",
            "error": str(exc),
            "elapsed_seconds": None,
        }


def write_training_records(records: Sequence[Dict[str, Any]], path: Path) -> None:
    payload = []
    for record in sorted(records, key=lambda item: str(item.get("sample_key", item.get("sample_id")))):
        description = str(record.get("final_description") or "").strip()
        if not description:
            continue
        payload.append(
            {
                "instruction": DESCRIPTION_INSTRUCTION,
                "input": record.get("function_code") or "",
                "output": description,
                "Task": "description",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Simple ToT baseline for vulnerability descriptions")
    parser.add_argument("--input", default="ToT/tot_inputs_train.jsonl", help="JSONL/JSON input samples")
    parser.add_argument("--output", required=False, help="JSONL output with generated descriptions")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path")
    parser.add_argument("--training_output", default=None, help="Optional training-ready JSON output")
    parser.add_argument("--model", default=os.environ.get("TOT_MODEL", "gpt-5"))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--min_branches", type=int, default=2)
    parser.add_argument("--max_branches", type=int, default=6)
    parser.add_argument("--synthesis_mode", choices=["consensus", "judge"], default="consensus")
    parser.add_argument("--use_cve_description", action="store_true", help="Optional stronger variant; off by default")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_output_tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry_run is set")

    samples = load_samples(args.input)
    if not samples:
        raise ValueError(f"No usable Description samples in {args.input}")
    samples = samples[args.start:]
    if args.limit is not None:
        samples = samples[: args.limit]

    if args.dry_run:
        sample = samples[0]
        generator = SimpleToTGenerator(
            client=object(),
            model=args.model,
            min_branches=args.min_branches,
            max_branches=args.max_branches,
            synthesis_mode=args.synthesis_mode,
            use_cve_description=args.use_cve_description,
        )
        prompt = generator._initialization_prompt(sample["function_code"], "(none)")
        print(json.dumps({
            "sample_id": sample["sample_id"],
            "sample_key": sample["sample_key"],
            "function_chars": len(sample["function_code"]),
            "synthesis_mode": args.synthesis_mode,
            "uses_graph_evidence": False,
            "uses_cve_description": args.use_cve_description,
        }, ensure_ascii=False, indent=2))
        print("\n--- branch initialization prompt ---\n")
        print(prompt[:8000])
        return 0

    if not os.environ.get(args.api_key_env):
        raise RuntimeError(f"Set {args.api_key_env} or pass --api_key_env; use --dry_run to inspect prompts without an API key")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = read_completed(output_path) if args.resume else {}
    done_keys = {
        key for key, record in completed.items()
        if record.get("final_description") and not record.get("error")
    }
    pending = [sample for sample in samples if sample["sample_key"] not in done_keys]

    if args.resume:
        print(f"Resuming: {len(done_keys)} completed, {len(pending)} pending, {len(samples)} selected")
    else:
        output_path.write_text("", encoding="utf-8")
        print(f"Running: {len(pending)} samples -> {output_path}")

    local = threading.local()
    records_by_key: Dict[str, Dict[str, Any]] = dict(completed)
    started = time.time()
    completed_count = 0

    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(run_one, args, sample, local): sample for sample in pending}
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "sample_id": sample["sample_id"],
                        "sample_key": sample["sample_key"],
                        "sample_idx": sample.get("sample_idx"),
                        "final_description": "",
                        "generated_description": "",
                        "description_status": "error",
                        "error": str(exc),
                    }
                records_by_key[str(record.get("sample_key", record.get("sample_id")))] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed_count += 1
                elapsed = time.time() - started
                rate = completed_count / elapsed if elapsed > 0 else 0
                eta = (len(pending) - completed_count) / rate if rate > 0 else 0
                flag = "ERR" if record.get("error") else "OK"
                chars = len(str(record.get("final_description") or ""))
                print(
                    f"[{completed_count}/{len(pending)}] {sample['sample_key']} {flag} "
                    f"{chars} chars {elapsed:.0f}s ETA {eta:.0f}s",
                    flush=True,
                )

    all_records = [records_by_key[key] for key in sorted(records_by_key)]
    for record in all_records:
        if "function_code" not in record:
            sample = next((item for item in samples if item["sample_key"] == record.get("sample_key")), None)
            if sample is not None:
                record["function_code"] = sample["function_code"]
            else:
                record["function_code"] = ""

    summary = summarize(all_records, len(samples))
    summary["output"] = str(output_path)
    summary["model"] = args.model
    summary["synthesis_mode"] = args.synthesis_mode
    summary["workers"] = args.workers
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Summary -> {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    errors = summary["errors"]
    if args.training_output:
        if errors and not args.allow_partial:
            print(
                f"Not writing training output: {errors} records still have errors. "
                "Re-run with --resume, or use --allow_partial.",
                file=sys.stderr,
            )
            return 1
        write_training_records(all_records, Path(args.training_output))
        print(f"Training records -> {args.training_output}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

