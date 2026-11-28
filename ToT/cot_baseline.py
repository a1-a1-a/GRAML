#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simple_tot import (
    DESCRIPTION_INSTRUCTION,
    NO_VULNERABILITY,
    load_samples,
    read_completed,
    sentence_count,
    write_training_records,
)
from tot_vr import ToTVRGenerator

PROMPT_VERSION = "cot-baseline-v1.0"


class CoTGenerator(ToTVRGenerator):

    @staticmethod
    def _cot_prompt(function_code: str) -> str:
        return (
            DESCRIPTION_INSTRUCTION
            + "\n\nAnalyze the following C/C++ function with step-by-step reasoning. "
            "Identify the security-sensitive operation, trace how untrusted input or program state "
            "reaches it, determine the missing or incorrect protection, and state the impact an "
            "attacker could achieve. Base every statement on the source code and do not invent "
            "facts. Keep the intermediate reasoning internal. Return only the final vulnerability "
            "description as one paragraph, in 3 to 5 sentences, without parentheses and without "
            "absolute line numbers.\n\n"
            f"Source Code:\n{function_code}"
        )

    @staticmethod
    def _clean_description(text: str) -> str:
        import re

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

    def generate_cot_description(
        self,
        function_code: str,
        sample_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        prompt = self._cot_prompt(function_code)
        description = self._clean_description(self._call_llm(prompt))
        description = self._strip_line_number_annotations(description)
        if not description:
            raise ValueError("The model returned an empty description")

        status = "cot"
        if description.strip() == NO_VULNERABILITY:
            status = "no_vulnerability"

        return {
            "sample_id": sample_id,
            "strategy": "cot",
            "prompt_version": PROMPT_VERSION,
            "model": self.model,
            "uses_graph_evidence": False,
            "uses_cve_description": False,
            "final_description": description,
            "generated_description": description,
            "description_status": status,
            "elapsed_seconds": round(time.time() - start, 3),
        }


def build_generator(args: argparse.Namespace) -> CoTGenerator:
    extra_body: Optional[Dict[str, Any]] = None
    if args.disable_thinking:
        extra_body = {"thinking": {"type": "disabled"}}
    return CoTGenerator(
        api_key=os.environ.get(args.api_key_env),
        model=args.model,
        base_url=args.base_url or None,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
        extra_body=extra_body,
        verbose=args.verbose,
    )


def run_one(args: argparse.Namespace, sample: Dict[str, Any], local: threading.local) -> Dict[str, Any]:
    try:
        generator = getattr(local, "generator", None)
        if generator is None:
            generator = build_generator(args)
            local.generator = generator
        result = generator.generate_cot_description(
            function_code=sample["function_code"],
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
            "strategy": "cot",
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "uses_graph_evidence": False,
            "uses_cve_description": False,
            "final_description": "",
            "generated_description": "",
            "description_status": "error",
            "error": str(exc),
            "elapsed_seconds": None,
        }


def summarize(records: Sequence[Dict[str, Any]], total_inputs: int) -> Dict[str, Any]:
    valid = [r for r in records if r.get("final_description") and not r.get("error")]
    descs = [str(r["final_description"]) for r in valid]
    word_counts = [len(item.split()) for item in descs]
    sent_counts = [sentence_count(item) for item in descs]

    def stats(values: List[int]) -> Dict[str, Any]:
        if not values:
            return {"min": 0, "median": 0, "mean": 0, "max": 0}
        return {
            "min": min(values),
            "median": statistics.median(values),
            "mean": round(statistics.mean(values), 3),
            "max": max(values),
        }

    status_counts: Dict[str, int] = {}
    for record in records:
        status = str(record.get("description_status") or ("error" if record.get("error") else "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "prompt_version": PROMPT_VERSION,
        "total_inputs": total_inputs,
        "records_written": len(records),
        "completed": len(valid),
        "errors": len(records) - len(valid),
        "no_vulnerability": sum(1 for item in descs if item.strip() == NO_VULNERABILITY),
        "word_count": stats(word_counts),
        "sentence_count_estimate": stats(sent_counts),
        "description_status_counts": status_counts,
        "uses_graph_evidence": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CoT baseline for vulnerability descriptions")
    parser.add_argument("--input", default="ToT/tot_inputs_train.jsonl", help="JSONL/JSON input samples")
    parser.add_argument("--output", default=None, help="JSONL output with generated descriptions")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path")
    parser.add_argument("--training_output", default=None, help="Optional training-ready JSON output")
    parser.add_argument("--model", default=os.environ.get("TOT_MODEL", "gpt-5"))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
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
        prompt = CoTGenerator._cot_prompt(sample["function_code"])
        print(json.dumps({
            "sample_id": sample["sample_id"],
            "sample_key": sample["sample_key"],
            "function_chars": len(sample["function_code"]),
            "strategy": "cot",
            "uses_graph_evidence": False,
        }, ensure_ascii=False, indent=2))
        print("\n--- CoT prompt ---\n")
        print(prompt[:8000])
        return 0

    if not os.environ.get(args.api_key_env):
        raise RuntimeError(
            f"Set {args.api_key_env} or pass --api_key_env; use --dry_run to inspect prompts without an API key"
        )

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
            record["function_code"] = sample["function_code"] if sample is not None else ""

    summary = summarize(all_records, len(samples))
    summary["output"] = str(output_path)
    summary["model"] = args.model
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
