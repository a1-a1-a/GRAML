import argparse
import csv
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = (
    "You are a security code vulnerability analyzer. "
    "Return exactly one word: Yes or No."
)

DEFAULT_FIXED_INSTRUCTION = (
    "Analyze the provided code snippet. If you detect any potential security "
    "vulnerability, return Yes. If the code appears secure and free from obvious "
    "vulnerabilities, return No. Output exactly one word: Yes or No."
)

_THREAD_LOCAL = threading.local()


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as file:
        text = file.read().decode("utf-8-sig")
        data = json.loads(text)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or a dict with data list: {path}")
    return data


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_detect_label(value: Any) -> str:
    text = str(value).strip()
    lower = text.lower()
    if lower in {"yes", "1", "true", "vulnerable"}:
        return "Yes"
    if lower in {"no", "0", "false", "benign", "safe", "secure"}:
        return "No"
    if "not vulnerable" in lower or "non-vulnerable" in lower or "non vulnerable" in lower:
        return "No"
    if "yes" in lower or "vulnerable" in lower or "true" in lower:
        return "Yes"
    if "no" in lower or "false" in lower:
        return "No"
    return "No"


def sample_label(sample: Dict[str, Any]) -> str:
    for key in ["output", "Output", "target", "label", "gold"]:
        if key in sample:
            return normalize_detect_label(sample[key])
    return "No"


def extract_prediction(text: str) -> str:
    clean = str(text or "").strip()
    first = re.split(r"\s+", clean, maxsplit=1)[0].strip(" .,:;`'\"[](){}").lower()
    if first == "yes":
        return "Yes"
    if first == "no":
        return "No"
    lower = clean.lower()
    yes_pos = lower.find("yes")
    no_pos = lower.find("no")
    if yes_pos >= 0 and (no_pos < 0 or yes_pos < no_pos):
        return "Yes"
    if no_pos >= 0:
        return "No"
    return "No"


def get_sample_code(sample: Dict[str, Any]) -> str:
    return str(sample.get("input", sample.get("Input", "")) or "").strip()


def truncate_code(code: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars > 0 and len(code) > max_chars:
        return code[:max_chars], True
    return code, False


def get_instruction(sample: Dict[str, Any], prompt_mode: str) -> str:
    if prompt_mode == "dataset":
        instruction = str(sample.get("instruction", sample.get("Instruction", "")) or "").strip()
        return instruction or DEFAULT_FIXED_INSTRUCTION
    return DEFAULT_FIXED_INSTRUCTION


def normalize_example_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "1", "true", "vulnerable"}:
        return "Yes"
    if text in {"no", "0", "false", "benign", "safe", "secure"}:
        return "No"
    raise ValueError(f"Few-shot example has an unsupported label: {value!r}")


def read_examples(path: str) -> List[Dict[str, str]]:
    with open(path, "rb") as file:
        data = json.loads(file.read().decode("utf-8-sig"))
    if isinstance(data, dict):
        examples = data.get("examples", data.get("data"))
    else:
        examples = data
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Few-shot examples file must contain a non-empty list: {path}")

    normalized = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"Few-shot example {index} is not an object: {path}")
        code = str(example.get("input", example.get("Input", "")) or "").strip()
        if not code:
            raise ValueError(f"Few-shot example {index} has an empty input: {path}")
        label_value = None
        for key in ["output", "label", "target", "answer"]:
            if key in example:
                label_value = example[key]
                break
        if label_value is None:
            raise ValueError(f"Few-shot example {index} has no label: {path}")
        normalized.append({"input": code, "output": normalize_example_label(label_value)})
    return normalized


def build_few_shot_user_prompt(
    sample: Dict[str, Any],
    prompt_mode: str,
    max_code_chars: int,
    examples: List[Dict[str, str]],
    max_example_code_chars: int,
) -> str:
    instruction = get_instruction(sample, prompt_mode)
    raw_code = get_sample_code(sample)
    if any(example["input"].strip() == raw_code for example in examples):
        raise ValueError("Target code is also one of the few-shot examples; refusing to evaluate.")
    code, truncated = truncate_code(raw_code, max_code_chars)

    blocks = [instruction, ""]
    for index, example in enumerate(examples, start=1):
        example_code, example_truncated = truncate_code(
            example["input"], max_example_code_chars
        )
        blocks.append(f"## Example {index}\n### Code\n{example_code}")
        if example_truncated:
            blocks.append(
                "[Note: This example was truncated to the configured maximum length.]"
            )
        blocks.append(f"### Label: {example['output']}")
        blocks.append("")

    blocks.append(f"## Target\n### Code\n{code}")
    if truncated:
        blocks.append("[Note: The code was truncated to the configured maximum length.]")
    blocks.append("### Label:")
    return "\n".join(blocks)


def build_user_prompt(
    sample: Dict[str, Any],
    prompt_mode: str,
    max_code_chars: int,
    examples: Optional[List[Dict[str, str]]] = None,
    max_example_code_chars: int = 2000,
) -> str:
    if examples:
        return build_few_shot_user_prompt(
            sample, prompt_mode, max_code_chars, examples, max_example_code_chars
        )

    code, truncated = truncate_code(get_sample_code(sample), max_code_chars)
    instruction = get_instruction(sample, prompt_mode)

    if prompt_mode == "dataset":
        user = f"{instruction}\n\n## Input:\n{code}"
    else:
        user = f"{instruction}\n\n## Code:\n{code}"

    if truncated:
        user += "\n\n[Note: The code was truncated to the configured maximum length.]"
    return user


def response_text_from_responses(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks).strip()


def usage_from_response(response: Any) -> Tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    input_tokens = int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or input_tokens + output_tokens)
    return input_tokens, output_tokens, total_tokens


def call_responses(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: Optional[float],
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[str, int, int, int]:
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if extra_body:
        payload["extra_body"] = extra_body
    try:
        response = client.responses.create(**payload)
    except Exception as error:
        if temperature is None:
            raise
        payload.pop("temperature", None)
        response = client.responses.create(**payload)
    text = response_text_from_responses(response)
    input_tokens, output_tokens, total_tokens = usage_from_response(response)
    return text, input_tokens, output_tokens, total_tokens


def call_chat_completions(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: Optional[float],
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[str, int, int, int]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if extra_body:
        payload["extra_body"] = extra_body

    errors = []
    for token_key in ["max_completion_tokens", "max_tokens"]:
        attempt = dict(payload)
        attempt[token_key] = max_output_tokens
        try:
            response = client.chat.completions.create(**attempt)
            text = response.choices[0].message.content or ""
            input_tokens, output_tokens, total_tokens = usage_from_response(response)
            return text, input_tokens, output_tokens, total_tokens
        except Exception as error:
            errors.append(error)
            if temperature is not None:
                attempt.pop("temperature", None)
                try:
                    response = client.chat.completions.create(**attempt)
                    text = response.choices[0].message.content or ""
                    input_tokens, output_tokens, total_tokens = usage_from_response(response)
                    return text, input_tokens, output_tokens, total_tokens
                except Exception as retry_error:
                    errors.append(retry_error)
    raise errors[-1]


def call_model(
    client: OpenAI,
    api_mode: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: Optional[float],
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[str, int, int, int, str]:
    if api_mode == "responses":
        text, input_tokens, output_tokens, total_tokens = call_responses(
            client, model, system_prompt, user_prompt, max_output_tokens, temperature, extra_body
        )
        return text, input_tokens, output_tokens, total_tokens, "responses"
    if api_mode == "chat_completions":
        text, input_tokens, output_tokens, total_tokens = call_chat_completions(
            client, model, system_prompt, user_prompt, max_output_tokens, temperature, extra_body
        )
        return text, input_tokens, output_tokens, total_tokens, "chat_completions"

    try:
        text, input_tokens, output_tokens, total_tokens = call_responses(
            client, model, system_prompt, user_prompt, max_output_tokens, temperature, extra_body
        )
        return text, input_tokens, output_tokens, total_tokens, "responses"
    except Exception:
        text, input_tokens, output_tokens, total_tokens = call_chat_completions(
            client, model, system_prompt, user_prompt, max_output_tokens, temperature, extra_body
        )
        return text, input_tokens, output_tokens, total_tokens, "chat_completions"


def get_thread_client(client_kwargs: Dict[str, Any]) -> OpenAI:
    client = getattr(_THREAD_LOCAL, "openai_client", None)
    if client is None:
        client = OpenAI(**client_kwargs)
        _THREAD_LOCAL.openai_client = client
    return client


def process_sample(
    index: int,
    sample: Dict[str, Any],
    args: argparse.Namespace,
    client_kwargs: Dict[str, Any],
    temperature: Optional[float],
    extra_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    client = get_thread_client(client_kwargs)
    target = sample_label(sample)
    examples = getattr(args, "few_shot_examples", None)
    num_shots = len(examples) if examples else 0
    user_prompt = build_user_prompt(
        sample,
        args.prompt_mode,
        args.max_code_chars,
        examples=examples,
        max_example_code_chars=args.max_example_code_chars,
    )

    last_error = None
    for attempt in range(args.max_retries):
        start = time.perf_counter()
        try:
            raw_text, input_tokens, output_tokens, total_tokens, api_mode_used = call_model(
                client=client,
                api_mode=args.api_mode,
                model=args.model,
                system_prompt=args.system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=args.max_output_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
            latency_sec = time.perf_counter() - start
            break
        except Exception as error:
            last_error = error
            wait_sec = min(60.0, 2.0 ** attempt)
            time.sleep(wait_sec)
    else:
        raise RuntimeError(f"Sample {index} failed after {args.max_retries} retries: {last_error}")

    prediction = extract_prediction(raw_text)
    estimated_cost = (
        input_tokens * args.input_cost_per_1m / 1_000_000
        + output_tokens * args.output_cost_per_1m / 1_000_000
    )
    return {
        "target": target,
        "prediction": prediction,
        "raw_output": raw_text,
        "correct": target == prediction,
        "model": args.model,
        "dataset": args.test_path,
        "prompt_mode": args.prompt_mode,
        "num_shots": num_shots,
        "api_mode": api_mode_used,
        "row_index": index,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_sec": latency_sec,
        "estimated_cost_usd": estimated_cost,
    }


def metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = tn = fp = fn = 0
    for row in rows:
        target = row["target"]
        prediction = row["prediction"]
        if target == "Yes" and prediction == "Yes":
            tp += 1
        elif target == "No" and prediction == "No":
            tn += 1
        elif target == "No" and prediction == "Yes":
            fp += 1
        elif target == "Yes" and prediction == "No":
            fn += 1
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def append_summary_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "model",
        "dataset",
        "prompt_mode",
        "num_shots",
        "api_mode",
        "total",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "total_latency_sec",
        "avg_latency_sec",
        "estimated_cost_usd",
        "output_file",
    ]
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    if not write_header:
        with open(path, newline="", encoding="utf-8") as existing_file:
            existing_fields = csv.DictReader(existing_file).fieldnames or []
        if existing_fields and existing_fields != fieldnames:
            print(
                f"Warning: {path} uses an older CSV schema; appending only its columns.",
                flush=True,
            )
            fieldnames = existing_fields
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def existing_completed_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    return {row["row_index"] for row in read_jsonl(path) if "row_index" in row}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT/OpenAI code-only vulnerability detection baseline.")
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_csv_path", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api_mode", choices=["auto", "responses", "chat_completions"], default="auto")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--prompt_mode", choices=["dataset", "fixed"], default="dataset")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_code_chars", type=int, default=0)
    parser.add_argument("--examples_path", default=None, help="Few-shot examples JSON. Omit for zero-shot.")
    parser.add_argument("--num_shots", type=int, default=0, help="Use the first N examples; 0 uses all.")
    parser.add_argument("--max_example_code_chars", type=int, default=2000)
    parser.add_argument("--max_output_tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no_temperature", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Write prompts without calling the API.")
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--request_timeout_sec", type=float, default=120.0)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--extra_body_json", default=None)
    parser.add_argument("--input_cost_per_1m", type=float, default=0.0)
    parser.add_argument("--output_cost_per_1m", type=float, default=0.0)
    args = parser.parse_args()

    data = read_json(args.test_path)
    if args.limit > 0:
        data = data[: args.limit]

    examples: Optional[List[Dict[str, str]]] = None
    if args.examples_path:
        examples = read_examples(args.examples_path)
        if args.num_shots > 0:
            if args.num_shots > len(examples):
                raise ValueError(
                    f"--num_shots={args.num_shots} exceeds the {len(examples)} examples "
                    f"available in {args.examples_path}"
                )
            examples = examples[: args.num_shots]
    elif args.num_shots != 0:
        raise ValueError("--num_shots requires --examples_path")

    if examples:
        yes_count = sum(1 for example in examples if example["output"] == "Yes")
        no_count = len(examples) - yes_count
        if yes_count != no_count:
            raise ValueError(
                "Few-shot examples must be label-balanced; "
                f"got {yes_count} Yes and {no_count} No."
            )

    args.few_shot_examples = examples

    if args.dry_run:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        dry_run_count = min(len(data), args.limit if args.limit > 0 else 1)
        with open(args.output_path, "w", encoding="utf-8") as output_file:
            for index, sample in enumerate(data[:dry_run_count]):
                prompt = build_user_prompt(
                    sample,
                    args.prompt_mode,
                    args.max_code_chars,
                    examples=examples,
                    max_example_code_chars=args.max_example_code_chars,
                )
                row = {
                    "row_index": index,
                    "target": sample_label(sample),
                    "num_shots": len(examples) if examples else 0,
                    "prompt": prompt,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"Dry run wrote {dry_run_count} prompt(s) to {args.output_path}; "
            f"shots={len(examples) if examples else 0}",
            flush=True,
        )
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key environment variable: {args.api_key_env}")

    client_kwargs = {"api_key": api_key, "timeout": args.request_timeout_sec}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if args.resume and os.path.exists(args.output_path):
        for existing_row in read_jsonl(args.output_path):
            if "num_shots" not in existing_row:
                continue
            existing_shots = int(existing_row.get("num_shots", 0) or 0)
            if existing_shots != (len(examples) if examples else 0):
                raise ValueError(
                    "Refusing to resume: the existing output file uses a different "
                    "few-shot configuration. Use a new --output_path."
                )
            break

    completed = existing_completed_keys(args.output_path) if args.resume else set()
    temperature = None if args.no_temperature else args.temperature
    extra_body = {}
    if args.extra_body_json:
        extra_body.update(json.loads(args.extra_body_json))
    if args.disable_thinking:
        extra_body.setdefault("thinking", {"type": "disabled"})
    if not extra_body:
        extra_body = None

    rows_for_metrics = []
    if args.resume and os.path.exists(args.output_path):
        rows_for_metrics.extend(read_jsonl(args.output_path))

    mode_used_counter = Counter()
    for row in rows_for_metrics:
        if row.get("api_mode"):
            mode_used_counter[row["api_mode"]] += 1

    pending_items = [
        (index, sample)
        for index, sample in enumerate(data)
        if index not in completed
    ]
    finished_count = len(rows_for_metrics)
    total_count = len(data)
    print(
        f"Loaded {total_count} samples, completed {finished_count}, pending {len(pending_items)}, "
        f"workers {max(1, args.num_workers)}",
        flush=True,
    )

    with open(args.output_path, "a" if args.resume else "w", encoding="utf-8") as output_file:
        if args.num_workers <= 1:
            for index, sample in pending_items:
                result = process_sample(index, sample, args, client_kwargs, temperature, extra_body)
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
                rows_for_metrics.append(result)
                mode_used_counter[result["api_mode"]] += 1
                finished_count += 1
                if args.progress_every > 0 and finished_count % args.progress_every == 0:
                    print(f"Progress {finished_count}/{total_count}", flush=True)
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_index = {
                    executor.submit(
                        process_sample, index, sample, args, client_kwargs, temperature, extra_body
                    ): index
                    for index, sample in pending_items
                }
                for future in as_completed(future_to_index):
                    result = future.result()
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                    rows_for_metrics.append(result)
                    mode_used_counter[result["api_mode"]] += 1
                    finished_count += 1
                    if args.progress_every > 0 and finished_count % args.progress_every == 0:
                        print(f"Progress {finished_count}/{total_count}", flush=True)
                    if args.sleep_sec > 0:
                        time.sleep(args.sleep_sec)

    metric_values = metrics(rows_for_metrics)
    total_input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in rows_for_metrics)
    total_output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in rows_for_metrics)
    total_tokens = sum(int(row.get("total_tokens", 0) or 0) for row in rows_for_metrics)
    total_latency = sum(float(row.get("latency_sec", 0) or 0) for row in rows_for_metrics)
    total_cost = sum(float(row.get("estimated_cost_usd", 0) or 0) for row in rows_for_metrics)
    summary = {
        "model": args.model,
        "dataset": args.test_path,
        "prompt_mode": args.prompt_mode,
        "num_shots": len(examples) if examples else 0,
        "api_mode": dict(mode_used_counter),
        **metric_values,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_latency_sec": total_latency,
        "avg_latency_sec": total_latency / metric_values["total"] if metric_values["total"] else 0,
        "estimated_cost_usd": total_cost,
        "output_file": args.output_path,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_csv_path:
        append_summary_csv(args.summary_csv_path, summary)


if __name__ == "__main__":
    main()
