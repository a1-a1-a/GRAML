"""
w/o Graph Evidence — Ablation Experiment (multi-threaded)
==========================================================
Generates vulnerability descriptions from raw source code using an
LLM with a one-step prompt (no CPG, no ToT-VR).
Uses ThreadPoolExecutor for concurrent API calls.
"""

import argparse, hashlib, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

import openai

DESCRIPTION_INSTRUCTION = (
    "You are a security code vulnerability analyzer. "
    "Analyze the following C code and describe any security vulnerabilities present. "
    "Write the final vulnerability explanation as one coherent paragraph "
    "in professional security-analysis style that clearly states "
    "what the vulnerability is, why it occurs as the root cause, "
    "how it can be triggered, and what impact an attacker may achieve. "
    "The output must be a single paragraph of 3 to 5 sentences, "
    "must not contain any parentheses, and may quote small relevant portions "
    "of the source code without including line numbers or bracket references. "
    "The writing must be fluent, concise, and technically professional. "
    "If you determine that there is no vulnerability, "
    'only output "There is no vulnerability."'
)

MAX_TOKENS = 1024
TEMPERATURE = 0.7
SAVE_LOCK = Lock()


def extract_code(input_text: str) -> str:
    match = re.search(
        r"\[Code\]\s*\n(.*?)(?:\n\n\[CPG Evidence\]|\Z)",
        input_text, re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return input_text.strip()


def code_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def load_samples(input_path: str) -> Tuple[List[Dict], List[Dict]]:
    with open(input_path, "rb") as f:
        data = json.loads(f.read().decode("utf-8-sig"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    samples, entries = [], []
    for sample in data:
        if sample.get("Task") == "description" and sample.get("output") != "There is no vulnerability.":
            samples.append(sample)
            code = extract_code(sample["input"])
            entries.append({
                "index": len(entries),
                "hash": code_hash(code),
                "code": code,
            })
    return samples, entries


def call_llm(entry: Dict, model: str, client_kwargs: Dict) -> Dict:
    client = openai.OpenAI(**client_kwargs)
    prompt = "Analyze this C code for security vulnerabilities. Only output the final description paragraph, nothing else.\n\n" + entry["code"]
    MAX_ATTEMPTS = 3
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DESCRIPTION_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            desc = resp.choices[0].message.content.strip()
            break
        except Exception as exc_info:
            attempts += 1
            if attempts < MAX_ATTEMPTS:
                time.sleep(2)
            else:
                desc = f"__API_ERROR__: {exc_info}"
    return {
        "index": entry["index"],
        "hash": entry["hash"],
        "code": entry["code"],
        "generated_description": desc,
    }


def save(results: List[Dict], path: Path):
    with SAVE_LOCK:
        results.sort(key=lambda x: x["index"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model", default="doubao-seed-2-0-lite-260428")
    parser.add_argument("--base_url", default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pure_code_file = output_dir / "wo_graph_evidence_input.json"
    result_file = output_dir / "wo_graph_evidence_output.json"

    # Step 1: Extract
    print("=" * 60)
    print("Step 1: Extracting raw code from training JSON")
    samples, entries = load_samples(str(input_path))
    total = len(entries)
    print(f"  Total samples to generate: {total}")
    print(f"  Avg code length: {sum(len(e['code']) for e in entries)//total} chars")

    with open(pure_code_file, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in e.items() if k != "code"} for e in entries], f, ensure_ascii=False, indent=2)
    print(f"  Saved metadata -> {pure_code_file}")
    print("  Full code in: " + str(result_file))

    # Step 2: Setup client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("API key required: --api_key or OPENAI_API_KEY")

    client_kwargs = {"api_key": api_key, "base_url": args.base_url}
    print(f"\nStep 2: {args.threads} threads -> {args.model} @ {args.base_url}")

    # Resume
    done_indices = set()
    results = []
    if args.resume and result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_indices = {r["index"] for r in results}
        print(f"  Resuming: {len(done_indices)}/{total} already done")

    pending = [e for e in entries if e["index"] not in done_indices]
    print(f"  Pending: {len(pending)}/{total}")

    # Step 3: Run
    completed = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(call_llm, e, args.model, client_kwargs): e for e in pending}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(pending) - completed) / rate if rate > 0 else 0
            desc_len = len(result["generated_description"])
            is_err = result["generated_description"].startswith("__API_ERROR__")
            flag = "ERR" if is_err else ("NV" if result["generated_description"] == "There is no vulnerability." else "OK")
            print(f"  [{completed}/{len(pending)}] idx={result['index']:3d} {flag} {desc_len:4d} chars  {elapsed:.0f}s  ETA {eta:.0f}s")
            if completed % 50 == 0 or completed == len(pending):
                save(results, result_file)

    save(results, result_file)

    # Summary
    total_done = len(results)
    errs = sum(1 for r in results if r["generated_description"].startswith("__API_ERROR__"))
    no_vul = sum(1 for r in results if r["generated_description"] == "There is no vulnerability.")
    ok = total_done - errs - no_vul
    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"  Done:  {total_done}")
    print(f"  Valid: {ok}  |  NoVul: {no_vul}  |  Errors: {errs}")
    print(f"  Time:  {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Saved: {result_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()