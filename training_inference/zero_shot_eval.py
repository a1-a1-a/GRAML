import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel

from infer_unsloth import (
    read_json,
    is_detection,
    normalize_detect_label,
)
from task_prompts import SYSTEM_PROMPT, PROMPT_BUILDERS, resolve_instruction


def metrics_from_rows(rows):
    tp = tn = fp = fn = 0
    for r in rows:
        target = r["target"]
        pred = r["prediction"]
        if target == "Yes" and pred == "Yes":
            tp += 1
        elif target == "No" and pred == "No":
            tn += 1
        elif target == "No" and pred == "Yes":
            fp += 1
        elif target == "Yes" and pred == "No":
            fn += 1
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def metrics_at_threshold(probs, labels, threshold):
    rows = [{
        "target": "Yes" if label else "No",
        "prediction": "Yes" if prob >= threshold else "No",
    } for prob, label in zip(probs, labels)]
    return metrics_from_rows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--prompt_style", required=True, choices=["mistral", "deepseek", "alpaca", "qwen"])
    p.add_argument("--datasets_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=None, help="Defaults to 8,192 for Mistral and 16,384 for all other backbones")
    p.add_argument("--threshold", type=float, default=0.5, help="Fallback threshold when --find_optimal is not set")
    p.add_argument("--valid_path", default=None, help="ID validation set used to select the decision threshold")
    p.add_argument("--find_optimal", action="store_true", help="Select the threshold on the validation set")
    p.add_argument("--metric", choices=["accuracy", "f1", "precision", "recall"], default="f1", help="Metric optimized during threshold selection")
    p.add_argument("--threshold_min", type=float, default=0.01)
    p.add_argument("--threshold_max", type=float, default=0.90)
    p.add_argument("--threshold_step", type=float, default=0.01)
    p.add_argument("--curve_path", default=None, help="Optional CSV path or directory for the validation threshold curve")
    p.add_argument("--force", action="store_true")
    p.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    args = p.parse_args()
    if args.max_seq_length is None:
        args.max_seq_length = 8192 if "mistral" in args.model_name.lower() else 16384

    build_fn = PROMPT_BUILDERS[args.prompt_style]

    datasets = json.loads(Path(args.datasets_json).read_text(encoding="utf-8-sig"))
    model_out = Path(args.out_dir) / args.model_name
    model_out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model once: {args.model_path}", file=sys.stderr, flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    yes_ids = sorted({x[-1] for x in [
        tokenizer.encode(" Yes", add_special_tokens=False),
        tokenizer.encode("Yes", add_special_tokens=False),
    ] if x})
    no_ids = sorted({x[-1] for x in [
        tokenizer.encode(" No", add_special_tokens=False),
        tokenizer.encode("No", add_special_tokens=False),
    ] if x})

    def get_yes_prob(prompt):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_seq_length,
        ).to("cuda")
        with torch.no_grad():
            logits = model(inputs.input_ids).logits[0, -1, :]
            yes_score = max(logits[t].item() for t in yes_ids)
            no_score = max(logits[t].item() for t in no_ids)
            probs = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
            return probs[1].item()

    threshold = args.threshold
    threshold_info = {"threshold": threshold, "source": "fixed"}
    if args.find_optimal:
        if not args.valid_path:
            p.error("--find_optimal requires --valid_path")
        print(f"Selecting threshold on {args.valid_path}", file=sys.stderr, flush=True)
        valid_data = read_json(args.valid_path)
        valid_data = [s for s in valid_data if is_detection(s)]
        if not valid_data:
            p.error("no detection samples found in --valid_path")

        v_probs = []
        v_labels = []
        for i, ex in enumerate(valid_data, 1):
            instr = resolve_instruction(ex)
            inp = ex.get("input", "")
            out = ex.get("output", "")
            prompt = build_fn(args.system_prompt, instr, inp)
            v_probs.append(get_yes_prob(prompt))
            v_labels.append(1 if normalize_detect_label(out) == "Yes" else 0)
            if i % 200 == 0:
                print(f"  validation: {i}/{len(valid_data)}", flush=True)

        start = int(round(args.threshold_min * 100))
        end = int(round(args.threshold_max * 100))
        step = max(1, int(round(args.threshold_step * 100)))
        best_score = -1.0
        best_threshold = args.threshold
        curve = []
        for th_int in range(start, end + 1, step):
            th = th_int / 100.0
            m = metrics_at_threshold(v_probs, v_labels, th)
            curve.append({"threshold": th, **m})
            score = m[args.metric]
            if score > best_score:
                best_score = score
                best_threshold = th
        threshold = best_threshold
        threshold_info = {
            "threshold": threshold,
            "source": "validation",
            "valid_path": args.valid_path,
            "metric": args.metric,
            "score": best_score,
            "num_validation_samples": len(valid_data),
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
        }
        (model_out / "validation_threshold.json").write_text(
            json.dumps(threshold_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Selected threshold ({args.metric}): {threshold:.4f} (score={best_score:.4f})", flush=True)

        if args.curve_path:
            curve_file = Path(args.curve_path)
            if curve_file.is_dir() or not curve_file.suffix:
                curve_file.mkdir(parents=True, exist_ok=True)
                curve_file = curve_file / f"{args.model_name}_validation_threshold_curve.csv"
            else:
                curve_file.parent.mkdir(parents=True, exist_ok=True)
            fields = ["threshold", "total", "tp", "tn", "fp", "fn",
                      "accuracy", "precision", "recall", "f1"]
            with curve_file.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in curve:
                    w.writerow({k: row.get(k, "") for k in fields})
            print(f"WROTE {curve_file}", flush=True)

    print(f"Using threshold {threshold:.4f} ({threshold_info['source']})", flush=True)

    all_metrics = []
    for ds in datasets:
        name = ds["name"]
        test_path = ds["path"]
        expected = int(ds["expected"])
        out_file = model_out / f"{name}.jsonl"
        metrics_file = model_out / f"{name}_metrics.json"

        if out_file.exists() and not args.force:
            n = sum(1 for _ in out_file.open("r", encoding="utf-8"))
            if n == expected and metrics_file.exists():
                m = json.loads(metrics_file.read_text(encoding="utf-8"))
                print(f"SKIP {args.model_name} {name} ({n}/{expected})", flush=True)
                all_metrics.append(m)
                continue
            if n == expected:
                rows = [json.loads(line) for line in out_file.open("r", encoding="utf-8") if line.strip()]
                m = metrics_from_rows(rows)
                m.update({"model": args.model_name, "model_path": args.model_path,
                          "dataset": name, "dataset_path": test_path,
                          "threshold": threshold,
                          "threshold_source": threshold_info["source"],
                          "output_file": str(out_file)})
                metrics_file.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                all_metrics.append(m)
                print(f"RECOVER {args.model_name} {name} metrics", flush=True)
                continue

        data = read_json(test_path)
        total_raw = len(data)
        data = [s for s in data if is_detection(s)]
        print(f"START {args.model_name} {name}: {len(data)}/{total_raw} detection samples", flush=True)

        tp = tn = fp = fn = 0
        t0 = time.time()
        with out_file.open("w", encoding="utf-8") as f:
            for i, ex in enumerate(data, 1):
                instr = resolve_instruction(ex)
                inp = ex.get("input", "")
                out = ex.get("output", "")
                prompt = build_fn(args.system_prompt, instr, inp)
                prob = get_yes_prob(prompt)
                pred = "Yes" if prob >= threshold else "No"
                target = normalize_detect_label(out)
                if target == "Yes" and pred == "Yes":
                    tp += 1
                elif target == "No" and pred == "No":
                    tn += 1
                elif target == "No" and pred == "Yes":
                    fp += 1
                elif target == "Yes" and pred == "No":
                    fn += 1
                f.write(json.dumps({
                    "target": target,
                    "prediction": pred,
                    "yes_prob": prob,
                    "correct": target == pred,
                    "model": args.model_path,
                    "threshold": threshold,
                    "threshold_source": threshold_info["source"],
                    "dataset": test_path,
                }, ensure_ascii=False) + "\n")
                if i % 200 == 0:
                    f.flush()
                    print(f"  {args.model_name} {name}: {i}/{len(data)}", flush=True)

        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        m = {"model": args.model_name, "model_path": args.model_path,
             "dataset": name, "dataset_path": test_path, "threshold": threshold,
             "threshold_source": threshold_info["source"],
             "valid_path": args.valid_path,
             "total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
             "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
             "elapsed_sec": time.time() - t0, "output_file": str(out_file)}
        metrics_file.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_metrics.append(m)
        print(f"END {args.model_name} {name}: F1={f1:.4f} Acc={acc:.4f} ({total} samples, {time.time()-t0:.1f}s)", flush=True)

    csv_path = model_out / "metrics.csv"
    fields = ["model", "model_path", "dataset", "dataset_path", "threshold",
              "threshold_source", "valid_path", "total",
              "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1",
              "elapsed_sec", "output_file"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in all_metrics:
            w.writerow({k: m.get(k, "") for k in fields})
    print(f"WROTE {csv_path}", flush=True)


if __name__ == "__main__":
    main()
