
import argparse
import json
import os
import sys
import csv
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel


os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        text = f.read().decode("utf-8-sig")
        data = json.loads(text)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data

def is_detection(sample: Dict[str, Any]) -> bool:
    t = None
    if "task" in sample: t = sample["task"]
    elif "category" in sample: t = sample["category"]
    elif "type" in sample: t = sample["type"]
    elif "Task" in sample: t = sample["Task"]
    
    if isinstance(t, str):
        ts = t.lower()
        if "detect" in ts or "detection" in ts or "det" in ts:
            return True
    
    instr = sample.get("instruction", "") or sample.get("Instruction", "")
    if isinstance(instr, str):
        s = instr.lower()
        if "detect" in s or "detection" in s:
            return True
    return False

def normalize_detect_label(v: str) -> str:
    s = str(v).strip()
    sl = s.lower()
    if "yes" in sl or "vulnerable" in sl or sl == "1" or "true" in sl:
        return "Yes"
    if "no" in sl or "not vulnerable" in sl or sl == "0" or "false" in sl:
        return "No"
    return "No"


def build_prompt_mistral(system_prompt: str, instruction: str, user_input: str) -> str:
    user = instruction.strip()
    if user_input:
        user = user + "\n## Input:\n" + str(user_input).strip()
    return f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n{user} [/INST]"

def build_prompt_deepseek(system_prompt: str, instruction: str, user_input: str) -> str:
    user = instruction.strip()
    if user_input:
        user = user + "\n" + str(user_input).strip()
    return f"{system_prompt}\n### Instruction:\n{user}\n### Response:\n"

def build_prompt_alpaca(system_prompt: str, instruction: str, user_input: str) -> str:
    user = instruction.strip()
    input_str = ""
    if user_input:
        input_str = f"\n\n### Input:\n{str(user_input).strip()}"
    return f"{system_prompt}\n\n### Instruction:\n{user}{input_str}\n\n### Response:\n"

def build_prompt_qwen(system_prompt: str, instruction: str, user_input: str) -> str:
    user = instruction.strip()
    if user_input:
        user = user + "\n" + str(user_input).strip()
    
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA 权重目录 (包含 adapter_config.json)")
    parser.add_argument("--base_model", type=str, default=None, help="基座模型路径 (如果 adapter_config 中未指定或需覆盖)")
    parser.add_argument("--test_path", type=str, required=True, help="测试集 JSON")
    parser.add_argument("--output_path", type=str, default="predictions.jsonl")
    parser.add_argument("--prompt_style", type=str, default="mistral", choices=["mistral", "deepseek", "alpaca", "qwen"])
    parser.add_argument("--valid_path", type=str, default=None, help="验证集 (用于搜索阈值)")
    parser.add_argument("--find_optimal", action="store_true", help="是否搜索最佳阈值")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "f1", "precision", "recall"], help="阈值搜索的优化指标")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--system_prompt", type=str, default="You are a security assistant for vulnerability detection.")
    parser.add_argument("--only_detection", action="store_true")
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--summary_csv_path", type=str, default=None)
    parser.add_argument("--threshold_min", type=float, default=0.1, help="阈值搜索起始点")
    parser.add_argument("--threshold_max", type=float, default=0.9, help="阈值搜索结束点")
    parser.add_argument("--threshold_step", type=float, default=0.01, help="阈值搜索步长")
    parser.add_argument("--curve_path", type=str, default=None, help="阈值搜索曲线记录CSV路径")
    parser.add_argument("--force_target_modules", action="store_true", help="强制指定target_modules（仅当需要手动加载LoRA时）")

    args = parser.parse_args()

  
    if args.prompt_style == "deepseek":
        build_fn = build_prompt_deepseek
    elif args.prompt_style == "alpaca":
        build_fn = build_prompt_alpaca
    elif args.prompt_style == "qwen":
        build_fn = build_prompt_qwen
    else:
        build_fn = build_prompt_mistral

    
    print(f"Loading model from {args.adapter_path}...", file=sys.stderr)
    
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.adapter_path, 
            max_seq_length = args.max_seq_length,
            dtype = None,
            load_in_4bit = args.load_in_4bit,
            local_files_only = True,
        )
    except Exception as e:
        print(f"Direct load failed ({e}), trying loading base + adapter...", file=sys.stderr)
        if args.base_model is None:
             raise ValueError("Please provide --base_model when loading LoRA adapters explicitly.")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.base_model,
            max_seq_length = args.max_seq_length,
            dtype = None,
            load_in_4bit = args.load_in_4bit,
            local_files_only = True,
        )
        
        
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Dummy value, will be overwritten by load_adapter
            target_modules = target_modules, 
            lora_alpha = 16,
        )
        
        model.load_adapter(args.adapter_path)

    FastLanguageModel.for_inference(model) 
    
    
    test_data = read_json(args.test_path)
    if args.only_detection:
        test_data = [s for s in test_data if is_detection(s)]
    
    
    yes_tokens = [tokenizer.encode(" Yes", add_special_tokens=False), tokenizer.encode("Yes", add_special_tokens=False)]
    no_tokens = [tokenizer.encode(" No", add_special_tokens=False), tokenizer.encode("No", add_special_tokens=False)]
    yes_tokens = [x for x in yes_tokens if len(x)>0]
    no_tokens = [x for x in no_tokens if len(x)>0]

    def get_yes_prob(prompt):
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            add_special_tokens=False, 
            truncation=True, 
            max_length=args.max_seq_length
        ).to("cuda")
        input_ids = inputs.input_ids
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :] 
            
            
            yes_score = max([logits[tokens[0]].item() for tokens in yes_tokens])
            no_score = max([logits[tokens[0]].item() for tokens in no_tokens])
            
           
            probs = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
            return probs[1].item()

    
    best_threshold = args.threshold
    if args.find_optimal and args.valid_path:
        print("Searching for optimal threshold on validation set...", file=sys.stderr)
        valid_data = read_json(args.valid_path)
        if args.only_detection:
            valid_data = [s for s in valid_data if is_detection(s)]
            
        v_probs = []
        v_labels = []
        
        from tqdm import tqdm
        for ex in tqdm(valid_data, desc="Validating"):
            instr = ex.get("instruction", "")
            inp = ex.get("input", "")
            out = ex.get("output", "")
            prompt = build_fn(args.system_prompt, instr, inp)
            
            prob = get_yes_prob(prompt)
            label = 1 if normalize_detect_label(out) == "Yes" else 0
            
            v_probs.append(prob)
            v_labels.append(label)
            
        
        best_score = 0
        target_metric = args.metric.lower()
        
       
        start = int(args.threshold_min * 100)
        end = int(args.threshold_max * 100)
        step = int(args.threshold_step * 100)
        if step < 1: step = 1
        
        threshold_records = [] 
        
        for th_int in range(start, end + 1, step):
            th = th_int / 100.0
            tp = tn = fp = fn = 0
            for p, l in zip(v_probs, v_labels):
                pred = 1 if p >= th else 0
                if l==1 and pred==1: tp+=1
                elif l==0 and pred==0: tn+=1
                elif l==0 and pred==1: fp+=1
                elif l==1 and pred==0: fn+=1
            
          
            total = tp + tn + fp + fn
            acc = (tp + tn) / total if total else 0
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            
            threshold_records.append({
                "model": args.adapter_path,
                "dataset": args.valid_path,
                "threshold": th,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            })
            
            score = {
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec
            }[target_metric]

            if score > best_score:
                best_score = score
                best_threshold = th
        print(f"Optimal threshold found ({target_metric}): {best_threshold} (Score: {best_score:.4f})", file=sys.stderr)

        
        if args.curve_path:
           
            curve_file = args.curve_path
            if os.path.isdir(args.curve_path) or (not os.path.splitext(args.curve_path)[1]):
                os.makedirs(args.curve_path, exist_ok=True)
                model_name = os.path.basename(os.path.normpath(args.adapter_path))
                dataset_name = os.path.splitext(os.path.basename(args.valid_path or args.test_path))[0]
                filename = f"{model_name}_{dataset_name}_curve.csv"
                curve_file = os.path.join(args.curve_path, filename)
                print(f"Auto-generated curve filename: {curve_file}", file=sys.stderr)
            else:
                os.makedirs(os.path.dirname(args.curve_path) or ".", exist_ok=True)

            write_header = not os.path.exists(curve_file)
            with open(curve_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["model", "dataset", "threshold", "accuracy", "precision", "recall", "f1"])
                for r in threshold_records:
                    writer.writerow([r["model"], r["dataset"], r["threshold"], r["accuracy"], r["precision"], r["recall"], r["f1"]])
            print(f"Threshold search curve saved to {curve_file}", file=sys.stderr)

    
    print(f"Running inference with threshold {best_threshold}...", file=sys.stderr)
    
   
    output_file = args.output_path
    if os.path.isdir(args.output_path) or (not os.path.splitext(args.output_path)[1]):
        os.makedirs(args.output_path, exist_ok=True)
        
        model_name = os.path.basename(os.path.normpath(args.adapter_path))
        
        dataset_name = os.path.splitext(os.path.basename(args.test_path))[0]
        
        filename = f"{model_name}_{dataset_name}_predictions.jsonl"
        output_file = os.path.join(args.output_path, filename)
        print(f"Auto-generated output filename: {output_file}", file=sys.stderr)
    else:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    
    results = []
    tp = tn = fp = fn = 0
    
    from tqdm import tqdm
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in tqdm(test_data, desc="Testing"):
            instr = ex.get("instruction", "")
            inp = ex.get("input", "")
            out = ex.get("output", "")
            prompt = build_fn(args.system_prompt, instr, inp)
            
            prob = get_yes_prob(prompt)
            pred_label = "Yes" if prob >= best_threshold else "No"
            true_label = normalize_detect_label(out)
            
    
            if true_label == "Yes" and pred_label == "Yes": tp += 1
            elif true_label == "No" and pred_label == "No": tn += 1
            elif true_label == "No" and pred_label == "Yes": fp += 1
            elif true_label == "Yes" and pred_label == "No": fn += 1
            
            res = {
                "instruction": instr,
                "input": inp,
                "target": true_label,
                "prediction": pred_label,
                "yes_prob": prob,
                "correct": true_label == pred_label,
                "model": args.adapter_path,
                "threshold": best_threshold,
                "dataset": args.test_path
            }
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    
    metrics = {
        "model": args.adapter_path,
        "dataset": args.test_path,
        "threshold": best_threshold,
        "max_seq_length": args.max_seq_length,
        "total": total,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "output_file": output_file
    }
    
    print("\n" + "="*20 + " Metrics " + "="*20)
    print(json.dumps(metrics, indent=2))
    
    
    if args.summary_csv_path:
        os.makedirs(os.path.dirname(args.summary_csv_path) or ".", exist_ok=True)
        write_header = not os.path.exists(args.summary_csv_path)
        with open(args.summary_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["model", "dataset", "threshold", "max_seq_length", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"])
            writer.writerow([
                args.adapter_path,
                args.test_path,
                best_threshold,
                args.max_seq_length,
                tp, tn, fp, fn,
                acc, prec, rec, f1
            ])
        print(f"Metrics appended to {args.summary_csv_path}", file=sys.stderr)

   
    if args.summary_path:
        with open(args.summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
