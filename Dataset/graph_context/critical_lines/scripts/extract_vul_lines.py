import hashlib, json, os, sys
from pathlib import Path

TRAIN_EVID = os.path.join(str(Path(__file__).resolve().parents[3]), "ablations", "direct_cpg",
    "Ultimate_train_cpg_description_only_plus_grouped_2019_2026_detection.json")
TRAIN_JSONL = os.path.join(str(Path(__file__).resolve().parents[3]), "graph_context", "critical_lines", "source_data", "train.jsonl")
DIVERSEVUL = os.path.join(str(Path(__file__).resolve().parents[3]), "graph_context", "critical_lines", "source_data", "diversevul_20230702.json")
MSR_PATH = r"PATH\TO\VulLLM\gpt5\MSR_data_cleaned.json"
OUTPUT = os.path.join(str(Path(__file__).resolve().parents[3]), "graph_context", "critical_lines", "output", "train_vul_lines.json")

def norm_code(code):
    lines = [l.rstrip() for l in code.splitlines()]
    return "\n".join(l for l in lines if l.strip())

def code_hash(code):
    return hashlib.sha256(norm_code(code).encode()).hexdigest()

def extract_code(inp):
    if "[Code]" not in inp:
        return inp.strip()
    start = inp.index("[Code]") + len("[Code]")
    code = inp[start:]
    if "[CPG Evidence]" in code:
        code = code[:code.index("[CPG Evidence]")]
    return code.strip()

# Load positive description samples
print("Loading description samples...")
with open(TRAIN_EVID, "r", encoding="utf-8") as f:
    data = json.load(f)
samples = []
for d in data:
    if d.get("Task") != "description":
        continue
    out = d.get("output", "").strip().lower()
    if out in ("no vulnerability", "n/a", "") or "no vulnerability" in out:
        continue
    samples.append(d)
print(f"Positive descriptions: {len(samples)}")

# Build target hash -> sample_idx
target = {}
for idx, s in enumerate(samples):
    code = extract_code(s["input"])
    h = code_hash(code)
    target[h] = idx

# ---- Source 1: train.jsonl (has location_label!) ----
print("\n=== Matching train.jsonl ===")
vuln_lines = {}  # sample_idx -> [line_numbers]
matched_train = set()
with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        fb = obj.get("func_before", "")
        if not fb:
            continue
        h = code_hash(fb)
        if h in target:
            idx = target[h]
            if idx not in matched_train:
                loc = obj.get("location_label", [])
                if loc and isinstance(loc, list):
                    vuln_lines[idx] = {"lines": loc, "source": "train.jsonl", "reliability": 1.0}
                else:
                    # Try diff as fallback
                    fa = obj.get("func_after", "")
                    if fa:
                        bl = fb.splitlines()
                        al = fa.splitlines()
                        from difflib import SequenceMatcher
                        sm = SequenceMatcher(None, bl, al)
                        diff_lines_set = set()
                        for tag, i1, i2, j1, j2 in sm.get_opcodes():
                            if tag in ("replace", "delete"):
                                for i in range(i1, i2):
                                    diff_lines_set.add(i + 1)
                        loc = sorted(diff_lines_set)
                    vuln_lines[idx] = {"lines": loc, "source": "train.jsonl(diff)", "reliability": 1.0}
                matched_train.add(idx)
print(f"Matched train.jsonl: {len(matched_train)}")

# ---- Source 2: diversevul (only has func, NO func_after) ----
print("\n=== Matching diversevul ===")
matched_dv = set()
with open(DIVERSEVUL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        func = obj.get("func", "")
        if not func:
            continue
        h = code_hash(func)
        if h in target:
            idx = target[h]
            if idx not in matched_train and idx not in matched_dv:
                # No func_after -> mark as needs_joern
                matched_dv.add(idx)
print(f"Matched diversevul (no vuln lines, needs Joern): {len(matched_dv)} (counted but vuln_lines unfilled)")

# ---- Source 3: MSR_data_cleaned (streaming, has func_after) ----
print("\n=== Matching MSR (streaming) ===")
matched_msr = set()
remaining = set(range(len(samples))) - matched_train
print(f"Remaining after train.jsonl: {len(remaining)}")
try:
    import ijson
    with open(MSR_PATH, "r", encoding="utf-8") as f:
        for key, entry in ijson.kvitems(f, ""):
            fb = entry.get("func_before", "")
            if not fb:
                continue
            h = code_hash(fb)
            if h in target:
                idx = target[h]
                if idx in matched_train or idx in matched_msr:
                    continue
                fa = entry.get("func_after", "")
                if fa:
                    bl = fb.splitlines()
                    al = fa.splitlines()
                    from difflib import SequenceMatcher
                    sm = SequenceMatcher(None, bl, al)
                    diff_set = set()
                    for tag, i1, i2, j1, j2 in sm.get_opcodes():
                        if tag in ("replace", "delete"):
                            for i in range(i1, i2):
                                diff_set.add(i + 1)
                    loc = sorted(diff_set)
                    vuln_lines[idx] = {"lines": loc, "source": "MSR", "reliability": 1.0}
                    matched_msr.add(idx)
except ImportError:
    print("ijson not available, skipping MSR")
except Exception as e:
    print(f"MSR error: {e}")

print(f"Matched MSR: {len(matched_msr)}")

# ---- Build output ----
output = []
total_matched = matched_train | matched_msr
for idx, s in enumerate(samples):
    code = extract_code(s["input"])
    rec = {"sample_idx": idx, "code": code[:200] + "...", "matched": idx in total_matched}
    if idx in vuln_lines:
        vl = vuln_lines[idx]
        rec.update({"vuln_lines": vl["lines"], "source": vl["source"], "reliability": vl["reliability"]})
    else:
        rec.update({"vuln_lines": [], "source": "unmatched", "reliability": 0.0, "match_type": "needs_joern"})
    output.append(rec)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# Summary
print(f"\n{'='*60}")
print(f"Total positive descriptions: {len(samples)}")
print(f"  train.jsonl (location_label): {len(matched_train)}")
print(f"  MSR (diff): {len(matched_msr)}")
print(f"  diversevul (matched, no lines): {len(matched_dv)}")
print(f"  Total with vuln lines: {len(total_matched)}")
print(f"  Unmatched: {len(samples) - len(total_matched)}")
print(f"Output: {OUTPUT}")
