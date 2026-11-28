import json, os
from pathlib import Path

TRAIN_EVID = os.path.join(str(Path(__file__).resolve().parents[3]), "ablations", "direct_cpg",
    "Ultimate_train_direct_cpg.json")
TRAIN_JSONL = os.path.join(str(Path(__file__).resolve().parents[3]), "graph_context", "critical_lines", "source_data", "train.jsonl")
DIVERSEVUL = os.path.join(str(Path(__file__).resolve().parents[3]), "graph_context", "critical_lines", "source_data", "diversevul_20230702.json")
MSR_PATH = os.environ.get("MSR_PATH", r"PATH\TO\MSR_data_cleaned.json")
OUTPUT = os.path.join(str(Path(__file__).resolve().parents[3]), "graph_context", "critical_lines", "output", "train_vul_lines.json")

def norm_code(code):
    lines = [l.rstrip() for l in code.splitlines()]
    return "\n".join(l for l in lines if l.strip())

def extract_code(inp):
    if "[Code]" not in inp:
        return inp.strip()
    start = inp.index("[Code]") + len("[Code]")
    code = inp[start:]
    if "[CPG Evidence]" in code:
        code = code[:code.index("[CPG Evidence]")]
    return code.strip()

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

target = {}
for idx, s in enumerate(samples):
    code = extract_code(s["input"])
    target[norm_code(code)] = idx

print("\n=== Matching train.jsonl ===")
vuln_lines = {}
matched_train = set()
with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        fb = obj.get("func_before", "")
        if not fb:
            continue
        key = norm_code(fb)
        if key in target:
            idx = target[key]
            if idx not in matched_train:
                loc = obj.get("location_label", [])
                if loc and isinstance(loc, list):
                    vuln_lines[idx] = {"lines": loc, "source": "train.jsonl", "reliability": 1.0}
                else:
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

print("\n=== Matching diversevul ===")
matched_dv = set()
with open(DIVERSEVUL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        func = obj.get("func", "")
        if not func:
            continue
        key = norm_code(func)
        if key in target:
            idx = target[key]
            if idx not in matched_train and idx not in matched_dv:
                matched_dv.add(idx)
print(f"Matched diversevul (no vuln lines, needs Joern): {len(matched_dv)} (counted but vuln_lines unfilled)")

print("\n=== Matching MSR (streaming) ===")
matched_msr = set()
remaining = set(range(len(samples))) - matched_train
print(f"Remaining after train.jsonl: {len(remaining)}")
try:
    import ijson
    with open(MSR_PATH, "r", encoding="utf-8") as f:
        for _entry_key, entry in ijson.kvitems(f, ""):
            fb = entry.get("func_before", "")
            if not fb:
                continue
            code_key = norm_code(fb)
            if code_key in target:
                idx = target[code_key]
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

output = []
total_matched = matched_train | matched_msr
for idx, s in enumerate(samples):
    code = extract_code(s["input"])
    rec = {"sample_idx": idx, "vuln_lines": vuln_lines[idx]["lines"] if idx in vuln_lines else []}
    output.append(rec)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Total positive descriptions: {len(samples)}")
print(f"  train.jsonl (location_label): {len(matched_train)}")
print(f"  MSR (diff): {len(matched_msr)}")
print(f"  diversevul (matched, no lines): {len(matched_dv)}")
print(f"  Total with vuln lines: {len(total_matched)}")
print(f"Output: {OUTPUT}")
