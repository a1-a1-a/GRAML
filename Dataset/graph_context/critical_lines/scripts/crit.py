import json, os, re, shutil, subprocess, sys, locale
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

SYS_ENC = locale.getpreferredencoding()

BASE = Path(__file__).resolve().parents[3]
INNO = BASE / "graph_context" / "critical_lines"
TRAIN = BASE / "ablations" / "direct_cpg" / "Ultimate_train_direct_cpg.json"
VL = INNO / "output" / "train_vul_lines.json"
OUT = INNO / "output" / "train_crit.json"
WD = BASE / "graph_context" / "cpg_evidence" / "joern_work_crit"
JP = os.environ.get("JOERN_PARSE") or shutil.which("joern-parse") or "joern-parse"
JX = os.environ.get("JOERN_EXPORT") or shutil.which("joern-export") or "joern-export"

REL_TYPE_TO_KEY = {"REACHING_DEF": "D", "DDG": "D", "CONTROL_DEPENDENCE": "C", "CDG": "C", "CFG_NEXT": "F", "CFG": "F"}
KEY_TO_WEIGHT = {"D": 0.50, "C": 0.30, "F": 0.20}
MC = 20
Z1 = 0.45
Z2 = 0.25
DV = 0.20
RD = 0.10
AV = 1.0
AJ = 0.7

def run(cmd, to=600):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=to, encoding=SYS_ENC, errors="replace")
    except Exception:
        return None

def ec(i):
    if "[Code]" not in i:
        return i.strip()
    s = i.index("[Code]") + 6
    c = i[s:]
    if "[CPG Evidence]" in c:
        c = c[:c.index("[CPG Evidence]")]
    return c.strip()

def vl(l):
    s = l.strip()
    if not s or s.startswith(("//", "/*", "*", "#")) or s in ("{", "}"):
        return False
    return True

def vls(c):
    ls = c.splitlines()
    return {i + 1 for i, l in enumerate(ls) if vl(l)}

def sc(code):
    for m in ["__user", "__kernel", "__iomem", "__force", "__init", "__exit",
              "__must_check", "__maybe_unused", "__read_mostly", "__ref",
              "OMX_IN", "OMX_OUT", "OMX_INOUT", "IN", "OUT", "INOUT",
              "TSRMLS_DC", "TSRMLS_CC", "TSRMLS_C", "TSRMLS_D"]:
        code = re.sub(r"" + re.escape(m) + r"", "", code)
    return re.sub(r"__always_inline", "inline", code)

def ls():
    with open(TRAIN, "r", encoding="utf-8") as f:
        d = json.load(f)
    ss = []
    for x in d:
        if x.get("Task") != "description":
            continue
        o = x.get("output", "").strip().lower()
        if o in ("no vulnerability", "n/a", "") or "no vulnerability" in o:
            continue
        ss.append(x)
    with open(VL, "r", encoding="utf-8") as f:
        vd = json.load(f)
    for i, s in enumerate(ss):
        c = ec(s["input"])
        v = vd[i] if i < len(vd) else {}
        s["_c"] = c
        s["_vl"] = v.get("vuln_lines", [])
        s["_r"] = v.get("reliability", 0.7 if not v.get("matched") else 1.0)
    return ss


def _clean_html(s):
    return (s or "").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def _parse_dot(dot_path):
    text = dot_path.read_text(encoding="utf-8", errors="replace")
    method = ""
    i = text.find("digraph")
    if i >= 0:
        j = text.find(chr(34), i)
        if j >= 0:
            k = text.find(chr(34), j + 1)
            if k >= 0:
                method = _clean_html(text[j + 1:k])
    node_line = {}
    marker = "[label = <"
    pos = 0
    while True:
        s = text.find(marker, pos)
        if s < 0:
            break
        closeq = text.rfind(chr(34), 0, s)
        if closeq >= 0:
            openq = text.rfind(chr(34), 0, closeq)
            if openq >= 0:
                nid = text[openq + 1:closeq].strip()
                body = text[s + len(marker):]
                e = body.find(">")
                lab = body[:e] if e >= 0 else body
                mm = re.search(r",\s*(\d+)\s*<BR", lab)
                if mm and nid.isdigit():
                    node_line[nid] = int(mm.group(1))
        pos = s + 1
    edges = []
    for line in text.splitlines():
        if "->" not in line:
            continue
        parts = line.split("->")
        if len(parts) < 2:
            continue
        src = parts[0].strip().strip(chr(34))
        dst_part = parts[1]
        dst = dst_part[:dst_part.find("[")].strip().strip(chr(34))
        if "DDG:" in line:
            t = "DDG"
        elif "CDG:" in line:
            t = "CDG"
        else:
            t = ""
        if src in node_line and dst in node_line:
            sl, dl = node_line[src], node_line[dst]
            if sl != dl:
                edges.append({"src": sl, "dst": dl, "_t": t})
    return method, node_line, edges


def _select_critical(code, vuln_lines, valid, edges):
    valid_set = set(valid)
    vuln = set(int(v) for v in vuln_lines) & valid_set
    pair_types = defaultdict(set)
    for e in edges:
        sl = e.get("src"); dl = e.get("dst"); et = e.get("type", "")
        if sl is None or dl is None:
            continue
        sl = int(sl); dl = int(dl)
        if sl == dl or sl not in valid_set or dl not in valid_set:
            continue
        if REL_TYPE_TO_KEY.get(et) is None:
            continue
        pair_types[(sl, dl)].add(et)

    a0 = {}
    for (i, j), ts in pair_types.items():
        keys = {REL_TYPE_TO_KEY[t] for t in ts}
        a0[(i, j)] = sum(KEY_TO_WEIGHT[k] for k in keys)

    if not vuln:
        deg = defaultdict(float)
        for (i, j), w in a0.items():
            deg[i] += w; deg[j] += w
        top = sorted(deg.items(), key=lambda x: -x[1])[:3]
        vuln = {l for l, _w in top if l in valid_set}
        if not vuln:
            return [], [], "no_usable_joern_edges", [], []

    involved = sorted(set(i for (i, j) in a0) | set(j for (i, j) in a0) | set(vuln))
    pos = {ln: p for p, ln in enumerate(involved)}
    n = len(involved)
    A0 = np.zeros((n, n))
    for (i, j), w in a0.items():
        A0[pos[i], pos[j]] = w

    B = A0 + A0.T
    deg = B.sum(axis=1)
    dinv = np.zeros(n)
    nz = deg > 1e-12
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    Di = np.diag(dinv)
    tildeA = Di @ B @ Di

    z0 = np.zeros(n)
    for v in vuln:
        z0[pos[v]] = 1.0 / len(vuln)

    z1 = tildeA @ z0
    z2 = tildeA @ z1
    m1 = z1.max() if z1.size else 0.0
    m2 = z2.max() if z2.size else 0.0
    z1h = z1 / m1 if m1 > 1e-12 else z1
    z2h = z2 / m2 if m2 > 1e-12 else z2

    cand_pos = [p for p in range(n) if involved[p] not in vuln and (z1h[p] + z2h[p]) > 1e-9]

    B_nbr = B > 1e-12
    def nbr_of(p):
        return {q for q in range(n) if B_nbr[p, q]}
    sel_pos = [pos[v] for v in vuln]

    def diversity_of(cp):
        types = set()
        cln = involved[cp]
        for sp in sel_pos:
            sln = involved[sp]
            for pair in ((cln, sln), (sln, cln)):
                if pair in pair_types:
                    types |= {REL_TYPE_TO_KEY[t] for t in pair_types[pair]}
        return len(types) / 3.0

    def redundancy_of(cp):
        nb = nbr_of(cp)
        if not nb or not sel_pos:
            return 0.0
        best = 0.0
        for sp in sel_pos:
            nj = nbr_of(sp)
            if not nj: continue
            union = len(nb | nj)
            if union == 0: continue
            jac = len(nb & nj) / union
            if jac > best: best = jac
        return best

    while len(sel_pos) < MC and cand_pos:
        best_p = None; best_s = -1e18
        for cp in cand_pos:
            s_ = Z1 * z1h[cp] + Z2 * z2h[cp] + DV * diversity_of(cp) - RD * redundancy_of(cp)
            if s_ > best_s:
                best_s = s_; best_p = cp
        if best_p is None or best_s <= 0:
            break
        sel_pos.append(best_p); cand_pos.remove(best_p)

    crit = sorted(involved[p] for p in sel_pos)
    cs = set(crit)
    code_lines = code.splitlines()
    crit_records = [
        {
            "line": line,
            "code": code_lines[line - 1].rstrip() if 0 < line <= len(code_lines) else "",
            "role": "vulnerability_anchor" if line in vuln else "graph_context",
        }
        for line in crit
    ]
    rels = [
        {"src": i, "dst": j, "type": t, "weight": round(KEY_TO_WEIGHT[REL_TYPE_TO_KEY[t]], 3)}
        for (i, j), ts in pair_types.items() if i in cs and j in cs
        for t in sorted(ts)
    ]
    rels.sort(key=lambda r: (r["src"], r["dst"], r["type"]))
    svl = sorted([v for v in vuln if v in cs])
    om = sorted([v for v in vuln if v not in cs])
    if len(vuln) > MC: status = "joern_multirelation_anchor_cap"
    elif not a0: status = "joern_ast_fallback"
    else: status = "joern_multirelation"
    return crit_records, rels, status, svl, om
def main():
    print("=" * 50)
    print("CRIT v2")
    print("=" * 50)
    ss = ls()
    n = len(ss)
    nv = sum(1 for s in ss if s["_r"] >= 0.99)
    print(f"[1] Samples: {n} (verified={nv}, joern-cand={n-nv})")

    print("[2] Joern per-sample PDG/CFG export...")
    WD.mkdir(parents=True, exist_ok=True)
    cd = WD / "c_files"; cd.mkdir(exist_ok=True)
    for i, s in enumerate(ss):
        with open(cd / f"func_{i:04d}.c", "w", encoding="utf-8") as f:
            f.write(sc(s["_c"]).replace("::", "__"))
    print(f"  {n} C files written")

    per_sample_edges = []
    edge_type_counter = Counter()
    tmp_cpg = WD / "_tmp_cpg.bin"
    for i, s in enumerate(ss):
        src_c = cd / f"func_{i:04d}.c"
        if tmp_cpg.exists():
            tmp_cpg.unlink()
        r = run([JP, str(src_c), "--output", str(tmp_cpg)], 600)
        if r is None or r.returncode != 0 or not tmp_cpg.exists():
            per_sample_edges.append([])
            continue
        edges = []
        for repr_name in ("pdg", "cfg"):
            out_dir = WD / f"_export_{repr_name}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            r2 = run([JX, str(tmp_cpg), "--repr", repr_name, "--format", "dot", "--out", str(out_dir)], 900)
            if r2 is None or r2.returncode != 0:
                continue
            for dp in sorted(out_dir.glob("*.dot")):
                m, _, dedges = _parse_dot(dp)
                if not m or m.startswith("<"):
                    continue
                for e in dedges:
                    t = e.get("_t")
                    if repr_name == "pdg":
                        if t in ("DDG", "CDG"):
                            edges.append({"src": e["src"], "dst": e["dst"], "type": t})
                    else:
                        edges.append({"src": e["src"], "dst": e["dst"], "type": "CFG"})
        per_sample_edges.append(edges)
        for e in edges:
            edge_type_counter[e["type"]] += 1
        if (i + 1) % 50 == 0:
            print(f"  exported {i + 1}/{n}...")
    ecnt = dict(edge_type_counter)
    print(f"  Edge types: {ecnt}")

    print("[3/4] Matrix construction and critical-line selection...")
    res = []
    for i, s in enumerate(ss):
        code = s["_c"]
        valid = vls(code)
        flat = per_sample_edges[i] if i < len(per_sample_edges) else []
        crit, rels, _status, svl, _om = _select_critical(code, s["_vl"], valid, flat)
        res.append(dict(sample_idx=i, critical_lines=crit, line_relations=rels,
                       selected_vulnerable_lines=svl))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}...")
    print(f"  {n}/{n} done")

    print("[5] Saving...")
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    tc = sum(len(r["critical_lines"]) for r in res)
    tr = sum(len(r["line_relations"]) for r in res)
    med = sorted([len(r["critical_lines"]) for r in res])[len(res) // 2] if res else 0
    print(f"  {OUT}")
    print(chr(10) + f"DONE | Crit:{tc} Med:{med} Rel:{tr}")

if __name__ == "__main__":
    main()
