import hashlib, json, os, re, subprocess, sys, locale
from collections import Counter, defaultdict
from pathlib import Path

SYS_ENC = locale.getpreferredencoding()

BASE = Path(__file__).resolve().parents[3]  # Dataset 目录
INNO = BASE / "graph_context" / "critical_lines"
TRAIN = BASE / "ablations" / "direct_cpg" / "Ultimate_train_cpg_description_only_plus_grouped_2019_2026_detection.json"
VL = INNO / "output" / "train_vul_lines.json"
OUT = INNO / "output" / "train_crit.json"
REP = INNO / "output" / "report.json"
WD = BASE / "graph_context" / "cpg_evidence" / "joern_work_crit"
SC_SC = INNO / "scripts" / "crit.sc"
J = r"PATH\TO\joern.bat"
JP = r"PATH\TO\joern-parse.bat"
EW = {"REACHING_DEF":0.50, "CONTROL_DEPENDENCE":0.30, "CFG_NEXT":0.20, "AST_CHILD":0.10}
MC = 20; Z1=0.45; Z2=0.25; DV=0.20; RD=0.10; AV=1.0; AJ=0.7

def run(cmd, to=600):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=to, encoding=SYS_ENC, errors="replace")
    except: return None

def nc(c):
    return "\n".join(l.rstrip() for l in c.splitlines() if l.strip())

def ec(i):
    if "[Code]" not in i: return i.strip()
    s = i.index("[Code]")+6; c = i[s:]
    if "[CPG Evidence]" in c: c = c[:c.index("[CPG Evidence]")]
    return c.strip()

def vl(s):
    ln = s.strip()
    if not ln or ln.startswith(("//","/*","*","#")) or ln in ("{","}"): return False
    return True

def vls(c):
    ls = c.splitlines()
    return {i+1 for i,l in enumerate(ls) if vl(l)}

def sc(code):
    for m in ["__user","__kernel","__iomem","__force","__init","__exit",
              "__must_check","__maybe_unused","__read_mostly","__ref",
              "OMX_IN","OMX_OUT","OMX_INOUT","IN","OUT","INOUT",
              "TSRMLS_DC","TSRMLS_CC","TSRMLS_C","TSRMLS_D"]:
        code = re.sub(r'\b'+re.escape(m)+r'\b','',code)
    return re.sub(r'__always_inline\b','inline',code)

def ls():
    with open(TRAIN,"r",encoding="utf-8") as f: d=json.load(f)
    ss=[]
    for x in d:
        if x.get("Task")!="description": continue
        o=x.get("output","").strip().lower()
        if o in ("no vulnerability","n/a","") or "no vulnerability" in o: continue
        ss.append(x)
    with open(VL,"r",encoding="utf-8") as f: vd=json.load(f)
    for i,s in enumerate(ss):
        c=ec(s["input"])
        v=vd[i] if i<len(vd) else {}
        s["_c"]=c; s["_vl"]=v.get("vuln_lines",[])
        s["_r"]=v.get("reliability",0.7 if not v.get("matched") else 1.0)
    return ss

def main():
    print("="*50)
    print("CRIT v2")
    print("="*50)
    ss=ls()
    n=len(ss); nv=sum(1 for s in ss if s["_r"]>=0.99)
    print(f"[1] Samples: {n} (verified={nv}, joern-cand={n-nv})")

    print("[2] Joern...")
    WD.mkdir(parents=True,exist_ok=True)
    cd=WD/"c_files"; cd.mkdir(exist_ok=True)
    for i,s in enumerate(ss):
        with open(cd/f"func_{i:04d}.c","w",encoding="utf-8") as f: f.write(sc(s["_c"]))
    print(f"  {n} C files written")

    cpg=WD/"cpg.bin"
    if not cpg.exists():
        r=run([JP,str(cd),"--output",str(cpg)],600)
        if r is None or r.returncode!=0: raise RuntimeError("joern-parse failed")
    print(f"  CPG: {cpg.stat().st_size/1e6:.1f} MB")

    ef=WD/"all_edges.json"
    if not ef.exists():
        # Use -D system properties
        r=run([J,"-DcpgFile="+str(cpg),"-DoutFile="+str(ef),"--script",str(SC_SC)],900)
        if r is None: raise RuntimeError("Scala failed")
        if not ef.exists():
            raise RuntimeError(f"Scala no output: {r.stderr[:300] if r.stderr else '?'}")
    print(f"  Edges: {ef.stat().st_size/1e6:.1f} MB")

    with open(ef,"r",encoding="utf-8") as f: ae=json.load(f)
    ecnt=Counter(e.get("type","?") for e in ae)
    print(f"  Edge types: {dict(ecnt)}")

    print("[3] Quality matrix...")
    fe=defaultdict(list)
    for e in ae:
        fn=e.get("src",{}).get("file","") or e.get("dst",{}).get("file","")
        fe[fn].append(e)
    sa=[]
    for i,s in enumerate(ss):
        fn=f"func_{i:04d}.c"
        edges=fe.get(fn,[])
        al=defaultdict(lambda: defaultdict(float))
        for e in edges:
            sl=e["src"].get("line"); dl=e["dst"].get("line"); et=e.get("type","")
            if sl is None or dl is None or sl==dl: continue
            w=EW.get(et,0.05)
            al[sl][dl]+=w; al[dl][sl]+=w*0.5
        af=[]
        for src,dsts in al.items():
            tw=sum(dsts.values())
            for dst,w in dsts.items(): af.append({"src":src,"dst":dst,"weight":w/max(tw,1.0)})
        sa.append(af)

    print("[4] Scoring...")
    res=[]; st=Counter()
    for i,s in enumerate(ss):
        valid=vls(s["_c"])
        vuln=set(s["_vl"])
        adj=sa[i]
        iv=s["_r"]>=0.99
        am={v:(AV if iv else AJ) for v in vuln}
        if not vuln and not iv:
            deg=defaultdict(int)
            for e in adj: deg[e["src"]]+=e["weight"]; deg[e["dst"]]+=e["weight"]
            top=sorted(deg.items(),key=lambda x:-x[1])[:3]
            vuln=set([l for l,d in top if l in valid])
            for v in vuln: am[v]=AJ
        if not vuln:
            status="no_usable_joern_edges"
            crit,rels,svl,om=[],[],[],[]
        else:
            # propagate
            scores={}
            if adj:
                nbr=defaultdict(float)
                for e in adj:
                    src,dst,w=e["src"],e["dst"],e["weight"]
                    for a in vuln:
                        if src==a and dst in valid: nbr[dst]+=w*am.get(a,0.7)
                th=defaultdict(float)
                for e in adj:
                    src,dst,w=e["src"],e["dst"],e["weight"]
                    for nn,ns in nbr.items():
                        if src==nn and dst in valid and dst not in vuln: th[dst]+=w*ns*0.5
                for a in vuln:
                    ac=sum(e["weight"] for e in adj if e["src"]==a or e["dst"]==a)
                    scores[a]=am.get(a,0.7)+0.1*min(ac,1.0)
                for l,s in nbr.items():
                    if l not in scores: scores[l]=s
                for l,s in th.items():
                    if l not in scores: scores[l]=s
            # select critical
            if not scores:
                crit=sorted(vuln)[:MC]
            else:
                ms=max(scores.values()) if scores else 1.0
                nm={l:s/ms for l,s in scores.items()}
                cands=list(set(list(nm.keys())+list(vuln)))
                cands=[c for c in cands if c in valid]
                fs={}
                for c in cands:
                    base=nm.get(c,0.0); vb=0.2 if c in vuln else 0.0
                    div=0.1*(1.0/(1.0+len([x for x in fs if abs(x-c)<=2])))
                    red=0.05*len([x for x in fs if abs(x-c)<=1])
                    fs[c]=Z1*base+Z2*vb+DV*div-RD*red
                sc2=sorted(fs.items(),key=lambda x:-x[1])
                crit=sorted([l for l,s in sc2[:MC]])
            # relations
            cs=set(crit)
            rels=[{"src":e["src"],"dst":e["dst"],"weight":round(e["weight"],3)}
                  for e in adj if e["src"] in cs and e["dst"] in cs]
            svl=sorted([v for v in vuln if v in cs])
            om=sorted([v for v in vuln if v not in cs])
            status=("joern_multirelation_anchor_cap" if len(vuln)>MC
                    else ("joern_ast_fallback" if not adj else "joern_multirelation"))
        st[status]+=1
        res.append(dict(sample_idx=i,critical_lines=crit,line_relations=rels,
                       critical_line_status=status,selected_vulnerable_lines=svl,
                       omitted_vulnerable_lines=om))
        if (i+1)%100==0: print(f"  {i+1}/{n}...")
    print(f"  {n}/{n} done")

    print("[5] Saving...")
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT,"w",encoding="utf-8") as f: json.dump(res,f,indent=2,ensure_ascii=False)
    tc=sum(len(r["critical_lines"]) for r in res)
    tr=sum(len(r["line_relations"]) for r in res)
    med=sorted([len(r["critical_lines"]) for r in res])[len(res)//2] if res else 0
    rp=dict(total=n,verified=nv,joern_cand=n-nv,total_crit=tc,total_rel=tr,
            median_crit=med,cap=MC,status=dict(st),edges=dict(ecnt))
    with open(REP,"w",encoding="utf-8") as f: json.dump(rp,f,indent=2,ensure_ascii=False)
    print(f"  {OUT}\n  {REP}")
    print(f"\nDONE | Crit:{tc} Med:{med} Rel:{tr} | {dict(st)}")

if __name__=="__main__": main()
