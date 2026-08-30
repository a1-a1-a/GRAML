import subprocess
from pathlib import Path

cpg = Path(r"D:\path\to\repo\Dataset\graph_context\cpg_evidence\joern_work_crit_example\cpg.bin")
outdir = Path(r"D:\path\to\repo\Dataset\graph_context\cpg_evidence\joern_work_crit_example\pdg_export")
je = r"PATH\TO\joern-cli\joern-export.bat"

cmd = [je, str(cpg), "--repr", "pdg", "--format", "dot", "--out", str(outdir)]
print("Running:", " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                   encoding="utf-8", errors="replace")
print("RC:", r.returncode)
print("STDOUT:", r.stdout[:500] if r.stdout else "none")
print("STDERR:", r.stderr[:500] if r.stderr else "none")
dots = list(outdir.glob("*.dot")) if outdir.exists() else []
print(f"DOT files: {len(dots)}")
if dots:
    with open(dots[0], "r") as f:
        print("Sample:", f.read()[:400])
