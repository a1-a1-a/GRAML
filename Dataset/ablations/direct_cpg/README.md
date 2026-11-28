# Direct CPG Supervision Dataset

This folder contains the Direct CPG supervision variant.

## File

- `Ultimate_train_direct_cpg.json`: 24,000 training records (Detection 21,537, Description 823, Localization 820, Assessment 820).

## Task construction

- The 411 vulnerable Description records use raw Joern CPG edge records as the target `output`.
- Their `input` is the function code used for Joern extraction, so `src` and
  `dst` line numbers index directly into the target input.
- The 412 non-vulnerable Description records keep the output `There is no vulnerability.`
- Detection, Localization, and Assessment records are unchanged from the final training split.

Each target starts with `[Joern CPG]` and then contains one JSON edge record per line:

```text
[Joern CPG]
{"type":"REACHING_DEF","src":15,"dst":41,"source":"joern-cpgql:reaching_def"}
{"type":"CONTROL_DEPENDENCE","src":15,"dst":41,"source":"joern-cpgql:control_dependence"}
{"type":"CFG_NEXT","src":15,"dst":16,"source":"joern-cpgql:cfg_next"}
```

The source records are in
`Dataset/graph_context/cpg_evidence/joern_cpg_evidence.json`. They contain
411 samples with `REACHING_DEF`, `CONTROL_DEPENDENCE`, `CFG_NEXT`, and
`AST_CHILD` edges.

This is target-side CPG supervision: the model learns to produce the Joern CPG
edge records rather than a natural-language vulnerability description.

## Rebuilding

```bash
python Dataset/scripts/build_direct_cpg_target.py \
  --train_path Dataset/id/expansion/Ultimate_train_expanded.json \
  --joern_path Dataset/graph_context/cpg_evidence/joern_cpg_evidence.json \
  --output_path Dataset/ablations/direct_cpg/Ultimate_train_direct_cpg.json
```

The script maps each Description input to its Joern CPG record, writes the edge
records into `output`, keeps the raw function code in `input`, and leaves
non-vulnerable and non-Description records unchanged.
