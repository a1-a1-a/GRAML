# Joern CPG Evidence

## `joern_cpg_evidence.json`

This file contains the Joern CPG edge records for the 411 vulnerable
Description samples in the main training split.

- Records: 411
- Edge types: `REACHING_DEF`, `CONTROL_DEPENDENCE`, `CFG_NEXT`, `AST_CHILD`
- Edge fields: `type`, `src`, `dst`, `source`
- `src` and `dst` are source line numbers
- `source` identifies the Joern export used for the edge

Each record also contains `data_flow_edges` (the `REACHING_DEF` subset) and
`control_flow_edges` (the `CONTROL_DEPENDENCE` subset).

For Joern parsing, scope-qualified names are flattened by replacing `::` with
`__`; the edge records retain the original source line numbers.

The records are consumed by `Dataset/scripts/build_direct_cpg_target.py` to
build the Direct CPG supervision target.

## Rebuilding

Generate the base control-flow, control-dependence, and AST records with
`Dataset/scripts/run_local_cpg_description_only.ps1`, then add the
data-dependence records:

```bash
python Dataset/scripts/augment_cpg_evidence_ddg.py
python Dataset/scripts/build_direct_cpg_target.py
```
