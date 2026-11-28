# Critical Line Selection

This folder contains the outputs and scripts for the graph-structured context
extraction step, which selects critical source lines and typed line relations
from the Code Property Graph (CPG).

## Outputs

- `output/train_crit.json`: selected critical lines and typed line relations for each vulnerable Description sample
- `output/train_vul_lines.json`: identified vulnerable line numbers for each sample

The outputs contain 411 samples with critical lines and typed line relations.

## Scripts

- `scripts/crit.py`: end-to-end pipeline that (1) exports DDG/CDG/CFG edges via Joern, (2) constructs the relation matrices and (3) selects critical lines using the scoring equations
- `scripts/export_graph_context.py`: exports the critical-line outputs from `Dataset/descriptions/vulnerability_descriptions.json`
- `scripts/extract_vul_lines.py`: extracts vulnerable lines from source data

## Requirements

- Python 3.10+ with `numpy`
- A local Joern installation (`joern-parse` and `joern-export`) for `crit.py`

Add Joern to `PATH`, or set `JOERN_PARSE` and `JOERN_EXPORT` to the local
executables before running.

`extract_vul_lines.py` expects external source files under `source_data/` and
an MSR dataset path in `MSR_PATH`.

## Rebuilding the outputs

```bash
python Dataset/graph_context/critical_lines/scripts/export_graph_context.py
```

This reads the canonical 411 Description records and writes `train_crit.json`
and `train_vul_lines.json` under `output/`.

## Recomputing from source with Joern

```bash
python Dataset/graph_context/critical_lines/scripts/crit.py
```

For every vulnerable Description sample, `crit.py`:

1. Writes the function to `c_files/func_XXXX.c` after replacing `::` with `__`
   in the parsing input; original line numbering is preserved.
2. Runs `joern-parse` on that single file to build a per-sample CPG.
3. Runs `joern-export --repr pdg` (yields `DDG` / `CDG` edges) and `joern-export --repr cfg` (yields `CFG` edges).
4. Parses the exported `.dot` files and maps graph edges back to source line numbers.
5. Applies the graph-context equations (mask, weighted relation matrix, symmetric normalization, two-hop propagation, greedy critical-score selection) to output `critical_lines` and typed `line_relations`.

`crit.py` parses and exports each sample individually, so a full run can take a while.

## Output schema

Each entry of `train_crit.json` has the form:

```json
{
  "sample_idx": 0,
  "critical_lines": [
    {"line": 1, "code": "int foo(...)", "role": "graph_context"},
    {"line": 14, "code": "if (keyring) {", "role": "vulnerability_anchor"}
  ],
  "line_relations": [
    {"src": 1, "dst": 14, "type": "DDG", "weight": 0.5},
    {"src": 1, "dst": 14, "type": "CFG", "weight": 0.2}
  ],
  "selected_vulnerable_lines": [14]
}
```

Relation entries keep their type; the selection weight is 0.50 for `DDG`,
0.30 for `CDG`, and 0.20 for `CFG`.
