# Critical Line Selection

This folder contains the outputs and scripts for the graph-structured context extraction step described in the paper (Sec. 2.2), which selects critical source lines and typed line relations from the Code Property Graph (CPG).

## Outputs

- `output/train_crit.json`: selected critical lines and typed line relations for each vulnerable description sample
- `output/train_vul_lines.json`: identified vulnerable lines with reliability scores (input to `crit.py`)
- `output/report.json`: summary statistics of the critical-line selection process

## Scripts

- `scripts/crit.py`: end-to-end pipeline that (1) exports DDG/CDG/CFG edges via Joern, (2) constructs the relation matrices from the paper, and (3) selects critical lines with the paper equations
- `scripts/extract_vul_lines.py`: extracts vulnerable lines from source data (produces `train_vul_lines.json`)

## Requirements

- Python 3.10+ with `numpy`
- A local Joern installation (`joern-parse` and `joern-export`)

Set the Joern paths at the top of `crit.py` (placeholders `PATH\TO\joern-parse.bat` and `PATH\TO\joern-export.bat`):

```python
JP = r"C:\...\joern-cli\joern-parse.bat"
JX = r"C:\...\joern-cli\joern-export.bat"
```

## How the pipeline works

For every vulnerable Description sample, `crit.py`:

1. Writes the function to `c_files/func_XXXX.c`.
2. Runs `joern-parse` on that single file to build a per-sample CPG.
3. Runs `joern-export --repr pdg` (yields `DDG` / `CDG` edges) and `joern-export --repr cfg` (yields `CFG` edges).
4. Parses the exported `.dot` files and maps graph edges back to source line numbers.
5. Applies the paper equations (mask, weighted relation matrix, symmetric normalization, two-hop propagation, greedy critical-score selection) to output `critical_lines` and typed `line_relations`.

## Usage

```bash
python scripts/crit.py
```

The output schema of `train_crit.json` is:

```json
{
  "sample_idx": 0,
  "critical_lines": [1, 3, 4, 6, 7],
  "line_relations": [{"src": 1, "dst": 7, "weight": 0.7}],
  "critical_line_status": "joern_multirelation",
  "selected_vulnerable_lines": [1],
  "omitted_vulnerable_lines": []
}
```

> Note: `crit.py` parses and exports each sample function individually, which is accurate but can take a while for a large dataset.

