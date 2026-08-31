# Critical Line Selection

This folder contains the outputs and scripts for the graph-structured context extraction step described in the paper, which selects critical source lines and typed line relations from the CPG.

## Outputs

- `output/train_crit.json`: selected critical lines and typed line relations for the training samples
- `output/train_vul_lines.json`: identified vulnerable lines with reliability scores
- `output/report.json`: summary statistics of the critical-line selection process

## Scripts

- `scripts/crit.py`: main script that runs Joern to build the CPG, select critical lines, and compute line relations
- `scripts/crit.sc`: Joern query script used to export control-flow / data-flow edges
- `scripts/extract_vul_lines.py`: extracts vulnerable lines from source data

## Usage

The scripts expect a Joern installation. Update the Joern paths at the top of `crit.py` (currently placeholders `PATH\TO\joern.bat` and `PATH\TO\joern-parse.bat`) before running:

```bash
python scripts/crit.py
```
