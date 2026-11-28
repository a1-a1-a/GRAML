# w/o Graph Evidence Ablation

This folder contains a generator for the w/o Graph Evidence training variant.

The variant keeps the 24,000-record training split and replaces only the 411
positive Description outputs with source-only descriptions. The generation
prompt contains the function code but no critical lines, typed relations, or
CPG evidence.

## Generate

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="<provider-endpoint>"

python Dataset/ablations/graph_evidence/generate_wo_graph_evidence.py \
  --train_path Dataset/id/expansion/Ultimate_train_expanded.json \
  --output_dir Dataset/ablations/graph_evidence \
  --output_train Dataset/ablations/graph_evidence/Ultimate_train_wo_graph_evidence.json \
  --model gpt-5 \
  --workers 4 \
  --resume
```

Inspect the prompt without an API call:

```bash
python Dataset/ablations/graph_evidence/generate_wo_graph_evidence.py --dry_run
```

The generated descriptions are written to `wo_graph_evidence_output.json`, and
the integrated training file is written to the path passed with
`--output_train`.
