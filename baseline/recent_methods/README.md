# Recent Method Baselines

Reproduction protocol for SAGE, LOSVER, and VulLLM. Each method is run from its
official release with its original modeling design, and the resulting
predictions are evaluated with the shared data and protocol below.

## Configurations

| Method | Configuration | Official implementation | Checkpoint entry |
|---|---|---|---|
| SAGE | `sage.json` | https://github.com/BDS-SDU/SAGE | Train the SAE locally; select on ID validation |
| LOSVER | `losver.json` | https://github.com/waroad/losver | https://doi.org/10.6084/m9.figshare.29192708 |
| VulLLM | `vulllm.json` | https://github.com/CGCL-codes/VulLLM | https://zenodo.org/records/10677069 |

The configurations cover ten backbone-specific runs: four for SAGE, one for
LOSVER, and five for VulLLM. Hyperparameters are stored in the JSON files.

LOSVER reruns its line localizer on every evaluation set so that localized lines
correspond to the evaluated source text. VulLLM uses the released three-task
formulation with detection, localization, and interpretation targets.

## Shared evaluation data

- Training: `Dataset/id/expansion/Ultimate_train_expanded.json`
- Validation: `Dataset/eval/ID_valid.json`
- ID test: `Dataset/eval/ID_test.json`
- OOD test: `Dataset/eval/*_test.json`
- Robustness: `Dataset/robustness/<dataset>/<dataset>_<category>.json`

SAGE and VulLLM select the decision threshold on the ID validation set by F1
and keep it fixed for every evaluation set. LOSVER uses its released fixed
threshold of 0.5.

## Prediction format

Store one JSONL or CSV row per function:

```json
{"sample_id": "...", "score": 0.73, "prediction": 1}
```

Use the same `sample_id` values as the evaluation files.
