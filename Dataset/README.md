# Dataset Guide

## Final data

| Split | File | Records |
|---|---|---:|
| Training | `id/expansion/Ultimate_train_expanded.json` | 24,000 |
| Validation | `eval/ID_valid.json` | 3,000 |
| ID test | `eval/ID_test.json` | 3,000 |
| OOD test | `eval/{CVEfixes,Devign,DiverseVul,Juliet,PrimeVul,ReVeal}_test.json` | see `eval/README.md` |

The training split contains Detection 21,537, Description 823, Localization 820,
and Assessment 820 records.

## Additional data

- `graph_context/`: critical lines and Joern CPG records for the 411 descriptions
- `ablations/`: multi-task, Direct CPG, and w/o Graph Evidence variants
- `robustness/`: three perturbation variants per test set
- `scripts/`: dataset construction scripts

## Data format

Each record uses `instruction`, `input`, `output`, and a task key (`Task`,
`task`, `category`, or `type`). Scripts accept a JSON list or an object with a
top-level `data` field.
