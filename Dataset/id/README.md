# In-Distribution Dataset

## Final split

- train: `Dataset/id/expansion/Ultimate_train_expanded.json` (24,000 samples)
- validation: `Dataset/eval/ID_valid.json` (3,000 samples; 1,500 Yes / 1,500 No)
- test: `Dataset/eval/ID_test.json` (3,000 samples; 1,500 Yes / 1,500 No)

The training split has four tasks:

| Task | Count |
|---|---:|
| Detection | 21,537 |
| Description | 823 |
| Localization | 820 |
| Assessment | 820 |

The Description task contains 411 vulnerable descriptions and 412
non-vulnerable records. The canonical description file is
`Dataset/descriptions/vulnerability_descriptions.json`.

## Folders

- `expansion/`: final training split
- `Dataset/descriptions/`: 411 vulnerability descriptions
