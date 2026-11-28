# Vulnerability Descriptions

## File

`Dataset/descriptions/vulnerability_descriptions.json`

- Records: 411
- Every record has a nonempty description and a `sample_id`

## Training file

The 24,000-instance training split is
`Dataset/id/expansion/Ultimate_train_expanded.json`.

The training file contains the 411 descriptions in the Description task. The
task has 823 records total: 411 vulnerable descriptions and 412 non-vulnerable
records with the output `There is no vulnerability.`
