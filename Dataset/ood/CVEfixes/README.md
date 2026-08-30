# CVEfixes OOD Split

This folder stores the GRAML-formatted out-of-distribution evaluation split derived from the `CVEfixes` source.

## File

- `test.json`: external test set used for OOD evaluation

## Use Case

Use this split when you want to test how well a model trained on the main GRAML in-distribution dataset transfers to the `CVEfixes` data source without retraining on it.
