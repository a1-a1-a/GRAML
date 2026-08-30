# Juliet OOD Split

This folder stores the GRAML-formatted out-of-distribution data derived from the `Juliet` source.

## Files

- `test.json`: external test split
- `valid.json`: external validation split

## Use Case

This is the only OOD source in the repository that also includes a validation file. It can be useful if you want a source-specific validation stage before reporting final OOD results on `test.json`.
