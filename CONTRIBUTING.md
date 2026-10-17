# Contributing

Keep changes focused and include the command that demonstrates the problem. Changes to candidate generation, perturbation, probability precision, or greedy selection should be treated as algorithm changes and evaluated explicitly.

Run the source checks and dependency-free tests before submitting:

```bash
python scripts/verify_repository.py
python test_mke_new_group_shapley.py
python -m unittest discover -s tests -v
```

For changes involving the GCN or graph loader, also run `python scripts/check_environment.py --smoke` in the intended machine-learning environment and evaluate a compatible checkpoint. Include the dataset/task and configuration used. Mathematical unit tests alone cannot establish GPU or checkpoint compatibility.

Preserve third-party attribution and the GPLv3 license. Do not commit raw datasets, weights, generated run folders, credentials, or machine-specific absolute paths. Keep experiment values unrounded in machine-readable output; round only when presenting them.

The source snapshot includes a checksum manifest. After intentional source changes, regenerate it with `python scripts/verify_repository.py --write-manifest`; verify a downloaded release with `--check-manifest` before making changes.
