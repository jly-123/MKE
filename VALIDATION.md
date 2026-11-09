# Release validation

## Publication update: 2026-11-09

The omitted source subdirectories have been restored, and six historical CSVs
and six best GCN checkpoints are now included. Source text is normalized to LF;
CSV and checkpoint bytes are preserved. The 52 lightweight tests, six CSV input
checks, source checks and regenerated source manifest pass. Artifact hashes and
known data/checkpoint limitations are documented in
[DATA_VALIDATION.md](docs/DATA_VALIDATION.md). This update does not include a
model or GPU validation run.

## Original archive checks

The following lightweight checks were performed locally for this source archive:

- 21 group-value and adapter tests, including comparison with an independent merged-player permutation oracle.
- 12 input/environment/runner tests, including invalid CSV handling, checkpoint overwrite protection, command construction, dry-run behavior and failed-job propagation.
- 3 predictor-summary tests covering accuracy, confusion-matrix orientation and invalid/empty input handling.
- 16 result-tool tests covering full-precision recomputation, negative values, node budgets, seeds, incomplete files, binary labels, separate predictors/tasks and macro-average coverage.
- Python 3.7 grammar and compilation checks for all Python files, with no runtime ML imports required.
- Zero Python comment tokens or module/class/function docstrings in the source snapshot.
- Six dataset configuration/catalog checks and input schema checks against the locally available CSVs.
- Command-line help and the default 18-job dry run.
- Relative documentation links, standalone local imports and archive checksum checks.

The 16 non-pipeline definitions in `mke_new.py`, including candidate extraction, the group game, greedy selection and GNN adapter, retain identical executable ASTs to the previous source release. The GCN, pooling, scoring code and `TrainModel` computations are unchanged. Release-only entry changes add dataset selection, configurable runtime settings, input checks, overwrite protection and richer run records. Default Mutagenicity inference and selection settings remain the same.

The archive is extracted into a separate directory and its source checks and all 52 tests are run again there. `FILES_SHA256.txt` records archive contents; use `python scripts/verify_repository.py --check-manifest` before making local changes.

No predictor training, complete GNN dataset evaluation or CUDA experiment was performed while preparing this archive. Torch/PyG/RDKit are not installed in the local validation interpreter. Matplotlib figures and RDKit molecular images were not rendered locally; figure-tool input logic and CLI parsing were checked. Use the working server environment for model, checkpoint and rendering validation. Test success is not a reproduction claim for historical numerical results.

The supplied GitHub Actions workflow runs lightweight checks on Python 3.9 and 3.12 after upload. It has not been executed on GitHub as part of local packaging.
