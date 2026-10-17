# MKE

**Knowledge-Guided Multi-Granularity Subgraph Selection for GNN Explanations**

[中文说明](README_zh.md) · [Installation](docs/INSTALLATION.md) · [Method](docs/METHOD.md) · [Experiments](docs/EXPERIMENTS.md)

MKE constructs molecular substructure candidates and selects combinations to explain a fixed graph classifier. Selection maximizes positive marginal gains in the **Shapley group value**, subject to a budget on the union of distinct selected nodes. Selected candidates remain intact; overlapping atoms are counted once in the node budget.

This repository contains the MKE method, GCN and molecular data utilities, predictor training/evaluation, experiment automation, result analysis, and tests. Data and trained weights are supplied separately.

## Repository contents

| Path | Purpose |
| --- | --- |
| `mke_new.py` | Candidate generation, group values, greedy selection and explanation evaluation |
| `train_gcn.py` | Optional predictor training and checkpoint selection |
| `evaluate_gcn.py` | Standalone predictor accuracy and confusion-matrix evaluation |
| `mke_support/` | GCN, pooling, molecular graphs, perturbation scoring and utilities |
| `config/` | Six dataset configurations, model parameters and runtime settings |
| `scripts/check_environment.py` | Dependency inventory and optional CPU model forward check |
| `scripts/check_inputs.py` | CSV schema, split, label and file-hash checks |
| `scripts/run_experiments.py` | Sequential datasets × budgets × seeds runner with logs |
| `scripts/summarize_results.py` | Validate and recompute unrounded metrics |
| `scripts/plot_results.py` | Metric figures in PNG, JPG and PDF |
| `scripts/render_explanations.py` | Molecular explanation drawings |
| `scripts/verify_repository.py` | Source, configuration and checksum checks |
| `test_mke_new_group_shapley.py`, `tests/` | Mathematical and experiment-tool tests |
| `data/`, `checkpoints/` | Input instructions and expected directory layouts |
| `docs/`, `.github/` | Detailed guides, lightweight CI and contribution templates |

## Quick start

Use the environment associated with the project's GCN. The model uses legacy PyG interfaces; `requirements.txt` is a dependency list, not a validated version lockfile. See [installation](docs/INSTALLATION.md).

```bash
python scripts/check_environment.py --smoke
```

Place the default inputs at:

```text
data/origin_data/Mutagenicity.csv
checkpoints/Mutagenicity/gcn_3l_best.pth
```

The CSV requires `smiles`, `Mutagenicity`, and `group`. The split values are `training`, `valid`, and `test`. The checkpoint contains a compatible `net` state dictionary.

```bash
python scripts/check_inputs.py --dataset Mutagenicity
python evaluate_gcn.py datasets=Mutagenicity device_id=0
python mke_new.py datasets=Mutagenicity device_id=0 target_sparsity=0.5
```

The predictor entry evaluates accuracy before explanation selection. The explanation entry includes every retained test graph, including empty explanations, in its reported means.

Existing server inputs can be used without copying:

```bash
python mke_new.py datasets=Mutagenicity device_id=0 input_csv=/absolute/path/Mutagenicity.csv models.gnn_saving_dir=/absolute/path/checkpoints
```

The checkpoint directory is the parent of the dataset subdirectory. Relative input/output paths resolve against this repository root. GPU 0 means the first device visible to the process.

## Datasets and budgets

| Hydra key | CSV filename | Label column | GCN hidden dimension |
| --- | --- | --- | --- |
| `Mutagenicity` | `Mutagenicity.csv` | `Mutagenicity` | 256 |
| `bbbp` | `BBBP.csv` | `BBBP` | 128 |
| `hERG` | `hERG.csv` | `hERG` | 256 |
| `Tox21_AhR` | `Tox21_AhR.csv` | `AhR` | 256 |
| `Bace` | `Bace.csv` | `Bace` | 256 |
| `clintox` | `clintox.csv` | `CT_TOX` | 256 |

Preserve the CSV's existing split assignments and pair the task column with its trained checkpoint. See [DATASETS.md](docs/DATASETS.md) for preprocessing, counts and label conventions.

```bash
python mke_new.py datasets=bbbp target_sparsity=0.6 device_id=0
python mke_new.py datasets=Bace target_sparsity=0.7 device_id=0
```

The budget is `floor((1 - target_sparsity) * num_nodes)`. Actual sparsity is `1 - selected_node_count / num_nodes`. Whole candidates, integer budgets and positive-gain stopping can leave budget unused; target and measured sparsity are recorded separately.

## Batch experiments

Preview the six datasets and three budgets without executing models, then run sequentially:

```bash
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --device 0
```

Choose datasets and sampling seeds:

```bash
python scripts/run_experiments.py --datasets Mutagenicity bbbp --sparsities 0.5 0.6 0.7 --seeds 0 1 2 --device 0
```

The runner checks inputs, records commands and logs, and stops on the first failed job. These explanation seeds vary permutation sampling on a shared fixed predictor. [EXPERIMENTS.md](docs/EXPERIMENTS.md) covers external paths, training seeds and runtime overrides.

## Results and figures

Each explanation run saves per-graph probabilities, Fidelity±, actual sparsity, selected nodes/candidates, stopping reasons, greedy traces, effective configuration and input/checkpoint hashes. See [output fields](docs/OUTPUTS.md).

```bash
python scripts/summarize_results.py --results results --output reports
python scripts/plot_results.py --input reports/aggregated.csv --output figures --format png jpg pdf
```

The summarizer recomputes metrics from probabilities and node sets. Different inputs, checkpoints, label tasks and computation settings stay separate. Negative Fidelity values remain signed, and means use unrounded values. Multiple compatible runs report sample standard deviations; a single run does not supply an estimated between-run deviation.

Request a complete six-dataset macro average explicitly:

```bash
python scripts/summarize_results.py --results results --output reports --datasets Mutagenicity bbbp hERG Tox21_AhR Bace clintox
```

Draw selected regions using actual `sample_index` values from a completed run:

```bash
python scripts/render_explanations.py --run /path/to/run_directory --samples 6904 6905 --output figures/explanations
```

SVG is the default; PNG is available with `--format png`.

## Method settings

- Exact enumeration for at most 8 original deduplicated candidates; otherwise 100 Monte Carlo permutations per group by default.
- Largest strictly positive group-value gain among feasible next candidates.
- Distinct-node budget, while preserving candidate identities in the cooperative game.
- Fixed clean predicted target class and predictor parameters during explanation.
- Default `split` perturbation removes edges incident to masked nodes and retains node feature rows.
- Per-graph sampling seed: `random_seed + 1000003 * dataset_index`.

Read [METHOD.md](docs/METHOD.md) and [CONFIGURATION.md](docs/CONFIGURATION.md) for the exact objective, candidate behavior and parameters.

## Training and tests

Use a new checkpoint directory to train another predictor:

```bash
python train_gcn.py datasets=Mutagenicity device_id=0 models.gnn_saving_dir=checkpoints_new
```

Training uses the supplied split, Adam, cross-entropy and validation-accuracy checkpoint selection. Existing best/latest weights are protected by default.

```bash
python scripts/verify_repository.py
python test_mke_new_group_shapley.py
python -m unittest discover -s tests -v
```

The lightweight tests require only the Python standard library. GitHub Actions runs these source, mathematical and tool checks. Dataset/GPU verification is separate; see [VALIDATION.md](VALIDATION.md) for the archive's tested scope.

[Troubleshooting](docs/TROUBLESHOOTING.md) · [GitHub setup](docs/GITHUB.md) · [Contributing](CONTRIBUTING.md) · [Citation](CITATION.md) · [License](LICENSE) · [Source notices](THIRD_PARTY_NOTICES.md)
