# MKE

**Knowledge-Guided Multi-Granularity Subgraph Selection for GNN Explanations**

[中文说明](README_zh.md) · [Method](docs/METHOD.md) · [Configuration](docs/CONFIGURATION.md) · [Experiments](docs/EXPERIMENTS.md)

MKE explains a trained graph classifier by selecting molecular substructure candidates. It uses Shapley group values and greedy positive-gain selection under a budget on the union of selected atoms.

The repository includes training and evaluation code, six experiment CSVs, and six historical GCN checkpoints. Dataset splits and known data issues are documented in [DATASETS.md](docs/DATASETS.md).

## Layout

```text
data/origin_data/   Molecular CSVs
checkpoints/       GCN weights, grouped by dataset
config/            Dataset, model and explanation settings
mke_support/       Data loading, GCN, pooling and scoring
scripts/           Experiment, analysis and visualization tools
tests/             Tool tests
docs/              Detailed guides
mke_new.py         MKE entry
train_gcn.py       Predictor training
evaluate_gcn.py    Predictor evaluation
```

## Quick start

Run from the repository root in an environment compatible with the GCN's legacy PyG interfaces; see [installation](docs/INSTALLATION.md).

```bash
python scripts/check_environment.py --smoke
python scripts/check_inputs.py --dataset Mutagenicity
python evaluate_gcn.py datasets=Mutagenicity device_id=0
python mke_new.py datasets=Mutagenicity device_id=0 target_sparsity=0.5
```

Dataset keys are `Mutagenicity`, `bbbp`, `hERG`, `Tox21_AhR`, `Bace`, and `clintox`. The node budget is `floor((1 - target_sparsity) * num_nodes)`; results record actual sparsity separately.

Tox21 AhR and ClinTox contain split overlaps, and AhR also contains conflicting labels. Review the [data notes](docs/DATASETS.md#known-data-issues) and checkpoint provenance before interpreting results.

## Training and batch experiments

```bash
python train_gcn.py datasets=Mutagenicity device_id=0 models.gnn_saving_dir=checkpoints_new
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --device 0
python scripts/summarize_results.py --results results --output reports
python scripts/plot_results.py --input reports/aggregated.csv --output figures
```

Training uses Adam, cross-entropy and validation-accuracy checkpoint selection. Use a new checkpoint directory for a new predictor. See the [experiment guide](docs/EXPERIMENTS.md) for seeds, external paths and molecular drawings.

## Source checks

```bash
python scripts/verify_repository.py --check-manifest
python test_mke_new_group_shapley.py
python -m unittest discover -s tests -v
```

The 52 lightweight tests use only the standard library; model and GPU validation require the ML environment.

[Output format](docs/OUTPUTS.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [GPLv3 license](LICENSE) · [Source attribution](THIRD_PARTY_NOTICES.md)
