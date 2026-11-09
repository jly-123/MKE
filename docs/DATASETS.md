# Molecular datasets

The six dataset configurations use a common CSV reader and molecular graph builder. Dataset names are case-sensitive because they select both a configuration and a checkpoint directory.

| Configuration (`datasets=...`) | CSV filename | Default binary label column | Checkpoint directory |
| --- | --- | --- | --- |
| `Mutagenicity` | `Mutagenicity.csv` | `Mutagenicity` | `checkpoints/Mutagenicity/` |
| `bbbp` | `BBBP.csv` | `BBBP` | `checkpoints/bbbp/` |
| `hERG` | `hERG.csv` | `hERG` | `checkpoints/hERG/` |
| `Tox21_AhR` | `Tox21_AhR.csv` | `AhR` | `checkpoints/Tox21_AhR/` |
| `Bace` | `Bace.csv` | `Bace` | `checkpoints/Bace/` |
| `clintox` | `clintox.csv` | `CT_TOX` | `checkpoints/clintox/` |

Place the CSVs in `data/origin_data/`, or set `datasets.dataset_root` to another directory. `input_csv` selects a single explicit file and takes precedence over the configured directory and filename.

```bash
python mke_new.py datasets=bbbp device_id=0
python mke_new.py datasets=Bace input_csv=/absolute/path/Bace.csv device_id=0
```

## CSV contract

Each file must contain:

- `smiles`: a nonempty SMILES string for each molecule.
- `group`: exactly `training`, `valid`, or `test`.
- The configured label column: finite binary values `0` or `1`.

Additional columns are allowed. All three splits and both label values must be present in the full input. Missing columns, duplicate column names, malformed rows, missing labels, and unknown split names fail validation before graph construction. The validator reports the file SHA-256, row count, split counts, and class counts.

Splits come directly from `group`; the program does not shuffle or regenerate the supplied split. In particular, `datasets.seed` does not change these assignments. Preserve the original CSV row order when repeating a run: the processed dataset index also determines the graph's Monte Carlo seed.

## Graph representation

RDKit parses each SMILES in input order. The nine atom features are the categorical indices of atomic number, chirality, total degree, formal charge, total hydrogen count, radical electron count, hybridization, aromaticity, and ring membership, using the fixed maps in `mke_support/data.py`. They are converted to floating-point inputs for the GCN; they are not one-hot encodings.

Each bond becomes two directed edges. The builder also records bond type, stereochemistry, and conjugation as three categorical bond features. Edges are sorted by source and destination. The GCN used here receives atom features, edge indices, and graph assignments; its forward call does not consume `edge_attr`.

Unparseable SMILES are skipped by the graph builder. The saved metadata therefore distinguishes raw CSV rows from valid graphs and reports the number skipped. `sample_index` is the zero-based index among valid graphs across the entire CSV, not a test-only index or an unconditional original CSV row number. Chemical feature values outside the fixed feature maps raise an error rather than being silently remapped.

The generated graph cache is stored under `data/processed/`. It is rebuilt from the selected CSV by the current entry. Do not run multiple jobs writing the same cache or cumulative workbook simultaneously from one checkout; use separate working copies or sequential runs.

## ClinTox endpoint

The local ClinTox CSV contains both `FDA_APPROVED` and `CT_TOX`. The default `CT_TOX` follows the existing controlled dataset interfaces used to prepare this release. It is one binary endpoint, not a two-task ClinTox model.

A checkpoint's filename or tensor dimensions do not establish which endpoint it was trained on. The inspected historical checkpoint does not record its label-column provenance. Use the training records associated with your checkpoint to establish that pairing. To use `FDA_APPROVED`, provide a checkpoint trained on that endpoint and select it explicitly:

```bash
python mke_new.py datasets=clintox datasets.label_column=FDA_APPROVED models.gnn_saving_dir=/absolute/path/fda_checkpoints device_id=0
```

The directory must contain `clintox/gcn_3l_best.pth`. Changing the label-column option alone does not retrain or change the checkpoint. New runs of `train_gcn.py` save the chosen endpoint and input-file hash in `training_inputs.json`.

## Inspected local input sizes

These are raw row counts from the local CSVs inspected during release preparation. They describe those files, not universal sizes for every public version of each dataset. Runtime graph counts may be smaller if SMILES parsing fails.

| CSV | Rows | Training | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| `Mutagenicity.csv` | 7,672 | 6,137 | 767 | 768 |
| `BBBP.csv` | 1,859 | 1,487 | 185 | 187 |
| `hERG.csv` | 9,876 | 7,900 | 987 | 989 |
| `Tox21_AhR.csv` | 8,166 | 6,533 | 817 | 816 |
| `Bace.csv` | 1,513 | 1,265 | 44 | 204 |
| `clintox.csv` | 1,491 | 1,192 | 149 | 150 |

BACE's observed split is deliberately reported as stored; it is not replaced by an assumed 80/10/10 split. Save the input hashes with experiment results to identify the exact version used.

See [configuration](CONFIGURATION.md), [checkpoint preparation](../checkpoints/README.md), and [output fields](OUTPUTS.md).

## Included historical files and known overlaps

The six CSVs are now included. See [DATA_VALIDATION.md](DATA_VALIDATION.md) and [`input_manifest.json`](../data/input_manifest.json) for exact input hashes and split/label issues, including Tox21 AhR and ClinTox overlaps. These files preserve historical splits; they are not a newly cleaned benchmark.
