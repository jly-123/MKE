# Datasets and checkpoints

Six experiment CSVs are included in `data/origin_data/`, with historical best GCN weights at `checkpoints/<dataset>/gcn_3l_best.pth`. Dataset keys are case-sensitive.

| Dataset key | CSV | Label column |
| --- | --- | --- |
| `Mutagenicity` | `Mutagenicity.csv` | `Mutagenicity` |
| `bbbp` | `BBBP.csv` | `BBBP` |
| `hERG` | `hERG.csv` | `hERG` |
| `Tox21_AhR` | `Tox21_AhR.csv` | `AhR` |
| `Bace` | `Bace.csv` | `Bace` |
| `clintox` | `clintox.csv` | `CT_TOX` |

## Input format

Each CSV requires a nonempty `smiles`, the selected binary label (`0` or `1`), and `group` values `training`, `valid`, or `test`. The checker validates columns, row structure, both classes and all three splits:

```bash
python scripts/check_inputs.py --dataset Mutagenicity
```

The loader preserves stored splits and CSV order. `datasets.seed` does not regenerate splits. To use an external CSV or weight directory:

```bash
python mke_new.py datasets=Bace input_csv=/path/to/Bace.csv models.gnn_saving_dir=/path/to/checkpoints
```

`input_csv` takes precedence over `datasets.dataset_root`. The checkpoint root is the parent of the dataset directory.

## Graph representation

RDKit builds nine categorical atom features: atomic number, chirality, degree, formal charge, hydrogen count, radical electrons, hybridization, aromaticity and ring membership. They are passed to the GCN as floating-point categorical indices. Bonds produce two directed edges; bond features are recorded but not consumed by this GCN.

Unparseable SMILES are skipped. Run metadata records raw rows, valid graphs and skipped rows. `sample_index` is the zero-based index among valid graphs across the full CSV. Unsupported feature values raise an error. Graph caches are written under `data/processed/`; use the sequential runner to avoid concurrent writes to the same cache.

## Known data issues

The included files retain original bytes, labels, row order and historical splits. These counts describe these CSVs before molecular parsing.

| Dataset | Rows | Training | Validation | Test | Exact SMILES shared by training and test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity | 7,672 | 6,137 | 767 | 768 | 0 |
| BBBP | 1,859 | 1,487 | 185 | 187 | 0 |
| hERG | 9,876 | 7,900 | 987 | 989 | 0 |
| Tox21 AhR | 8,166 | 6,533 | 817 | 816 | 221 |
| BACE | 1,513 | 1,265 | 44 | 204 | 0 |
| ClinTox | 1,491 | 1,192 | 149 | 150 | 1 |

- Tox21 AhR has 6,713 unique exact SMILES and 1,453 duplicate rows. There are 467 SMILES in multiple splits: 223 overlap training/validation, 221 training/test, and 41 validation/test. Pairwise counts overlap and cannot be added. For 44 SMILES, both AhR labels occur.
- ClinTox has seven duplicate rows by exact SMILES and one training/test overlap, with no conflicting `CT_TOX` labels for identical strings.
- The other four CSVs have no repeated exact SMILES. BACE's validation set has 44 rows; its stored split is preserved.

Comparison strips surrounding whitespace only; it does not canonicalize molecules or check scaffolds. Zero exact-string overlap does not establish molecular independence. For a new benchmark, review conflicting records, split by standardized molecular identity, and retrain the predictor. Preserve the historical version when reproducing its original protocol.

中文：Tox21_AhR 有 221 个训练/测试重复 SMILES，44 个 SMILES 对应冲突标签；ClinTox 有 1 个训练/测试重复。现有文件保留历史划分，新实验应先核对并清理数据，再重新划分和训练。

## Checkpoints and ClinTox endpoint

Checkpoints contain a `net` state dictionary. BBBP uses three GCN layers of width 128; other datasets use width 256. All configurations use nine input features, two output classes and max pooling; see [model configuration](CONFIGURATION.md).

Historical checkpoint hashes and sizes were verified, but their original training input hashes and label endpoints are not established by the supplied records. Filenames and output dimensions do not establish task provenance. Load only trusted checkpoint files.

ClinTox contains `FDA_APPROVED` and `CT_TOX`; the default configuration selects `CT_TOX`. Confirm the checkpoint's training endpoint independently, or train a new model on the intended task. To select `FDA_APPROVED` with matching weights:

```bash
python mke_new.py datasets=clintox datasets.label_column=FDA_APPROVED models.gnn_saving_dir=/path/to/fda_checkpoints
```

Changing the label option does not retrain the checkpoint. New training writes `training_inputs.json` and `training_config.yaml`, recording the endpoint and CSV hash; see [training](EXPERIMENTS.md#predictor-training).

## File identity and validation

[`input_manifest.json`](../data/input_manifest.json) records CSV hashes, sizes, labels, counts and duplicate statistics. [`checkpoint_manifest.json`](../checkpoints/checkpoint_manifest.json) records weight hashes, sizes and provenance limits. [`FILES_SHA256.txt`](../FILES_SHA256.txt) covers source and documentation; it excludes CSV and weight payloads.

All six CSVs pass schema checks and all six checkpoint files exist. The source has 52 passing lightweight tests. Full molecular parsing, checkpoint loading, training and GPU evaluation have not been verified in the publication environment; run the [environment smoke check](INSTALLATION.md) and predictor evaluation in a compatible ML environment.
