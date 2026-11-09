# Published data and checkpoint validation

Publication update: 2026-11-09.

## Included artifacts

The repository includes six historical experiment CSVs in `data/origin_data/`
and six historical best GCN checkpoints in `checkpoints/<dataset>/`.
The CSV bytes, row order, labels, split assignments and checkpoint bytes are
unchanged from the supplied local experiment files.

- [`data/input_manifest.json`](../data/input_manifest.json) records CSV SHA-256,
  sizes, task columns, class counts, split counts, duplicates and overlaps.
- [`checkpoints/checkpoint_manifest.json`](../checkpoints/checkpoint_manifest.json)
  records checkpoint SHA-256, sizes and provenance limitations.
- [`FILES_SHA256.txt`](../FILES_SHA256.txt) covers source and documentation.
  Its verifier intentionally excludes CSV and checkpoint payloads; use the two
  artifact manifests to identify those payloads.

## CSV validation and known issues

All six files pass the supplied CSV validator: each has nonempty `smiles`,
the configured binary label, and `group` values `training`, `valid` and `test`.
Both binary classes are present in each full file. These are raw CSV row
counts, not counts after molecular parsing.

| Dataset | Rows | Training | Validation | Test | Exact SMILES shared by training and test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity | 7,672 | 6,137 | 767 | 768 | 0 |
| BBBP | 1,859 | 1,487 | 185 | 187 | 0 |
| hERG | 9,876 | 7,900 | 987 | 989 | 0 |
| Tox21 AhR | 8,166 | 6,533 | 817 | 816 | 221 |
| BACE | 1,513 | 1,265 | 44 | 204 | 0 |
| ClinTox | 1,491 | 1,192 | 149 | 150 | 1 |

Tox21 AhR contains 6,713 unique exact SMILES strings and 1,453 duplicate rows.
There are 467 unique SMILES present in multiple splits: 223 in training and
validation, 221 in training and test, and 41 in validation and test. Pairwise
counts overlap for molecules in all three splits and must not be added together.
For 44 exact SMILES strings, the file contains both AhR labels 0 and 1.
These records need source-level review before constructing a cleaned benchmark.

ClinTox contains seven duplicate rows by exact SMILES; one unique SMILES occurs
in both training and test. No conflicting `CT_TOX` labels were found for identical
SMILES strings.

Exact SMILES comparison strips surrounding whitespace only. No RDKit
canonicalization, salt removal, stereochemistry normalization or scaffold-based
split review was performed. Zero exact-string overlap does not prove absence of
molecular or scaffold overlap.

The published files preserve historical experiment inputs rather than silently
changing the original evaluation protocol. For a new benchmark, resolve duplicate
and conflicting records, split by standardized molecular identity, and retrain
the predictor with the resulting split. A changed CSV/split defines a new
experiment; retain its hashes and training records.

## Historical checkpoint limitations

The six checkpoint files were copied from the local experiment's dataset-specific
best-checkpoint paths. Their sizes and hashes were verified, but they were not
loaded, retrained or evaluated during this update. The available historical
training records do not establish exact input-file hashes or label endpoints.
Do not treat adjacent CSV files, filenames or matching output dimensions as proof
that a checkpoint was trained on a particular task or file version.

In particular, `clintox` defaults to `CT_TOX`, while the CSV also contains
`FDA_APPROVED`. Both tasks use two output classes. Confirm the historical training
endpoint independently, or train a new model on the intended endpoint in a new
checkpoint directory. New training runs save `training_inputs.json` and
`training_config.yaml` to record the selected task and input hash.

## Checks performed

The complete source package passes 52 standard-library tests covering group
values, selection, adapters, predictor summaries, input checks and experiment
tools. Source/configuration and source hash checks also pass. All six CSVs pass
input schema validation and all six checkpoint paths exist.

The publication-validation Python environment lacks Torch, PyG and RDKit.
No full molecular parsing, model forward pass, predictor training, checkpoint
loading, GPU experiment or reproduction of paper metrics was performed here.
Run the environment smoke check and predictor evaluation in a compatible runtime
before interpreting explanation results.

```bash
python scripts/check_environment.py --smoke
python scripts/check_inputs.py --dataset Mutagenicity
python evaluate_gcn.py datasets=Mutagenicity device_id=0
python mke_new.py datasets=Mutagenicity device_id=0 target_sparsity=0.5
```

## 中文说明

仓库附带六份历史实验 CSV 和六个最佳 GCN 检查点，保留原文件字节、行序、标签及划分，
并提供 SHA-256 清单。Tox21_AhR 的训练集与测试集有 221 个相同 SMILES，44 个相同
SMILES 对应冲突标签；ClinTox 有 1 个 SMILES 同时出现在训练集和测试集。

这些文件用于保留历史输入，不代表已完成数据清洗。正式开展新实验前，应核对重复与冲突记录，
按标准化分子身份进行划分，并重新训练预测器。历史权重的训练输入哈希和标签端点没有完整记录；
尤其不能仅凭文件名或输出维度确认 ClinTox 权重对应 `CT_TOX` 还是 `FDA_APPROVED`。

源码的 52 项轻量测试和六份 CSV 格式检查已通过；本次没有执行实际训练、权重加载、
完整分子解析或 GPU 实验，不能据此声称复现论文指标。
