# Configuration reference

The explanation and training entries use Hydra with `config/config.yaml`. Run commands from the repository root. Relative data, checkpoint, result, and record-directory paths are resolved against that root, not against Hydra's generated run directory.

Select a dataset with the configuration-group override, for example `datasets=bbbp`. Changing only `datasets.dataset_name` leaves the old CSV filename and label configuration in place and is not the intended way to switch datasets.

## Common options

| Option | Default | Meaning |
| --- | --- | --- |
| `datasets` | `Mutagenicity` | Dataset configuration; see [dataset table](DATASETS.md). |
| `random_seed` | `0` | Base Monte Carlo seed; also the optional trainer's initialization seed. |
| `device_id` | `0` | Logical CUDA device index among the devices visible to the process. CPU is used when CUDA is unavailable. |
| `target_sparsity` | `0.5` | Fraction targeted for exclusion; finite value in `[0, 1]`. |
| `input_csv` | `null` | Optional explicit CSV path, overriding the dataset directory and filename. |
| `overwrite_checkpoint` | `false` | Trainer permission to replace an existing best/latest checkpoint. |
| `models.gnn_saving_dir` | `checkpoints` | Root containing dataset checkpoint subdirectories. |
| `datasets.dataset_root` | `data/origin_data` | CSV directory when `input_csv` is unset. |
| `datasets.csv_filename` | Dataset-specific | Filename resolved inside the CSV directory. |
| `datasets.label_column` | Dataset-specific | Binary endpoint, which must agree with checkpoint training. |
| `datasets.seed` | `2` | Retained data-loader configuration field; CSV groups directly define splits. |
| `explainers.explanation_result_dir` | `results` | Root for unique per-run output directories. |
| `record_filename` | `result_jsons_mke_new` | Legacy record-directory option; authoritative results are the per-run files described in [outputs](OUTPUTS.md). |

```bash
python mke_new.py datasets=hERG target_sparsity=0.7 random_seed=0 device_id=0
python mke_new.py datasets=Bace datasets.dataset_root=/data/molecules models.gnn_saving_dir=/data/checkpoints explainers.explanation_result_dir=/data/results
```

`device_id=0` refers to the first visible CUDA device. If `CUDA_VISIBLE_DEVICES` is set, this may differ from the machine's physical GPU numbering. An invalid visible-device index is rejected.

## Explanation controls

| Option | Default | Meaning |
| --- | ---: | --- |
| `performance.exact_threshold` | `8` | Use exact enumeration when the original deduplicated candidate count is at most this value. |
| `performance.sample_num` | `100` | Permutation samples for each uncached approximate group value. |
| `performance.batch_size` | `1024` | Maximum payoff-query and masked-inference batch size. |
| `performance.explain_batch_size` | `32` | Number of test molecules processed together by the entry. |
| `performance.substructure_workers` | `8` | Maximum candidate-generation worker processes; use `1` for serial execution. |

These configuration values must be positive integers. `exact_threshold` and `sample_num` affect the score estimator; they are not only speed settings. Reducing inference and molecule batch sizes lowers peak memory. Candidate workers use CPU processes and host memory.

```bash
python mke_new.py datasets=Mutagenicity performance.batch_size=64 performance.explain_batch_size=4 performance.substructure_workers=1
```

The active explainer options are stored under `explainers.param.<dataset>`. All supplied configurations use `subgraph_building_method: split` and `reward_method: group_shapley`. The reward-method field names the output directory; selection uses the implemented group-value objective.

To change the perturbation convention deliberately:

```bash
python mke_new.py datasets=bbbp explainers.param.bbbp.subgraph_building_method=zero_filling
```

Supported modes and their semantics are described in [the method document](METHOD.md). Record the mode when comparing runs.

## Predictor and training options

All supplied models are three-layer GCNs with ReLU, normalized adjacency, max pooling, and a linear two-class output. The trainer uses Adam and cross-entropy. Supplied learning rate and weight decay are `0.001` and `0.0005`.

| Dataset key | Hidden dimensions | Training batch size | Maximum epochs | Classifier dropout |
| --- | --- | ---: | ---: | ---: |
| `Mutagenicity` | `256, 256, 256` | 256 | 500 | 0.4 |
| `bbbp` | `128, 128, 128` | 32 | 300 | 0.0 |
| `hERG` | `256, 256, 256` | 128 | 500 | 0.0 |
| `Tox21_AhR` | `256, 256, 256` | 128 | 500 | 0.0 |
| `Bace` | `256, 256, 256` | 64 | 1,000 | 0.0 |
| `clintox` | `256, 256, 256` | 64 | 1,000 | 0.0 |

Training model options are nested under `models.param.<dataset>`. For example:

```bash
python train_gcn.py datasets=bbbp models.gnn_saving_dir=checkpoints_new models.param.bbbp.num_epochs=300 device_id=0
```

The supplied configurations set GNN dropout to zero, use no hidden classifier layers, and set scheduler milestones/gamma to `null`. `num_early_stop: 0` disables early stopping. Checkpoints are selected by validation accuracy. During explanation, dropout is disabled by evaluation mode, explicit self-loops are added to the input graph, and automatic GCN loop insertion is disabled. Training uses the configured automatic self-loop insertion.

Changing architecture options requires a matching checkpoint. The compatibility helper handles legacy names; required model parameter names and shapes must match. Training defaults are configuration settings, not evidence of how an arbitrary external checkpoint was produced. The optional trainer writes `training_config.yaml` and `training_inputs.json` alongside its new checkpoints.

## Run records

Each explanation run saves its resolved configuration, data and checkpoint hashes, active label, sampling settings, metric scope, and per-graph selection decisions. Reuse these records when rerunning or aggregating experiments. See [output fields](OUTPUTS.md).
