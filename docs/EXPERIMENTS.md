# Experiment guide

Run commands from the repository root. Every script under `scripts/` supports `--help` without loading machine-learning libraries.

## Inputs and predictor

```bash
python scripts/check_inputs.py --dataset Mutagenicity --data-root /path/to/csv_directory --checkpoint-root /path/to/checkpoints
python evaluate_gcn.py datasets=Mutagenicity device_id=0 input_csv=/path/to/Mutagenicity.csv models.gnn_saving_dir=/path/to/checkpoints
```

The input checker verifies file presence, columns, binary labels, exact split names and hashes. It does not parse molecules or load weights. The predictor entry builds the graphs and loads the actual checkpoint. It writes `predictions.csv`, `predictor_summary.json`, `predictor_metadata.json`, and `config.yaml` under `results/<dataset>/gcn/predictor_evaluation/run_*/`. The summary includes accuracy, counts and a confusion matrix. These files are separate from explanation outputs.

## Single explanation run

```bash
python mke_new.py datasets=Mutagenicity device_id=0 target_sparsity=0.5 random_seed=0
```

Changing the seed changes local permutation sampling while the predictor stays fixed. Graph-specific seeds also depend on the constructed dataset index, so preserve input order and preprocessing. Use `explainers.explanation_result_dir=/path/to/results` to select another output root.

Dataset-specific Hydra settings need their nested key:

```bash
python mke_new.py datasets=Mutagenicity explainers.param.Mutagenicity.subgraph_building_method=split
```

See [CONFIGURATION.md](CONFIGURATION.md) for all fields. The effective configuration is recorded with each run.

## Batch execution

```bash
python scripts/run_experiments.py --datasets Mutagenicity bbbp hERG Tox21_AhR Bace clintox --sparsities 0.5 0.6 0.7 --seeds 0 --device 0 --dry-run
```

Remove `--dry-run` to execute. The preview prints the planned arguments without checking data, dependencies or weights. External directories and repeated sampling seeds can be supplied as follows:

```bash
python scripts/run_experiments.py --data-root /path/to/csv_directory --checkpoint-root /path/to/checkpoints --results-root /path/to/mke_results --device 0
python scripts/run_experiments.py --datasets Mutagenicity --seeds 0 1 2 --sparsities 0.5 0.6 0.7 --device 0
```

Runtime overrides are repeatable:

```bash
python scripts/run_experiments.py --datasets Mutagenicity --override performance.batch_size=128 --override performance.explain_batch_size=4
```

All requested input CSVs and checkpoints are checked before the first job. Each job's arguments, log, status, exit code and timestamps are recorded in the batch directory. Execution stops on failure; remaining jobs stay pending. After resolving the issue, invoke only the remaining settings. Repeated completed runs with the same seed are rejected by the summarizer unless explicitly allowed.

## Predictor training

```bash
python train_gcn.py datasets=Mutagenicity random_seed=0 device_id=0 models.gnn_saving_dir=checkpoints_new
```

Training preserves the supplied data split. Best weights are written under `checkpoints_new/Mutagenicity/gcn_3l_best.pth`. `training_config.yaml` and `training_inputs.json` record the configuration, task and CSV identity. Existing best/latest weights are protected by default.

Train multiple datasets or seeds sequentially:

```bash
python scripts/run_experiments.py --mode train --seeds 0 --checkpoint-root checkpoints_new --device 0
python scripts/run_experiments.py --mode train --datasets Mutagenicity --seeds 0 1 2 --checkpoint-root checkpoints_new
```

With multiple training seeds, the checkpoint parents are `checkpoints_new/seed_0`, `seed_1`, and `seed_2`. Point explanation runs at the intended one. Multiple explanation seeds on one predictor and separately trained predictors represent different experiments; the summarizer keeps distinct checkpoint identities separate.

Intentional replacement requires `--overwrite-checkpoints` on the runner or `overwrite_checkpoint=true` on the training entry.

## Aggregation

```bash
python scripts/summarize_results.py --results /path/to/mke_results --output reports
```

- `per_run.csv`: recomputed dataset means, accuracy and counts for each completed run.
- `aggregated.csv`: equal-weight mean of run means, seeds and sample standard deviation within each compatible dataset/input/predictor/protocol group.
- `macro_average.csv`: generated only with `--datasets`; equal-weight means over the specified datasets after run aggregation. Coverage and seed lists must match.

The tool rejects incomplete files and contradictory recorded means. Single-run standard deviations stay blank. Legacy outputs without graph sizes or file hashes are marked with the resulting validation limits. Use a result root containing the intended experiment series; mixed protocols are kept separate and may prevent an unambiguous macro average.

## Figures

```bash
python scripts/plot_results.py --input reports/aggregated.csv --output figures --format png jpg pdf
python scripts/plot_results.py --input reports/aggregated.csv --output figures_actual --x actual --format pdf
```

The default x-axis is target sparsity; `--x actual` uses measured mean sparsity. Lines connect measured points without smoothing, clipping or enforcing a trend. `plotted_points.csv` records each plotted value and series identity. Figures show means; standard deviations remain available in the table.

```bash
python scripts/render_explanations.py --run /path/to/explanation_run --samples 6904 6905 --output figures/molecules --format svg
```

Replace example indices with values from the run's CSV. The drawing highlights original atom IDs and bonds whose endpoints are both selected. Atom indices are shown for inspection.
