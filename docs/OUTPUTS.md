# Results and selection traces

Each explanation run creates a unique directory:

```text
results/<dataset>/gcn/mke_new_group_shapley/group_shapley/run_<timestamp>/
  mke_new_explanations.csv
  group_selection_trace.jsonl
  run_metadata.json
  run_config.yaml
  summary.json
```

The root can be changed with `explainers.explanation_result_dir`. The default directory name includes the dataset, model, method, configured reward tag, and a timestamp with microseconds.

## Per-graph CSV

`mke_new_explanations.csv` contains one row for every valid test graph. It includes empty explanations and incorrect clean predictions.

| Fields | Meaning |
| --- | --- |
| `sample_index` | Index in the full processed dataset after unparseable SMILES are skipped; not a test-only index. |
| `molecule_smiles`, `explanation_smiles` | Original input and RDKit fragment rendering of the selected nodes. An empty or unrenderable fragment has no explanation SMILES. |
| `explanation_nodes` | JSON list of selected original atom indices. |
| `label`, `prediction` | Binary ground-truth endpoint and fixed clean predicted class. |
| `num_nodes` | Original graph node count. |
| `target_probability` | Clean probability used by the shared scoring path. |
| `explain_value` | Probability after masking the selected node set under the configured perturbation. |
| `explain_no_mask` | Probability under the inverse perturbation retaining the selected set. |
| `fidelity_plus`, `fidelity_minus` | Clean probability minus the respective perturbed probability. |
| `target_sparsity`, `sparsity` | Requested sparsity and actual value `1 - selected_nodes / num_nodes`. |
| `node_budget` | Integer upper bound `floor((1 - target_sparsity) * num_nodes)`. |
| `candidate_count` | Number of nonempty deduplicated candidate node sets. |
| `selected_candidate_indices` | JSON list of selected candidate identities. |
| `group_score` | Final merged-player Shapley group value or its Monte Carlo estimate. |
| `score_method`, `sampling_seed` | `exact` or `monte_carlo`, and the graph-specific seed. |
| `empty_explanation`, `stop_reason` | Whether no nodes were selected and why greedy search stopped. |

The fragment SMILES is for inspection; it does not redefine the perturbation used in scoring. Under `split`, inference retains the full feature matrix while changing edges as detailed in [the method document](METHOD.md).

## Selection trace

`group_selection_trace.jsonl` contains one JSON object per evaluated graph. Each object records:

- The deduplicated candidate node sets and their original candidate-index mappings.
- Target class, original node count, node budget, requested and actual sparsity.
- Selected candidate identities, selected nodes, final score, and stop reason.
- `proposals`: each budget-feasible proposal evaluated at each greedy round, its group score, previous score, marginal gain, and acceptance flag.
- `history`: accepted proposals in order.
- Exact/Monte Carlo mode, threshold, sample count, seed, and counts of payoff queries, payoff batches, computed group scores, and permutations.
- Initial class-selection target probability and recomputed scoring target probability, plus the masked/retained probabilities and Fidelity values.

Infeasible proposals are excluded before score evaluation and do not appear as evaluated proposals. Cached group values may be reused. The trace is sufficient to inspect the selection sequence and its measured gains without presenting each candidate as an independent explanation.

## Configuration and provenance

`run_config.yaml` is the resolved runtime configuration. `run_metadata.json` records the algorithm, input path/hash and validation report, valid/skipped graph counts, label column, checkpoint path/hash, entry-code hash, test count, budget rule, candidate normalization, seed rule, estimator settings, masking convention, precision convention, and metric scope.

Paths in runtime metadata refer to the machine that ran the experiment. Review them before publicly sharing result files. File hashes identify exact inputs; matching a dataset name alone is insufficient.

## Dataset summary

`summary.json` contains arithmetic means of raw per-graph Fidelity+ and Fidelity-, mean actual sparsity, predictor accuracy, requested sparsity, evaluated/empty counts, runtime, seed, run ID, and result directory.

The timer starts after data and checkpoint initialization, immediately before the explanation loop. It includes candidate extraction, explanation inference, trace writing, and the final per-graph CSV export. It does not measure predictor training or initial data/model loading.

A cumulative `mke_new_group_shapley_results.xlsx` is also updated beside the entry script. It is a convenience index; the per-run CSV, trace, configuration, and summary are the detailed record. Run jobs sequentially within one checkout to avoid concurrent writes to this workbook and shared graph-cache files.

## Aggregation and plotting

Use saved numerical values, not rounded console text, for further aggregation. Console summaries display four decimal places; CSV/JSON fields are written without an explicit four-decimal rounding step. Their precision is still limited by the numerical inference path, including mixed precision.

Retain signed Fidelity values. Report actual sparsity alongside the requested budget. Do not replace a negative measurement with zero or relabel actual sparsity as an exact achieved target. When averaging over datasets, distinguish a macro-average of dataset means from a pooled average over molecules. Repeated runs and budgets from the same dataset are not automatically independent datasets.

Plot the measured points without enforcing a desired trend. Different sparsity runs can select different candidate groups, and group-value optimization does not guarantee monotonic Fidelity. Label axes with the requested or actual sparsity according to the values actually plotted. See the repository's experiment workflow for the supplied collection and plotting commands.
