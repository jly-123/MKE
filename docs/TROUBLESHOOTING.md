# Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Invalid device id` | Inspect `nvidia-smi` and `CUDA_VISIBLE_DEVICES`, then use a valid logical device index; normally start with `device_id=0`. |
| Out of memory while moving the GCN to CUDA | Check free GPU memory and other processes. Explanation batch settings do not reduce memory already occupied by other jobs. |
| Out of memory during explanation | Try `performance.batch_size=128 performance.explain_batch_size=4`; record the change in the run configuration because numerical batching can change floating-point results. |
| `__check_input__`, `__collect__`, or inspector errors | The PyG version is incompatible with the retained GCN. Use the working experiment environment and run the CPU model smoke check. |
| Missing checkpoint or tensor shape mismatch | Check the dataset directory, three-layer model, 9 atom features, 2 output classes, and dataset-specific hidden dimension. The checkpoint needs a `net` state dictionary. |
| Missing column or invalid split/label | Run `scripts/check_inputs.py`. The loader requires `smiles`, the selected binary task column, and exact `training`/`valid`/`test` assignments. |
| Actual sparsity exceeds target | Whole candidates and the integer node budget can leave unused nodes; positive-gain stopping can leave additional budget unused. Report the recorded actual value. |
| Negative Fidelity | Probability differences can be negative. Keep the signed value; do not replace it with zero. |
| Different results across machines | Compare CSV and checkpoint hashes, label/task, library stack, mask convention, precision, batching, and sampling seed. A seed does not guarantee bitwise CUDA equality. |
| No positive-gain candidate | An empty explanation is a valid stopping outcome and remains in the dataset mean. Inspect `group_selection_trace.jsonl`. |
| Result summarizer rejects duplicate seeds | Use the experiment manifest to choose the intended runs, or explicitly allow repeated-seed runs. Do not present repeated seed executions as independent random seeds. |
| Training refuses an existing checkpoint | Choose a new `models.gnn_saving_dir`. Use `overwrite_checkpoint=true` only when intentionally replacing its best/latest weights. |

Hydra prints the effective configuration at launch and writes its own output metadata. Paths for data, weights, and results are resolved against the repository root. For an additional traceback, set `HYDRA_FULL_ERROR=1` before running the command.

The data builder skips SMILES that RDKit cannot parse. The run metadata records raw row counts, valid graph counts, and skipped rows; inspect these before combining results. Test indices refer to the constructed graph dataset, after these skipped rows.

Run multiple experiments with the provided sequential runner. Simultaneous jobs writing the same graph cache or cumulative Excel file are not supported; use separate repository working copies and output locations if independent concurrent execution is required.
