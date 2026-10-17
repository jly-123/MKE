# Source and dependency notices

This package was extracted from a local MKE project based on **DIG: Dive into Graphs**, maintained by DIVE Lab, Texas A&M University. The source repository's `setup.py` identifies `dive_into_graphs`, links to <https://github.com/divelab/DIG>, and declares GPLv3; its `dig/version.py` reports version 1.0.0. The original repository's GPL version 3 license is included as `LICENSE`.

This extraction retains required support code; it must not be understood as a claim that every included utility was originally authored for MKE. The local repository contains modifications, and this package does not assert byte identity with an upstream DIG release.

| Packaged source | Local source/provenance |
| --- | --- |
| `mke_new.py` | Local MKE candidate generation and new merged-player Shapley group-value greedy entry |
| `mke_support/gnn.py` | `benchmarks/xgraph/gnnNets.py`, GCN/model construction code |
| `mke_support/pooling.py` | Required pooling base from `dig/xgraph/models/models.py` |
| `mke_support/scoring.py` | Required graph perturbation and batched probability scorer from `dig/xgraph/method/subgraphx_optimized.py` |
| `mke_support/data.py` | Local molecular graph builder from `dig/xgraph/dataset/my_molecular_data.py` |
| `mke_support/utils.py` | Required utilities from `benchmarks/xgraph/utils.py`, dataloader from `benchmarks/xgraph/dataset.py`, and checkpoint conversion from `dig/xgraph/utils/compatibility.py` |
| `train_gcn.py` | Optional predictor trainer derived from `benchmarks/xgraph/train_gnns.py` |
| `config/` | Six molecular dataset/GCN configurations from the local experiment configuration; CSV/task mappings follow the existing dataset interfaces |
| `evaluate_gcn.py`, `scripts/`, `mke_support/release.py`, `tests/` | Release utilities for predictor evaluation, input checking, experiment orchestration, results and validation |

Packaging changes make imports, input/output paths, and configuration self-contained. The explanation core, candidate extraction, GCN, and default `split` scoring behavior are retained from the current local implementation. Other explanation methods have not been bundled.

External libraries are listed in `requirements.txt`; their implementations are not vendored in this archive. Their own licenses and notices apply to the separately installed distributions. No datasets, pretrained weights, or model results are distributed here.
