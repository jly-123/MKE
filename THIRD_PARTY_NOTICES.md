# Source and dependency notices

MKE includes modified support code from **DIG: Dive into Graphs**, maintained by DIVE Lab, Texas A&M University: <https://github.com/divelab/DIG>. The original GPLv3 license is included as [LICENSE](LICENSE).

The table identifies retained components and their source paths in the local DIG-based project. These files include local modifications.

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


External libraries are listed in `requirements.txt`; their own licenses apply to separately installed distributions. Included data and checkpoint details are in [DATASETS.md](docs/DATASETS.md).
