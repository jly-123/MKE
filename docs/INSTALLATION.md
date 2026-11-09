# Installation

Run commands from the repository root. The scripts use Hydra configuration and do not require installing this repository as a Python package.

## Runtime environment

Use the project's working GCN environment when available. `requirements.txt` lists dependencies without pinned versions. The GCN uses legacy PyG interfaces (`__check_input__`, `__collect__` and the message-passing inspector), so PyTorch, PyG, torch-scatter and torch-sparse must be installed as a compatible set for the selected Python/CUDA version.

```bash
python scripts/check_environment.py --smoke --output environment-report.json
python -m pip freeze > environment-server.txt
```

The smoke check constructs the GCN and runs a small CPU forward pass; it does not load a checkpoint. Record the working environment with experiment results.

## Optional figures

```bash
python -m pip install -r requirements-plot.txt
```

Metric plots use Matplotlib. Molecular drawings use RDKit; PNG output requires its Cairo support, while SVG is the default.

## GPU selection

`device_id=0` selects the first CUDA device visible to the process. For example:

```bash
CUDA_VISIBLE_DEVICES=2 python mke_new.py device_id=0
```

The entry uses CPU when CUDA is unavailable. The [README](../README.md#source-checks) lists standard-library checks that run without the ML dependencies.
