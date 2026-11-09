# Installation and environment

Run commands from the repository root. The main entry is a Python script with Hydra configuration; installing this repository as a Python package is unnecessary.

## Use the existing experiment environment

The safest starting point for an existing checkpoint is the Python environment that was used to run the project's GCN. Activate that environment, then inspect it:

```bash
python --version
python scripts/check_environment.py
python scripts/check_environment.py --smoke --output environment-report.json
python -m pip freeze > environment-server.txt
```

The environment checker does not install or upgrade packages. The optional smoke check constructs the included GCN and performs a small CPU forward pass. It checks the actual model interface, rather than treating a successful import as sufficient. It does not load an experiment checkpoint or run a dataset.

`requirements.txt` lists runtime dependencies. It is not a version lockfile. The custom GCN uses legacy PyTorch Geometric interfaces such as `__check_input__`, `__collect__`, and its message-passing inspector. A newer PyG release may remove these interfaces. PyTorch, PyG, torch-scatter, and torch-sparse must be installed as a compatible set for the selected Python/CUDA environment. Do not replace a working machine-learning stack with the latest packages just to run this source release.

When creating another machine's environment, use the recorded working environment and the matching platform-specific PyTorch/PyG wheels. Retain the environment report with experiment results. A `pip freeze` file records packages but does not itself supply CUDA drivers or portable binary wheels.

## Optional figures

```bash
python -m pip install -r requirements-plot.txt
```

The figure tool uses Matplotlib; molecular rendering uses RDKit already listed in runtime requirements. PNG molecular output additionally requires RDKit's Cairo drawing support. SVG is the default and avoids that requirement.

## Source checks without machine-learning libraries

```bash
python scripts/verify_repository.py
python test_mke_new_group_shapley.py
python -m unittest discover -s tests -v
```

These commands use only Python's standard library. The source is checked against Python 3.7 grammar. GitHub Actions runs the lightweight tests on Python 3.9 and 3.12; this CI matrix is not a declaration that the legacy GNN dependency stack supports either environment.

## GPU selection

`device_id=0` selects the first CUDA device visible to the Python process. For example, on Linux:

```bash
CUDA_VISIBLE_DEVICES=2 python mke_new.py device_id=0
```

This selects the device exposed as logical GPU 0, after visibility filtering. If CUDA is unavailable, the entry uses CPU. GPU out-of-memory errors while loading the model usually require checking existing GPU usage before changing explanation settings.
