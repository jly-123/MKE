import argparse
import importlib
import importlib.util
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mke_support.release import DATASET_NAMES


DEPENDENCIES = (("torch", "torch"), ("torch_geometric", "torch-geometric"),
                ("torch_scatter", "torch-scatter"), ("torch_sparse", "torch-sparse"),
                ("numpy", "numpy"), ("pandas", "pandas"), ("networkx", "networkx"),
                ("rdkit", "rdkit"), ("hydra", "hydra-core"), ("omegaconf", "omegaconf"),
                ("openpyxl", "openpyxl"))


def installed_version(distribution):
    try:
        from importlib import metadata
    except ImportError:
        return None
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def inventory():
    packages = {}
    for module, distribution in DEPENDENCIES:
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, AttributeError):
            present = False
        packages[module] = {"installed": present, "distribution": distribution,
                            "version": installed_version(distribution)}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "python_compatible": sys.version_info >= (3, 7), "packages": packages,
            "scope": "Package discovery does not establish binary or model API compatibility; use --smoke for a CPU forward pass."}


def smoke_check(dataset):
    imported = {}
    for module, _ in DEPENDENCIES:
        imported[module] = importlib.import_module(module)
    torch = imported["torch"]
    OmegaConf = imported["omegaconf"].OmegaConf
    from mke_support.gnn import get_gnnNets
    model_config = OmegaConf.load(str(ROOT / "config" / "models" / "gcn.yaml"))
    model_config.param = model_config.param[dataset]
    torch.manual_seed(0)
    model = get_gnnNets(input_dim=9, output_dim=2, model_config=model_config).cpu().eval()
    features = torch.zeros((3, 9), dtype=torch.float32)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        output = model(features, edges, batch)
    if tuple(output.shape) != (1, 2) or not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("CPU forward returned invalid logits: {}".format(output))
    return {"status": "passed", "device": "cpu", "dataset_config": dataset,
            "logit_shape": list(output.shape), "torch_cuda_build": torch.version.cuda,
            "scope": "Synthetic graph forward pass; no checkpoint, training, GPU run, or experiment reproduction."}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect the environment; optionally check the actual GCN API with a CPU forward pass.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="Mutagenicity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = inventory()
    missing = [name for name, details in report["packages"].items() if not details["installed"]]
    success = report["python_compatible"] and not missing
    if missing:
        report["missing_packages"] = missing
    if args.smoke:
        if success:
            try:
                report["smoke"] = smoke_check(args.dataset)
            except Exception as error:
                report["smoke"] = {"status": "failed", "error": "{}: {}".format(type(error).__name__, error)}
                success = False
        else:
            report["smoke"] = {"status": "skipped", "reason": "Required packages or Python version unavailable"}
    report["status"] = "passed" if success else "failed"
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
