import argparse
import ast
import hashlib
import io
import json
import sys
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "results", "reports",
            "figures", "logs", "outputs", "multirun", ".hydra", "processed", "build", "dist"}


def source_files(root):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in EXCLUDED for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pth", ".pt", ".xlsx", ".zip"}:
            continue
        if relative.parts[:2] == ("data", "origin_data") and path.suffix == ".csv":
            continue
        if path.name in {"FILES_SHA256.txt", "environment-report.json", "environment-server.txt",
                         "training_config.yaml", "training_inputs.json"}:
            continue
        yield path


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root):
    checked = 0
    for path in source_files(root):
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8-sig")
        kwargs = {"feature_version": 7} if sys.version_info >= (3, 8) else {}
        tree = ast.parse(source, filename=str(path), **kwargs)
        compile(tree, str(path), "exec")
        comments = [tok for tok in tokenize.generate_tokens(io.StringIO(source).readline)
                    if tok.type == tokenize.COMMENT]
        docstrings = [node for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                      and ast.get_docstring(node) is not None]
        if comments or docstrings:
            raise ValueError("Unexpected source comments/docstrings in {}".format(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            else:
                continue
            if any(name.split(".")[0] in {"dig", "benchmarks"} for name in names):
                raise ValueError("Non-standalone import in {}".format(path))
        checked += 1
    catalog = json.loads((root / "config/dataset_catalog.json").read_text(encoding="utf-8"))
    for dataset, spec in catalog.items():
        path = root / "config/datasets" / (dataset + ".yaml")
        config = path.read_text(encoding="utf-8")
        for key, value in (("dataset_name", dataset), ("csv_filename", spec["csv_filename"]),
                           ("label_column", spec["label_column"])):
            if "{}: {}".format(key, value) not in config.splitlines():
                raise ValueError("Dataset catalog/config inconsistency: {} {}".format(dataset, key))
        for group in ("models/gcn.yaml", "explainers/mke.yaml"):
            if "  {}:".format(dataset) not in (root / "config" / group).read_text(encoding="utf-8").splitlines():
                raise ValueError("Missing {} settings for {}".format(group, dataset))
    print("Verified {} Python files and {} dataset configurations".format(checked, len(catalog)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check source syntax, standalone imports, configs and release hashes.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write-manifest", action="store_true")
    group.add_argument("--check-manifest", action="store_true")
    args = parser.parse_args(argv)
    verify(ROOT)
    manifest = ROOT / "FILES_SHA256.txt"
    files = list(source_files(ROOT))
    actual = {path.relative_to(ROOT).as_posix(): checksum(path) for path in files}
    if args.write_manifest:
        manifest.write_text("".join("{}  {}\n".format(value, name) for name, value in sorted(actual.items())), encoding="utf-8")
        print("Wrote {} file hashes".format(len(actual)))
    if args.check_manifest:
        recorded = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            value, name = line.split("  ", 1)
            if name in recorded:
                raise ValueError("Duplicate manifest entry: {}".format(name))
            recorded[name] = value
        if recorded != actual:
            mismatched = sorted(name for name in set(actual) | set(recorded)
                                if actual.get(name) != recorded.get(name))
            raise ValueError("Manifest mismatch: {}".format(", ".join(mismatched)))
        print("Verified {} file hashes".format(len(actual)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
