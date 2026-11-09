import argparse
import csv
import datetime
import itertools
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mke_support.release import DATASET_NAMES, file_sha256, load_catalog, validate_input_csv


def resolved(path):
    return Path(path).expanduser().resolve()


def hydra_value(value):
    return json.dumps(str(value).replace("\\", "/"), ensure_ascii=False)


def validate_options(args):
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Dataset names must be unique")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        raise ValueError("Seeds must be distinct integers in [0, 2**32)")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in args.sparsities):
        raise ValueError("Target sparsity must be finite and in [0, 1]")
    if len(set(args.sparsities)) != len(args.sparsities):
        raise ValueError("Target sparsities must be unique")
    if args.device < 0:
        raise ValueError("GPU device index must be nonnegative")
    protected = ("datasets", "models.gnn_saving_dir", "explainers.explanation_result_dir",
                 "device_id", "random_seed", "target_sparsity", "input_csv", "record_filename",
                 "overwrite_checkpoint")
    for override in args.override:
        if "=" not in override:
            raise ValueError("Each --override must have key=value form")
        key = override.split("=", 1)[0].lstrip("+~")
        if not key or any(key == name or key.startswith(name + ".") for name in protected):
            raise ValueError("Override {!r} conflicts with runner-managed inputs or outputs".format(override))
    if args.mode == "explain" and args.overwrite_checkpoints:
        raise ValueError("--overwrite-checkpoints applies only to --mode train")


def training_checkpoint_root(args, seed):
    root = resolved(args.checkpoint_root)
    return root / "seed_{}".format(seed) if len(args.seeds) > 1 else root


def preflight(args, catalog):
    report = {}
    for dataset in args.datasets:
        entry = catalog[dataset]
        csv_path = resolved(args.data_root) / entry["csv_filename"]
        dataset_report = {"csv": validate_input_csv(csv_path, entry["label_column"])}
        if args.mode == "explain":
            checkpoint = resolved(args.checkpoint_root) / dataset / "gcn_3l_best.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError("Required checkpoint not found: {}".format(checkpoint))
            dataset_report["checkpoint"] = {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}
        else:
            for seed in args.seeds:
                checkpoint_dir = training_checkpoint_root(args, seed) / dataset
                existing = [path for path in checkpoint_dir.glob("gcn_3l_*.pth") if path.is_file()]
                if existing and not args.overwrite_checkpoints:
                    raise FileExistsError("Training would replace checkpoints in {}. Use a new --checkpoint-root or explicitly pass --overwrite-checkpoints.".format(checkpoint_dir))
        report[dataset] = dataset_report
    return report


def build_jobs(args, catalog, batch_dir):
    jobs = []
    entry_script = ROOT / ("mke_new.py" if args.mode == "explain" else "train_gcn.py")
    sparsities = args.sparsities if args.mode == "explain" else [None]
    for number, (dataset, seed, sparsity) in enumerate(itertools.product(args.datasets, args.seeds, sparsities), 1):
        tag = "{:03d}_{}_seed_{}".format(number, dataset, seed)
        if sparsity is not None:
            tag += "_sparsity_{}".format(format(sparsity, ".12g").replace(".", "p"))
        job_dir = batch_dir / tag
        checkpoint_root = resolved(args.checkpoint_root) if args.mode == "explain" else training_checkpoint_root(args, seed)
        argv = [sys.executable, str(entry_script), "datasets={}".format(dataset),
                "device_id={}".format(args.device), "random_seed={}".format(seed),
                "datasets.dataset_root={}".format(hydra_value(resolved(args.data_root))),
                "input_csv={}".format(hydra_value(resolved(args.data_root) / catalog[dataset]["csv_filename"])),
                "models.gnn_saving_dir={}".format(hydra_value(checkpoint_root)),
                "explainers.explanation_result_dir={}".format(hydra_value(job_dir / "results")),
                "record_filename={}".format(hydra_value(job_dir / "records")),
                "hydra.run.dir={}".format(hydra_value(job_dir / "hydra"))]
        if sparsity is not None:
            argv.append("target_sparsity={}".format(repr(sparsity)))
        if args.mode == "train":
            argv.append("overwrite_checkpoint={}".format(str(args.overwrite_checkpoints).lower()))
        argv.extend(args.override)
        jobs.append({"dataset": dataset, "seed": seed, "target_sparsity": sparsity,
                     "checkpoint_root": str(checkpoint_root), "working_directory": str(ROOT),
                     "directory": str(job_dir), "log": str(job_dir / "console.log"),
                     "argv": argv, "status": "pending"})
    return jobs


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def execute_jobs(jobs, manifest_path, manifest):
    for job in jobs:
        Path(job["directory"]).mkdir(parents=True, exist_ok=True)
        job["status"] = "running"
        job["started_at"] = timestamp()
        write_manifest(manifest_path, manifest)
        print("Running: {}".format(json.dumps(job["argv"], ensure_ascii=False)), flush=True)
        process = None
        try:
            with open(job["log"], "w", encoding="utf-8") as log:
                process = subprocess.Popen(job["argv"], cwd=str(ROOT), stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, universal_newlines=True,
                                           encoding="utf-8", errors="replace", bufsize=1,
                                           env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"))
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                process.stdout.close()
                job["returncode"] = process.wait()
            job["status"] = "passed" if job["returncode"] == 0 else "failed"
        except KeyboardInterrupt:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            job["status"] = "interrupted"
            job["returncode"] = 130
        except OSError as error:
            job["status"] = "failed"
            job["returncode"] = 1
            job["error"] = str(error)
        job["finished_at"] = timestamp()
        write_manifest(manifest_path, manifest)
        if job["returncode"] != 0:
            manifest["status"] = job["status"]
            manifest["finished_at"] = timestamp()
            write_manifest(manifest_path, manifest)
            return job["returncode"] if job["returncode"] > 0 else 1
    manifest["status"] = "passed"
    manifest["finished_at"] = timestamp()
    write_manifest(manifest_path, manifest)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MKE sequentially across dataset, sparsity, and seed settings, recording logs and a manifest.")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "origin_data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "batch_runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=("explain", "train"), default="explain")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--overwrite-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_options(args)
        catalog = load_catalog(ROOT)
        run_id = "{}_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:8])
        batch_dir = resolved(args.results_root) / run_id
        jobs = build_jobs(args, catalog, batch_dir)
        manifest = {"mode": args.mode, "created_at": timestamp(), "status": "planned",
                    "dry_run": args.dry_run, "batch_directory": str(batch_dir), "jobs": jobs}
        if args.dry_run:
            manifest["validation"] = "Skipped for dry run; paths and installed ML libraries have not been checked."
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            return 0
        manifest["inputs"] = preflight(args, catalog)
        manifest["status"] = "running"
        batch_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = batch_dir / "manifest.json"
        write_manifest(manifest_path, manifest)
        result = execute_jobs(jobs, manifest_path, manifest)
        print("Batch manifest: {}".format(manifest_path))
        return result
    except (OSError, ValueError, csv.Error) as error:
        print("Experiment runner failed: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
