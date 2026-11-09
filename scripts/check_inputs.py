import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mke_support.release import DATASET_NAMES, file_sha256, load_catalog, validate_input_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate the molecular CSV and checkpoint location without importing ML libraries.")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="Mutagenicity")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "origin_data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--label-column")
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    try:
        entry = load_catalog(ROOT)[args.dataset]
        csv_path = args.input_csv or args.data_root / entry["csv_filename"]
        report = {"dataset": args.dataset, "csv": validate_input_csv(csv_path, args.label_column or entry["label_column"])}
        if not args.skip_checkpoint:
            checkpoint = (args.checkpoint_root.expanduser() / args.dataset / "gcn_3l_best.pth").resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError("Required checkpoint not found: {}".format(checkpoint))
            report["checkpoint"] = {"path": str(checkpoint), "sha256": file_sha256(checkpoint), "bytes": checkpoint.stat().st_size}
        report["status"] = "passed"
        report["scope"] = "CSV schema and file checks only; molecular parsing and checkpoint loading occur in the model entry."
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, csv.Error) as error:
        print("Input validation failed: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
