import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = {
    "fidelity_plus": "Fidelity+", "fidelity_minus": "Fidelity-",
    "accuracy": "Predictor accuracy", "sparsity": "Actual mean sparsity"
}


def read_points(path, datasets=None):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No aggregated rows found")
    available = {row["dataset"] for row in rows}
    if datasets and not set(datasets).issubset(available):
        raise ValueError("Requested datasets are absent from aggregated.csv")
    groups = defaultdict(list)
    for row in rows:
        if datasets and row["dataset"] not in datasets:
            continue
        for key in ["target_sparsity"] + [name + "_mean" for name in METRICS]:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError("Nonfinite value in " + key)
        for key in ("target_sparsity", "sparsity_mean", "accuracy_mean"):
            if not 0 <= row[key] <= 1:
                raise ValueError(key + " must be in [0, 1]")
        key = tuple(row[field] for field in (
            "dataset", "mask", "input_identity", "checkpoint_identity", "protocol_id",
            "test_indices_sha256", "sparsity_source", "seeds", "label_column"
        ))
        groups[key].append(row)
    for key, points in groups.items():
        if len({row["target_sparsity"] for row in points}) != len(points):
            raise ValueError("Duplicate target sparsity within one plot series")
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot measured MKE metrics from aggregated.csv without smoothing, clipping or enforcing monotonicity. Connected lines only link measured points.")
    parser.add_argument("--input", required=True, help="aggregated.csv produced by summarize_results.py")
    parser.add_argument("--output", required=True, help="Directory for figures and plotted_points.csv")
    parser.add_argument("--datasets", nargs="+", help="Optional dataset subset")
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRICS), default=["fidelity_plus", "fidelity_minus"], help="Metrics to plot")
    parser.add_argument("--x", choices=("target", "actual"), default="target", help="Use target sparsity (default) or measured actual sparsity")
    parser.add_argument("--format", nargs="+", choices=("png", "jpg", "pdf"), default=["png"], help="One or more output formats")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        if args.dpi <= 0:
            raise ValueError("--dpi must be positive")
        groups = read_points(args.input, args.datasets)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ValueError("Plotting requires matplotlib; install it in your plotting environment")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        x_column = "target_sparsity" if args.x == "target" else "sparsity_mean"
        exported = []
        for series_number, (key, points) in enumerate(sorted(groups.items()), 1):
            for row in points:
                exported.append(dict(row, series_id=series_number))
        for metric in args.metrics:
            fig, ax = plt.subplots(figsize=(6.8, 4.5))
            for series_number, (key, points) in enumerate(sorted(groups.items()), 1):
                points = sorted(points, key=lambda row: row[x_column])
                ax.plot([row[x_column] for row in points], [row[metric + "_mean"] for row in points], marker="o", label="{} | {} | series {}".format(key[0], key[1], series_number))
            ax.set_xlabel("Target sparsity" if args.x == "target" else "Actual mean sparsity")
            ax.set_ylabel(METRICS[metric])
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize="small")
            fig.tight_layout()
            for suffix in args.format:
                fig.savefig(str(output / (metric + "_vs_" + args.x + "_sparsity." + suffix)), dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
        with (output / "plotted_points.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
            writer.writeheader()
            writer.writerows(exported)
        print("Plotted {} measured series. Wrote {}".format(len(groups), output.resolve()))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, "error: {}\n".format(exc))


if __name__ == "__main__":
    main()
