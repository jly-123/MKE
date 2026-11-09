import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


METRICS = ("fidelity_plus", "fidelity_minus", "sparsity", "accuracy")
PER_RUN_FIELDS = (
    "dataset", "label_column", "target_sparsity", "mask", "seed", "num_evaluated", "num_correct",
    "num_empty_explanations", "fidelity_plus", "fidelity_minus", "sparsity", "accuracy",
    "input_identity", "checkpoint_identity", "identity_validation", "protocol_id",
    "test_indices_sha256", "sparsity_source", "run_id", "run_directory", "protocol_json"
)
GROUP_FIELDS = (
    "dataset", "target_sparsity", "mask", "label_column", "input_identity", "checkpoint_identity",
    "identity_validation", "protocol_id", "test_indices_sha256", "sparsity_source"
)
AGGREGATE_FIELDS = GROUP_FIELDS + (
    "num_runs", "num_seeds", "seeds", "graphs_per_run", "aggregation"
) + tuple(name + suffix for name in METRICS for suffix in ("_mean", "_std"))
PROTOCOL_KEYS = (
    "algorithm", "objective", "node_budget_rule", "candidate_normalization",
    "exact_threshold", "threshold_applies_to", "mc_permutations_per_group",
    "per_graph_seed_rule", "mc_cache", "probability_precision", "target_class",
    "metric_scope", "gpu_config", "predictor_config", "model_config",
    "software_versions", "source_sha256", "code_sha256", "implementation_sha256"
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def fail(message):
    raise ValueError(message)


def finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        fail("{} must be a finite number".format(name))
    if not math.isfinite(number):
        fail("{} must be finite".format(name))
    return number


def integer(value, name, minimum=0):
    if isinstance(value, bool):
        fail("{} must be an integer".format(name))
    number = finite(value, name)
    if number != int(number) or number < minimum:
        fail("{} must be an integer >= {}".format(name, minimum))
    return int(number)


def probability(value, name):
    number = finite(value, name)
    if not 0 <= number <= 1:
        fail("{} must be in [0, 1]".format(name))
    return number


def check_close(actual, expected, name):
    if not math.isclose(finite(actual, name), expected, rel_tol=1e-9, abs_tol=1e-10):
        fail("{} is inconsistent: recorded {}, recomputed {}".format(name, actual, expected))


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle, parse_constant=lambda token: fail("Nonfinite JSON value: " + token))
    if not isinstance(value, dict):
        fail("{} must contain a JSON object".format(path))
    canonical(value)
    return value


def identity(metadata, key, path_key):
    value = metadata.get(key)
    if value is not None:
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
            fail("{} must be a 64-character SHA256 digest".format(key))
        return "sha256:" + value.lower(), True
    path = metadata.get(path_key)
    if not isinstance(path, str) or not path.strip():
        fail("Missing {} and {}".format(key, path_key))
    return "path:" + path, False


def read_run(directory):
    directory = Path(directory)
    metadata = read_json(directory / "run_metadata.json")
    summary = read_json(directory / "summary.json")
    dataset = metadata["dataset"]
    mask = metadata["subgraph_building_method"]
    if not isinstance(dataset, str) or not dataset.strip() or not isinstance(mask, str) or not mask.strip():
        fail("Dataset and perturbation method must be nonempty strings")
    label_column = metadata.get("label_column", "unspecified")
    if not isinstance(label_column, str) or not label_column.strip():
        fail("label_column must be a nonempty string")
    seed = integer(metadata["sampling_seed"], "sampling_seed")
    target = probability(metadata["target_sparsity"], "target_sparsity")
    expected_count = integer(metadata["test_count"], "test_count", 1)
    input_id, input_verified = identity(metadata, "input_sha256", "input_csv")
    checkpoint_id, checkpoint_verified = identity(metadata, "checkpoint_sha256", "checkpoint")
    input_report = metadata.get("input_report")
    if input_report is not None:
        if not isinstance(input_report, dict):
            fail("input_report must be an object")
        if "sha256" in input_report and input_report["sha256"] != metadata.get("input_sha256"):
            fail("input_report.sha256 disagrees with input_sha256")
    protocol = {key: metadata[key] for key in PROTOCOL_KEYS if key in metadata}
    for required in ("algorithm", "exact_threshold", "mc_permutations_per_group", "metric_scope", "gpu_config"):
        if required not in protocol:
            fail("Missing protocol metadata: " + required)
    integer(protocol["exact_threshold"], "exact_threshold")
    integer(protocol["mc_permutations_per_group"], "mc_permutations_per_group", 1)
    if not isinstance(protocol["gpu_config"], dict):
        fail("gpu_config must be an object")
    for gpu_key, meta_key in (("exact_threshold", "exact_threshold"), ("sample_num", "mc_permutations_per_group")):
        if gpu_key in protocol["gpu_config"]:
            check_close(protocol["gpu_config"][gpu_key], float(protocol[meta_key]), "gpu_config." + gpu_key)
    with (directory / "mke_new_explanations.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "sample_index", "label", "prediction", "target_probability", "explain_value",
            "explain_no_mask", "fidelity_plus", "fidelity_minus", "sparsity", "target_sparsity",
            "explanation_nodes"
        }
        missing = required_fields.difference(reader.fieldnames or [])
        if missing:
            fail("Missing CSV fields: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if len(rows) != expected_count:
        fail("Incomplete run: expected {} graphs, found {}".format(expected_count, len(rows)))
    has_num_nodes = all(row.get("num_nodes", "") != "" for row in rows)
    if any(row.get("num_nodes", "") != "" for row in rows) and not has_num_nodes:
        fail("num_nodes is missing from some rows")
    values = {name: [] for name in METRICS}
    indices = set()
    empty_count = 0
    for number, row in enumerate(rows, 2):
        prefix = "row {} ".format(number)
        index = integer(row["sample_index"], prefix + "sample_index")
        if index in indices:
            fail("Duplicate sample_index: {}".format(index))
        indices.add(index)
        label = integer(row["label"], prefix + "label")
        prediction = integer(row["prediction"], prefix + "prediction")
        if label not in (0, 1) or prediction not in (0, 1):
            fail(prefix + "label and prediction must be binary (0 or 1)")
        clean = probability(row["target_probability"], prefix + "target_probability")
        removed = probability(row["explain_value"], prefix + "explain_value")
        retained = probability(row["explain_no_mask"], prefix + "explain_no_mask")
        plus, minus = clean - removed, clean - retained
        check_close(row["fidelity_plus"], plus, prefix + "fidelity_plus")
        check_close(row["fidelity_minus"], minus, prefix + "fidelity_minus")
        check_close(row["target_sparsity"], target, prefix + "target_sparsity")
        if row.get("group_score", "") != "":
            finite(row["group_score"], prefix + "group_score")
        if row.get("candidate_count", "") != "":
            integer(row["candidate_count"], prefix + "candidate_count")
        nodes = json.loads(row["explanation_nodes"])
        if not isinstance(nodes, list) or any(isinstance(node, bool) or not isinstance(node, int) or node < 0 for node in nodes):
            fail(prefix + "explanation_nodes must be a list of nonnegative integers")
        if len(nodes) != len(set(nodes)):
            fail(prefix + "explanation_nodes contains duplicates")
        sparsity = probability(row["sparsity"], prefix + "sparsity")
        if has_num_nodes:
            count = integer(row["num_nodes"], prefix + "num_nodes", 1)
            if any(node >= count for node in nodes):
                fail(prefix + "explanation node is outside the graph")
            sparsity = 1 - len(nodes) / count
            check_close(row["sparsity"], sparsity, prefix + "sparsity")
        if row.get("node_budget", "") != "":
            budget = integer(row["node_budget"], prefix + "node_budget")
            if len(nodes) > budget:
                fail(prefix + "explanation exceeds node budget")
            if has_num_nodes and metadata.get("node_budget_rule") == "floor((1 - target_sparsity) * num_nodes)":
                if budget != math.floor((1 - target) * count):
                    fail(prefix + "node budget is inconsistent with target sparsity")
        if row.get("sampling_seed", "") != "" and metadata.get("per_graph_seed_rule") == "sampling_seed + 1000003 * dataset_index":
            if integer(row["sampling_seed"], prefix + "sampling_seed") != seed + 1000003 * index:
                fail(prefix + "sampling seed disagrees with per-graph seed rule")
        empty = len(nodes) == 0
        if row.get("empty_explanation", "") != "":
            recorded = row["empty_explanation"].lower()
            if recorded not in ("true", "false", "1", "0") or (recorded in ("true", "1")) != empty:
                fail(prefix + "empty_explanation is inconsistent")
        if empty and sparsity != 1:
            fail(prefix + "empty explanation must have sparsity 1")
        empty_count += int(empty)
        for name, value in zip(METRICS, (plus, minus, sparsity, float(label == prediction))):
            values[name].append(value)
    means = {name: math.fsum(items) / len(items) for name, items in values.items()}
    for name in ("fidelity_plus", "fidelity_minus", "sparsity"):
        check_close(summary[name], means[name], "summary." + name)
    if integer(summary["num_evaluated"], "summary.num_evaluated") != expected_count:
        fail("summary.num_evaluated is inconsistent")
    if integer(summary["num_empty_explanations"], "summary.num_empty_explanations") != empty_count:
        fail("summary.num_empty_explanations is inconsistent")
    for key, expected in (("dataset_name", dataset), ("sampling_seed", seed), ("metric_scope", protocol["metric_scope"]), ("run_id", metadata.get("run_id"))):
        if key in summary and summary[key] != expected:
            fail("summary.{} disagrees with run_metadata.json".format(key))
    if "accuracy" in summary:
        check_close(summary["accuracy"], means["accuracy"], "summary.accuracy")
    if "predictor_accuracy" in summary:
        check_close(summary["predictor_accuracy"], means["accuracy"], "summary.predictor_accuracy")
    if "target_sparsity" in summary:
        check_close(summary["target_sparsity"], target, "summary.target_sparsity")
    if "label_column" in summary and summary["label_column"] != label_column:
        fail("summary.label_column disagrees with run_metadata.json")
    for key in ("input_sha256", "checkpoint_sha256"):
        if key in summary and summary[key] != metadata.get(key):
            fail("summary.{} disagrees with run_metadata.json".format(key))
    result = {
        "dataset": dataset, "label_column": label_column, "target_sparsity": target, "mask": mask, "seed": seed,
        "num_evaluated": expected_count, "num_correct": int(math.fsum(values["accuracy"])),
        "num_empty_explanations": empty_count, "input_identity": input_id,
        "checkpoint_identity": checkpoint_id,
        "identity_validation": "sha256" if input_verified and checkpoint_verified else "path_only_or_partial",
        "protocol_id": digest(protocol), "test_indices_sha256": digest(sorted(indices)),
        "sparsity_source": "recomputed_from_nodes" if has_num_nodes else "recorded_no_num_nodes",
        "run_id": metadata.get("run_id", directory.name), "run_directory": str(directory.resolve()),
        "protocol_json": canonical(protocol)
    }
    result.update(means)
    return result


def collect_runs(results):
    root = Path(results)
    if not root.is_dir():
        fail("Result root is not a directory: {}".format(root))
    directories = set()
    for filename in ("run_metadata.json", "summary.json", "mke_new_explanations.csv"):
        directories.update(path.parent for path in root.rglob(filename))
    if not directories:
        fail("No MKE run files found under {}".format(root))
    records = []
    for directory in sorted(directories):
        try:
            records.append(read_run(directory))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            fail("{}: {}".format(directory, exc))
    return records


def aggregate(records, allow_repeated_seed=False):
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in GROUP_FIELDS)].append(record)
    output = []
    for key, group in sorted(grouped.items()):
        seeds = [item["seed"] for item in group]
        if len(seeds) != len(set(seeds)) and not allow_repeated_seed:
            fail("Repeated seed for dataset {} target {} and the same predictor/protocol; use --allow-repeated-seed to weight every run equally".format(key[0], key[1]))
        row = dict(zip(GROUP_FIELDS, key))
        row.update({
            "num_runs": len(group), "num_seeds": len(set(seeds)), "seeds": canonical(sorted(seeds)),
            "graphs_per_run": group[0]["num_evaluated"], "aggregation": "equal_weight_per_run"
        })
        for name in METRICS:
            values = [item[name] for item in group]
            row[name + "_mean"] = math.fsum(values) / len(values)
            row[name + "_std"] = statistics.stdev(values) if len(values) > 1 else ""
        output.append(row)
    return output


def macro_average(aggregated, datasets):
    requested = set(datasets)
    if len(requested) != len(datasets) or not requested:
        fail("--datasets must contain distinct dataset names")
    if not requested.issubset({row["dataset"] for row in aggregated}):
        fail("Requested macro-average datasets are missing")
    settings = defaultdict(list)
    for row in aggregated:
        if row["dataset"] in requested:
            settings[(row["target_sparsity"], row["mask"], row["protocol_id"])].append(row)
    output = []
    for key, rows in sorted(settings.items()):
        names = [row["dataset"] for row in rows]
        if set(names) != requested or len(names) != len(requested):
            fail("Macro average requires exactly one predictor/input group for every requested dataset at each setting")
        if len({row["seeds"] for row in rows}) != 1:
            fail("Macro average requires matching seed lists across datasets")
        result = {
            "target_sparsity": key[0], "mask": key[1], "protocol_id": key[2],
            "datasets": canonical(sorted(requested)), "num_datasets": len(rows),
            "seeds": rows[0]["seeds"], "aggregation": "equal_weight_per_dataset_after_equal_weight_per_run",
            "dataset_identities": canonical({row["dataset"]: {field: row[field] for field in ("input_identity", "checkpoint_identity", "label_column", "test_indices_sha256")} for row in rows})
        }
        for name in METRICS:
            result[name + "_mean"] = math.fsum(row[name + "_mean"] for row in rows) / len(rows)
        output.append(result)
    return output


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate complete MKE runs and recompute full-precision metrics from per-graph probabilities. No clipping or rounding is applied.")
    parser.add_argument("--results", required=True, help="Root containing run_metadata.json, summary.json and mke_new_explanations.csv for every run")
    parser.add_argument("--output", required=True, help="Directory for per_run.csv and aggregated.csv")
    parser.add_argument("--allow-repeated-seed", action="store_true", help="Allow repeated seeds within one protocol; every run then receives equal weight")
    parser.add_argument("--datasets", nargs="+", help="Explicit dataset names for an additional equal-dataset macro_average.csv; complete coverage and matching seed lists are required")
    args = parser.parse_args(argv)
    try:
        records = collect_runs(args.results)
        aggregated = aggregate(records, args.allow_repeated_seed)
        macro = macro_average(aggregated, args.datasets) if args.datasets else None
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "per_run.csv", records, PER_RUN_FIELDS)
        write_csv(output / "aggregated.csv", aggregated, AGGREGATE_FIELDS)
        if macro:
            write_csv(output / "macro_average.csv", macro, list(macro[0]))
        print("Validated {} runs in {} groups. Wrote {}".format(len(records), len(aggregated), output.resolve()))
        if any(row["identity_validation"] != "sha256" for row in records):
            print("Legacy runs lack one or both file fingerprints; path identities were kept separate and cannot verify file contents.", file=sys.stderr)
        if any(row["sparsity_source"] == "recorded_no_num_nodes" for row in records):
            print("Legacy rows lack num_nodes; their sparsity values were range-checked and averaged, not independently reconstructed.", file=sys.stderr)
    except (OSError, ValueError) as exc:
        parser.exit(2, "error: {}\n".format(exc))


if __name__ == "__main__":
    main()
