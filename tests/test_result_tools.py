import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / (name + ".py")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary_tool = load_script("summarize_results")
plot_tool = load_script("plot_results")


class ResultToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, name="run_a", seed=0, checkpoint="a", dataset="Mutagenicity"):
        path = self.root / name
        path.mkdir()
        metadata = {
            "dataset": dataset, "sampling_seed": seed, "target_sparsity": 0.5,
            "subgraph_building_method": "split", "test_count": 2,
            "algorithm": "mke_new_group_shapley", "input_csv": "/data/" + dataset + ".csv",
            "checkpoint": "/weights/" + checkpoint + ".pth", "input_sha256": "1" * 64,
            "checkpoint_sha256": checkpoint * 64, "run_id": name,
            "exact_threshold": 8, "mc_permutations_per_group": 100,
            "metric_scope": "all_test_graphs_including_empty_explanations",
            "gpu_config": {"sample_num": 100, "exact_threshold": 8},
            "node_budget_rule": "floor((1 - target_sparsity) * num_nodes)",
            "per_graph_seed_rule": "sampling_seed + 1000003 * dataset_index"
        }
        rows = []
        for index, clean, removed, retained, label, prediction, nodes in (
            (10, 0.8123456789123, 0.9023456789234, 0.7123456789321, 0, 0, [0, 1]),
            (15, 0.6654321987654, 0.3654321987123, 0.7654321987564, 1, 0, [])
        ):
            rows.append({
                "sample_index": index, "label": label, "prediction": prediction,
                "target_probability": clean, "explain_value": removed, "explain_no_mask": retained,
                "fidelity_plus": clean - removed, "fidelity_minus": clean - retained,
                "sparsity": 1 - len(nodes) / 4, "target_sparsity": 0.5,
                "explanation_nodes": json.dumps(nodes), "num_nodes": 4, "node_budget": 2,
                "sampling_seed": seed + 1000003 * index, "empty_explanation": not nodes
            })
        summary = {
            "dataset_name": dataset, "sampling_seed": seed, "run_id": name,
            "num_evaluated": 2, "num_empty_explanations": 1,
            "metric_scope": metadata["metric_scope"]
        }
        for key in ("fidelity_plus", "fidelity_minus", "sparsity"):
            summary[key] = math.fsum(row[key] for row in rows) / len(rows)
        self.write_fixture(path, metadata, summary, rows)
        return path, metadata, summary, rows

    def write_fixture(self, path, metadata, summary, rows):
        (path / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        with (path / "mke_new_explanations.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_recomputes_unrounded_metrics_accuracy_and_empty_rows(self):
        path, metadata, summary, rows = self.fixture()
        result = summary_tool.read_run(path)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["num_correct"], 1)
        self.assertEqual(result["num_empty_explanations"], 1)
        self.assertEqual(result["sparsity"], 0.75)
        expected = math.fsum(row["target_probability"] - row["explain_value"] for row in rows) / 2
        self.assertEqual(result["fidelity_plus"], expected)
        self.assertNotEqual(result["fidelity_plus"], round(expected, 4))
        self.assertLess(rows[0]["fidelity_plus"], 0)
        self.assertEqual(result["sparsity_source"], "recomputed_from_nodes")

    def test_missing_files_and_incomplete_run_rejected(self):
        path, metadata, summary, rows = self.fixture()
        (path / "summary.json").unlink()
        with self.assertRaisesRegex(ValueError, "summary.json"):
            summary_tool.collect_runs(self.root)
        self.write_fixture(path, metadata, summary, rows[:1])
        with self.assertRaisesRegex(ValueError, "Incomplete run"):
            summary_tool.read_run(path)

    def test_nonfinite_csv_and_metadata_rejected(self):
        path, metadata, summary, rows = self.fixture()
        rows[0]["target_probability"] = "nan"
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "finite"):
            summary_tool.read_run(path)
        rows[0]["target_probability"] = 0.8
        metadata["gpu_config"]["sample_num"] = float("inf")
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
            summary_tool.read_run(path)

    def test_missing_columns_and_probability_mismatch_rejected(self):
        path, metadata, summary, rows = self.fixture()
        for row in rows:
            del row["label"]
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "Missing CSV fields"):
            summary_tool.read_run(path)
        path, metadata, summary, rows = self.fixture("run_b")
        rows[0]["fidelity_plus"] = 0
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "fidelity_plus is inconsistent"):
            summary_tool.read_run(path)

    def test_summary_mismatch_rejected(self):
        path, metadata, summary, rows = self.fixture()
        summary["fidelity_minus"] = 0.8
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "summary.fidelity_minus"):
            summary_tool.read_run(path)

    def test_predictor_accuracy_summary_is_crosschecked(self):
        path, metadata, summary, rows = self.fixture()
        summary["predictor_accuracy"] = 0.5
        self.write_fixture(path, metadata, summary, rows)
        self.assertEqual(summary_tool.read_run(path)["accuracy"], 0.5)
        summary["predictor_accuracy"] = 1.0
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "summary.predictor_accuracy"):
            summary_tool.read_run(path)

    def test_nonbinary_labels_and_predictions_rejected(self):
        path, metadata, summary, rows = self.fixture()
        for field in ("label", "prediction"):
            original = rows[0][field]
            rows[0][field] = 2
            self.write_fixture(path, metadata, summary, rows)
            with self.assertRaisesRegex(ValueError, "must be binary"):
                summary_tool.read_run(path)
            rows[0][field] = original

    def test_wrong_node_sparsity_and_duplicate_indices_rejected(self):
        path, metadata, summary, rows = self.fixture()
        rows[0]["sparsity"] = 0.1
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "sparsity is inconsistent"):
            summary_tool.read_run(path)
        rows[0]["sparsity"] = 0.5
        rows[1]["sample_index"] = rows[0]["sample_index"]
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "Duplicate sample_index"):
            summary_tool.read_run(path)

    def test_repeated_seed_requires_explicit_opt_in(self):
        self.fixture()
        self.fixture("run_b")
        records = summary_tool.collect_runs(self.root)
        with self.assertRaisesRegex(ValueError, "Repeated seed"):
            summary_tool.aggregate(records)
        result = summary_tool.aggregate(records, allow_repeated_seed=True)
        self.assertEqual(result[0]["num_runs"], 2)
        self.assertEqual(result[0]["num_seeds"], 1)

    def test_distinct_predictors_and_protocols_are_not_merged(self):
        self.fixture()
        self.fixture("run_b", checkpoint="b")
        path, metadata, summary, rows = self.fixture("run_c")
        metadata["gpu_config"]["batch_size"] = 64
        self.write_fixture(path, metadata, summary, rows)
        result = summary_tool.aggregate(summary_tool.collect_runs(self.root))
        self.assertEqual(len(result), 3)

    def test_multiple_seeds_are_averaged_without_rounding(self):
        self.fixture()
        path, metadata, summary, rows = self.fixture("run_b", seed=1)
        rows[0]["explain_value"] -= 0.125678912345
        rows[0]["fidelity_plus"] = rows[0]["target_probability"] - rows[0]["explain_value"]
        summary["fidelity_plus"] = math.fsum(row["fidelity_plus"] for row in rows) / 2
        self.write_fixture(path, metadata, summary, rows)
        records = summary_tool.collect_runs(self.root)
        result = summary_tool.aggregate(records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["num_seeds"], 2)
        self.assertEqual(result[0]["fidelity_plus_mean"], math.fsum(row["fidelity_plus"] for row in records) / 2)
        self.assertGreater(result[0]["fidelity_plus_std"], 0)

    def test_different_labels_are_separate_groups(self):
        path, metadata, summary, rows = self.fixture()
        metadata["label_column"] = "CT_TOX"
        self.write_fixture(path, metadata, summary, rows)
        path, metadata, summary, rows = self.fixture("run_b")
        metadata["label_column"] = "FDA_APPROVED"
        self.write_fixture(path, metadata, summary, rows)
        result = summary_tool.aggregate(summary_tool.collect_runs(self.root))
        self.assertEqual(len(result), 2)
        self.assertEqual({row["label_column"] for row in result}, {"CT_TOX", "FDA_APPROVED"})

    def test_legacy_sparsity_and_path_identity_are_labeled(self):
        path, metadata, summary, rows = self.fixture()
        metadata.pop("input_sha256")
        metadata.pop("checkpoint_sha256")
        for row in rows:
            del row["num_nodes"]
        self.write_fixture(path, metadata, summary, rows)
        result = summary_tool.read_run(path)
        self.assertEqual(result["sparsity_source"], "recorded_no_num_nodes")
        self.assertEqual(result["identity_validation"], "path_only_or_partial")
        self.assertTrue(result["input_identity"].startswith("path:"))

    def test_seed_target_and_hash_mismatches_rejected(self):
        path, metadata, summary, rows = self.fixture()
        rows[0]["sampling_seed"] = 99
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "sampling seed"):
            summary_tool.read_run(path)
        rows[0]["sampling_seed"] = 10000030
        rows[0]["target_sparsity"] = 0.6
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "target_sparsity"):
            summary_tool.read_run(path)
        metadata["input_sha256"] = "invalid"
        self.write_fixture(path, metadata, summary, rows)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            summary_tool.read_run(path)

    def test_macro_requires_complete_unambiguous_coverage(self):
        self.fixture(dataset="A")
        self.fixture("run_b", dataset="B")
        aggregates = summary_tool.aggregate(summary_tool.collect_runs(self.root))
        macro = summary_tool.macro_average(aggregates, ["A", "B"])
        self.assertEqual(macro[0]["num_datasets"], 2)
        self.assertEqual(macro[0]["accuracy_mean"], 0.5)
        with self.assertRaisesRegex(ValueError, "missing"):
            summary_tool.macro_average(aggregates, ["A", "B", "C"])
        self.fixture("run_c", dataset="A", checkpoint="b")
        aggregates = summary_tool.aggregate(summary_tool.collect_runs(self.root))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            summary_tool.macro_average(aggregates, ["A", "B"])

    def test_cli_outputs_and_plot_input_keep_negative_values(self):
        path, metadata, summary, rows = self.fixture()
        for row in rows:
            row["explain_value"] = 0.99
            row["fidelity_plus"] = row["target_probability"] - row["explain_value"]
        summary["fidelity_plus"] = math.fsum(row["fidelity_plus"] for row in rows) / 2
        self.write_fixture(path, metadata, summary, rows)
        output = self.root / "tables"
        summary_tool.main(["--results", str(self.root), "--output", str(output)])
        self.assertTrue((output / "per_run.csv").is_file())
        groups = plot_tool.read_points(output / "aggregated.csv")
        self.assertEqual(len(groups), 1)
        points = next(iter(groups.values()))
        self.assertLess(points[0]["fidelity_plus_mean"], 0)
        self.assertEqual(points[0]["target_sparsity"], 0.5)
        self.assertEqual(points[0]["sparsity_mean"], 0.75)
        with self.assertRaisesRegex(ValueError, "absent"):
            plot_tool.read_points(output / "aggregated.csv", ["missing"])


if __name__ == "__main__":
    unittest.main()
