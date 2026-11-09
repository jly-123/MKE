import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mke_support.release import DATASET_NAMES, file_sha256, load_catalog, validate_input_csv


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / (name + ".py")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("run_experiments")
input_checker = load_script("check_inputs")
environment_checker = load_script("check_environment")


class ReleaseToolsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mke release ")
        self.root = Path(self.temporary.name)
        self.csv = self.root / "Mutagenicity.csv"
        self.catalog = {"Mutagenicity": {"csv_filename": "Mutagenicity.csv", "label_column": "target"}}
        self.csv.write_text("smiles,target,group\nCC,0,training\nCO,1,valid\nCN,0,test\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, **updates):
        values = dict(datasets=["Mutagenicity"], seeds=[0], sparsities=[0.5, 0.6, 0.7],
                      device=0, data_root=self.root, checkpoint_root=self.root / "checkpoints",
                      results_root=self.root / "results", mode="explain", override=[],
                      overwrite_checkpoints=False, dry_run=False)
        values.update(updates)
        return argparse.Namespace(**values)

    def test_input_counts_and_checksum_preserve_original(self):
        before = self.csv.read_bytes()
        result = validate_input_csv(self.csv, "target")
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["groups"], {"training": 1, "valid": 1, "test": 1})
        self.assertEqual(result["classes"], {"0": 2, "1": 1})
        self.assertEqual(result["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(self.csv.read_bytes(), before)

    def test_bom_and_numeric_binary_labels(self):
        self.csv.write_text("smiles,target,group\nCC,0.0,training\nCO,1.0,valid\nCN,0,test\n", encoding="utf-8-sig")
        self.assertEqual(validate_input_csv(self.csv, "target")["count"], 3)

    def test_schema_and_rows_reject_silent_reinterpretation(self):
        invalid = [
            "smiles,target,target,group\nCC,0,1,training\n",
            "smiles,target,\nCC,0,training\n",
            "smiles,other,group\nCC,0,training\n",
            "smiles,target,group\nCC,0,training \nCO,1,valid\nCN,0,test\n",
            "smiles,target,group\nCC,0,training\nCO,1,valid\n",
            "smiles,target,group\nCC,2,training\nCO,1,valid\nCN,0,test\n",
            "smiles,target,group\nCC,NaN,training\nCO,1,valid\nCN,0,test\n",
            "smiles,target,group\nCC,,training\nCO,1,valid\nCN,0,test\n",
            "smiles,target,group\n,0,training\nCO,1,valid\nCN,0,test\n",
            "smiles,target,group\nCC,0,training,extra\nCO,1,valid\nCN,0,test\n",
            "smiles,target,group\nCC,0,training\nCO,0,valid\nCN,0,test\n",
            "smiles,target,group\n",
        ]
        for data in invalid:
            with self.subTest(data=data):
                self.csv.write_text(data, encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_input_csv(self.csv, "target")

    def test_catalog_validates_six_names_and_required_fields(self):
        config_dir = self.root / "config"
        config_dir.mkdir()
        path = config_dir / "dataset_catalog.json"
        catalog = {name: {"csv_filename": name + ".csv", "label_column": name} for name in DATASET_NAMES}
        path.write_text(json.dumps(catalog), encoding="utf-8")
        self.assertEqual(load_catalog(self.root), catalog)
        catalog["bbbp"]["csv_filename"] = "../outside.csv"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_catalog(self.root)

    def test_grid_preserves_all_settings_and_argv_paths(self):
        args = self.args(seeds=[0, 2])
        runner.validate_options(args)
        jobs = runner.build_jobs(args, self.catalog, self.root / "batch")
        self.assertEqual(len(jobs), 6)
        self.assertEqual({(job["seed"], job["target_sparsity"]) for job in jobs},
                         {(0, 0.5), (0, 0.6), (0, 0.7), (2, 0.5), (2, 0.6), (2, 0.7)})
        self.assertEqual(len({job["directory"] for job in jobs}), 6)
        first = jobs[0]["argv"]
        self.assertEqual(first[0], sys.executable)
        self.assertTrue(Path(first[1]).is_absolute())
        csv_override = next(value for value in first if value.startswith("input_csv="))
        self.assertEqual(json.loads(csv_override.split("=", 1)[1]), str(self.csv).replace("\\", "/"))

    def test_reserved_overrides_and_invalid_settings_are_rejected(self):
        invalid_args = [self.args(override=["input_csv=unchecked.csv"]),
                        self.args(override=["+datasets.label_column=other"]),
                        self.args(override=["models.gnn_saving_dir=other"]),
                        self.args(override=["overwrite_checkpoint=true"]),
                        self.args(seeds=[0, 0]), self.args(sparsities=[float("nan")]),
                        self.args(sparsities=[1.1]), self.args(device=-1)]
        for args in invalid_args:
            with self.assertRaises(ValueError):
                runner.validate_options(args)
        runner.validate_options(self.args(override=["models.param.Mutagenicity.num_epochs=1"]))

    def test_preflight_requires_checkpoint_for_explain_only(self):
        with self.assertRaises(FileNotFoundError):
            runner.preflight(self.args(), self.catalog)
        report = runner.preflight(self.args(mode="train"), self.catalog)
        self.assertNotIn("checkpoint", report["Mutagenicity"])
        checkpoint = self.root / "checkpoints" / "Mutagenicity" / "gcn_3l_best.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"test checkpoint placeholder")
        report = runner.preflight(self.args(), self.catalog)
        self.assertEqual(report["Mutagenicity"]["checkpoint"]["sha256"], file_sha256(checkpoint))
        with self.assertRaises(FileExistsError):
            runner.preflight(self.args(mode="train"), self.catalog)
        runner.preflight(self.args(mode="train", overwrite_checkpoints=True), self.catalog)

    def test_training_seeds_get_separate_checkpoint_roots(self):
        jobs = runner.build_jobs(self.args(mode="train", seeds=[0, 1]), self.catalog, self.root / "batch")
        self.assertEqual(len(jobs), 2)
        self.assertNotEqual(jobs[0]["checkpoint_root"], jobs[1]["checkpoint_root"])
        self.assertTrue(all(job["target_sparsity"] is None for job in jobs))
        self.assertTrue(all(not any(value.startswith("target_sparsity=") for value in job["argv"]) for job in jobs))
        self.assertTrue(all("overwrite_checkpoint=false" in job["argv"] for job in jobs))
        enabled = runner.build_jobs(self.args(mode="train", overwrite_checkpoints=True), self.catalog, self.root / "other")
        self.assertIn("overwrite_checkpoint=true", enabled[0]["argv"])

    def test_dry_run_uses_no_ml_or_subprocess_and_writes_nothing(self):
        output_root = self.root / "not-created"
        with mock.patch.object(runner, "load_catalog", return_value=self.catalog), \
                mock.patch.object(runner, "preflight", side_effect=AssertionError("Preflight called")), \
                mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Process started")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = runner.main(["--datasets", "Mutagenicity", "--results-root", str(output_root), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(output.getvalue())["jobs"]), 3)
        self.assertFalse(output_root.exists())

    def test_runner_records_failure_and_stops_pending_jobs(self):
        args = self.args()
        batch_dir = self.root / "batch"
        batch_dir.mkdir()
        jobs = runner.build_jobs(args, self.catalog, batch_dir)
        manifest = {"jobs": jobs, "status": "running"}
        fake_process = mock.Mock()
        fake_process.stdout = io.StringIO("diagnostic output\n")
        fake_process.wait.return_value = 7
        with mock.patch.object(runner.subprocess, "Popen", return_value=fake_process) as popen, \
                contextlib.redirect_stdout(io.StringIO()):
            code = runner.execute_jobs(jobs, batch_dir / "manifest.json", manifest)
        self.assertEqual(code, 7)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(jobs[0]["status"], "failed")
        self.assertEqual(jobs[1]["status"], "pending")
        self.assertEqual(Path(jobs[0]["log"]).read_text(encoding="utf-8"), "diagnostic output\n")
        persisted = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "failed")
        self.assertNotIn("shell", popen.call_args[1])

    def test_input_cli_checks_only_the_requested_paths(self):
        with mock.patch.object(input_checker, "load_catalog", return_value=self.catalog), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = input_checker.main(["--dataset", "Mutagenicity", "--input-csv", str(self.csv), "--skip-checkpoint"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["csv"]["count"], 3)

    def test_missing_dependencies_report_failure_without_smoke(self):
        report = {"python_compatible": True, "packages": {"torch": {"installed": False}}}
        with mock.patch.object(environment_checker, "inventory", return_value=report), \
                mock.patch.object(environment_checker, "smoke_check", side_effect=AssertionError("Smoke called")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = environment_checker.main(["--smoke"])
        self.assertEqual(code, 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["smoke"]["status"], "skipped")
        self.assertEqual(result["missing_packages"], ["torch"])


if __name__ == "__main__":
    unittest.main()
