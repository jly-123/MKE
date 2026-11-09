import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT / "evaluate_gcn.py").read_text(encoding="utf-8"))
function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "summarize_predictions")
module = ast.parse("")
module.body = [function]
namespace = {}
exec(compile(ast.fix_missing_locations(module), "evaluate_gcn.py", "exec"), namespace)
summarize_predictions = namespace["summarize_predictions"]


class PredictorSummaryTest(unittest.TestCase):
    def test_confusion_orientation_counts_and_unrounded_accuracy(self):
        rows = [{"label": 0, "prediction": 0}, {"label": 0, "prediction": 1},
                {"label": 1, "prediction": 1}, {"label": 1, "prediction": 0},
                {"label": 1, "prediction": 0}, {"label": 1, "prediction": 1}]
        summary = summarize_predictions(rows)
        self.assertEqual(summary["confusion_matrix"], [[1, 1], [2, 2]])
        self.assertEqual(summary["label_counts"], {"0": 2, "1": 4})
        self.assertEqual(summary["prediction_counts"], {"0": 3, "1": 3})
        self.assertEqual(summary["count"], 6)
        self.assertEqual(summary["num_correct"], 3)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summarize_predictions(rows[:3])["accuracy"], 2 / 3)

    def test_empty_test_set_cannot_report_accuracy(self):
        with self.assertRaises(ValueError):
            summarize_predictions([])

    def test_unknown_classes_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_predictions([{"label": 2, "prediction": 0}])


if __name__ == "__main__":
    unittest.main()
