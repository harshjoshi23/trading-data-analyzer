"""Run both normal CLI entry points with an isolated output directory."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.evaluation import forecast_metrics


ROOT = Path(__file__).resolve().parents[1]


class TrainingTest(unittest.TestCase):
    def test_both_cli_entry_points_score_original_prices(self):
        for script in ("pipeline.py", "ongoing_dev_pipeline.py"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "artifacts"
                result = subprocess.run(
                    [sys.executable, str(ROOT / "src" / script), "--epochs", "1",
                     "--output-dir", str(output)],
                    cwd=directory, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["model"]["observations"], 26)
                self.assertEqual(report["model"]["observations"], report["persistence_baseline"]["observations"])
                self.assertTrue((output / "model.pth").is_file())
                self.assertTrue((output / "predictions.csv").is_file())
                self.assertEqual(json.loads((output / "evaluation.json").read_text()), report)

    def test_baseline_and_direction_metrics(self):
        report = forecast_metrics([11, 9, 10], [10, 10, 10], [10, 10, 10])
        self.assertAlmostEqual(report["mae"], 2 / 3)
        self.assertEqual(report["moving_days"], 2)
        self.assertEqual(report["direction_accuracy_on_moving_days"], 0)
        with self.assertRaises(ValueError):
            forecast_metrics([1], [1, 2], [1])
