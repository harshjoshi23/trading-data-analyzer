"""Exercise preprocessing functions without starting model training."""

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd


class StationarityTest(unittest.TestCase):
    def test_nonfinite_prices_are_cleaned_with_numpy_two(self):
        for name in ("pipeline.py", "ongoing_dev_pipeline.py"):
            with self.subTest(pipeline=name):
                source = Path(__file__).resolve().parents[1] / "src" / name
                tree = ast.parse(source.read_text())
                function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_stationary")
                namespace = {"np": np, "adfuller": Mock(return_value=(0, 0.01))}
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
                frame = pd.DataFrame({"Close": [10.0, np.inf, 12.0]})
                result = namespace["make_stationary"](frame)
                self.assertTrue(np.isfinite(result).all())
                self.assertEqual(result.tolist(), [10.0, 10.0, 12.0])

    def test_stationarity_decision_uses_training_period(self):
        for name in ("pipeline.py", "ongoing_dev_pipeline.py"):
            with self.subTest(pipeline=name):
                source = Path(__file__).resolve().parents[1] / "src" / name
                tree = ast.parse(source.read_text())
                function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_stationary")
                statistic = Mock(return_value=(0, 0.01))
                namespace = {"np": np, "adfuller": statistic}
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
                frame = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 10000.0]})
                namespace["make_stationary"](frame, fit_end=3)
                self.assertEqual(statistic.call_args.args[0].tolist(), [1.0, 2.0, 3.0])
