"""Holdout changes must not affect training samples or scaler parameters."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import prepare_sequences


class EvaluationTest(unittest.TestCase):
    def test_holdout_outlier_does_not_change_training(self):
        features = np.column_stack((np.arange(10), np.arange(10) * 100))
        original = prepare_sequences(features, train_rows=7, seq_length=3)
        changed = features.copy()
        changed[7:] = 100000
        modified = prepare_sequences(changed, train_rows=7, seq_length=3)
        np.testing.assert_array_equal(original[0], modified[0])
        np.testing.assert_array_equal(original[2], modified[2])
        np.testing.assert_array_equal(original[4].data_max_, [6, 600])
        self.assertEqual(original[2].shape, (4, 1))
        self.assertEqual(original[3].shape, (3, 1))
        self.assertGreater(modified[3][0, 0], 1)
        restored = original[4].inverse_transform(np.column_stack((original[3], np.zeros_like(original[3]))))
        np.testing.assert_allclose(restored[:, 0], [7, 8, 9])

    def test_short_training_period_rejected(self):
        with self.assertRaisesRegex(ValueError, "training targets"):
            prepare_sequences(np.arange(10).reshape(-1, 1), train_rows=3, seq_length=3)
