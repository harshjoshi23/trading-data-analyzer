"""Observed data must be valid before stationarity fitting or training."""

from pathlib import Path
import tempfile
import unittest

from src.training import load_stock


class StockInputTest(unittest.TestCase):
    def test_dates_sorted_and_only_observed_dates_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text("Date,Name,Close,Volume\n03-01-2022,X,12,200\n01-01-2022,X,10,100\n")
            daily, observed = load_stock(path, "X")
            self.assertEqual(len(daily), 3)
            self.assertEqual(len(observed), 2)
            self.assertEqual(daily.Close.tolist(), [10, 10, 12])

    def test_invalid_observed_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text("Date,Name,Close,Volume\n01-01-2022,X,inf,100\n")
            with self.assertRaisesRegex(ValueError, "finite"):
                load_stock(path, "X")
