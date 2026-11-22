"""Reproducible chronological LSTM evaluation against last-close persistence."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .evaluation import prepare_sequences, forecast_metrics
else:
    from evaluation import prepare_sequences, forecast_metrics


class LSTMModel(nn.Module):
    def __init__(self, enhanced):
        super().__init__()
        hidden, layers = (100, 3) if enhanced else (50, 2)
        self.lstm = nn.LSTM(
            2 if enhanced else 1, hidden, layers,
            batch_first=True, dropout=0.2 if enhanced else 0,
        )
        self.output = nn.Linear(hidden, 1)

    def forward(self, values):
        sequence, _ = self.lstm(values)
        return self.output(sequence[:, -1, :])


def load_stock(path, stock):
    data = pd.read_csv(path)
    data = data.loc[data["Name"] == stock].copy()
    if data.empty:
        raise ValueError(f"No observations for {stock!r}.")
    data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y", errors="raise")
    data = data.set_index("Date").sort_index()
    if data.index.has_duplicates:
        raise ValueError("Stock dates must be unique.")
    for column in ("Close", "Volume"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all():
            raise ValueError(f"{column} must contain finite observed values.")
    observed = data.index
    return data.asfreq("D", method="ffill"), observed


def run(args, enhanced):
    if args.epochs < 1 or args.threads < 1:
        raise ValueError("Epochs and threads must be positive.")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("Output destination already exists; choose a fresh --output-dir for each run.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    data, observed = load_stock(args.data, args.stock)
    split_row = int(len(data) * 0.8)
    if split_row <= 61 or split_row >= len(data):
        raise ValueError("Need at least 62 training calendar rows and a holdout.")
    cutoff = data.index[split_row]
    train_close = data["Close"].iloc[:split_row]
    # A constant training series is already stationary for this workflow.
    p_value = float(adfuller(train_close)[1]) if train_close.nunique() > 1 else 0.0
    differenced = p_value > 0.05
    target = data["Close"].diff().dropna() if differenced else data["Close"]
    features = target.to_numpy()[:, None]
    if enhanced:
        features = np.column_stack((features, data.loc[target.index, "Volume"].to_numpy()))
    train_rows = int((target.index < cutoff).sum())
    x_train, x_test, y_train, y_test, scaler = prepare_sequences(features, train_rows)
    dates = target.index[train_rows:]
    observed_mask = dates.isin(observed)
    if not observed_mask.any():
        raise ValueError("Holdout must contain observed market dates.")

    model = LSTMModel(enhanced)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=64 if enhanced else 32,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        if enhanced else None
    )
    best_loss = float("inf")
    best_state = None
    for _ in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)
        mean_loss = total_loss / len(x_train)
        if scheduler is not None:
            scheduler.step(mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scaled_predictions = model(torch.tensor(x_test, dtype=torch.float32)).numpy()
    padding = np.zeros((len(y_test), features.shape[1] - 1))
    prediction = scaler.inverse_transform(np.column_stack((scaled_predictions, padding)))[:, 0]
    actual_target = scaler.inverse_transform(np.column_stack((y_test, padding)))[:, 0]
    previous = data["Close"].shift(1).loc[dates].to_numpy()
    actual = data["Close"].loc[dates].to_numpy()
    predicted_price = prediction + previous if differenced else prediction
    reconstructed = actual_target + previous if differenced else actual_target
    if not np.allclose(reconstructed, actual):
        raise ValueError("Transformed targets do not align with held-out prices.")

    actual, predicted_price, previous = (
        values[observed_mask] for values in (actual, predicted_price, previous)
    )
    scores = forecast_metrics(actual, predicted_price, previous)
    baseline = forecast_metrics(actual, previous, previous)
    result = {
        "variant": "enhanced" if enhanced else "standard",
        "stock": args.stock,
        "seed": args.seed,
        "epochs": args.epochs,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "training_end": str(data.index[split_row - 1].date()),
        "holdout_start": str(dates[observed_mask][0].date()),
        "holdout_end": str(dates[observed_mask][-1].date()),
        "training_samples": len(x_train),
        "adf_training_p_value": p_value,
        "differenced": differenced,
        "scoring": "one-step observed market dates, original Close units",
        "model": scores,
        "persistence_baseline": baseline,
        "beats_baseline_mae": scores["mae"] < baseline["mae"],
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({
        "date": dates[observed_mask],
        "actual_close": actual,
        "predicted_close": predicted_price,
        "persistence_close": previous,
    }).to_csv(args.output_dir / "predictions.csv", index=False)
    torch.save({
        "state_dict": best_state,
        "variant": result["variant"],
        "seed": args.seed,
        "scaler_min": scaler.min_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "differenced": differenced,
        "data_sha256": result["data_sha256"],
    }, args.output_dir / "model.pth")
    (args.output_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(enhanced=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "Stock_Market_Data.csv")
    parser.add_argument("--stock", default="01.Bank")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100 if enhanced else 20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / ("enhanced" if enhanced else "standard"))
    args = parser.parse_args()
    try:
        result = run(args, enhanced)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
