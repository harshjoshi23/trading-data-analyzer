"""Chronological sequence preparation with training-only feature scaling."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_sequences(features, train_rows, seq_length=60):
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Features must be a finite two-dimensional array.")
    if not 0 < seq_length < train_rows < len(values):
        raise ValueError("Need training targets and held-out targets after the sequence window.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[:train_rows])
    scaled = scaler.transform(values)
    inputs = np.array([scaled[i:i + seq_length] for i in range(len(scaled) - seq_length)])
    targets = scaled[seq_length:, :1]
    train_size = train_rows - seq_length
    return inputs[:train_size], inputs[train_size:], targets[:train_size], targets[train_size:], scaler


def forecast_metrics(actual, predicted, previous):
    """Score one-step price forecasts and their implied direction."""
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    previous = np.asarray(previous, dtype=float).reshape(-1)
    if not len(actual) or actual.shape != predicted.shape or actual.shape != previous.shape:
        raise ValueError("Need equally sized, non-empty forecast arrays.")
    if not all(np.isfinite(values).all() for values in (actual, predicted, previous)):
        raise ValueError("Forecast arrays must contain finite values.")
    errors = predicted - actual
    moving = actual != previous
    direction = (
        float(np.mean(np.sign(predicted[moving] - previous[moving]) == np.sign(actual[moving] - previous[moving])))
        if moving.any() else None
    )
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "direction_accuracy_on_moving_days": direction,
        "moving_days": int(moving.sum()),
        "observations": len(actual),
    }
