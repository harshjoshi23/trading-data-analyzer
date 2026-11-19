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
