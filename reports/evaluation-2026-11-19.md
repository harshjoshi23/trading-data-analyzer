# Retraining evaluation, 19 November 2026

All six LSTM runs had higher held-out MAE than last-close persistence.
These results provide insufficient evidence of a useful forecasting advantage.

## Protocol

- Existing 01.Bank CSV observations from January through June 2022.
- Training ends 25 May 2022. Evaluation covers 26 observed dates from 26 May through 30 June.
- Calendar gaps are forward-filled for input history. Scoring includes observed market dates only.
- The final 20% of calendar dates are held out. Stationarity and scaling use training data only.
- Standard architecture uses 50 hidden units, two layers and 20 epochs.
- Enhanced architecture uses Close changes and Volume, 100 hidden units, three layers and 100 epochs.
- Both architectures use seeds 42, 43 and 44. Every run is reported.
- The lowest training-loss checkpoint is evaluated. Holdout metrics never choose the checkpoint.
- Differenced predictions are reconstructed to original Close units using the previous observed Close.
- Evaluation is rolling one-step prediction with available historical observations.
- A repeated standard seed-42 run reproduced its JSON result exactly on this environment.

## Results

| Model | Mean MAE | Mean RMSE | MAE above persistence |
|---|---|---|---|
| Last-close persistence | 0.23808 | 0.33263 | 0% |
| standard | 0.25480 | 0.34544 | 7.02% |
| enhanced | 0.24121 | 0.33498 | 1.32% |

Every LSTM run had 50% direction accuracy over the 24 evaluation days with a nonzero actual change.
The small sample, anonymized instrument and 2022 data limit generalization.
This experiment measures forecast errors. Profitability, transaction costs, slippage,
latency and live-market performance remain unmeasured.

## Reproduce

Install torch, numpy, pandas, scikit-learn and statsmodels in a virtual environment.
From the repository root, run both entry points for each seed.

```sh
for seed in 42 43 44; do
  python src/pipeline.py --seed "$seed" --output-dir "results/standard-$seed"
  python src/ongoing_dev_pipeline.py --seed "$seed" --output-dir "results/enhanced-$seed"
done
```

Each run saves model.pth, predictions.csv and evaluation.json in its output directory.
The JSON includes the input CSV hash and dependency versions. Existing historical
weight and prediction files have not been replaced.

## Next decision

Use persistence as the benchmark for further evaluation. Assess a larger,
properly licensed dataset with several instruments and walk-forward windows
before revisiting model complexity. A voice-assisted research product can first
serve portfolio queries, cited market summaries and paper-order preparation,
with explicit user confirmation and an audit trail.
