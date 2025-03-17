# Trading Data Analyzer

This project focuses on analyzing stock market data for the stock '01.Bank' using Exploratory Data Analysis (EDA) and Long Short-Term Memory (LSTM) models for time series forecasting. The dataset (`Stock_Market_Data.csv`) contains daily stock prices with columns: `Date`, `Name`, `Open`, `High`, `Low`, `Close`, and `Volume`. The project includes an EDA notebook and two forecasting pipelines: `pipeline.py` (baseline) and `ongoing_dev_pipeline.py` (improved version).

## Project Structure

- **data/**: Contains the dataset `Stock_Market_Data.csv`.
- **src/**: Contains the scripts:
  - `EDA_latest_analysis.ipynb`: Jupyter Notebook for Exploratory Data Analysis.
  - `pipeline.py`: Baseline script for preprocessing and LSTM-based forecasting.
  - `ongoing_dev_pipeline.py`: Enhanced script with improved LSTM forecasting.
  - `lstm_01.Bank_best.pth`: Saved model weights from the best training run.
  - `predictions_01.Bank.npy`: Saved predictions for further analysis.
- **README.md**: This file.

## Exploratory Data Analysis (EDA)

The EDA is performed in `EDA_latest_analysis.ipynb` to understand the stock data for '01.Bank' and prepare it for modeling. Below is a summary of the analyses conducted:

### 1. Data Loading and Preprocessing
- Loaded the dataset and converted the `Date` column to datetime format.
- Selected data for '01.Bank', set the date as the index, and ensured daily frequency using forward filling.
- Checked for missing values (none found after preprocessing) and confirmed data types.

### 2. Summary Statistics
- Generated summary statistics for '01.Bank':
  - **Count**: 180 data points.
  - **Mean Close Price**: 21.302112.
  - **Min Close Price**: 19.170000.
  - **Max Close Price**: 23.370000.
  - **Mean Volume**: 1,161,698.
  - **Max Volume**: 2,844,397.

### 3. Visualizations
The following plots were generated to explore the data:

- **Closing Prices Over Time**:
  - Shows the trend of closing prices from January to July 2022, revealing a general downward trend with fluctuations.
  - Plot: See `Closing Prices Over Time for 01.Bank` in the notebook (Cell 7).

- **Trading Volume Over Time**:
  - Displays the trading volume, highlighting periods of high trading activity (e.g., peaks around 2.5M).
  - Plot: See `Trading Volume Over Time for 01.Bank` in the notebook (Cell 8).

- **Rolling Statistics of Closing Prices**:
  - Plots the closing price alongside its 30-day rolling mean and standard deviation, showing the trend and volatility.
  - The rolling mean confirms the downward trend, while the standard deviation indicates periods of higher volatility.
  - Plot: See `Rolling Statistics of Closing Prices for 01.Bank` in the notebook (Cell 9).

- **Correlation Heatmap**:
  - Displays correlations between `Open`, `High`, `Low`, `Close`, and `Volume`.
  - Strong correlations (0.99–1.00) between price-related features (`Open`, `High`, `Low`, `Close`), but a weaker correlation (0.51–0.54) with `Volume`.
  - Plot: See `Correlation Heatmap of Stock Features for 01.Bank` in the notebook (Cell 10).

- **Stationary Closing Prices**:
  - After applying differencing to make the series stationary (confirmed by ADF test: p-value 0.8068 > 0.05 initially), the differenced closing prices are plotted.
  - Shows reduced trend with more random fluctuations, suitable for modeling.
  - Plot: See `Stationary Closing Prices for 01.Bank` in the notebook (Cell 11).

- **Seasonal Decomposition**:
  - Decomposes the closing prices into trend, seasonal, and residual components using a 30-day period.
  - Confirms the downward trend, with a weak seasonal pattern and noticeable residuals.
  - Plot: See `Seasonal Decomposition of Closing Prices for 01.Bank` in the notebook (Cell 12).

- **Distribution of Daily Returns**:
  - A histogram of daily returns (calculated as percentage change in closing price) shows a near-normal distribution centered around 0, with some outliers.
  - Indicates relatively stable returns with occasional extreme movements.
  - Plot: See `Distribution of Daily Returns for 01.Bank` in the notebook (Cell 14).

## Forecasting Pipelines

### Baseline Pipeline (`pipeline.py`)
The baseline forecasting pipeline is implemented in `pipeline.py`, focusing on predicting the stationary closing prices using an LSTM model. The script performs the following steps:

1. **Data Loading and Preprocessing**:
   - Loads the dataset and selects data for '01.Bank'.
   - Ensures daily frequency and handles missing/infinite values.
   - Makes the `Close` price series stationary using differencing (ADF test confirms non-stationarity initially).

2. **LSTM Model Training**:
   - Prepares sequences with a length of 60 days.
   - Splits data into 80% training and 20% testing sets.
   - Trains a 2-layer LSTM model with 50 hidden units for 20 epochs using the Adam optimizer and Mean Squared Error (MSE) loss.
   - Training output:
     - Epoch 5 Loss: 0.012496
     - Epoch 10 Loss: 0.014591
     - Epoch 15 Loss: 0.012447
     - Epoch 20 Loss: 0.012219

3. **Evaluation**:
   - Evaluates the model on the test set, achieving an MSE of 0.0387 on the differenced (stationary) series.
   - The low MSE indicates reasonable predictive performance for the stationary data.

### Enhanced Pipeline (`ongoing_dev_pipeline.py`)
The enhanced forecasting pipeline in `ongoing_dev_pipeline.py` builds on the baseline with several improvements for better performance and scalability. The script includes the following enhancements:

1. **Data Loading and Preprocessing**:
   - Same as the baseline but includes `Volume` as an additional feature to leverage its correlation with price movements (EDA insight: correlation 0.51–0.54 with `Close`).

2. **LSTM Model Training**:
   - Prepares sequences with a length of 60 days and splits data into 80% training and 20% testing sets.
   - Trains a 3-layer LSTM model with 100 hidden units for 100 epochs, using dropout (0.2) to prevent overfitting.
   - Incorporates a learning rate scheduler (`ReduceLROnPlateau`) to adjust the learning rate dynamically, improving convergence.
   - Saves the best model weights (`lstm_01.Bank_best.pth`) based on the lowest training loss.
   - Training output:
     - Epoch 1 Loss: 0.375410
     - Epoch 5 Loss: 0.028969
     - Epoch 30 Loss: 0.010347 (lowest)
     - Epoch 100 Loss: 0.014833

3. **Evaluation**:
   - Evaluates the model on the test set, achieving an MSE of 0.0364, MAE of 0.1273, and RMSE of 0.1907 on the differenced series.
   - Saves predictions (`predictions_01.Bank.npy`) for further analysis.
   - The improved MSE (0.0364 vs. 0.0387) indicates better predictive performance compared to the baseline.

### Improvements in `ongoing_dev_pipeline.py`
- **Increased Model Complexity**: 3 layers and 100 hidden units (vs. 2 layers and 50 units) for capturing more complex patterns.
- **Additional Feature**: Uses `Volume` alongside `Close` to improve predictions.
- **More Epochs**: Trains for 100 epochs (vs. 20) with a scheduler to ensure better convergence.
- **Enhanced Evaluation**: Reports MSE, MAE, and RMSE for a comprehensive assessment.
- **Model Persistence**: Saves the best model and predictions for reuse.

## How to Run

1. **Environment Setup**:
   - Activate the virtual environment: `source new_env/bin/activate`.
   - Install dependencies: `pip install -r requirements.txt`.

2. **Run EDA**:
   - Open `EDA_latest_analysis.ipynb` in Jupyter Notebook and run all cells to explore the data and view visualizations.

3. **Run Pipelines**:
   - Navigate to the `src/` directory: `cd src`.
   - Run the baseline pipeline: `python3 pipeline.py`.
   - Run the enhanced pipeline: `python3 ongoing_dev_pipeline.py`.
   - Expected output includes ADF test results, training loss per epoch, and evaluation metrics.

## Dependencies

- Python 3.10
- Libraries: pandas, numpy, matplotlib, seaborn, statsmodels, torch, sklearn (listed in `requirements.txt`)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.