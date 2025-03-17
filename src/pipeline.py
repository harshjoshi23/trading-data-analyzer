# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

# Load the data
def load_data(file_path):
    """Load stock market data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to make data stationary
def make_stationary(data, column='Close'):
    """Check stationarity and apply differencing if needed."""
    # Ensure data is finite
    if not np.isfinite(data[column]).all():
        print(f"Warning: '{column}' contains non-finite values. Cleaning data...")
        data[column] = data[column].replace([np.inf, -np.inf], np.NaN).fillna(method='ffill')
        if data[column].isna().any():
            data[column] = data[column].fillna(method='bfill')  # Backup fill
    result = adfuller(data[column].dropna())
    print(f'ADF Statistic for {column}: {result[0]}')
    print(f'p-value for {column}: {result[1]}')
    if result[1] > 0.05:
        print(f"Series '{column}' is non-stationary, applying differencing.")
        return data[column].diff().dropna()
    print(f"Series '{column}' is stationary.")
    return data[column]

# Load and preprocess data
data = load_data('../data/Stock_Market_Data.csv')
if data is not None:
    # Select data for one stock (e.g., '01.Bank')
    stock_name = '01.Bank'
    data_stock = data[data['Name'] == stock_name].copy()
    if data_stock.empty:
        print(f"Error: No data found for stock '{stock_name}'. Check stock names with data['Name'].unique().")
    else:
        data_stock.set_index('Date', inplace=True)
        data_stock = data_stock.drop(columns=['Name'])
        
        # Handle missing or infinite values and ensure daily frequency
        data_stock = data_stock.replace([np.inf, -np.inf], np.NaN)
        data_stock = data_stock.asfreq('D', method='ffill')
        
        # Make the Close price stationary
        stationary_close = make_stationary(data_stock, 'Close')
        if stationary_close is None or len(stationary_close) == 0:
            print("Error: Stationary data is empty after processing.")
        else:
            # Prepare data for LSTM
            print("\nPreparing data for LSTM...")
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(stationary_close.values[:, None])

            # Create sequences
            def create_sequences(data, seq_length=60):
                """Create input-output sequences for LSTM."""
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i + seq_length])
                    y.append(data[i + seq_length])
                return np.array(X), np.array(y)

            seq_length = 60
            X, y = create_sequences(scaled_data)
            if len(X) == 0:
                print(f"Error: Not enough data points ({len(scaled_data)}) for sequence length {seq_length}.")
            else:
                # Split into train and test sets
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                # Convert to PyTorch tensors
                X_train = torch.Tensor(X_train)
                y_train = torch.Tensor(y_train)
                X_test = torch.Tensor(X_test)
                y_test = torch.Tensor(y_test)

                # Define LSTM model
                class LSTMModel(nn.Module):
                    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
                        super(LSTMModel, self).__init__()
                        self.hidden_size = hidden_size
                        self.num_layers = num_layers
                        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                        self.fc = nn.Linear(hidden_size, 1)

                    def forward(self, x):
                        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                        out, _ = self.lstm(x, (h0, c0))
                        out = self.fc(out[:, -1, :])
                        return out

                # Set device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {device}")

                # Initialize model, loss, and optimizer
                model = LSTMModel(input_size=1, hidden_size=50, num_layers=2).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # Create DataLoader
                batch_size = 32
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Train the model
                num_epocs = 20
                print("Training LSTM model...")
                for epoch in range(num_epocs):
                    model.train()
                    total_loss = 0
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    avg_loss = total_loss / len(train_loader)
                    if (epoch + 1) % 5 == 0:
                        print(f'Epoch [{epoch+1}/{num_epocs}], Loss: {avg_loss:.6f}')

                # Evaluate the model
                model.eval()
                with torch.no_grad():
                    X_test = X_test.to(device)
                    predictions = model(X_test).cpu().numpy()
                    predictions = scaler.inverse_transform(predictions)
                    y_test_inv = scaler.inverse_transform(y_test.cpu().numpy())

                # Calculate MSE
                mse = np.mean((y_test_inv - predictions) ** 2)
                print(f'Mean Squared Error on Test Set: {mse:.4f}')