import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')  
    data.set_index('Date', inplace=True)
    return data.dropna()  # Remove missing values