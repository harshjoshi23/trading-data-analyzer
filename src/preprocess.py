def make_stationary(data, column='Close'):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(data[column])
    if result[1] > 0.05:
        return data[column].diff().dropna()
    return data[column]