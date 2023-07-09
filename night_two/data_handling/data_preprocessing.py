import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(filename):
    """
    Preprocesses historical market data for machine learning.

    Args:
    filename (str): the name of the CSV file containing the historical data.

    Returns:
    pandas.DataFrame: the preprocessed data.
    """
    # Load data from CSV file
    data = pd.read_csv(filename, parse_dates=True, index_col=0)

    # Fill any missing values
    data.fillna(method='ffill', inplace=True)

    # Normalize the data
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    # Save the preprocessed data to a new CSV file
    data.to_csv('preprocessed_' + filename)

    return data

if __name__ == "__main__":
    # Preprocess the data
    preprocess_data('russell_2000_daily.csv')