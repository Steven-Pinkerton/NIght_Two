import yfinance as yf

def download_data(ticker, start_date, end_date, period='1d'):
    """
    Downloads historical market data for a given ticker within a date range.

    Args:
    ticker (str): the ticker symbol for the security.
    start_date (str): the start date for the data range in format 'YYYY-MM-DD'.
    end_date (str): the end date for the data range in format 'YYYY-MM-DD'.
    period (str): the frequency of data ('1d' for daily, '1wk' for weekly, '1mo' for monthly). Default is '1d'.

    Returns:
    pandas.DataFrame: the historical market data for the given ticker and date range.
    """
    # Download historical data as a pandas DataFrame
        # Download historical data as a pandas DataFrame
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=period)
    except Exception as e:
        print(f"Error occurred while trying to download data for {ticker}: {e}")
        return None
    
    return data


def save_data(data, filename):
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
    data (pandas.DataFrame): the DataFrame to save.
    filename (str): the name of the CSV file.
    """
    try:
        data.to_csv(filename)
    except Exception as e:
        print(f"Error occurred while trying to save data to {filename}: {e}")


if __name__ == "__main__":
    # Download daily data for the Russell 2000 Index
    data = download_data('^RUT', '2010-01-01', '2023-06-28', period='1d')
    
    # Save the data to a CSV file
    save_data(data, 'russell_2000_daily.csv')