import pandas as pd
from pathlib import Path


def load_kraken_data(ticker: str, denomination="USD", interval_in_minutes=1):
    valid_intervals = [1, 5, 15, 60, 720, 1440]
    assert interval_in_minutes in valid_intervals, "Valid intervals include {}".format(valid_intervals)
    filename = "{}{}_{}.csv".format(ticker, denomination, interval_in_minutes)
    data_path = (Path(__file__).parent / "../data/raw/kraken/{}".format(filename)).resolve()
    df = pd.read_csv(data_path, names=["Date", "Open", "High", "Low", "Close", "Volume", "Trades"])
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    return df