import pandas as pd
from .model import MLPClassifierV0


def load_kraken_data(ticker1, ticker2, timeframe):
    df = pd.read_csv("../data/raw/kraken/{}{}_{}.csv".format(ticker1, ticker2, timeframe))
    ## TODO: append columns
    return df

df = load_kraken_data("XBT", "USD", 60)
sma_hourly_v0_btc = MLPClassifierV0(df, "SMANNHourly1")
sma_hourly_v0_btc.get_labels(df)
sma_hourly_v0_btc.get_features(df)
