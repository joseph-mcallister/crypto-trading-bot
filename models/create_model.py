import pandas as pd
from .model import SMANNHourlyV0


def load_kraken_data(ticker1, ticker2, timeframe):
    df = pd.read_csv("../data/raw/kraken/{}{}_{}.csv".format(ticker1, ticker2, timeframe))
    ## TODO: append columns
    return df

df = load_kraken_data("XBT", "USD", 60)
sma_hourly_v0_btc = SMANNHourlyV0(df, "SMANNHourly1")
sma_hourly_v0_btc.get_labels(df)
sma_hourly_v0_btc.get_features(df)
