import datetime
import unittest
from .data_utils import load_kraken_data

class KrakenTests(unittest.TestCase):
    def test_load_kraken_data(self):
        df = load_kraken_data("XBT", "USD", 15)
        assert len(df) > 0
        required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            assert col in list(df.columns)
        assert df["Date"].iloc[0] < datetime.datetime(2016, 1, 1)


if __name__ == '__main__':
    unittest.main()
