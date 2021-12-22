import unittest
import numpy as np
import pandas as pd
from models import SMANNHourlyV0, SMANNHourlyXGBoost
from .model_utils import moving_avg, moving_avg_diff, perc_change, binary_labels
from pathlib import Path
from sklearn import metrics


class ModelUtils(unittest.TestCase):
    @staticmethod
    def test_moving_average():
        arr = np.array([1, 10, 100, 1000])
        assert moving_avg(arr, 1)[-1] == 1000
        assert moving_avg(arr, 2)[-1] == (1000 + 100) / 2
        assert moving_avg(arr, 3)[-2] == sum(arr[:3]) / 3
        assert arr.size == len(arr)

    @staticmethod
    def test_moving_avg_diff():
        arr = np.array([0, 1000, 100, 500])
        moving_avg_2 = moving_avg(arr, 2)
        moving_avg_3 = moving_avg(arr, 3)
        moving_avg_3_2 = moving_avg_diff(moving_avg_2, moving_avg_3)
        expected = ((500 + 100) / 2 - (500 + 100 + 1000) / 3) / ((500 + 100 + 1000) / 3)
        assert moving_avg_3_2[-1] - expected < 0.0001
        assert np.isnan(moving_avg_3_2[0])

    @staticmethod
    def test_perc_change():
        arr = np.random.random(10)
        tmp = perc_change(arr)
        assert tmp[-1] == (arr[9] - arr[8]) / arr[8]
        assert perc_change(np.array([100, 200, 300]), shift=2)[1] - 3 < 0.00001
        assert np.isnan(tmp[0])

    @staticmethod
    def test_binary_labels():
        arr = [1, 1.3, 1.5, 1.4, 1.7, 0.9]
        binary_labels_1 = binary_labels(arr, shift=1)
        binary_labels_2 = binary_labels(arr, shift=2)
        assert np.isnan(binary_labels_1[0])
        assert binary_labels_1[1] == 1
        assert binary_labels_1[3] == 0
        assert binary_labels_2[2] == 1
        assert binary_labels_2[3] == 1
        assert binary_labels_2[5] == 0

    @staticmethod
    def test_smann_v0():
        data_path = (Path(__file__).parent / "../data/raw/kraken/XBTUSD_1.csv").resolve()
        df = pd.read_csv(data_path, names=["Date", "Open", "High", "Low", "Close", "Volume", "Trades"])
        df['Date'] = pd.to_datetime(df['Date'], unit='s')
        df = df.iloc[::60, :]
        model = SMANNHourlyV0(df, "test_1")
        # x_(t-1) predicts y_t
        assert model.get_features().shape[0] == df[model.period:].shape[0]
        model.train()
        assert len(model.x_train) + len(model.x_test) == df[model.period:].shape[0] - 2
        labels = model.get_labels()
        baseline = max(np.where(labels == 0)[0].shape[0], np.where(labels == 1)[0].shape[0]) / len(labels)
        print("Baseline Accuracy: ", baseline)
        y_test_pred = model.infer(model.x_test)
        y_train_pred = model.infer(model.x_train)
        print("Train Accuracy:", metrics.accuracy_score(model.y_train, y_train_pred))
        print("Test Accuracy:", metrics.accuracy_score(model.y_test, y_test_pred))
        assert metrics.accuracy_score(model.y_test, y_test_pred) > 0.5

if __name__ == '__main__':
    unittest.main()
