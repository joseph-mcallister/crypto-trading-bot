import datetime
import unittest
import numpy as np

from data.data_utils import load_kraken_data
from models import MLPClassifierV0, XGBoostV0
from models.features import HourlySMA
from .model_utils import moving_avg, moving_avg_diff, perc_change, binary_labels, trinary_labels
from sklearn import metrics
from .labels import BinaryLabels, TrinaryLabels

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
    def test_trinary_labels():
        arr = [1, 1.3, 1.5, 1.4, 1.7, 0.9]
        trinary_labels_1 = trinary_labels(arr, shift=1, threshold=0.15)
        assert trinary_labels_1[1] == 2
        assert trinary_labels_1[3] == 1
        assert trinary_labels_1[5] == 0
        trinary_labels_2 = trinary_labels(arr, shift=2, threshold=0.3)
        assert np.isnan(trinary_labels_2[0])
        assert not np.isnan(trinary_labels_2[2])
        assert trinary_labels_2[2] == 2
        assert trinary_labels_2[5] == 0
        assert trinary_labels_2[3] == 1

    @staticmethod
    def test_model():
        df = load_kraken_data("XBT", "USD", 60)
        df = df[df["Date"] >= datetime.datetime(2017, 1, 1)]
        feature_store = HourlySMA()
        model = XGBoostV0(df, feature_store, TrinaryLabels(0.001), "hourly_sma_mlp")
        period = feature_store.get_period()
        assert feature_store.get_features(df).shape[0] == df[period:].shape[0]
        model.train()
        assert len(model.x_train) + len(model.x_test) == df[period:].shape[0] - 2

        labels = model.get_labels()
        possible_labels = np.unique(labels)
        common_guess_baseline = 0
        for label in possible_labels:
            perc_dist =  len(labels[labels == label]) / len(labels)
            print("Label {} distribution: {}%".format(label, perc_dist*100))
            if perc_dist > common_guess_baseline:
                common_guess_baseline = perc_dist
        print("Baseline Accuracy {}: ".format(common_guess_baseline))

        y_test_pred = model.infer(model.x_test)
        y_train_pred = model.infer(model.x_train)
        print("Train Accuracy:", metrics.accuracy_score(model.y_train, y_train_pred))
        print("Test Accuracy:", metrics.accuracy_score(model.y_test, y_test_pred))
        assert metrics.accuracy_score(model.y_train, y_train_pred) > common_guess_baseline, "Does not outperform baseline on train set"
        assert metrics.accuracy_score(model.y_test, y_test_pred) > common_guess_baseline, "Does not outperform baseline on test set"


if __name__ == '__main__':
    unittest.main()
