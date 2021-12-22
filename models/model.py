from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib
import pickle

from .model_utils import binary_labels, moving_avg, moving_avg_diff

class Model(ABC):

    def __init__(self, df: pd.DataFrame, name: str):
        self.df: pd.DataFrame = df
        self.name = name
        self.period = None
        self.model = None
        self.x_train: np.ndarry = None
        self.x_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None

    @abstractmethod
    def split_data(self) -> None:
        ...

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def infer(self, df: pd.DataFrame = None) -> np.ndarray:
        if self.df is None and df is None:
            raise "No data frame provided"

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        return binary_labels()

    def save(self, path):
        filename = path + '/' + self.name
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        filename = path + '/' + self.name
        pickle.load(open(filename, 'rb'))


class SMANNHourlyV0(Model):

    def __init__(self, df, name):
        super().__init__(df, name)
        self.period = 24*31

    def split_data(self):
        assert self.df is not None, "Data frame must be provided"
        x = self.get_features(self.df)
        y = self.get_labels()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.33, shuffle=False)
        self.x_train, self.x_test = self.x_train[:len(self.x_train)-1], self.x_test[:len(self.x_test)-1]
        self.y_train, self.y_test = self.y_train[1:], self.y_test[1:]

    def train(self):
        self.split_data()
        self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 20, 20), random_state=1, max_iter=10000)
        self.model.fit(self.x_train, self.y_train)

    def infer(self, x: np.ndarray):
        return self.model.predict(x)

    def get_features(self, df: pd.DataFrame = None) -> np.ndarray:
        df = df if df is not None else self.df
        moving_avg_1_hours = moving_avg(df["Close"], 1)
        moving_avg_2_hours = moving_avg(df["Close"], 2)
        moving_avg_3_hours = moving_avg(df["Close"], 3)
        moving_avg_12_hours = moving_avg(df["Close"], 12)
        moving_avg_24_hours = moving_avg(df["Close"], 24)
        moving_avg_48_hours = moving_avg(df["Close"], 48)
        moving_avg_1_week = moving_avg(df["Close"], 24 * 7)
        moving_avg_1_month = moving_avg(df["Close"], 24 * 31)

        mv_1h_2h = moving_avg_diff(moving_avg_1_hours, moving_avg_2_hours)
        mv_1h_3h = moving_avg_diff(moving_avg_1_hours, moving_avg_3_hours)
        mv_3h_12h = moving_avg_diff(moving_avg_3_hours, moving_avg_12_hours)
        mv_12h_24h = moving_avg_diff(moving_avg_12_hours, moving_avg_24_hours)
        mv_24h_48h = moving_avg_diff(moving_avg_24_hours, moving_avg_48_hours)
        mv_48h_1w = moving_avg_diff(moving_avg_48_hours, moving_avg_1_week)
        mv_1w_1m = moving_avg_diff(moving_avg_1_week, moving_avg_1_month)

        x = np.stack((mv_1h_2h, mv_1h_3h, mv_3h_12h, mv_12h_24h, mv_24h_48h, mv_48h_1w, mv_1w_1m), axis=1)
        return x[self.period:]

    def get_labels(self) -> pd.DataFrame:
        return binary_labels(self.df["Close"])[self.period:]


class SMANNHourlyXGBoost(Model):

    def __init__(self, df, name):
        super().__init__(df, name)
        self.period = 24*31

    def split_data(self):
        assert self.df is not None, "Data frame must be provided"
        x = self.get_features(self.df)
        y = self.get_labels()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.33, shuffle=False)
        self.x_train, self.x_test = self.x_train[:len(self.x_train)-1], self.x_test[:len(self.x_test)-1]
        self.y_train, self.y_test = self.y_train[1:], self.y_test[1:]

    def train(self):
        self.split_data()
        self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=10000)
        self.model.fit(self.x_train, self.y_train)

    def infer(self, x: np.ndarray):
        return self.model.predict(x)

    def get_features(self, df: pd.DataFrame = None) -> np.ndarray:
        df = df if df is not None else self.df
        moving_avg_1_hours = moving_avg(df["Close"], 1)
        moving_avg_2_hours = moving_avg(df["Close"], 2)
        moving_avg_3_hours = moving_avg(df["Close"], 3)
        moving_avg_12_hours = moving_avg(df["Close"], 12)
        moving_avg_24_hours = moving_avg(df["Close"], 24)
        moving_avg_48_hours = moving_avg(df["Close"], 48)
        moving_avg_1_week = moving_avg(df["Close"], 24 * 7)
        moving_avg_1_month = moving_avg(df["Close"], 24 * 31)

        mv_1h_2h = moving_avg_diff(moving_avg_1_hours, moving_avg_2_hours)
        mv_1h_3h = moving_avg_diff(moving_avg_1_hours, moving_avg_3_hours)
        mv_3h_12h = moving_avg_diff(moving_avg_3_hours, moving_avg_12_hours)
        mv_12h_24h = moving_avg_diff(moving_avg_12_hours, moving_avg_24_hours)
        mv_24h_48h = moving_avg_diff(moving_avg_24_hours, moving_avg_48_hours)
        mv_48h_1w = moving_avg_diff(moving_avg_48_hours, moving_avg_1_week)
        mv_1w_1m = moving_avg_diff(moving_avg_1_week, moving_avg_1_month)

        x = np.stack((mv_1h_2h, mv_1h_3h, mv_3h_12h, mv_12h_24h, mv_24h_48h, mv_48h_1w, mv_1w_1m), axis=1)
        return x[self.period:]

    def get_labels(self) -> pd.DataFrame:
        return binary_labels(self.df["Close"])[self.period:]