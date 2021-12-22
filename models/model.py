from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pickle

from .features import FeatureStore
from .model_utils import binary_labels

class Model(ABC):

    def __init__(self, df: pd.DataFrame, feature_store: FeatureStore, name: str):
        self.df: pd.DataFrame = df
        self.name = name
        self.period = None
        self.model = None
        self.period = feature_store.get_period()
        self.get_features = feature_store.get_features
        self.x_train: np.ndarry = None
        self.x_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None

    def split_data(self) -> None:
        assert self.df is not None, "Data frame must be provided"
        x = self.get_features(self.df)
        y = self.get_labels()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.33, shuffle=False)
        self.x_train, self.x_test = self.x_train[:len(self.x_train)-1], self.x_test[:len(self.x_test)-1]
        self.y_train, self.y_test = self.y_train[1:], self.y_test[1:]

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


class MLPClassifierV0(Model):

    def train(self):
        self.split_data()
        self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 20, 20), random_state=1, max_iter=10000)
        self.model.fit(self.x_train, self.y_train)

    def infer(self, x: np.ndarray):
        return self.model.predict(x)

    def get_labels(self) -> pd.DataFrame:
        return binary_labels(self.df["Close"])[self.period:]

# def XGBoostV0(Model):
#
#     def
