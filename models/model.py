from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

from .features import FeatureStore
from .labels import LabelGenerator
from .model_utils import binary_labels
import xgboost as xgb

from enum import Enum

class Model(ABC):

    def __init__(self, df: pd.DataFrame, feature_store: FeatureStore, label_generator: LabelGenerator, name: str):
        self.df: pd.DataFrame = df
        self.name = name
        self.model = None
        self.period = feature_store.get_period()
        self.feature_store = feature_store
        self.label_generator = label_generator
        self.x_train: np.ndarry = None
        self.x_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None
        self.split_data()

    def split_data(self) -> None:
        assert self.df is not None, "Data frame must be provided"
        x = self.feature_store.get_features(self.df)
        y = self.get_labels()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.33, shuffle=False)
        self.x_train, self.x_test = self.x_train[:len(self.x_train)-1], self.x_test[:len(self.x_test)-1]
        self.y_train, self.y_test = self.y_train[1:], self.y_test[1:]

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def infer(self, df: pd.DataFrame = None) -> np.ndarray:
        ...

    def get_labels(self) -> np.ndarray:
        return self.label_generator.get_labels(self.df)[self.period:]

    def save(self, path):
        filename = path + '/' + self.name
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        filename = path + '/' + self.name
        pickle.load(open(filename, 'rb'))


class MLPClassifierV0(Model):

    def __init__(self, df: pd.DataFrame, feature_store: FeatureStore, label_generator, name: str):
        super().__init__(df, feature_store, label_generator, name)

    def train(self):
        self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=70)
        self.model.fit(self.x_train, self.y_train)

    def infer(self, x: np.ndarray):
        return self.model.predict(x)


class XGBoostV0(Model):

    def __init__(self, df: pd.DataFrame, feature_store: FeatureStore, label_generator, name: str):
        super().__init__(df, feature_store, label_generator, name)

    def train(self):
        self.model = xgb.XGBClassifier(use_label_encoder=False)
        # self.model = xgb.XGBClassifier(max_depth=3, gamma=500, use_label_encoder=False, eval_metric="logloss")
        eval_metric = "error" if len(np.unique(self.get_labels())) == 2 else "merror"
        self.model.fit(self.x_train, self.y_train, eval_metric=eval_metric, eval_set=[(self.x_test, self.y_test)])

    def infer(self, x: np.ndarray):
        return self.model.predict(x)


