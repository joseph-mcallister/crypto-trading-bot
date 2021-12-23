from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .model_utils import binary_labels, trinary_labels

class LabelGenerator(ABC):

    @abstractmethod
    def get_labels(self, df: pd.DataFrame) -> np.ndarray:
        ...

class BinaryLabels(LabelGenerator):

    def get_labels(self, df: pd.DataFrame):
        return binary_labels(df["Close"].to_numpy())

class TrinaryLabels(LabelGenerator):

    def __init__(self, threshold):
        self.threshold = threshold

    def get_labels(self, df: pd.DataFrame):
        return trinary_labels(df["Close"].to_numpy(), threshold=self.threshold)
