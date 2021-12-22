from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from .model_utils import moving_avg, moving_avg_diff

class FeatureStore(ABC):

    @staticmethod
    @abstractmethod
    def get_features(df: pd.DataFrame) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def get_period() -> int:
        ...

class HourlySMA(FeatureStore):

    @staticmethod
    def get_period():
        return 24 * 31

    def get_features(df: pd.DataFrame) -> np.ndarray:
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
        return x[HourlySMA.get_period():]
