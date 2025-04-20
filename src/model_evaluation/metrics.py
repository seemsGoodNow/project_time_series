"""
Модуль с используемыми функциями потерь и метриками для оценки моделей
"""
from abc import ABC, abstractmethod
from typing import Sequence, Union

import numpy as np
import pandas as pd


class Metric:
    @abstractmethod
    def calc(self, y_true, y_pred) -> float:
        pass


class SimpleTargetLoss(Metric):
    
    def __init__(
        self, 
        key_rate: float = 7,
        additional_rate_diff: float = 0.5,
        surplus_rate_diff: float = -0.9,
        deficit_rate_diff: float = 1.0,
        n_days_in_year: int = 1
    ):
        self._key_rate = key_rate
        self._additional_rate = self._key_rate + additional_rate_diff
        self._surplus_rate = self._key_rate + surplus_rate_diff
        self._deficit_rate = self._key_rate + deficit_rate_diff
        self._n_days_in_year = n_days_in_year
    
    def calc(self, y_true: Sequence, y_pred: Sequence) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        diff = y_pred - y_true
        loss = np.abs(diff) * (
            (diff > 0) * self._deficit_rate
            + (diff < 0) * (self._additional_rate - self._additional_rate)
        ) / 100 / self._n_days_in_year
        return np.mean(loss) 


class MaxAE(Metric):
    def __init__(self):
        pass

    def calc(self, y_true: Sequence, y_pred: Sequence) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.max(np.abs(y_pred-y_true))
    
    
class MAE(Metric):
    def __init__(self):
        pass

    def calc(self, y_true: Sequence, y_pred: Sequence) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_pred-y_true))


class TargetLoss(Metric):
    
    def __init__(
        self, 
        rate: pd.Series, 
        max_ae: float = 0.42, 
        max_ae_penalty: Union[int, float] = 1e6
    ):
        self._rate = rate
        self._max_ae = max_ae
        self._max_ae_penalty = max_ae_penalty
        
    def calc(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        pass
        
        