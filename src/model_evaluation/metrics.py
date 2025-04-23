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
        

class NumCriticalErrors(Metric):
    def __init__(self):
        pass

    def calc(self, y_true: Sequence, y_pred: Sequence) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.count_nonzero(np.abs(y_pred-y_true) > 0.42)
    
    
class MoneyLoss(Metric):
    
    def __init__(
        self, 
        rate: pd.Series = None, 
        max_ae: float = 0.42, 
        max_ae_penalty: Union[int, float] = 1,
        n_days_in_year: int = 365
    ):
        self._rate = rate
        self._max_ae = max_ae
        self._max_ae_penalty = max_ae_penalty
        self._n_days_in_year = n_days_in_year
        
    def calc(self, y_true: pd.Series, y_pred: pd.Series) -> float:
            
        diff = y_pred - y_true
        
        pos_diff = diff.where(diff > 0)
        # в случае перепрогноза экономически теряем
        pos_loss = pos_diff * (self._rate.reindex(pos_diff.index) + 1) / 100 / self._n_days_in_year

        neg_diff = diff.where(diff <= 0)
        # в случае недопрогноза экономически теряем
        neg_loss = neg_diff.abs() * 0.014 / self._n_days_in_year
        # 0.014 = ( (rate + 0.5) - (rate - 0.9) ) / 100 

        return pos_loss.sum() + neg_loss.sum()


class TargetLoss(Metric):
    
    def __init__(
        self, 
        rate: pd.Series = None, 
        max_ae: float = 0.42, 
        max_ae_penalty: Union[int, float] = 1,
        n_days_in_year: int = 365
    ):
        self._rate = rate
        self._max_ae = max_ae
        self._max_ae_penalty = max_ae_penalty
        self._n_days_in_year = n_days_in_year
        
    def calc(self, y_true: pd.Series, y_pred: pd.Series) -> float:
            
        diff = y_pred - y_true
        
        pos_diff = diff.where(diff > 0)
        # в случае перепрогноза экономически теряем
        pos_loss = pos_diff * (self._rate.reindex(pos_diff.index) + 1) / 100 / self._n_days_in_year

        neg_diff = diff.where(diff <= 0)
        # в случае недопрогноза экономически теряем
        neg_loss = neg_diff.abs() * 0.014 / self._n_days_in_year
        # 0.014 = ( (rate + 0.5) - (rate - 0.9) ) / 100
    
        # дополнительно штрафуем модель за превышение отклонения от допустимой границы требований заказчика      
        add_loss = (diff.abs() > self._max_ae).sum() * self._max_ae_penalty

        return pos_loss.sum() + neg_loss.sum() + add_loss


class LossForCatBoost:
    
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        result = []
        delta = 10**(-6)
        max_ae = 0.42
        add_penalty = 10
        for i in range(len(targets)):
            diff = targets[i] - approxes[i]
            der1 = 0 if abs(diff) < delta else (4.2 if diff>0 else -1.4)
            # 4.2 is average ruonia during 2021
            # 1.4 == 0.9 - (0.05)
            der2 = 0
            result.append((der1, der2))
        return result
    
class MetricForCatBoost:
    
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))
    
    def is_max_optimal(self):
        return False
    
    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 0.0
        max_ae = 0.42
        add_penalty = 100 * 365 * 1
        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            diff = approx[i] - target[i]
            error_sum += w * (4.2 * diff if diff > 0 else -1.4 * diff)
            if diff > max_ae or diff < -max_ae:
                error_sum += w * add_penalty
        return error_sum, weight_sum