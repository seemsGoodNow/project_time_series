"""
Модуль с используемыми функциями потерь и метриками для оценки моделей
"""

from typing import Sequence

import numpy as np


KEY_RATE = 7
ADDITIONAL_RATE = KEY_RATE + 0.5
SURPLUS_RATE = KEY_RATE - 0.9
DEFICIT_RATE = KEY_RATE + 1
N_DAYS_IN_YEAR = 1 # for better visibility instead of 365

def target_loss(y_true: Sequence, y_pred: Sequence) -> float:
    """Целевая ошибка

    Parameters
    ----------
    y_true : Sequence
        _description_
    y_pred : Sequence
        _description_

    Returns
    -------
    float
        _description_
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_pred - y_true
    loss = np.abs(diff) * (
        (diff > 0) * DEFICIT_RATE
        + (diff < 0) * (ADDITIONAL_RATE - SURPLUS_RATE)
    ) / 100 / N_DAYS_IN_YEAR
    return np.mean(loss)

def maximum_absolute_error(y_true: Sequence, y_pred: Sequence) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.max(np.abs(y_pred-y_true))
