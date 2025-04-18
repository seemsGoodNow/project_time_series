"""
Модуль с используемыми функциями потерь и метриками для оценки моделей
"""

from typing import Sequence

import numpy as np

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
    rate_filepath = 'data/rounia.xlsx'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate_df = pd.read_excel(rate_filepath, engine="openpyxl")
    rate_df['date'] = pd.to_datetime(rate_df['DT'])
    rate_df = rate_df[['date', 'ruo']].rename(columns={'ruo': 'rate'})
    rate_df = rate_df.sort_values('date').set_index('date')
    rate = np.array(rate_df.iloc[idx]['rate'])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_pred - y_true
    positive_diff = diff.clip(min=0)
    negative_diff = diff.clip(max=0)

    # экономические потери
    loss = positive_diff * (rate + 1) / 100 - negative_diff * (rate + 0.5 - (rate - 0.9)) / 100

    # требование заказчика
    MAX_AE = 0.42
    MAX_AE_IMPORTANCE = 1e6 # допустим, за ошибку прогноза больше 0.42 по абсолютному значению доп.штраф 1млн
    excess_level = abs(diff) - MAX_AE
    excess_level[excess_level > 0] = MAX_AE_IMPORTANCE
    
    loss += excess_level

    return np.mean(loss)

def maximum_absolute_error(y_true: Sequence, y_pred: Sequence) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.max(np.abs(y_pred-y_true))
