"""
Модуль содержит функции для разбиения временного промежутка на train и test
и проведения кросс-валидации
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple

import pandas as pd
import numpy as np

from .metrics import Metric, SimpleTargetLoss, MaxAE, MAE


@dataclass(frozen=True, kw_only=True)
class CrossValidationResult:
    metrics: Dict[str, List[float]] # key is metric name, value - metric value in fold
    models: List # list of trained models during cross-validation
    mean_metrics: Dict[str, float] # key is metric name, value - mean metric value

    def __repr__(self):
        s = 'CrossValidationResult('
        for key, val in self.__dict__.items():
            s += f'\n   {key} = {val}'
        s += '\n)'
        return s
    
def perform_cross_val(
    model_class,
    model_params: dict,
    splits: list,
    loss: Metric = SimpleTargetLoss(),
    additional_metrics: Dict[str, Metric] = {'max_ae': MaxAE(), 'mae': MAE()},
    X=None,
    y=None,
) -> CrossValidationResult:
    models = []
    if additional_metrics is None:
        additional_metrics = {}
    cv_loss = {
        key: [] for key in additional_metrics
    }
    cv_loss['target_loss'] = []
    for split in splits:
        if X is None or y is None:
            X_train, y_train = split[0]
            X_test, y_test = split[1]
        else:
            train_idx, test_idx = split
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        models.append(model)
        y_pred = model.predict(X_test)
        for key, metric in additional_metrics.items():
            cv_loss[key].append(metric.calc(y_test, y_pred))
        cv_loss['target_loss'].append(loss.calc(y_test, y_pred))
        
    return CrossValidationResult(
        metrics=cv_loss,
        mean_metrics={key: np.mean(vals) for key, vals in cv_loss.items()},
        models=models
    )

def split_period_for_cross_val(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    test_size_weeks: int = 1,
    train_size_weeks: int = 4,
    n_folds: int = 5,
    seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    days = pd.date_range(min_date, max_date, freq='1D', inclusive='both')
    random.seed(seed)
    # Defining borders for weeks 
    # (if border is near to min/max -> ensures that week is full)
    min_date_week_start = (
        min_date.to_period('W').start_time 
        + pd.DateOffset(days=1) # to Mondays
    )
    if (min_date_week_start - min_date).days < 0:
        min_date_week_start += pd.DateOffset(weeks=1)
    max_date_week_start = (
        max_date.to_period('W').start_time 
        + pd.DateOffset(days=1) # to Mondays
    )
    if (max_date - max_date_week_start).days < 0:
        max_date_week_start -= pd.DateOffset(weeks=1)

    # Defining borders for week starts which can be chosen as separator of train and test
    # (for min date must be enough train data and for max date must be enough test data)
    min_possible_date = min_date_week_start + pd.DateOffset(weeks=train_size_weeks)
    max_possible_date = max_date_week_start - pd.DateOffset(weeks=test_size_weeks)
    # all possible week starts
    week_starts = pd.date_range(
        min_possible_date, max_possible_date, freq='1W', inclusive='both'
    ) + pd.DateOffset(days=1) # to Mondays
    # randomly choose n_folds from all week_starts
    if n_folds > len(week_starts):
        raise ValueError('n_folds must be less or equal than number of suitable weeks presented between min_date and max_date')
    chosen_weeks = random.sample(sorted(week_starts), n_folds)
    idx_splits = []
    for week_start in chosen_weeks:
        week_start_idx = np.where(days >= week_start)[0][0]
        train_idx = np.arange(week_start_idx - train_size_weeks*7, week_start_idx)
        test_idx = np.arange(week_start_idx, week_start_idx + test_size_weeks*7)
        idx_splits.append((train_idx, test_idx))
    return idx_splits
