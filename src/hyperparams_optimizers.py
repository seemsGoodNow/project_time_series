"""
Модуль содержит подходы для подбора гиперпараметров модели
"""


from typing import Optional, Dict, NoReturn, List, Tuple
from abc import ABC, abstractmethod

import pandas as pd
import optuna
from catboost import CatBoostRegressor

from .model_evaluation.metrics import SimpleTargetLoss, MAE, MaxAE

from .model_evaluation.cross_validation import (
    CrossValidationResult,
    perform_cross_val,
)


class BaseHyperparamsOptimizer(ABC):
    
    @abstractmethod
    def get_optimized_params(self):
        pass


class BaselineHyperparamsOptimizer(BaseHyperparamsOptimizer):
    def __init__(
        self,
        splits: List[Tuple[List[int], List[int]]],
        parameter_ranges: Optional[Dict] = None,
        cross_val_score_kws: Optional[Dict] = None,
        optuna_n_trials: int = 20,
        model_class = CatBoostRegressor
    ):
        self.splits = splits
        self.__define_parameter_ranges(parameter_ranges)
        self.__define_cross_val_score_kws(cross_val_score_kws)
        
        self.static_params = {
            key: val for key, val in self.parameter_ranges.items() if not callable(val)
        }
        self.optuna_n_trials = optuna_n_trials
        self.model_class = model_class
    
    def get_optimized_params(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        # Defining objective for optuna search
        def objective(trial: optuna.trial.Trial) -> float:
            params = self._get_evaluated_trial_params(trial) # create a dict from passed callables
            loss = perform_cross_val(
                model_class=self.model_class,
                model_params=params,
                splits=self.splits,
                X=X, y=y,
                **self.cross_val_score_kws
            ).mean_metrics['target_loss']
            return loss
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.optuna_n_trials)
        return study.best_params
    
    def get_fitted_models_and_metrics(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        params: Dict
    ) -> CrossValidationResult:
        cross_val_result: CrossValidationResult = perform_cross_val(
            model_class=self.model_class,
            model_params=params,
            splits=self.splits,
            X=X, y=y,
            **self.cross_val_score_kws
        )
        return cross_val_result
    
    def _get_evaluated_trial_params(self, trial: optuna.trial.Trial):
        return {
            key: val(trial) if callable(val) else val 
            for key, val in self.parameter_ranges.items()
        }
    
    def __define_parameter_ranges(self, parameter_ranges: Optional[Dict] = None) -> NoReturn:
        self.parameter_ranges = {
            'iterations': lambda trial: trial.suggest_int('iterations', 100, 2000),
            'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'depth': lambda trial: trial.suggest_int('depth', 2, 7),
            'l2_leaf_reg': lambda trial: trial.suggest_float('l2_leaf_reg', 0.0001, 10, log=True),
            'random_strength': lambda trial: trial.suggest_float('random_strength', 0.1, 10),
            'bagging_temperature': lambda trial: trial.suggest_float('bagging_temperature', 0, 2),
            'max_ctr_complexity': lambda trial: trial.suggest_int('max_ctr_complexity', 1, 4),
            'early_stopping_rounds': 25,
            'verbose': False
        }
        if parameter_ranges is not None:
            self.parameter_ranges.update(parameter_ranges)
    
    def __define_cross_val_score_kws(self, cross_val_score_kws: Optional[Dict] = None) -> NoReturn:
        self.cross_val_score_kws = {
            'loss': SimpleTargetLoss(),
            'additional_metrics': {'mae': MAE(), 'max_ae': MaxAE()},
        }
        if cross_val_score_kws is not None:
            self.cross_val_score_kws.update(cross_val_score_kws)
