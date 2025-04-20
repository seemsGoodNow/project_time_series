"""
Модуль, содержащий различные модели, тестируемые для решения задачи о предсказании
"""


from abc import abstractmethod, ABC
from typing import Dict, Tuple, List, NoReturn
import collections

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from .feature_engineering.strategies import BaselineFeatureEngineerStrategy
from .feature_selection.selectors import SelectFromModelEmbeddedFeatureSelector
from .hyperparams_optimizers import BaselineHyperparamsOptimizer
from .model_evaluation.metrics import (
    SimpleTargetLoss,
    MaxAE,
    MAE
)
from .model_evaluation.cross_validation import (
    split_period_for_cross_val
)


class BaseModel(ABC):
    
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def prepare_data(self):
        pass
    
    @abstractmethod
    def forecast(self):
        pass


class BaselineModel(BaseModel):
    def __init__(
        self,
        engineer_strategy_kws: Dict = {},
        feature_selector_kws: Dict = {},
        hyperparams_optimizer_kws: Dict = {},
        cross_val_split_kws: Dict = {}
    ):
        self.model_class = CatBoostRegressor
        self.__define_hyperparams_optimizer_kws(hyperparams_optimizer_kws)
        self.__define_cross_val_split_kws(cross_val_split_kws)
        self.test_size_days = self.cross_val_split_kws['test_size_weeks'] * 7
        self.__define_engineer_strategy_kws(engineer_strategy_kws)
        self.__define_feature_selector_kws(feature_selector_kws)
    
    def fit(self, df: pd.DataFrame, taxes: pd.DataFrame, holidays: pd.DataFrame):
        X, y = self.prepare_data(
            df=df, taxes=taxes, holidays=holidays, inference=False
        )
        self.splits = split_period_for_cross_val(
            min_date=X.index.min(), max_date=X.index.max(), **self.cross_val_split_kws
        )
        self.hyperparams_optimizer_kws['splits'] = self.splits
        # defining cat_features heuristics
        n_unique_column_vals = X.nunique()
        self.cat_features = n_unique_column_vals[n_unique_column_vals <= 3].index
        # defining end of train and range for test
        self.train_end = X.index.max()
        next_day = self.train_end + pd.DateOffset(days=1)
        self.expected_test_range = (next_day, next_day + pd.DateOffset(days=self.test_size_days-1))
        # select features using cross-validation
        self.selected_features = self._select_features(X=X, y=y)
        self.cat_features = [item for item in self.cat_features if item in self.selected_features]
        # Find best hyperparams combination
        self.hyperparams_optimizer_kws['parameter_ranges']['cat_features'] = self.cat_features
        opt = BaselineHyperparamsOptimizer(**self.hyperparams_optimizer_kws)
        self.best_params = opt.get_optimized_params(X[self.selected_features], y)
        # fit models with best params for averaging predictions
        cross_val_result = opt.get_fitted_models_and_metrics(
            X[self.selected_features], y, self.best_params | {'verbose': False}
        )
        self.mean_metrics = cross_val_result.mean_metrics
        self.models = cross_val_result.models
        return self
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        taxes: pd.DataFrame,
        holidays: pd.DataFrame,
        inference=False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        fe = BaselineFeatureEngineerStrategy(**self.engineer_strategy_kws)
        df = fe.build_features(
            df=df,
            taxes=taxes, 
            holidays=holidays
        )
        df = df.set_index(self.engineer_strategy_kws['date_col'])
        X, y = df.drop(columns=[self.engineer_strategy_kws['target']]), df[self.engineer_strategy_kws['target']]
        if inference:
            X = X[self.selected_features]
        return X, y
    
    def forecast(
        self,
        df: pd.DataFrame,
        taxes: pd.DataFrame,
        holidays: pd.DataFrame,
    ) -> pd.Series:
        X, _ = self.prepare_data(
            df=df, taxes=taxes, holidays=holidays, inference=True
        )
        predictions = np.zeros(X.shape[0])
        for model in self.models:
            predictions += model.predict(X)
        mean_prediction = predictions / len(self.models)
        result = pd.Series(mean_prediction)
        result.index = X.index
        return result
        
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        features = []
        min_num_of_iter_where_feature_presented = int(len(self.splits) / 2)
        for split in self.splits:
            train_idx, test_idx = split
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            selector = SelectFromModelEmbeddedFeatureSelector(**self.feature_selector_kws)
            features.extend(selector.select_features(X_train, y_train))
        selected_features = [
            feature
            for feature, freq in
            collections.Counter(features).most_common()
            if freq >= min_num_of_iter_where_feature_presented
        ]
        return selected_features
        
    def __define_feature_selector_kws(self, feature_selector_kws: Dict) -> NoReturn:
        self.feature_selector_kws = {
            'threshold': 'median', 'estimator': self.model_class(verbose=False)
        }
        self.feature_selector_kws.update(feature_selector_kws)
        
    def __define_engineer_strategy_kws(self, engineer_strategy_kws: Dict) -> NoReturn:
        self.engineer_strategy_kws = {
            'date_col': 'date', 'target': 'balance', 'lags': np.arange(self.test_size_days, 31)
        }
        self.engineer_strategy_kws.update(engineer_strategy_kws)
    
    def __define_hyperparams_optimizer_kws(self, hyperparams_optimizer_kws: Dict) -> NoReturn:
        # define default set of kwargs passed to optimizer module
        self.hyperparams_optimizer_kws = {
            'parameter_ranges': {
                'iterations': lambda trial: trial.suggest_int('iterations', 100, 2000),
                'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'depth': lambda trial: trial.suggest_int('depth', 2, 7),
                'l2_leaf_reg': lambda trial: trial.suggest_float('l2_leaf_reg', 0.0001, 10, log=True),
                'random_strength': lambda trial: trial.suggest_float('random_strength', 0.1, 10),
                'bagging_temperature': lambda trial: trial.suggest_float('bagging_temperature', 0, 2),
                'max_ctr_complexity': lambda trial: trial.suggest_int('max_ctr_complexity', 1, 4),
                'early_stopping_rounds': 25,
                'verbose': False
            },
            'cross_val_score_kws': {
                'loss': SimpleTargetLoss(),
                'additional_metrics': {
                    'mae': MAE(),
                    'max_ae': MaxAE(),
                },
            },
            'optuna_n_trials': 20,
            'model_class': self.model_class
        }
        # update kwargs by key if something was passed
        if hyperparams_optimizer_kws:
            for key, val in hyperparams_optimizer_kws.items():
                if key not in ['optuna_n_trials', 'model_class']:
                    self.hyperparams_optimizer_kws[key].update(val)
                else:
                    self.hyperparams_optimizer_kws[key] = val
    
    def __define_cross_val_split_kws(self, cross_val_split_kws: Dict) -> NoReturn:
        self.cross_val_split_kws = {
            'test_size_weeks': 1,
            'train_size_weeks': 12,
            'n_folds': 5,
            'seed': 42
        }
        if cross_val_split_kws is not None:
            self.cross_val_split_kws.update(cross_val_split_kws)
     
