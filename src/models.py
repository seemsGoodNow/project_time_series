"""
Модуль, содержащий различные модели, тестируемые для решения задачи о предсказании
"""

from abc import abstractmethod, ABC
from typing import Dict, Tuple, List, NoReturn
import collections
import warnings

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

from .feature_engineering.strategies import BaselineFeatureEngineerStrategy, ExternalFactorsFeatureEngineerStrategy
from .feature_selection.selectors import SelectFromModelEmbeddedFeatureSelector
from .hyperparams_optimizers import BaselineHyperparamsOptimizer
from .model_evaluation.metrics import (
    SimpleTargetLoss,
    MaxAE,
    MAE,
    LossForCatBoost,
    MetricForCatBoost
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


class CatBoostRegressorWrapper:
    def __init__(self, **kwargs):
        """
        Класс-обертка для CatBoost c кастомным predict

        :param holiday_col: название колонки-флага праздника
        :param nowork_col: название колонки-флага нерабочего дня
        :param kwargs: параметры для CatBoostRegressor
        """
        self.model = CatBoostRegressor(**kwargs)
        self.holiday_col = 'is_holiday'
        self.nowork_col = 'is_nowork'

    def fit(self, X, y=None, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        preds = self.model.predict(X, **kwargs)

        if isinstance(X, pd.DataFrame):
            if self.holiday_col in X.columns and self.nowork_col in X.columns:
                mask = (X[self.holiday_col] == 1) | (X[self.nowork_col] == 1)
                preds[mask.values] = 0.0
            else:
                warnings.warn(
                    f"Колонка {self.holiday_col} или {self.nowork_col} не найдена — предсказания не обнуляются.")
        else:
            raise ValueError("X должен быть pandas.DataFrame, чтобы использовать флаги")

        return preds

    def __getattr__(self, name):
        return getattr(self.model, name)


class BaselineModel(BaseModel):
    """Модель с catboost под капотом, использующая налоги и признаки ряда"""
    def __init__(
        self,
        engineer_strategy_kws: Dict = {},
        feature_selector_kws: Dict = {},
        hyperparams_optimizer_kws: Dict = {},
        cross_val_split_kws: Dict = {}
    ):
        self.model_class = CatBoostRegressorWrapper
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
        self.opt = BaselineHyperparamsOptimizer(**self.hyperparams_optimizer_kws)
        self.best_params = self.opt.get_optimized_params(X[self.selected_features], y)
        # fit models with best params for averaging predictions
        cross_val_result = self.opt.get_fitted_models_and_metrics(
            X[self.selected_features], y, self.best_params | {'verbose': False}
        )
        self.mean_metrics = cross_val_result.mean_metrics
        self.models = cross_val_result.models
        # Learn last available configuration on last data
        self.model_params = self.hyperparams_optimizer_kws['parameter_ranges']
        self.model_params.update(self.best_params)
        max_week_start = X.index.to_period('W').start_time.max() + pd.DateOffset(days=1)
        if max_week_start > X.index.max():
            max_week_start -= pd.DateOffset(weeks=1)
        self.last_train_period = (
            max_week_start - pd.DateOffset(weeks=self.cross_val_split_kws['train_size_weeks']),
            max_week_start
        )
        self.last_X_train = X[
            (X.index >= self.last_train_period[0])
            & (X.index < self.last_train_period[1])
        ][self.selected_features]
        self.last_y_train = y[
            (y.index >= self.last_train_period[0])
            & (y.index < self.last_train_period[1])
        ]
        self.model = self.model_class(**self.model_params)
        self.model.fit(self.last_X_train, self.last_y_train)
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
        mean_prediction: bool = True,
    ) -> pd.Series:
        X, _ = self.prepare_data(
            df=df, taxes=taxes, holidays=holidays, inference=True
        )
        if mean_prediction:
            predictions = np.zeros(X.shape[0])
            for model in self.models:
                predictions += model.predict(X)
            mean_prediction = predictions / len(self.models)
            result = pd.Series(mean_prediction)
            result.index = X.index
        else:
            result = pd.Series(self.model.predict(X))
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
            or feature in ['is_holiday', 'is_nowork']
        ]
        return selected_features
        
    def __define_feature_selector_kws(self, feature_selector_kws: Dict) -> NoReturn:
        self.feature_selector_kws = {
            'threshold': 'median', 'estimator': self.model_class(verbose=False)
        }
        self.feature_selector_kws.update(feature_selector_kws)
        
    def __define_engineer_strategy_kws(self, engineer_strategy_kws: Dict) -> NoReturn:
        self.engineer_strategy_kws = {
            'date_col': 'date', 'target': 'balance', 'lags': np.arange(1, 31)
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
                'verbose': False,
                # "loss_function": LossForCatBoost(),
                # "eval_metric": MetricForCatBoost()
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


class ExternalFactorsModel(BaseModel):
    """Модель с catboost под капотом, использующая налоги, признаки ряда и внешние факторы"""
    def __init__(
        self,
        engineer_strategy_kws: Dict = {},
        feature_selector_kws: Dict = {},
        hyperparams_optimizer_kws: Dict = {},
        cross_val_split_kws: Dict = {}
    ):
        self.model_class = CatBoostRegressorWrapper
        self.__define_hyperparams_optimizer_kws(hyperparams_optimizer_kws)
        self.__define_cross_val_split_kws(cross_val_split_kws)
        self.test_size_days = self.cross_val_split_kws['test_size_weeks'] * 7
        self.__define_engineer_strategy_kws(engineer_strategy_kws)
        self.__define_feature_selector_kws(feature_selector_kws)

    def fit(self, df: pd.DataFrame, taxes: pd.DataFrame, holidays: pd.DataFrame,
            moex: pd.DataFrame, usd: pd.DataFrame, inflation: pd.DataFrame):
        X, y = self.prepare_data(
            df=df, taxes=taxes, holidays=holidays,
            moex=moex, usd=usd, inflation=inflation, inference=False
        )
        self.splits = split_period_for_cross_val(
            min_date=X.index.min(), max_date=X.index.max(), **self.cross_val_split_kws
        )
        self.hyperparams_optimizer_kws['splits'] = self.splits
        n_unique_column_vals = X.nunique()
        self.cat_features = n_unique_column_vals[n_unique_column_vals <= 3].index
        self.train_end = X.index.max()
        next_day = self.train_end + pd.DateOffset(days=1)
        self.expected_test_range = (next_day, next_day + pd.DateOffset(days=self.test_size_days - 1))
        self.selected_features = list(set(self._select_features(X=X, y=y) + ['is_holiday', 'is_nowork']))
        self.cat_features = [item for item in self.cat_features if item in self.selected_features]
        self.hyperparams_optimizer_kws['parameter_ranges']['cat_features'] = self.cat_features
        opt = BaselineHyperparamsOptimizer(**self.hyperparams_optimizer_kws)
        self.best_params = opt.get_optimized_params(X[self.selected_features], y)
        cross_val_result = opt.get_fitted_models_and_metrics(
            X[self.selected_features], y, self.best_params | {'verbose': False}
        )
        self.mean_metrics = cross_val_result.mean_metrics
        self.models = cross_val_result.models
        # Learn last available configuration on last data
        self.model_params = self.hyperparams_optimizer_kws['parameter_ranges']
        self.model_params.update(self.best_params)
        max_week_start = X.index.to_period('W').start_time.max() + pd.DateOffset(days=1)
        if max_week_start > X.index.max():
            max_week_start -= pd.DateOffset(weeks=1)
        self.last_train_period = (
            max_week_start - pd.DateOffset(weeks=self.cross_val_split_kws['train_size_weeks']),
            max_week_start
        )
        self.last_X_train = X[
            (X.index >= self.last_train_period[0])
            & (X.index < self.last_train_period[1])
        ][self.selected_features]
        self.last_y_train = y[
            (y.index >= self.last_train_period[0])
            & (y.index < self.last_train_period[1])
        ]
        self.model = self.model_class(**self.model_params)
        self.model.fit(self.last_X_train, self.last_y_train)
        return self

    def prepare_data(
        self,
        df: pd.DataFrame,
        taxes: pd.DataFrame,
        holidays: pd.DataFrame,
        moex: pd.DataFrame,
        usd: pd.DataFrame,
        inflation: pd.DataFrame,
        inference=False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        fe = ExternalFactorsFeatureEngineerStrategy(**self.engineer_strategy_kws)
        df = fe.build_features(
            df=df,
            taxes=taxes,
            holidays=holidays,
            moex=moex,
            usd=usd,
            inflation=inflation
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
        moex: pd.DataFrame,
        usd: pd.DataFrame,
        inflation: pd.DataFrame,
        mean_prediction: bool = True,
    ) -> pd.Series:
        X, _ = self.prepare_data(
            df=df, taxes=taxes, holidays=holidays,
            moex=moex, usd=usd, inflation=inflation, inference=True
        )
        if mean_prediction:
            predictions = np.zeros(X.shape[0])
            for model in self.models:
                predictions += model.predict(X)
            mean_prediction = predictions / len(self.models)
            result = pd.Series(mean_prediction)
            result.index = X.index
        else:
            result = pd.Series(self.model.predict(X))
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
            for feature, freq in collections.Counter(features).most_common()
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
            'date_col': 'date', 'target': 'balance', 'lags': np.arange(1, 31),
            'moex_lags': np.arange(1, 30),
            'usd_lags': np.arange(1, 30),
            'inflation_lags': np.arange(1, 6)
        }
        self.engineer_strategy_kws.update(engineer_strategy_kws)

    def __define_hyperparams_optimizer_kws(self, hyperparams_optimizer_kws: Dict) -> NoReturn:
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