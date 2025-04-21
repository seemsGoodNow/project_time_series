# TODO: Добавить base-model параметр для передачи в selectors
from typing import Dict, Optional, Union
from abc import ABC, abstractmethod
from sklearn.base import clone
import numpy as np

from sklearn.feature_selection import (
    SelectFromModel,  # встроенные
    RFE,              # оберточные
    mutual_info_regression  # фильтрационные
)
from catboost import CatBoostRegressor


class BaseFeatureSelector(ABC):
    @abstractmethod
    def select_features(self) -> list[str]:
        pass


class SelectFromModelEmbeddedFeatureSelector(BaseFeatureSelector):
    """Встроенный метод с поддержкой кастомных моделей"""
    def __init__(
        self, 
        estimator=None, 
        threshold: Union[str, float] = "0.5*median", 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.estimator = estimator if estimator is not None else CatBoostRegressor(verbose=False)
        self.threshold = threshold

    def select_features(self, X, y) -> list[str]:
        selector = SelectFromModel(
            estimator=self.estimator,
            threshold=self.threshold
        )
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()

class WrapperFeatureSelector(BaseFeatureSelector):
    """Оберточный метод (RFE) с кастомными моделями"""
    def __init__(
        self,
        estimator: Optional[object] = None,
        n_features_to_select: Union[int, float, None] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.estimator = estimator or CatBoostRegressor(verbose=False)
        self.n_features_to_select = n_features_to_select

        # Проверка совместимости модели
        if not (hasattr(self.estimator, "coef_") or hasattr(self.estimator, "feature_importances_")):
            raise ValueError("Estimator must implement coef_ or feature_importances_")

    def select_features(self, X, y) -> list[str]:
        # Расчет числа признаков
        if isinstance(self.n_features_to_select, float):
            n_features = int(X.shape[1] * self.n_features_to_select)
        elif self.n_features_to_select is None:
            n_features = X.shape[1] // 2
        else:
            n_features = self.n_features_to_select

        selector = RFE(
            estimator=clone(self.estimator),
            n_features_to_select=n_features,
        )
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()


class FilterFeatureSelector(BaseFeatureSelector):
    """Фильтрационный метод (Mutual Information)"""
    def select_features(self, X, y):
        mi = mutual_info_regression(X, y)
        return X.columns[mi > np.median(mi)]

