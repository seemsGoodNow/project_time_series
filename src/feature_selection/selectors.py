from typing import Dict, Optional, Union
from abc import ABC, abstractmethod
from sklearn.base import clone
import numpy as np

from sklearn.feature_selection import (
    SelectFromModel,
    RFE,
    mutual_info_regression
)
from catboost import CatBoostRegressor


class BaseFeatureSelector(ABC):
    @abstractmethod
    def select_features(self) -> list[str]:
        pass


class SelectFromModelEmbeddedFeatureSelector(BaseFeatureSelector):
    def __init__(self, **kwargs):
        self.kwargs = {
            'threshold': '0.5*median',
            'estimator': CatBoostRegressor(verbose=False),
        }
        if kwargs is not None:
            self.kwargs.update(kwargs)
            
    def select_features(self, X, y):
        selector = SelectFromModel(**self.kwargs)
        selector.fit(X, y)
        return X.columns[selector.get_support()]


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