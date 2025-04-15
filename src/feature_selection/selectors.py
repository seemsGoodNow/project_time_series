# TODO: Добавить base-model параметр для передачи в selectors
from typing import Dict
from abc import ABC, abstractmethod

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


# class WrapperFeatureSelector(BaseFeatureSelector):
#     """Оберточный метод (Recursive Feature Elimination)"""
#     def select_features(self, X, y):
#         selector = RFE(
#             estimator=RandomForestRegressor(),
#             n_features_to_select=X.shape[1]//2
#         )
#         selector.fit(X, y)
#         return X.columns[selector.get_support()]


# class FilterFeatureSelector(BaseFeatureSelector):
#     """Фильтрационный метод (Mutual Information)"""
#     def select_features(self, X, y):
#         mi = mutual_info_regression(X, y)
#         return X.columns[mi > np.median(mi)]

