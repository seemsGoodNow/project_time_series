from typing import Dict, Optional, Union
from abc import ABC, abstractmethod
from sklearn.base import clone
import numpy as np

from sklearn.feature_selection import (
    SelectFromModel,  # встроенные
    RFE,              # оберточные
    mutual_info_regression  # фильтрационные (нелинейный)
)
from catboost import CatBoostRegressor


class BaseFeatureSelector(ABC):
    """Абстрактный базовый класс для всех селекторов признаков.
    
    Наследники должны реализовать метод select_features().
    """
    @abstractmethod
    def select_features(self) -> list[str]:
        """Выбирает важные признаки из набора данных.
        
        Returns:
            list[str]: Список названий отобранных признаков
        """
        pass


class SelectFromModelEmbeddedFeatureSelector(BaseFeatureSelector):
    """Встроенный метод отбора признаков на основе важности признаков модели.

    Использует sklearn.feature_selection.SelectFromModel под капотом.

    Args:
        estimator (object, optional): Модель с атрибутами feature_importances_ или coef_. 
            По умолчанию CatBoostRegressor(verbose=False).
        threshold (Union[str, float]): Порог для отбора признаков. Может быть:
            - числовое значение
            - строка вида "0.5*median" (умножение медианного значения)
        **kwargs: Дополнительные аргументы для базового класса
    """
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
        """Выполняет отбор признаков на основе важности признаков модели.
        
        Args:
            X (pd.DataFrame): Матрица признаков
            y (pd.Series): Целевая переменная
            
        Returns:
            list[str]: Список названий отобранных признаков
            
        Raises:
            ValueError: Если модель не имеет атрибутов feature_importances_ или coef_
        """
        selector = SelectFromModel(
            estimator=self.estimator,
            threshold=self.threshold
        )
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()

class WrapperFeatureSelector(BaseFeatureSelector):
    """Оберточный метод (Recursive Feature Elimination) для отбора признаков.

    Использует sklearn.feature_selection.RFE под капотом.

    Args:
        estimator (object, optional): Модель с атрибутами feature_importances_ или coef_.
            По умолчанию CatBoostRegressor(verbose=False).
        n_features_to_select (Union[int, float, None]): Количество признаков для отбора:
            - int: точное количество
            - float: доля от общего числа признаков
            - None: половина признаков (по умолчанию)
        **kwargs: Дополнительные аргументы для базового класса
            
    Raises:
        ValueError: Если модель не имеет атрибутов feature_importances_ или coef_
    """
    def __init__(
        self,
        estimator: Optional[object] = None,
        n_features_to_select: Union[int, float, None] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.estimator = estimator or CatBoostRegressor(verbose=False)
        self.n_features_to_select = n_features_to_select

        if not (hasattr(self.estimator, "coef_") or hasattr(self.estimator, "feature_importances_")):
            raise ValueError("Estimator must implement coef_ or feature_importances_")

    def select_features(self, X, y) -> list[str]:
        """Выполняет рекурсивный отбор признаков (RFE).
        
        Args:
            X (pd.DataFrame): Матрица признаков
            y (pd.Series): Целевая переменная
            
        Returns:
            list[str]: Список названий отобранных признаков
        """
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
    """Фильтрационный метод отбора признаков на основе Mutual Information.
    
    Отбирает признаки с mutual information выше медианного значения.
    """
    def select_features(self, X, y):
        """Выполняет отбор признаков с помощью взаимной информации.
        
        Args:
            X (pd.DataFrame): Матрица признаков
            y (pd.Series): Целевая переменная
            
        Returns:
            list[str]: Список названий отобранных признаков
        """
        mi = mutual_info_regression(X, y)
        return X.columns[mi > np.median(mi)]
