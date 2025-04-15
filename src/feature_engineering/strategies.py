"""
Модуль со стратегиями (последовательностями действий) для реализации генерации фичей модели
"""


from abc import ABC, abstractmethod
from typing import Optional, Iterable, Sequence, Dict

import pandas as pd

from .engineers import (
    TimeSeriesFeatureEngineer,
    TaxFeatureEngineer
)


class BaseFeatureEngineerStrategy(ABC):
    @abstractmethod
    def build_features(self):
        pass
    

class BaselineFeatureEngineerStrategy(BaseFeatureEngineerStrategy):
    def __init__(
        self,
        target: str = 'balance',
        date_col: str = 'date',
        lags: Optional[Iterable[int]] = None,
    ):
        self.time_series_fe = TimeSeriesFeatureEngineer(target=target, date_col=date_col, lags=lags)
        self.tax_fe = TaxFeatureEngineer(target=target, date_col=date_col)
    
    def build_features(
        self,
        df: pd.DataFrame, 
        taxes: pd.DataFrame, 
        holidays: Dict[str, Sequence[pd.Timestamp]]
    ) -> pd.DataFrame:
        out = self.time_series_fe.build_features(df=df, holidays=holidays)
        out = self.tax_fe.build_features(df=out, taxes=taxes)
        return out
