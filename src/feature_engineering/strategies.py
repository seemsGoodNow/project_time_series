"""
Модуль со стратегиями (последовательностями действий) для реализации генерации фичей модели
"""


from abc import ABC, abstractmethod
from typing import Optional, Iterable, Sequence, Dict

import pandas as pd

from .engineers import (
    TimeSeriesFeatureEngineer,
    TaxFeatureEngineer,
    MOEXFeatureEngineer,
    UsdRubFeatureEngineer,
    InflationFeatureEngineer
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
        self.target = target
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
        # Зануляем таргет в праздники и выходные
        out.loc[(out['is_holiday'] == 1) | (out['is_nowork'] == 1), self.target] = 0
        return out


class ExternalFactorsFeatureEngineerStrategy(BaseFeatureEngineerStrategy):
    def __init__(
        self,
        target: str = 'balance',
        date_col: str = 'date',
        lags: Optional[Iterable[int]] = None,
        moex_lags: Optional[Iterable[int]] = None,
        usd_lags: Optional[Iterable[int]] = None,
        inflation_lags: Optional[Iterable[int]] = None,
    ):
        self.target = target
        # Признаки из временного ряда таргета
        self.time_series_fe = TimeSeriesFeatureEngineer(target=target, date_col=date_col, lags=lags)
        self.tax_fe = TaxFeatureEngineer(target=target, date_col=date_col)

        # Внешние признаки
        self.moex_fe = MOEXFeatureEngineer(date_col=date_col, lags=moex_lags)
        self.usd_fe = UsdRubFeatureEngineer(date_col=date_col, lags=usd_lags)
        self.inflation_fe = InflationFeatureEngineer(date_col=date_col, lags=inflation_lags)

    def build_features(
        self,
        df: pd.DataFrame,
        taxes: pd.DataFrame,
        holidays: Dict[str, Sequence[pd.Timestamp]],
        moex: pd.DataFrame,
        usd: pd.DataFrame,
        inflation: pd.DataFrame,
    ) -> pd.DataFrame:
        # Временные признаки на основе таргета
        out = self.time_series_fe.build_features(df=df, holidays=holidays)
        out = self.tax_fe.build_features(df=out, taxes=taxes)

        # Внешние признаки
        moex_feats = self.moex_fe.build_features(moex)
        usd_feats = self.usd_fe.build_features(usd)
        inflation_feats = self.inflation_fe.build_features(inflation)

        # Объединение всех фичей
        out = out.merge(moex_feats, on='date', how='left')
        out = out.merge(usd_feats, on='date', how='left')
        out = out.merge(inflation_feats, on='date', how='left')

        out.loc[(out['is_holiday'] == 1) | (out['is_nowork'] == 1), self.target] = 0
        return out.dropna()