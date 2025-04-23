"""
Модуль со стратегиями (последовательностями действий) для реализации генерации фичей модели
"""


from abc import ABC, abstractmethod
from typing import Optional, Iterable, Sequence, Dict

import pandas as pd

from .engineers import (
    TimeSeriesFeatureEngineer,
    TaxFeatureEngineer,
    ExternalFactorsFeatureEngineer
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
        inflation_df: Optional[pd.DataFrame] = None,
        currency_df: Optional[pd.DataFrame] = None,
        moex_df: Optional[pd.DataFrame] = None,
    ):
        self.target = target
        self.date_col = date_col
        self.time_series_fe = TimeSeriesFeatureEngineer(target=target, date_col=date_col, lags=lags)
        self.tax_fe = TaxFeatureEngineer(target=target, date_col=date_col)

        self.external_fe = ExternalFactorsFeatureEngineer(
            inflation_df=inflation_df,
            currency_df=currency_df,
            moex_df=moex_df,
            date_col=date_col
        )

    def build_features(
        self,
        df: pd.DataFrame,
        taxes: pd.DataFrame,
        holidays: Dict[str, Sequence[pd.Timestamp]],
        **kwargs
    ) -> pd.DataFrame:
        # Временные признаки на основе таргета
        out = self.time_series_fe.build_features(df=df, holidays=holidays)
        out = self.tax_fe.build_features(df=out, taxes=taxes)

        # Признаки внешней среды (инфляция, курс, индекс)
        out = self.external_fe.build_features(out)

        # Проставляем 0 в таргете на праздники и нерабочие дни
        out.loc[(out['is_holiday'] == 1) | (out['is_nowork'] == 1), self.target] = 0
        return out.dropna()
