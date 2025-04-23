"""
Модуль с классами для генерации фичей для датасета
"""


from typing import Iterable, Optional, NoReturn, Dict, Sequence
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseFeatureEngineer(ABC):
    """Интерфейс для классов-инженеров фичей
    """
    @abstractmethod
    def build_features(self):
        pass


class TimeSeriesFeatureEngineer(BaseFeatureEngineer):
    """Инженер фичей временного ряда для таргета:
        1. Лаги (с 1 по 30)
        2. День недели, месяца, месяц
        3. Флаги того, что день недели между 15 и 20, между 20 и 25, между 25 и 30.
    """
    def __init__(
        self, 
        target: str = 'balance',
        date_col: str = 'date',
        lags: Optional[Iterable[int]] = None,
    ):
        self.target = target
        self.date_col = date_col
        if lags is None:
            self.lags = np.arange(1, 30)
        else:
            self.lags = lags

    def build_features(
        self, 
        df: pd.DataFrame, 
        holidays: Dict[str, Sequence[pd.Timestamp]]
    ) -> pd.DataFrame:
        df = df.sort_values(self.date_col).copy()
        self._add_holidays_features(df=df, holidays=holidays)
        # Зануляем таргет в праздники и выходные
        df.loc[(df['is_holiday'] == 1) | (df['is_nowork'] == 1), self.target] = 0
        self._add_lag_features(df)
        self._add_date_features(df)
        df = df.dropna()
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> NoReturn:
        for lag in self.lags:
            df[self.target + f'__lag_{lag}'] = df[self.target].shift(lag)

    def _add_date_features(self, df: pd.DataFrame) -> NoReturn:
        df['day_of_week'] = df[self.date_col].dt.dayofweek
        df['day_of_month'] = df[self.date_col].dt.day
        df['month'] = df[self.date_col].dt.month
        df['is_after_25_day'] = (df['day_of_month'] >= 25).astype(int)
        df['is_after_20_day'] = (
            (df['day_of_month'] >= 20)
            & (df['day_of_month'] < 25)
        ).astype(int)
        df['is_after_15_day'] = (
            (df['day_of_month'] >= 15)
            & (df['day_of_month'] < 20)
        ).astype(int)
    
    def _add_holidays_features(
        self, 
        df: pd.DataFrame, 
        holidays: Dict[str, Sequence[pd.Timestamp]]
    ) -> NoReturn:
        df['is_holiday'] = df[self.date_col].isin(holidays['holidays']).astype(int)
        df['is_preholidays'] = df[self.date_col].isin(holidays['preholidays']).astype(int)
        df['is_nowork'] = df[self.date_col].isin(holidays['nowork']).astype(int)


class TaxFeatureEngineer(BaseFeatureEngineer):
    """Инженер налоговых фичей, которые удалось спарсить из К+"""
    
    def __init__(
        self,
        target: str = 'balance',
        date_col: str = 'date',
    ):
        self.date_col = date_col
    
    def build_features(self, df: pd.DataFrame, taxes: pd.DataFrame) -> pd.DataFrame:
        # Датафрейм, где в столбцах тип налога, а по индексу дата
        encoded_taxes = (
            pd.get_dummies(taxes, columns=['tax_type'])
            .drop(columns=['tax_subtype'])
            .groupby([self.date_col])
            .sum()
            .astype(int)
            .reset_index()
        )
        assert encoded_taxes.shape[0] == np.unique(encoded_taxes[self.date_col]).shape[0]
        train_data = pd.merge(
            left=df,
            right=encoded_taxes,
            on=self.date_col,
            how='left'
        ).sort_values(self.date_col)
        # Фильтр налогов за пределами нужного промежутка
        train_data = train_data[
            (train_data[self.date_col] >= df[self.date_col].min())
            & (train_data[self.date_col] <= df[self.date_col].max())
        ]
        for col in encoded_taxes.columns:
            if col != self.date_col:
                train_data[col] = train_data[col].fillna(0).astype(int)
        return train_data


class MOEXFeatureEngineer(BaseFeatureEngineer):
    """Инженер фичей, связанных с индексом Мосбиржи (лаги изменения в прошлый день)
    """
    def __init__(
        self,
        target: str = 'MOEX',
        date_col: str = 'date',
        lags: Optional[Iterable[int]] = None,
        windows: Optional[Iterable[int]] = None
    ):
        self.target = target
        self.date_col = date_col
        self.lags = lags if lags is not None else np.arange(1, 30)
        self.windows = windows if windows is not None else [3, 5, 10, 15, 30]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(self.date_col).copy()
        self._add_lag_features(df)
        df = df.dropna()
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> NoReturn:
        for lag in self.lags:
            df[f'{self.target}__lag_{lag}'] = df[self.target].shift(lag)

    def _add_rolling_stats(self, df: pd.DataFrame) -> NoReturn:
        for window in self.windows:
            df[f'{self.target}__roll_mean_{window}'] = df[self.target].rolling(window).mean()
            df[f'{self.target}__roll_std_{window}'] = df[self.target].rolling(window).std()

    def _add_binary_flags(self, df: pd.DataFrame) -> NoReturn:
        df[f'{self.target}__is_growth'] = (df[self.target] > 0).astype(int)
        df[f'{self.target}__is_drop'] = (df[self.target] < 0).astype(int)

    def _add_trend_features(self, df: pd.DataFrame) -> NoReturn:
        for window in self.windows:
            roll_mean = df[self.target].rolling(window).mean()
            df[f'{self.target}__trend_{window}'] = df[self.target] - roll_mean


class UsdRubFeatureEngineer(BaseFeatureEngineer):
    """Инженер фичей, связанных с курсом доллара (лаги изменения в прошлый день)
    """
    
    def __init__(
        self,
        target: str = 'usd/rub',
        date_col: str = 'date',
        lags: Optional[Iterable[int]] = None,
        windows: Optional[Iterable[int]] = None
    ):
        self.target = target
        self.date_col = date_col
        self.lags = lags if lags is not None else np.arange(1, 30)
        self.windows = windows if windows is not None else [3, 5, 10, 15, 30]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(self.date_col).copy()
        self._add_lag_features(df)
        df = df.dropna()
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> NoReturn:
        for lag in self.lags:
            df[f'{self.target}__lag_{lag}'] = df[self.target].shift(lag)

    def _add_rolling_stats(self, df: pd.DataFrame) -> NoReturn:
        for window in self.windows:
            df[f'{self.target}__roll_mean_{window}'] = df[self.target].rolling(window).mean()
            df[f'{self.target}__roll_std_{window}'] = df[self.target].rolling(window).std()

    def _add_binary_flags(self, df: pd.DataFrame) -> NoReturn:
        df[f'{self.target}__is_growth'] = (df[self.target] > 0).astype(int)
        df[f'{self.target}__is_drop'] = (df[self.target] < 0).astype(int)

    def _add_trend_features(self, df: pd.DataFrame) -> NoReturn:
        for window in self.windows:
            roll_mean = df[self.target].rolling(window).mean()
            df[f'{self.target}__trend_{window}'] = df[self.target] - roll_mean


class InflationFeatureEngineer(BaseFeatureEngineer):
    """Инженер фичей, связанных с инфляцией (лаги изменения в прошлый день)
    """
    
    def __init__(
        self,
        target: str = 'inflation',
        date_col: str = 'date',
        lags: Optional[Iterable[int]] = None,
        windows: Optional[Iterable[int]] = None
    ):
        self.target = target
        self.date_col = date_col
        self.lags = lags if lags is not None else [1, 2, 3, 6]
        self.windows = windows if windows is not None else [3, 6]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(self.date_col).copy()
        self._add_lag_features(df)
        df = df.dropna()
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> NoReturn:
        for lag in self.lags:
            df[f'{self.target}__lag_{lag}'] = df[self.target].shift(lag)

    def _add_rolling_stats(self, df: pd.DataFrame) -> NoReturn:
        for window in self.windows:
            df[f'{self.target}__roll_mean_{window}'] = df[self.target].rolling(window).mean()
            df[f'{self.target}__roll_std_{window}'] = df[self.target].rolling(window).std()

    def _add_binary_flags(self, df: pd.DataFrame) -> NoReturn:
        df[f'{self.target}__is_growth'] = (df[self.target].diff() > 0).astype(int)
        df[f'{self.target}__is_drop'] = (df[self.target].diff() < 0).astype(int)

    def _add_trend_features(self, df: pd.DataFrame) -> NoReturn:
        for window in self.windows:
            roll_mean = df[self.target].rolling(window).mean()
            df[f'{self.target}__trend_{window}'] = df[self.target] - roll_mean