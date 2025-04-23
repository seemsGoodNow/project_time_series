"""
Модуль с классами для генерации фичей для датасета
"""


from typing import Iterable, Optional, NoReturn, Dict, Sequence, List
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseFeatureEngineer(ABC):
    @abstractmethod
    def build_features(self):
        pass


class TimeSeriesFeatureEngineer(BaseFeatureEngineer):
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
        # self._add_rolling_features(df)
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
    def __init__(
        self, 
        target: str = 'balance',
        date_col: str = 'date',
    ):
        self.target = target
        self.date_col = date_col
    
    def build_features(self, df: pd.DataFrame, taxes: pd.DataFrame) -> pd.DataFrame:
        encoded_taxes = (
            pd.get_dummies(taxes, columns=['tax_type'])
            .drop(columns=['tax_subtype'])
            .groupby(['date'])
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
        train_data = train_data[
            (train_data[self.date_col] >= df[self.date_col].min())
            & (train_data[self.date_col] <= df[self.date_col].max())
        ]
        for col in encoded_taxes.columns:
            if col != self.date_col:
                train_data[col] = train_data[col].fillna(0).astype(int)
        return train_data


class ExternalFactorsFeatureEngineer(BaseFeatureEngineer):
    def __init__(
        self,
        inflation_df: pd.DataFrame,
        currency_df: pd.DataFrame,
        moex_df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        date_col: str = 'date',
    ):
        self.inflation_df = inflation_df.copy()
        self.currency_df = currency_df.copy()
        self.moex_df = moex_df.copy()
        self.date_col = date_col
        self.target_cols = target_cols if target_cols is not None else ['inflation', 'usd/rub', 'MOEX']
        self.inflation_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.inflation_windows = [3, 6]
        self.other_lags = np.arange(1, 30)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(self.date_col).copy()
        df = df.set_index(self.date_col)

        # --- Merge инфляции ---
        self.inflation_df['year'] = self.inflation_df.index.year
        self.inflation_df['month'] = self.inflation_df.index.month
        df['year'] = df.index.year
        df['month'] = df.index.month
        df = df.merge(self.inflation_df, on=['year', 'month'], how='left')
        df.index = df.index  # сохраняем индекс после мержа
        df = df.drop(columns=['year', 'month'])

        # --- Merge курсов и индекса ---
        df = df \
            .merge(self.currency_df, left_index=True, right_index=True, how='left') \
            .merge(self.moex_df, left_index=True, right_index=True, how='left')

        # --- ffill и bfill ---
        df[['usd/rub', 'MOEX']] = df[['usd/rub', 'MOEX']].ffill().bfill()

        # --- Пересчет в процентное изменение к предыдущему дню ---
        df['usd/rub'] = df['usd/rub'].pct_change() * 100
        df['MOEX'] = df['MOEX'].pct_change() * 100

        # Убираем первый день, где были NaN после pct_change
        df = df.dropna(subset=['usd/rub', 'MOEX'])

        # Добавление признаков
        for target in self.target_cols:
            if target == 'inflation':
                self._add_lag_features(df, target, self.inflation_lags)
                self._add_rolling_stats(df, target, self.inflation_windows)
                self._add_binary_flags(df, target, is_diff=True)
            else:
                self._add_lag_features(df, target, self.other_lags)
                self._add_binary_flags(df, target, is_diff=False)

        df = df.dropna()
        return df

    def _add_lag_features(self, df: pd.DataFrame, target: str, lags: Iterable[int]) -> NoReturn:
        for lag in lags:
            df[f'{target}__lag_{lag}'] = df[target].shift(lag)

    def _add_rolling_stats(self, df: pd.DataFrame, target: str, windows: Iterable[int]) -> NoReturn:
        for window in windows:
            df[f'{target}__roll_mean_{window}'] = df[target].rolling(window).mean()
            df[f'{target}__roll_std_{window}'] = df[target].rolling(window).std()

    def _add_binary_flags(self, df: pd.DataFrame, target: str, is_diff: bool) -> NoReturn:
        if is_diff:
            df[f'{target}__is_growth'] = (df[target].diff() > 0).astype(int)
            df[f'{target}__is_drop'] = (df[target].diff() < 0).astype(int)
        else:
            df[f'{target}__is_growth'] = (df[target] > 0).astype(int)
            df[f'{target}__is_drop'] = (df[target] < 0).astype(int)


