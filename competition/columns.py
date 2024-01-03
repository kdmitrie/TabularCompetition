from __future__ import annotations

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare
from typing import List, Dict, Tuple

from competition.const import TabularCompetitionDataFrameType, translate


class TabularCompetitionColumn(ABC):
    figsize = (6, 4)

    def __init__(self, df: pd.DataFrame, name: str):
        self.type = None
        self.df = df
        self.name = name
        self.values = self.df[name].value_counts()

    def get_missing(self, df: TabularCompetitionDataFrameType = TabularCompetitionDataFrameType.TRAIN) -> int:
        return sum(
            self.df if df is TabularCompetitionDataFrameType.ALL else self.df[self.df.df == df][self.name].isna())

    @abstractmethod
    def display_distribution(self, dfs: TabularCompetitionDataFrameType | List[
        TabularCompetitionDataFrameType] = TabularCompetitionDataFrameType.TRAIN) -> plt.figure: pass

    @abstractmethod
    def compare_distributions(self, dfs: List[TabularCompetitionDataFrameType]) -> Tuple[str, List]: pass

    @abstractmethod
    def categorical_distributions(self, dfs: List[TabularCompetitionDataFrameType], columns: List[str]) -> Tuple[
        Dict[str, str], Dict[str, List], List]:
        pass

    @abstractmethod
    def numerical_distributions(self, dfs: List[TabularCompetitionDataFrameType],
                                columns: List[str]) -> plt.figure: pass


class TabularCompetitionNumericalColumn(TabularCompetitionColumn):
    """Numerical column class"""
    type = 'numerical'

    def display_distribution(self, dfs: TabularCompetitionDataFrameType | List[
        TabularCompetitionDataFrameType] = TabularCompetitionDataFrameType.TRAIN) -> None:
        pass

    def compare_distributions(self, dfs: List[TabularCompetitionDataFrameType]) -> Tuple[str, List]:
        pass

    def categorical_distributions(self, dfs: List[TabularCompetitionDataFrameType], columns: List[str]) -> Tuple[
        Dict[str, str], Dict[str, List], List]:
        pass

    def numerical_distributions(self, dfs: List[TabularCompetitionDataFrameType], columns: List[str]) -> plt.figure:
        pass


class TabularCompetitionCategoricalColumn(TabularCompetitionColumn):
    """Categorical column class"""
    type = 'categorical'

    def values_in_2nd_not_in_1st(self, df1: TabularCompetitionDataFrameType | List[
        TabularCompetitionDataFrameType] = TabularCompetitionDataFrameType.TRAIN,
                                 df2: TabularCompetitionDataFrameType = TabularCompetitionDataFrameType.SUBMIT) -> set:
        if isinstance(df1, TabularCompetitionDataFrameType):
            df1 = [df1, ]
        vals1 = self.df[self.df.df.isin(list(df1))][self.name].unique()
        vals2 = self.df[self.df.df == df2][self.name].dropna().unique()
        return set(vals2) - set(vals1)

    def display_distribution(self, dfs: TabularCompetitionDataFrameType | List[
        TabularCompetitionDataFrameType] = TabularCompetitionDataFrameType.TRAIN) -> plt.figure:
        if isinstance(dfs, TabularCompetitionDataFrameType):
            dfs = [dfs, ]
        df = self.df[self.df.df.isin(dfs)]
        fig = plt.figure(figsize=self.figsize)
        sns.histplot(data=df, x=self.name, stat='probability', hue=df.df.replace({df: translate(df) for df in dfs}),
                     multiple='dodge', common_norm=False)
        return fig

    def compare_distributions(self, dfs: List[TabularCompetitionDataFrameType]) -> Tuple[str, List]:
        freqs = self.df[[self.name, 'df']].value_counts().unstack()
        stats = []
        for n1, df1 in enumerate(dfs):
            dt1 = freqs[df1]
            for n2 in range(n1 + 1, len(dfs)):
                df2 = dfs[n2]
                dt2 = freqs[df2]
                stats.append((df1, df2, chisquare(dt1, dt2 * sum(dt1) / sum(dt2))))
        return 'scipy.stats.chisquare', stats

    def categorical_distributions(self, dfs: List[TabularCompetitionDataFrameType], columns: List[str]) -> Tuple[
        Dict[str, str], Dict[str, List], List]:
        vals = self.values.keys()
        headings = {df: vals for df in dfs}
        row_headings = {col: self.df[col].unique() for col in columns}

        data = []
        for col, col_values in row_headings.items():
            for col_value in col_values:
                row = []
                for df, my_values in headings.items():
                    for my_value in my_values:
                        row.append(str(len(self.df[(self.df.df == df) & (self.df[self.name] == my_value) & (
                            self.df[col].isna() if pd.isna(col_value) else self.df[col] == col_value)])))

                data.append(row)
        headings = {translate(df): [f'{self.name}={val}' for val in vals] for df, vals in headings.items()}
        return headings, row_headings, data

    def numerical_distributions(self, dfs: List[TabularCompetitionDataFrameType], columns: List[str]) -> plt.figure:
        fig_cols = 3
        fig_rows = int(np.ceil(len(columns) / fig_cols))
        fig = plt.figure(figsize=(self.figsize[0] * fig_cols, self.figsize[1] * fig_rows))
        subfigs = fig.subfigures(nrows=fig_rows, ncols=fig_cols, wspace=0.07).ravel()

        for n, col in enumerate(columns):
            subfigs[n].suptitle(col, fontsize='x-large')
            ax = subfigs[n].subplots(1, len(dfs), sharex=True)
            for m, df in enumerate(dfs):
                ax[m].set_title(translate(df))
                sns.kdeplot(data=self.df[self.df.df == df], x=col, hue=self.name, common_norm=False, ax=ax[m])
        return fig


class TabularCompetitionBinaryColumn(TabularCompetitionCategoricalColumn):
    """Binary column class"""
    type = 'binary'


class TabularCompetitionUnaryColumn(TabularCompetitionCategoricalColumn):
    """Unary column class - this kind of column is not useful at all"""
    type = 'unary'

    def display_distribution(self, dfs: TabularCompetitionDataFrameType | List[
        TabularCompetitionDataFrameType] = TabularCompetitionDataFrameType.TRAIN) -> None:
        print('Not implemented')
