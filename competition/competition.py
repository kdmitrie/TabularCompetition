from __future__ import annotations

import pandas as pd
from .display import DisplayCompetition
from .const import TabularCompetitionDataFrameType
from .columns import TabularCompetitionBinaryColumn, TabularCompetitionCategoricalColumn, TabularCompetitionNumericalColumn
from typing import Callable, List, Optional
from sklearn.decomposition import PCA


class TabularCompetition:
    def __init__(self, train_csv: str, submit_csv: str, original_csv: str = None,
                 original_preprocessing: Optional[Callable] = None, df_preprocessing: Optional[Callable] = None,
                 display: Optional[DisplayCompetition] = None):
        self.metric = None
        self.target = None
        self.display = display

        # We load all the data and mark it
        self.train_df = pd.read_csv(train_csv)
        self.train_df['df'] = TabularCompetitionDataFrameType.TRAIN

        self.submit_df = pd.read_csv(submit_csv)
        self.submit_df['df'] = TabularCompetitionDataFrameType.SUBMIT

        if df_preprocessing is not None:
            self.train_df = df_preprocessing(self.train_df)
            self.submit_df = df_preprocessing(self.submit_df)

        # If external dataframe is specified, we load and preprocess it
        if original_csv is not None:
            self.original_df = pd.read_csv(original_csv)
            self.original_df['df'] = TabularCompetitionDataFrameType.ORIGINAL
            if original_preprocessing is not None:
                self.original_df = original_preprocessing(self.original_df)
            self.df = pd.concat([self.train_df, self.submit_df, self.original_df[self.train_df.columns]])
            self.dfs = [TabularCompetitionDataFrameType.TRAIN, TabularCompetitionDataFrameType.SUBMIT,
                        TabularCompetitionDataFrameType.ORIGINAL]
        else:
            self.drop_original()

        self.df.reset_index(inplace=True)
        self.df.drop('index', axis=1, inplace=True)
        self.__build_columns()

    def __build_columns(self) -> None:
        self.columns = {}
        self.numerical_columns = []
        self.categorical_columns = []
        for column_name in set(self.df.columns) - {'id', 'df'}:
            column_values = len(self.df[column_name].value_counts())

            if column_values < 2:
                column = TabularCompetitionBinaryColumn(self.df, column_name)
            elif column_values == 2:
                column = TabularCompetitionBinaryColumn(self.df, column_name)
            elif column_values <= 50:
                column = TabularCompetitionCategoricalColumn(self.df, column_name)
            else:
                column = TabularCompetitionNumericalColumn(self.df, column_name)

            self.columns[column_name] = column
            if isinstance(column, TabularCompetitionNumericalColumn):
                self.numerical_columns.append(column_name)
            if isinstance(column, TabularCompetitionCategoricalColumn):
                self.categorical_columns.append(column_name)

        self.columns = dict(sorted(self.columns.items()))
        self.numerical_columns = sorted(self.numerical_columns)
        self.categorical_columns = sorted(self.categorical_columns)

    def set_target(self, target: str | List[str]) -> None:
        """Set the target(s) columns for the competition"""
        if isinstance(target, str):
            self.target = [target, ]
        elif isinstance(target, list):
            self.target = target
        else:
            raise TypeError('Target must be a string or a list of strings')

        if not set(self.target).issubset(self.df.columns):
            raise ValueError(f'Target {target} not in the dataframe columns')

    def set_metric(self, metric: Callable) -> None:
        """Sets the metric used in the competition"""
        self.metric = metric

    def apply_pca_numerical(self):
        """Applies PCA transform on all numerical columns except target"""
        numerical_columns = [col for col in self.numerical_columns if col not in self.target]
        numerical_columns_pca = [f'PC{n + 1}' for n in range(len(numerical_columns))]
        data = PCA().fit_transform(self.df[numerical_columns])
        self.df.drop(numerical_columns, axis=1, inplace=True)
        self.df[numerical_columns_pca] = data
        self.__build_columns()

    def drop_original(self) -> None:
        """Drops the original dataset"""
        self.original_df = None
        self.df = pd.concat([self.train_df, self.submit_df])
        self.dfs = [TabularCompetitionDataFrameType.TRAIN, TabularCompetitionDataFrameType.SUBMIT]
