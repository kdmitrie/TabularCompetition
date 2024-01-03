import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
import numpy as np


from competition.const import TabularCompetitionDataFrameType, translate
from competition.competition import TabularCompetition
from competition.display import DisplayCompetition
from competition.columns import TabularCompetitionCategoricalColumn, TabularCompetitionColumn

class TabularCompetitionFAQ:
    def __init__(self, competition: TabularCompetition, display: DisplayCompetition):
        self.competition = competition
        self.display = display

    def print(self, section: str) -> None:
        """Generates answers to FAQ"""
        self.display.h2(f'{section} Frequently Asked Questions')
        self.display.start_section(style='.column')
        self.faq_missing_values(section + '1.')
        self.display.stop_section()

        self.display.start_section(style='.column')
        self.faq_duplicates(section + '2.')
        self.display.stop_section()

        self.display.start_section(style='.column')
        self.faq_in_test_not_in_train(section + '3.')
        self.display.stop_section()

    def faq_missing_values(self, section: str) -> None:
        data = {}
        target_columns = []
        normal_columns = []
        for column_name, column in self.competition.columns.items():
            data[column_name] = [column.get_missing(df) for df in self.competition.dfs]
            if sum(data[column_name]):
                if column_name in self.competition.target:
                    target_columns.append(column_name)
                else:
                    normal_columns.append(column_name)

        self.display.h3(f'{section} Are there any missing values?')
        self.display.p(f'Let\'s count missing values for every column and for every dataframe:')
        self.display.table(headings=[translate(df) for df in self.competition.dfs], data=data, select=lambda x: x > 0)
        if len(target_columns):
            self.display.p(
                f'There are missing values in target column(s) ' + ', '.join(target_columns) + ', but that is okay!')
        if len(normal_columns):
            self.display.p(f'There are missing values in following column(s):' + ', '.join(normal_columns))

    def faq_duplicates(self, section: str) -> None:
        dataframes = [[TabularCompetitionDataFrameType.TRAIN, ],
                      [TabularCompetitionDataFrameType.SUBMIT, ],
                      [TabularCompetitionDataFrameType.TRAIN, TabularCompetitionDataFrameType.SUBMIT, ],
                      ]
        if self.competition.original_df is not None:
            dataframes += [[TabularCompetitionDataFrameType.ORIGINAL, ],
                           [TabularCompetitionDataFrameType.TRAIN, TabularCompetitionDataFrameType.ORIGINAL],
                           [TabularCompetitionDataFrameType.SUBMIT, TabularCompetitionDataFrameType.ORIGINAL],
                           [TabularCompetitionDataFrameType.TRAIN, TabularCompetitionDataFrameType.SUBMIT,
                            TabularCompetitionDataFrameType.ORIGINAL]]

        data = {}
        for ds in dataframes:
            df = self.competition.df[self.competition.df.df.isin(ds)]
            dfs_name = ' + '.join(translate(d) for d in ds)
            data[dfs_name] = [
                sum(df.drop(['id', 'df'], axis=1).duplicated()),
                sum(df.drop(['id', 'df'] + self.competition.target, axis=1).duplicated())
            ]
        self.display.h3(f'{section} Are there any duplicates?')
        self.display.p(
            f'We calculate the duplicates within each dataframe and each combination of dataframes, including or excluding target column(s):')
        self.display.table(headings=['Number of full duplicated rows',
                                     'Number of duplicated rows without ' + ', '.join(self.competition.target)],
                           data=data, select=lambda x: x > 0)

    def faq_in_test_not_in_train(self, section: str) -> None:
        dataframes = [[TabularCompetitionDataFrameType.TRAIN, ], ]
        if self.competition.original_df is not None:
            dataframes += [[TabularCompetitionDataFrameType.ORIGINAL, ],
                           [TabularCompetitionDataFrameType.TRAIN, TabularCompetitionDataFrameType.ORIGINAL], ]

        data = {}
        for column_name, column in self.competition.columns.items():
            if not column_name in self.competition.target and isinstance(column, TabularCompetitionCategoricalColumn):
                data[column_name] = [', '.join(str(item) for item in column.values_in_2nd_not_in_1st(df1=ds,
                                                                                                     df2=TabularCompetitionDataFrameType.SUBMIT))
                                     for ds in dataframes]

        self.display.h3(
            f'{section} Are there any values of categorical columns, which present only in test, but not in train/original data?')
        self.display.p(f'We iterate through all categorical columns and concern the sets of values in each dataframe:')
        self.display.table(headings=[' + '.join(translate(d) for d in ds) for ds in dataframes], data=data,
                           select=lambda x: len(x))


class TabularCompetitionColumnsAnalyzer:
    pvalue_level = 0.05

    def __init__(self, competition: TabularCompetition, display: DisplayCompetition):
        self.competition = competition
        self.display = display

    def describe_target_columns(self, section: str) -> None:
        """Goes through target columns and display some outline"""
        self.display.h2(f'{section} The exploration of target column(s)')
        for n, column_name in enumerate(self.competition.target):
            self.describe_column(f'{section}{n + 1}.', self.competition.columns[column_name])

    def describe_column(self, section: str, column: TabularCompetitionColumn) -> None:
        """describes a single column"""
        self.display.start_section(style='.column')
        self.display.h3(f'{section} `{column.name}`')

        if column.name not in self.competition.target:
            dfs = self.competition.dfs
        else:
            dfs = [df for df in self.competition.dfs if df != TabularCompetitionDataFrameType.SUBMIT]

        # 1. General information
        n = 1
        self.display.h4(
            f'{section}{n}. `{column.name}` has {len(column.values)} distinct values ⇒ it is a {column.type} column')

        # 2. Plot the distribution of values
        n += 1
        fig = column.display_distribution(dfs)
        self.display.h4(f'{section}{n}. Below are the distributions of `{column.name}`')
        self.display.fig(fig)

        # 3. Statistical tests of all dataframes
        if len(dfs) > 1:
            n += 1
            test_name, stats = column.compare_distributions(dfs)
            self.display.h4(
                f'{section}{n}. We apply `{test_name}` test to determine, whether the distributions of `{column.name}` are equal in different dataframes')
            self.display.p(f'The null hypothesis is that the frequencies of values are identical in both dataframes.')
            for df1, df2, stat in stats:
                if stat.pvalue < self.pvalue_level:
                    strs = ('below', 'We reject the null hypothesis and conclude that the distributions are different.')
                else:
                    strs = ('above', 'We don\'t have sufficient data to reject the null hypothesis')
                self.display.p(
                    f'Comparing the {translate(df1)} and {translate(df2)} datatsets results in p-value of {stat.pvalue:.1e}, which is {strs[0]} the threshold level of {self.pvalue_level}. <b>{strs[1]}</b>')

        # 4. Distribution of current column and all categorical columns
        if len(self.competition.categorical_columns) - (column.name in self.competition.categorical_columns):
            n += 1
            self.display.h4(
                f'{section}{n}. We analyse the dependence of `{column.name}` on categorical columns in different dataframes')
            headings, row_headings, data = column.categorical_distributions(dfs, list(
                set(self.competition.categorical_columns) - {column.name}))
            self.display.table2level(headings, row_headings, data)

        # 5. Distribution of current column and all numerical columns
        if len(self.competition.categorical_columns) - (column.name in self.competition.categorical_columns):
            n += 1
            fig = column.numerical_distributions(dfs, list(set(self.competition.numerical_columns) - {column.name}))
            self.display.h4(
                f'{section}{n}. We analyse the dependence of `{column.name}` on numerical columns in different dataframes')
            self.display.fig(fig)
        self.display.stop_section()

    def describe_cross_columns(self, section: str) -> None:
        """Describes the dependencies between all columns"""
        self.display.h2(f'{section} The dependencies between all columns')

        for n, df in enumerate(self.competition.dfs):
            data = self.competition.df[self.competition.df.df == df][self.competition.numerical_columns]
            fig = plt.figure(figsize=(24, 24))
            sns.heatmap(data.corr(), annot=True)

            self.display.h3(f'{section}{n + 1}. The heatmap for {translate(df)} dataframe')
            self.display.fig(fig)


class TabularCompetitionAV:
    threshold = 0.05

    def __init__(self, competition: TabularCompetition, display: DisplayCompetition):
        self.competition = competition
        self.display = display
        self.clfs = []

    def add_classifier(self, clf: BaseEstimator) -> None:
        self.clfs.append(clf)

    def set_CV(self, cv: BaseCrossValidator) -> None:
        self.cv = cv

    def print(self, section: str, info: bool = True) -> None:
        if info:
            strs = ['All available dataframes are combined together.',
                    'The problem is posed of identifying the test dataframe, i.e. it is a simple classification problem.',
                    ', '.join([clf.__class__.__name__ for clf in
                               self.clfs]) + ' algorithms are used to solve the problem independently.',
                    self.cv.__class__.__name__ + ' is used to perform cross-validation.',
                    'AUC ROC metrics is used. If it turnes out to be about 0.5, it is hard to separate test data from other data. Otherwise, column engineering is needed or droping some items from the dataframe, which differ from test.']
            self.display.p(
                'Adversarial validation is used to check if the suggested dataframes can be distingushed. We do the following.')
            self.display.ol(strs)
        self.validation_all(section)

    def validation_all(self, section: str) -> None:
        n = 1
        for n1, df1 in enumerate(self.competition.dfs):
            for n2 in range(n1 + 1, len(self.competition.dfs)):
                df2 = self.competition.dfs[n2]
                self.validation(f'{section}{n}.', df1, df2)
                n += 1

    def validation(self, section: str, df1: TabularCompetitionDataFrameType, df2: TabularCompetitionDataFrameType) -> None:
        data = self.competition.df.drop(self.competition.target + ['id'], axis=1)
        data = data[data.df.isin((df1, df2))]
        y = data.df.replace({df1: 0, df2: 1}).to_numpy()
        X = data.drop('df', axis=1).to_numpy()

        headings = [f'Fold {n + 1}' for n in range(self.cv.get_n_splits())] + ['Mean']
        data = {}
        overall_mean = 0
        for clf_n, clf in enumerate(self.clfs):
            dt = []
            for n, (train_index, val_index) in enumerate(self.cv.split(X, y)):
                clf.fit(X[train_index], y[train_index])
                pred = clf.predict_proba(X[val_index])[:, 1]
                dt.append(roc_auc_score(y[val_index], pred))

            mn = np.mean(dt)
            dt.append(mn)
            overall_mean += mn
            data[clf.__class__.__name__] = [f'{v:.3f}' for v in dt]
        overall_mean /= len(self.clfs)

        strs = [
            f'It is inside the range [{0.5 - self.threshold}, {0.5 + self.threshold}]. So we conclude, <b>the {translate(df1)} and {translate(df2)} dataframes are similar</b>.',
            f'It is outside the range [{0.5 - self.threshold}, {0.5 + self.threshold}]. So we conclude, <b>the {translate(df1)} and {translate(df2)} dataframes are different</b>.', ]

        self.display.start_section(style='.column')
        self.display.h2(f'{section} Comparing {translate(df1)} and {translate(df2)} dataframes')
        self.display.table(headings, data, select=lambda x: float(x) > self.threshold)
        self.display.p(f'Overall mean score is {overall_mean:.3f}. ' + (
            strs[0] if abs(overall_mean - 0.5) < self.threshold else strs[1]))
        self.display.stop_section()
