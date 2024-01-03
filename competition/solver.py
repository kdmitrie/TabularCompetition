from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, BaseCrossValidator
import optuna
import sklearn

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.ensemble import StackingRegressor, StackingClassifier

from .competition import TabularCompetition
from .const import TabularCompetitionDataFrameType
from .display import DisplayCompetition


class TabularCompetitionSolver(ABC):
    predict_functions = {'regression': 'predict', 'categorical': 'predict_proba', 'binary': 'predict_proba'}

    def __init__(self, competition: TabularCompetition, display: DisplayCompetition):
        self.competition = competition
        self.display = display
        self.estimators = []
        self.results = []

        # Input data
        self.X, self.y, self.submit_X = self.__get_data()
        self.split_train_test()

    def set_CV(self, cv: BaseCrossValidator) -> None:
        self.cv = cv

    def add_estimator(self, estimator: BaseEstimator) -> None:
        self.estimators.append(estimator)

    def set_score_function(self, score: Callable, direction: str = 'minimize') -> None:
        self.score = score
        self.score_direction = direction

    def split_train_test(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=random_state)

    def optimize_parameters(self, estimator_class: BaseEstimator, n_trials: int = 256,
                            parameters: Optional[Dict] = None) -> Dict:
        # Optimize the given `estimator` parameters using OPTUNA
        init_parameters = {} if parameters is None else parameters

        best_params = []
        cv_scores = []
        test_scores = []
        estimators = []

        for target_n, tgt in enumerate(self.competition.target):
            # Find the best CV solution
            study = optuna.create_study(direction=self.score_direction, pruner='MedianPruner')

            problem_type_shape = self.__get_problem_type_and_shape(estimator_class, self.y_train[:, target_n])
            objective = self.__get_objective(self.y_train[:, target_n], estimator_class, problem_type_shape, parameters)
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)
            params = {**study.best_params, **init_parameters}

            # Train it on the CV data and get the test score
            _, pred = self.__train_estimator(estimator_class(**params), self.X_train, self.y_train[:, target_n],
                                             self.X_test, problem_type_shape[0])
            test_score = self.score(self.y_test, pred)

            # Train on the whole data
            estimator, _ = self.__train_estimator(estimator_class(**params), self.X, self.y[:, target_n], self.X_test,
                                                  problem_type_shape[0])

            best_params.append(params)
            cv_scores.append(study.best_trial.value)
            test_scores.append(test_score)
            estimators.append(estimator)

        result = {'best_params': best_params, 'cv_scores': cv_scores, 'test_scores': test_scores,
                  'estimators': estimators}
        self.results.append(result)
        return result

    def use_parameters(self, estimator_class: BaseEstimator, parameters: Dict) -> Dict:
        cv_scores = []
        test_scores = []
        estimators = []

        for target_n, tgt in enumerate(self.competition.target):
            params = parameters[target_n]
            problem_type_shape = self.__get_problem_type_and_shape(estimator_class, self.y_train[:, target_n])

            # Train it on the CV data and get the test score
            _, pred = self.__train_estimator(estimator_class(**params), self.X_train, self.y_train[:, target_n],
                                             self.X_test, problem_type_shape[0])
            test_score = self.score(self.y_test, pred)

            # Train on the whole data
            estimator, _ = self.__train_estimator(estimator_class(**params), self.X, self.y[:, target_n], self.X_test,
                                                  problem_type_shape[0])

            cv_scores.append(None)
            test_scores.append(test_score)
            estimators.append(estimator)

        result = {'best_params': parameters, 'cv_scores': cv_scores, 'test_scores': test_scores,
                  'estimators': estimators}
        self.results.append(result)
        return result

    def __get_data(self) -> Tuple[np.array, np.array]:
        # Select available dataframes
        available_dfs = [df for df in self.competition.dfs if df != TabularCompetitionDataFrameType.SUBMIT]
        df = self.competition.df.loc[self.competition.df.df.isin(available_dfs)]
        df = df.drop(['id', 'df'], axis=1)

        submit_df = self.competition.df.loc[self.competition.df.df == TabularCompetitionDataFrameType.SUBMIT]
        submit_df = submit_df.drop(['id', 'df'], axis=1)

        # Fetch X and y data as a numpy arrays
        X_columns = [col for col in df.columns if not col in self.competition.target]
        X, y, submit_X = df[X_columns].to_numpy(), df[self.competition.target].to_numpy(), submit_df[
            X_columns].to_numpy()

        return X, y, submit_X

    def __get_objective(self, y: np.array, estimator_class: BaseEstimator, problem_type_shape: Tuple[str, Tuple],
                        parameters: Optional[Dict]) -> Callable:
        # Return an objective function for OPTUNA optimization
        problem_type, shape = problem_type_shape

        parameters = {} if parameters is None else parameters

        def objective(trial: optuna.Trial) -> float:
            base_estimator = self.__create_estimator(estimator_class, trial, parameters=parameters)
            results = np.zeros(shape, dtype=float)
            for n, (train_index, val_index) in enumerate(self.cv.split(self.X_train, y)):
                X_train, y_train = self.X_train[train_index], y[train_index]
                X_val, y_val = self.X_train[val_index], y[val_index]
                _, pred = self.__train_estimator(base_estimator, X_train, y_train, X_val, problem_type)
                results[val_index] += pred
            return self.score(y, results / (n + 1))

        return objective

    def __get_problem_type_and_shape(self, estimator_class, y) -> Tuple[str, Tuple]:
        if sklearn.base.is_classifier(estimator_class):
            num_unique = len(np.unique(y))
            if num_unique > 2:
                return 'categorical', (len(y), num_unique)
            return 'binary', (len(y),)
        return 'regression', (len(y),)

    def __train_estimator(self, base_estimator: BaseEstimator, X_train: np.array, y_train: np.array, X_val: np.array,
                          problem_type: str):
        estimator = sklearn.base.clone(base_estimator)
        fit_params = self.__get_fit_params(estimator)
        estimator.fit(X_train, y_train, **fit_params)
        pred = getattr(estimator, self.predict_functions[problem_type])(X_val)
        return estimator, (pred if problem_type == 'categorical' else pred[:, 1])

    def __get_fit_params(self, estimator: BaseEstimator) -> Dict:
        for cl in [XGBRegressor, XGBClassifier, CatBoostRegressor, CatBoostClassifier]:
            if isinstance(estimator, cl):
                return {'verbose': False}
        return {}

    def __create_estimator(self, estimator_class: BaseEstimator, trial: optuna.Trial, parameters: Dict = {}):
        # This function automatically sets the search space for estimator_class
        # and returns the estimator object for futher fitting

        # Linear estimators differ
        if estimator_class in [Lasso, Ridge]:
            parameters['alpha'] = trial.suggest_float('alpha', 1e-5, 1e5, log=True)
            return estimator_class(**parameters)

        if estimator_class in [LogisticRegression]:
            parameters['C'] = trial.suggest_float('C', 1e-5, 1e5, log=True)
            return estimator_class(**parameters)

        # CatBoost differs from other models
        if estimator_class in [CatBoostRegressor, CatBoostClassifier]:
            parameters['iterations'] = trial.suggest_int('iterations', 50, 2000, log=True)
            parameters['max_depth'] = trial.suggest_int('max_depth', 2, 10)
            parameters['learning_rate'] = trial.suggest_float('learning_rate', 0, 0.3)
            return estimator_class(**parameters)

        class_params = estimator_class().get_params().keys()
        if 'n_estimators' in class_params:
            parameters['n_estimators'] = trial.suggest_int('n_estimators', 1, 256, log=True)

        if 'iterations' in class_params:
            parameters['iterations'] = trial.suggest_int('iterations', 50, 2000, log=True)

        if 'max_depth' in class_params:
            parameters['max_depth'] = trial.suggest_int('max_depth', 2, 16)

        if 'depth' in class_params:
            parameters['depth'] = trial.suggest_int('depth', 2, 10)

        if 'min_samples_leaf' in class_params:
            parameters['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)

        if 'num_leaves' in class_params:
            parameters['num_leaves'] = trial.suggest_int('num_leaves', 2, 10)

        if 'learning_rate' in class_params:
            parameters['learning_rate'] = trial.suggest_float('learning_rate', 0, 0.3)

        if 'subsample' in class_params:
            parameters['subsample'] = trial.suggest_float('subsample', 0, 1)

        return estimator_class(**parameters)

    def print_results(self) -> None:
        """Generates results table"""
        row_headings = {}
        headings = {target: ['Parameters', 'CV', 'Test'] for target in self.competition.target}
        data = []
        for result in self.results:
            row_headings[self.__get_estimator_name(result['estimators'][0])] = ['']
            row = []
            for item in zip(result['best_params'], result['cv_scores'], result['test_scores']):
                row.append(str(item[0]))
                row.append(str(item[1]))
                row.append(str(item[2]))
            data.append(row)

        self.display.start_section(style='.column')
        self.display.table2level(headings, row_headings, data)
        self.display.stop_section()

    def stack(self, estimator_classes: List[BaseEstimator], final_estimator: BaseEstimator) -> Dict:
        estimators = []
        for cl in estimator_classes:
            for result in self.results:
                if isinstance(result['estimators'][0], cl):
                    break
            estimators.append(result['estimators'])

        best_params = []
        cv_scores = []
        test_scores = []
        stacked_estimators = []
        for target_n, target_estimators in enumerate(zip(*estimators)):
            problem_type, _ = self.__get_problem_type_and_shape(target_estimators[0], self.y_train[target_n])
            stacking_method = StackingRegressor if problem_type == 'regression' else StackingClassifier

            stacking_estimator = stacking_method([(self.__get_estimator_name(te), te) for te in target_estimators],
                                                 final_estimator)

            # Train it on the CV data and get the test score
            _, pred = self.__train_estimator(stacking_estimator, self.X_train, self.y_train[:, target_n], self.X_test,
                                             problem_type)
            test_score = self.score(self.y_test, pred)

            # Train on the whole data
            estimator, _ = self.__train_estimator(stacking_estimator, self.X, self.y[:, target_n], self.X_test,
                                                  problem_type)

            best_params.append({})
            cv_scores.append('-')
            test_scores.append(test_score)
            stacked_estimators.append(estimator)

        result = {'best_params': best_params, 'cv_scores': cv_scores, 'test_scores': test_scores,
                  'estimators': stacked_estimators}
        self.results.append(result)
        return result

    def __get_estimator_name(self, estimator: BaseEstimator) -> str:
        return ''.join([c for c in estimator.__class__.__name__ if c.isupper()])

    def predict(self, estimators: List[BaseEstimator]) -> List:
        predictions = []
        for estimator in estimators:
            predictions.append(estimator.predict(self.submit_X))
        return predictions

    def predict_proba(self, estimators: List[BaseEstimator]) -> List:
        predictions = []
        for estimator in estimators:
            predictions.append(estimator.predict_proba(self.submit_X))
        return predictions
