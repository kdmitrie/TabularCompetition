import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from competition.competition import TabularCompetition
from competition.display import DisplayCompetition
from competition.eda import TabularCompetitionFAQ, TabularCompetitionColumnsAnalyzer, TabularCompetitionAV
from competition.solver import TabularCompetitionSolver

TRAIN_CSV = "./data/binary_classification/train.csv"
TEST_CSV = "./data/binary_classification/test.csv"
EXTERNAL_CSV = "./data/binary_classification/orig.csv"

DISPLAY_FAQ = True
N_TRIALS = 3


# We perform the preprocessing of the external dataframe to make it consistent with generated dataframes
def original_preprocessing(df):
    df.rename(columns={'RowNumber': 'id'}, inplace=True)

    # Drop items with NaN in Age, Geography, HasCrCard or IsActiveMember columns, respectfully
    df.drop([9, 6, 4, 8], axis=0, inplace=True)

    return df_preprocessing(df)


# We perform the preprocessing of the whole dataframe
def df_preprocessing(df):
    # One-hot encoding of Geography
    df.Gender.replace({'Male': 0, 'Female': 1}, inplace=True)
    for country in ['France', 'Spain']:
        df['in_' + country] = (df.Geography == country).astype(int)
    df.drop('Geography', axis=1, inplace=True)

    # Surname: one hot encoding
    # df = df.join(other=df_surnames, on='Surname')
    # df['count'].fillna(0, inplace=True)
    # df['mean'].fillna(df_target_mean, inplace=True)
    # df.rename({'mean': 'Surname_mean', 'count': 'Surname_count'}, axis=1, inplace=True)

    # Just drop the Surname column
    df.drop('Surname', axis=1, inplace=True)

    # Drop CustomerID
    df.drop('CustomerId', axis=1, inplace=True)

    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Surname pre-calculation
    df_train = pd.read_csv(TRAIN_CSV)
    df_target_mean = df_train.Exited.mean()
    df_surnames = df_train.groupby(by='Surname').Exited.agg(['mean', 'count'])

    competition = TabularCompetition(train_csv=TRAIN_CSV,
                                     submit_csv=TEST_CSV,
                                     original_csv=EXTERNAL_CSV,
                                     original_preprocessing=original_preprocessing,
                                     df_preprocessing=df_preprocessing)

    competition.set_target('Exited')
    competition.set_metric(roc_auc_score)
    competition_display = DisplayCompetition()

    if DISPLAY_FAQ:
        competition_display.h1('1. Exploratory Data Analysis')

        faq = TabularCompetitionFAQ(competition=competition, display=competition_display)
        faq.print(section='1.1.')

        column_analysis = TabularCompetitionColumnsAnalyzer(competition=competition, display=competition_display)
        column_analysis.describe_target_columns(section='1.2.')
        column_analysis.describe_cross_columns(section='1.3.')

        av = TabularCompetitionAV(competition=competition, display=competition_display)
        av.add_classifier(RandomForestClassifier())
        av.add_classifier(XGBClassifier())
        av.add_classifier(LGBMClassifier())
        av.set_CV(StratifiedKFold(n_splits=5, shuffle=True, random_state=0))
        av.print(section='1.4.')

    # Do not include the original data
    competition.drop_original()

    tcs = TabularCompetitionSolver(competition=competition, display=competition_display)

    tcs.set_CV(StratifiedKFold(n_splits=5, shuffle=True, random_state=0))
    tcs.set_score_function(roc_auc_score, 'maximize')
    tcs.split_train_test()

    rf_results = tcs.optimize_parameters(RandomForestClassifier, n_trials=N_TRIALS)
    # rf_results = tcs.use_parameters(RandomForestClassifier, [{'n_estimators': 256, 'max_depth': 15, 'min_samples_leaf': 8}])

    xgb_results = tcs.optimize_parameters(XGBClassifier, n_trials=N_TRIALS)
    # xgb_results = tcs.use_parameters(XGBClassifier, [{'n_estimators': 123, 'max_depth': 4, 'learning_rate': 0.21001401029627081, 'subsample': 0.9966588774680264}])

    cb_results = tcs.optimize_parameters(CatBoostClassifier, n_trials=N_TRIALS)
    # cb_results = tcs.use_parameters(CatBoostClassifier, [{'iterations': 308, 'max_depth': 5, 'learning_rate': 0.07887693553154532}])

    gb_results = tcs.optimize_parameters(GradientBoostingClassifier, n_trials=N_TRIALS)
    # gb_results = tcs.use_parameters(GradientBoostingClassifier, [{'n_estimators': 192, 'max_depth': 6, 'min_samples_leaf': 3, 'learning_rate': 0.051424514052856085, 'subsample': 0.8282254513081093}	])

    lgb_results = tcs.optimize_parameters(LGBMClassifier, n_trials=N_TRIALS)
    # lgb_results = tcs.use_parameters(LGBMClassifier, [{'n_estimators': 256, 'max_depth': 6, 'num_leaves': 10, 'learning_rate': 0.10559841062803202, 'subsample': 0.48968310648298274}])

    stack_all_results = tcs.stack(estimator_classes=[RandomForestClassifier, XGBClassifier, CatBoostClassifier, LGBMClassifier], final_estimator=LogisticRegression())

    competition_display.h1('2. OPTUNA results')
    tcs.print_results()

    competition_display.h1('3. Submit')
    target = competition.target[0]
    data_submit = pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')

    results = {
        'rf': rf_results,
        'xgb': xgb_results,
        'cb': cb_results,
        'gb': gb_results,
        'lgb': lgb_results,
        'stacked': stack_all_results
    }

    for n, (label, result) in enumerate(results.items()):
        data_submit[target] = tcs.predict_proba(result['estimators'])[0][:, 1]
        fname = f'submission_{label}.csv'
        data_submit[['id', target]].to_csv(fname, index=False)

        competition_display.h2(f'3.{n}. {fname}')
