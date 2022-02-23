from typing import Tuple
import folktables
import sklearn
from folktables.acs import adult_filter
import numpy as np
import pandas as pd
from sklearn import preprocessing


def assert_no_nan(df, msg: str):
    assert pd.isnull(df).values.sum() == 0, msg
    return


def scale_data(df_tr, df_te, TGT="target", SENS="sensitive"):
    """Scale numeric train/test data columns using the *train* data statistics.
    """
    assert_no_nan(df_tr, "null values detected in training data; cannot scale")
    assert_no_nan(df_te, "null values detected in test data; cannot scale")

    scaler = preprocessing.StandardScaler()
    columns_to_scale = [c for c, dtype in df_tr.dtypes.items()
                        if c not in (TGT, SENS)
                        and dtype != 'object']
    unscaled_columns = set(df_tr.columns) - set(columns_to_scale)
    df_tr_scaled = pd.DataFrame(scaler.fit_transform(df_tr[columns_to_scale]),
                                columns=columns_to_scale)
    df_tr_out = pd.concat((df_tr_scaled, df_tr[unscaled_columns]), axis=1)
    df_te_scaled = pd.DataFrame(scaler.transform(df_te[columns_to_scale]),
                                columns=columns_to_scale)
    df_te_out = pd.concat((df_te_scaled, df_te[unscaled_columns]), axis=1)

    # Check that no null values were introduced
    assert_no_nan(df_tr_out, "null values introduced during scaling")
    assert_no_nan(df_te_out, "null values introduced during scaling")
    return df_tr_out, df_te_out


def make_dummy_cols(df_tr, df_te, drop_first=True) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    assert_no_nan(df_tr, "nan values detected in train data")
    assert_no_nan(df_te, "nan values detected in test data")

    n_train = len(df_tr)

    # Concatenate for dummy creation, so all columns are in both splits.
    df = pd.concat((df_tr, df_te))
    df_dummies = pd.get_dummies(df, drop_first=drop_first)
    df_tr_out = df_dummies[:n_train]
    df_te_out = df_dummies[n_train:]

    assert_no_nan(df_tr_out, "nan values introduced in train data")
    assert_no_nan(df_te_out, "nan values introduced in test data")

    return df_tr_out, df_te_out


def acs_data_to_df(
        features: np.ndarray, label: np.ndarray,
        group: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Build a DataFrame from the result of folktables.BasicProblem.df_to_numpy().
    """
    ary = np.concatenate(
        (features, label.reshape(-1, 1), group.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(ary, columns=feature_names + ['target', 'sensitive'])
    return df


def get_acs_data_source(year: int):
    return folktables.ACSDataSource(survey_year=str(year),
                                    horizon='1-Year',
                                    survey='person')


def get_adult_dataset(states=("CA",), year=2018):
    """Fetch the Adult dataset."""
    data_source = get_acs_data_source(year)
    acs_data = data_source.get_data(states=states, download=True)
    feature_names = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP',
                     'RELP', 'WKHP',
                     # 'SEX', 'RAC1P',
                     ]
    problem = folktables.BasicProblem(
        features=feature_names,
        target='PINCP',
        target_transform=lambda x: x > 50000,
        group='RAC1P',
        # 'White alone' vs. all other categories (RAC1P) or
        # 'Male' vs. Female (SEX)
        group_transform=lambda x: x == 1,
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    features, label, group = problem.df_to_numpy(acs_data)

    df = acs_data_to_df(features, label, group, feature_names)
    return df


def x_y_split(df) -> Tuple[np.ndarray, np.ndarray]:
    """Split dataframe into X (n,d) and y (n,)."""
    y = df.pop('target').values
    X = df.values
    return X, y


def x_y_g_split(df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataframe into X (n,d) and y (n,)."""
    y = df.pop('target').values
    g = df.pop('sensitive').values
    X = df.values
    return X, y, g


def train_test_split(df, test_size: float = 0.1
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = sklearn.model_selection.train_test_split(df, test_size=test_size)
    tr.reset_index(inplace=True, drop=True)
    te.reset_index(inplace=True, drop=True)
    return tr, te