from typing import Tuple

import folktables
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split

LR_MODEL = "LR"


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


def get_adult_dataset(states=("CA",)):
    """Fetch the Adult dataset."""
    acs_data = folktables.data_source.get_data(states=states, download=True)
    features, label, group = folktables.ACSIncome.df_to_numpy(acs_data)
    feature_names = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP',
                     'RELP', 'WKHP', 'SEX', 'RAC1P', ]
    df = acs_data_to_df(features, label, group, feature_names)
    return df


def x_y_split(df) -> Tuple[np.ndarray, np.ndarray]:
    """Split dataframe into X (n,d) and y (n,)."""
    y = df.pop('target').values
    X = df.values
    return X, y


def get_model(model_type: str):
    """Fetch the specified model."""
    if model_type == LR_MODEL:
        return sklearn.linear_model.LogisticRegression(penalty='none')


def main():
    df = get_adult_dataset()
    tr, te = sklearn.model_selection.train_test_split(df, test_size=0.1)
    X_tr, y_tr = x_y_split(tr)
    X_te, y_te = x_y_split(te)

    model = get_model(LR_MODEL)
    model.fit(X_tr, y_tr)

    # TODO(jpgard): compute ABROCA on test data after refactor.


if __name__ == "__main__":
    main()
