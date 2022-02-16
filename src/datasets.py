import folktables
from folktables.acs import adult_filter
import numpy as np
import pandas as pd


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
                     'RELP', 'WKHP', 'SEX', 'RAC1P', ]
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
