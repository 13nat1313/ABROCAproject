import folktables
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
    features, label, group = folktables.ACSIncome.df_to_numpy(acs_data)
    feature_names = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP',
                     'RELP', 'WKHP', 'SEX', 'RAC1P', ]
    df = acs_data_to_df(features, label, group, feature_names)
    return df