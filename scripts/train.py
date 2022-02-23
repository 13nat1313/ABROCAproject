import argparse
import pprint
import time
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn_extra.robust import RobustWeightedClassifier
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient

from src.abroca import abroca_from_predictions

from src.datasets import get_adult_dataset, make_dummy_cols, scale_data

# Vanilla logistic regression model
LR_MODEL = "LR"

# L2-regularized logistic regression model
L2LR_MODEL = "L2LR"

# RobustWeightedClassifier with Huber weighting
RWC_MODEL = "RWC"

# Reductions-based equalized odds-constrained model via fairlearn
EO_REDUCTION = "EO_REDUCTION"

# Postprocessing-based equalized odds model via aif360


VALID_MODELS = [LR_MODEL, RWC_MODEL, EO_REDUCTION, L2LR_MODEL]


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


class ExponentiatedGradientWrapper(ExponentiatedGradient):
    """Wraps ExponentiatedGradient to provide all needed sklearn-type methods.
    """

    def predict_proba(self, X):
        return self._pmf_predict(X)


def get_model(model_type: str):
    """Fetch the specified model."""
    if model_type == LR_MODEL:
        return sklearn.linear_model.LogisticRegression(penalty='none')
    elif model_type == L2LR_MODEL:
        return sklearn.linear_model.LogisticRegressionCV(penalty='l2')
    elif model_type == EO_REDUCTION:
        base_estimator = sklearn.linear_model.LogisticRegression()
        constraint = EqualizedOdds()
        model = ExponentiatedGradientWrapper(base_estimator, constraint)
        return model
    elif model_type == RWC_MODEL:
        return RobustWeightedClassifier(weighting="huber")
    else:
        raise ValueError(f"unsupported model type: {model_type}")


def evaluate(model: sklearn.linear_model, X_te: np.ndarray, y_te: np.ndarray,
             g_te: np.ndarray):
    """Compute classification metrics for the model."""
    y_hat_probs = model.predict_proba(X_te)
    loss = sklearn.metrics.log_loss(y_true=y_te, y_pred=y_hat_probs)
    abroca = abroca_from_predictions(y_true=y_te, y_pred=y_hat_probs[:, 1],
                                     g=g_te)
    return {"loss": loss, "abroca": abroca}


def fit(model, X: np.ndarray, y: np.ndarray, g: np.ndarray = None):
    if isinstance(model, ExponentiatedGradient):
        # requires sensitive features as kwargs
        assert g is not None, "g is required for exponentiated gradient."
        model.fit(X=X, y=y, sensitive_features=g)
    else:
        model.fit(X, y)
    return model


def train_test_split(df, test_size: float = 0.1
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = sklearn.model_selection.train_test_split(df, test_size=test_size)
    tr.reset_index(inplace=True, drop=True)
    te.reset_index(inplace=True, drop=True)
    return tr, te


def main(model_type: str = LR_MODEL, scale=True, make_dummies=True):
    start = time.time()
    df = get_adult_dataset()
    tr, te = train_test_split(df)
    if scale:
        tr, te = scale_data(df_tr=tr, df_te=te)
    if make_dummies:
        tr, te = make_dummy_cols(df_tr=tr, df_te=te)
    X_tr, y_tr, g_tr = x_y_g_split(tr)
    X_te, y_te, g_te = x_y_g_split(te)

    model = get_model(model_type)
    model = fit(model, X_tr, y_tr, g_tr)
    metrics = evaluate(model, X_te, y_te, g_te)
    print(f"metrics for model_type {model_type}:")
    pprint.pprint(metrics)
    print("completed in {}s".format(time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str,
                        choices=VALID_MODELS)
    args = parser.parse_args()
    main(**vars(args))
