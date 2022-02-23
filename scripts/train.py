import argparse
import pprint
import time

import numpy as np
import sklearn
import sklearn.linear_model
from fairlearn.reductions import ExponentiatedGradient

from datasets import x_y_g_split, train_test_split
from models import get_model, LR_MODEL, VALID_MODELS
from src.abroca import abroca_from_predictions

from src.datasets import get_adult_dataset, make_dummy_cols, scale_data


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
