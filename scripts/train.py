"""
Script for training a specified model and evaluating it on a given dataset.

Usage:

python scripts/train.py --model_type="L2LR" --dataset_name="adult"

python scripts/train.py --model_type="LR" --dataset_name="affect"
"""
import argparse
import pprint
import time

import numpy as np
import sklearn
import sklearn.linear_model
from fairlearn.reductions import ExponentiatedGradient

from src.datasets import ADULT_DATASET, VALID_DATASETS, get_dataset
from src.models import get_model, LR_MODEL, VALID_MODELS
from src.abroca import abroca_from_predictions
from src import datasets


def evaluate(model: sklearn.linear_model, X_te: np.ndarray, y_te: np.ndarray,
             g_te: np.ndarray):
    """Compute classification metrics for the model."""
    y_hat_probs = model.predict_proba(X_te)
    y_hat_labels = model.predict(X_te)
    loss = sklearn.metrics.log_loss(y_true=y_te, y_pred=y_hat_probs)
    acc = sklearn.metrics.accuracy_score(y_true=y_te, y_pred=y_hat_labels)
    abroca = abroca_from_predictions(y_true=y_te, y_pred=y_hat_probs[:, 1],
                                     g=g_te)
    return {"loss": loss, "abroca": abroca, "accuracy": acc}


def fit(model, X: np.ndarray, y: np.ndarray, g: np.ndarray = None):
    if isinstance(model, ExponentiatedGradient):
        # requires sensitive features as kwargs
        assert g is not None, "g is required for exponentiated gradient."
        model.fit(X=X, y=y, sensitive_features=g)
    else:
        model.fit(X, y)
    return model


def main(model_type: str = LR_MODEL, dataset_name: str = ADULT_DATASET,
         use_balanced: bool = False,
         scale=True, make_dummies=True):
    start = time.time()
    df = get_dataset(dataset_name)
    tr, te = datasets.train_test_split(df)
    if make_dummies:
        tr, te = datasets.make_dummy_cols(df_tr=tr, df_te=te)
    if scale:
        tr, te = datasets.scale_data(df_tr=tr, df_te=te)
    X_tr, y_tr, g_tr = datasets.x_y_g_split(tr)
    X_te, y_te, g_te = datasets.x_y_g_split(te)

    print(f"[INFO] training set has shape {X_tr.shape}")
    print(f"[INFO] training label distribution:")
    print(np.unique(y_tr, return_counts=True))
    print(f"[INFO] test label distribution:")
    print(np.unique(y_te, return_counts=True))

    model = get_model(model_type, use_balanced)
    model = fit(model, X_tr, y_tr, g_tr)
    metrics = evaluate(model, X_te, y_te, g_te)
    print(f"metrics for model_type {model_type}:")
    pprint.pprint(metrics)
    print("completed in {}s".format(time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str,
                        choices=VALID_MODELS)
    parser.add_argument("--dataset_name", default=ADULT_DATASET, type=str,
                        choices=VALID_DATASETS)
    parser.add_argument(
        "--use_balanced", default=False, action="store_true",
        help="If true, use class-balanced weights; otherwise no weights applied.")
    args = parser.parse_args()
    main(**vars(args))
