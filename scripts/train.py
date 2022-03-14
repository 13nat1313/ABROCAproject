"""
Script for training a specified model and evaluating it on a given dataset.

Usage:

python scripts/train.py --model_type="L2LR" --dataset_name="adult"

python scripts/train.py --model_type="LR" --dataset_name="affect"

python scripts/train.py --model_type="DRO" --dataset_name="affect"

python scripts/train.py --model_type="IW" --dataset_name="affect"

"""
import argparse
import pprint
import time

from fairlearn.reductions import ExponentiatedGradient
import numpy as np
import sklearn
import sklearn.linear_model
import wandb

from src.datasets import ADULT_DATASET, VALID_DATASETS, get_dataset
from src.models import get_model, LR_MODEL, IMPORANCE_WEIGHTING_MODEL, VALID_MODELS
from src.abroca import abroca_from_predictions
from src import datasets, torchutils
from src.config import DEFAULT_CONFIGS, CONFIG_FNS
from src.torchutils import PytorchRegressor


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


def fit(model, X: np.ndarray, y: np.ndarray, g: np.ndarray = None,
        X_val=None, y_val=None, g_val=None,
        **fit_kwargs):
    if isinstance(model, ExponentiatedGradient):
        # requires sensitive features as kwargs
        assert g is not None, "g is required for exponentiated gradient."
        model.fit(X=X, y=y, sensitive_features=g, **fit_kwargs)
    elif isinstance(model, PytorchRegressor):
        model.fit(X, y, g, X_val=X_val, y_val=y_val, g_val=g_val, **fit_kwargs)
    else:
        model.fit(X, y, **fit_kwargs)
    return model


def main(model_type: str, dataset_name: str = ADULT_DATASET,
         use_balanced: bool = False,
         scale=True, make_dummies=True):
    default_config = DEFAULT_CONFIGS[model_type]

    wandb.init(project="abroca", mode="disabled",
               config=default_config)
    config = wandb.config

    # unpack config into criterion and fit kwargs
    model_type = config["model_type"]
    criterion_kwargs, opt_kwargs, fit_kwargs = CONFIG_FNS[model_type](config)

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

    model = get_model(model_type, use_balanced, d_in=X_tr.shape[1],
                      criterion_kwargs=criterion_kwargs)
    if "optimizer" in config:
        opt = torchutils.get_optimizer(config["optimizer"],
                                       model, **opt_kwargs)
        fit_kwargs.update({"optimizer": opt})
    # fit_model = models.fit_regressor(X_tr, y_tr, X_val=X_te, y_val=y_te,
    #                                  model=model,
    #                                  **fit_kwargs)

    model = fit(model, X=X_tr, y=y_tr, g=g_tr,
                X_val=X_te, y_val=y_te, g_val=g_te,
                **fit_kwargs)
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
