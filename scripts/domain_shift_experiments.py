"""
Script for tdomain shift experiments.
"""
import argparse
import pprint
import time

import numpy as np
import pandas as pd
import wandb

from src.datasets import ADULT_DATASET, AFFECT_DATASET, VALID_DATASETS, \
    get_dataset
from src.models import get_model
from src import datasets, torchutils
from src.config import DEFAULT_CONFIGS, CONFIG_FNS
from src.experiment_utils import fit, evaluate

SPLIT_COLS_BY_DATASET = {
    ADULT_DATASET: 'ST',
    AFFECT_DATASET: None  # TODO
}


def train_models(models_to_train, X_tr, y_tr, g_tr, X_te, y_te, g_te):
    trained_models = dict()
    for model_type in models_to_train:
        config = DEFAULT_CONFIGS[model_type]
        criterion_kwargs, opt_kwargs, fit_kwargs = CONFIG_FNS[model_type](
            config)
        model = get_model(model_type, d_in=X_tr.shape[1],
                          criterion_kwargs=criterion_kwargs)
        if "optimizer" in config:
            opt = torchutils.get_optimizer(config["optimizer"],
                                           model, **opt_kwargs)
            fit_kwargs.update({"optimizer": opt})

        model = fit(model, X=X_tr, y=y_tr, g=g_tr,
                    X_val=X_te, y_val=y_te, g_val=g_te,
                    **fit_kwargs)
        trained_models[model_type] = model
    return trained_models


def run_domain_shift_experiment(df: pd.DataFrame,
                                models_to_train, domain_split_feature,
                                test_domain_val,
                                make_dummies=True, scale=True):
    print(
        f"[INFO] splitting on column {domain_split_feature} == {test_domain_val}")
    tr, te = datasets.domain_split(df, split_col=domain_split_feature,
                                   test_values=[test_domain_val, ])
    if make_dummies:
        tr, te = datasets.make_dummy_cols(df_tr=tr, df_te=te)
    if scale:
        tr, te = datasets.scale_data(df_tr=tr, df_te=te)
    X_tr, y_tr, g_tr = datasets.x_y_g_split(tr)
    X_te, y_te, g_te = datasets.x_y_g_split(te)

    print(f"[INFO] training set has shape {X_tr.shape}")
    print(f"[INFO] training label distribution:")
    print(np.unique(y_tr, return_counts=True))
    print(f"[INFO] test set has shape {X_te.shape}")
    print(f"[INFO] test label distribution:")
    print(np.unique(y_te, return_counts=True))

    trained_models = train_models(models_to_train, X_tr=X_tr, y_tr=y_tr,
                                  g_tr=g_tr, X_te=X_te, y_te=y_te, g_te=g_te)
    per_model_metrics = []
    for model_type, model in trained_models.items():
        metrics = evaluate(model, X_te, y_te, g_te)
        metrics["domain_split_feature"] = domain_split_feature
        metrics["domain_split_test_value"] = test_domain_val
        metrics["model_type"] = model_type
        per_model_metrics.append(metrics)
        print(f"metrics for model_type {model_type}:")
        pprint.pprint(metrics)
    return per_model_metrics


def main(dataset_name: str = ADULT_DATASET,
         scale=True, make_dummies=True):
    wandb.init(project="abroca", mode="disabled")
    models_to_train = list(DEFAULT_CONFIGS.keys())

    start = time.time()
    df = get_dataset(dataset_name)
    domain_split_feature = SPLIT_COLS_BY_DATASET[dataset_name]
    metrics_list = []

    for test_domain_val in sorted(df[domain_split_feature].unique().tolist())[
                           :2]:
        metrics = run_domain_shift_experiment(df, models_to_train,
                                              domain_split_feature,
                                              test_domain_val,
                                              make_dummies=make_dummies,
                                              scale=scale)
        metrics_list.extend(metrics)
    pd.DataFrame(metrics_list).to_csv("results.csv")
    print("completed in {}s".format(time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=None, type=str,
                        choices=VALID_DATASETS)
    args = parser.parse_args()
    main(**vars(args))
