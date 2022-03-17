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

import numpy as np

import wandb

from src.datasets import ADULT_DATASET, VALID_DATASETS, get_dataset
from src.models import get_model, VALID_MODELS, DORO_MODEL
from src import datasets, torchutils
from src.config import DEFAULT_CONFIGS, CONFIG_FNS
from src.experiment_utils import fit, evaluate


def main(model_type: str = DORO_MODEL, dataset_name: str = ADULT_DATASET,
         scale=True, make_dummies=True):
    default_config = DEFAULT_CONFIGS[model_type]

    wandb.init(project="abroca", config=default_config)
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

    model = get_model(model_type, d_in=X_tr.shape[1],
                      criterion_kwargs=criterion_kwargs)
    if "optimizer" in config:
        opt = torchutils.get_optimizer(config["optimizer"],
                                       model, **opt_kwargs)
        fit_kwargs.update({"optimizer": opt})

    model = fit(model, X=X_tr, y=y_tr, g=g_tr,
                X_val=X_te, y_val=y_te, g_val=g_te,
                **fit_kwargs)
    metrics = evaluate(model, X_te, y_te, g_te)
    print(f"metrics for model_type {model_type}:")
    pprint.pprint(metrics)
    print("completed in {}s".format(time.time() - start))


if __name__ == "__main__":
    main()
