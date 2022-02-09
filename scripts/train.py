import argparse
import pprint
from typing import Tuple

import numpy as np
import sklearn
import sklearn.linear_model
from sklearn_extra.robust import RobustWeightedClassifier

from src.datasets import get_adult_dataset

LR_MODEL = "LR"
RWC_MODEL = "RWC"


def x_y_split(df) -> Tuple[np.ndarray, np.ndarray]:
    """Split dataframe into X (n,d) and y (n,)."""
    y = df.pop('target').values
    X = df.values
    return X, y


def get_model(model_type: str):
    """Fetch the specified model."""
    if model_type == LR_MODEL:
        return sklearn.linear_model.LogisticRegression(penalty='none')
    elif model_type == RWC_MODEL:
        return RobustWeightedClassifier(weighting="huber")
    else:
        raise ValueError(f"unsupported model type: {model_type}")


def evaluate(model: sklearn.linear_model, X_te: np.ndarray, y_te: np.ndarray):
    """Compute classification metrics for the model."""
    y_hat_probs = model.predict_proba(X_te)
    loss = sklearn.metrics.log_loss(y_true=y_te, y_pred=y_hat_probs)
    # TODO(jpgard): compute and return abroca here.
    return {"loss": loss}


def main(model_type: str = LR_MODEL):
    df = get_adult_dataset()
    tr, te = sklearn.model_selection.train_test_split(df, test_size=0.1)
    X_tr, y_tr = x_y_split(tr)
    X_te, y_te = x_y_split(te)

    model = get_model(model_type)
    model.fit(X_tr, y_tr)
    metrics = evaluate(model, X_te, y_te)
    print(f"metrics for model_type {model_type}:")
    pprint.pprint(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str)
    args = parser.parse_args()
    main(**vars(args))
