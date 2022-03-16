from fairlearn.reductions import ExponentiatedGradient
import numpy as np
import sklearn
import sklearn.linear_model

from src.abroca import abroca_from_predictions
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