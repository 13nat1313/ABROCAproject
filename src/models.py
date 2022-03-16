import numpy as np
import sklearn
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn_extra.robust import RobustWeightedClassifier
from src import torchutils

# Vanilla logistic regression model
LR_MODEL = "LR"
LR_MODEL_BALANCED = "LR_BAL"

# L2-regularized logistic regression model
L2LR_MODEL = "L2LR"
L2LR_MODEL_BALANCED = "L2LR_BAL"

# RobustWeightedClassifier with Huber weighting
RWC_MODEL = "RWC"

# "Fast-DRO" model with chi-square constraint
FAST_DRO_MODEL = "DRO"

# DORO model
DORO_MODEL = "DORO"

# Group DRO
GROUP_DRO_MODEL = "GROUPDRO"

# Importance weighting
IMPORANCE_WEIGHTING_MODEL = "IW"

# Reductions-based equalized odds-constrained model via fairlearn
EO_REDUCTION = "EO_REDUCTION"

# Postprocessing-based equalized odds model via aif360
VALID_MODELS = [LR_MODEL, RWC_MODEL, EO_REDUCTION, L2LR_MODEL, FAST_DRO_MODEL,
                IMPORANCE_WEIGHTING_MODEL, DORO_MODEL, GROUP_DRO_MODEL]


def get_model(model_type: str, d_in: int = None,
              criterion_kwargs=None):
    """Fetch the specified model."""
    if model_type in (LR_MODEL, LR_MODEL_BALANCED):
        return sklearn.linear_model.LogisticRegression(
            penalty='none',
            class_weight="balanced" if model_type == LR_MODEL_BALANCED else None)
    elif model_type in (L2LR_MODEL, L2LR_MODEL_BALANCED):
        return sklearn.linear_model.LogisticRegressionCV(
            penalty='l2',
            class_weight="balanced" if model_type == L2LR_MODEL_BALANCED else None
        )
    if model_type == EO_REDUCTION:
        base_estimator = sklearn.linear_model.LogisticRegression()
        constraint = EqualizedOdds()
        model = ExponentiatedGradientWrapper(base_estimator, constraint)
        return model

    elif model_type == RWC_MODEL:
        return RobustWeightedClassifier(weighting="huber")

    elif model_type in (FAST_DRO_MODEL, DORO_MODEL):
        return torchutils.PytorchRegressor(
            d_in=d_in,
            criterion_kwargs=criterion_kwargs,
            model_type=model_type)

    elif model_type == GROUP_DRO_MODEL:
        # Pull group_dro_step_size out of config dict, since it is the only
        # element not passed to the super class constructor.
        group_weights_step_size = criterion_kwargs.pop(
            "group_weights_step_size")
        return torchutils.GroupDRORegressor(
            group_weights_step_size=group_weights_step_size,
            d_in=d_in,
            criterion_kwargs=criterion_kwargs,
            model_type=model_type)

    elif model_type == IMPORANCE_WEIGHTING_MODEL:
        return torchutils.PytorchRegressor(
            d_in=d_in,
            criterion_kwargs=criterion_kwargs,
            model_type=FAST_DRO_MODEL)
    else:
        raise ValueError(f"unsupported model type: {model_type}")


class ExponentiatedGradientWrapper(ExponentiatedGradient):
    """Wraps ExponentiatedGradient to provide all needed sklearn-type methods.
    """

    def predict_proba(self, X):
        return self._pmf_predict(X)
