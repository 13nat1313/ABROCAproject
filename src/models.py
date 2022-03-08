import sklearn
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn_extra.robust import RobustWeightedClassifier
from src import torchutils

# Vanilla logistic regression model
LR_MODEL = "LR"

# L2-regularized logistic regression model
L2LR_MODEL = "L2LR"

# RobustWeightedClassifier with Huber weighting
RWC_MODEL = "RWC"

# "Fast-DRO" model with chi-square constraint
FAST_DRO_MODEL = "DRO"

# Reductions-based equalized odds-constrained model via fairlearn
EO_REDUCTION = "EO_REDUCTION"

# Postprocessing-based equalized odds model via aif360
VALID_MODELS = [LR_MODEL, RWC_MODEL, EO_REDUCTION, L2LR_MODEL, FAST_DRO_MODEL]


def get_model(model_type: str, use_balanced: bool = False, d_in: int = None,
              criterion_kwargs=None):
    """Fetch the specified model."""
    if model_type == LR_MODEL:
        return sklearn.linear_model.LogisticRegression(
            penalty='none',
            class_weight="balanced" if use_balanced else None)
    elif model_type == L2LR_MODEL:
        return sklearn.linear_model.LogisticRegressionCV(
            penalty='l2',
            class_weight="balanced" if use_balanced else None)
    assert not use_balanced, "use_balanced only supported for LR/L2LR"
    if model_type == EO_REDUCTION:
        base_estimator = sklearn.linear_model.LogisticRegression()
        constraint = EqualizedOdds()
        model = ExponentiatedGradientWrapper(base_estimator, constraint)
        return model
    elif model_type == RWC_MODEL:
        return RobustWeightedClassifier(weighting="huber")
    elif model_type == FAST_DRO_MODEL:
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
