import sklearn
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn_extra.robust import RobustWeightedClassifier

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


def get_model(model_type: str, use_balanced: bool = False):
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
    else:
        raise ValueError(f"unsupported model type: {model_type}")


class ExponentiatedGradientWrapper(ExponentiatedGradient):
    """Wraps ExponentiatedGradient to provide all needed sklearn-type methods.
    """

    def predict_proba(self, X):
        return self._pmf_predict(X)
