import numpy as np
from scipy import interpolate
from scipy import integrate
from sklearn.metrics import roc_curve, auc


def interpolate_roc_fun(fpr, tpr, n_grid):
    """https://github.com/VaibhavKaushik3220/abroca/blob/main/abroca/compute_abroca.py"""
    roc_approx = interpolate.interp1d(x=fpr, y=tpr)
    x_new = np.linspace(0, 1, num=n_grid)
    y_new = roc_approx(x_new)
    return x_new, y_new


def compute_abroca(fpr_0, tpr_0, fpr_1, tpr_1, n_grid=10000, lb=0., ub=1.):
    """Compute the value of the abroca statistic."""

    # compare minority to majority class; accumulate absolute difference btw ROC curves to slicing statistic
    majority_roc_x, majority_roc_y = interpolate_roc_fun(fpr_0, tpr_0, n_grid)
    minority_roc_x, minority_roc_y = interpolate_roc_fun(fpr_1, tpr_1, n_grid)

    # use function approximation to compute slice statistic via piecewise linear function
    assert list(majority_roc_x) == list(
        minority_roc_x), "Majority and minority FPR are different"
    between_roc_dist = interpolate.interp1d(x=majority_roc_x,
                              y=(majority_roc_y - minority_roc_y))
    abs_between_roc_dist = lambda x: abs(between_roc_dist(x))
    slice, _ = integrate.quad(func=abs_between_roc_dist, a=lb, b=ub)

    return slice


def abroca_from_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                            g: np.ndarray) -> float:
    """Compute ABROCA from ground-truth, predicted labels and group labels."""
    assert np.all(np.logical_or(g == 1, g == 0)), "expect binary group labels."
    assert(np.all(y_true.shape==y_pred.shape))

    # Fetch indices of each subgroup.
    idxs_0 = (g == 0)
    idxs_1 = (g == 1)

    # Compute abroca.
    fpr_0, tpr_0, _ = roc_curve(y_true=y_true[idxs_0], y_score=y_pred[idxs_0])
    fpr_1, tpr_1, _ = roc_curve(y_true=y_true[idxs_1], y_score=y_pred[idxs_1])

    abroca = compute_abroca(fpr_0, tpr_0, fpr_1, tpr_1)
    return abroca
