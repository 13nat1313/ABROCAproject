import math
import numpy as np
from ortools.linear_solver import pywraplp


class RoLR:
    """Sklear-style class implementing RoLR algorithm of Feng et al. (2014)."""

    def __init__(self, n_1: int, p: int):
        self.p = p
        self.n_1 = n_1
        return

    @staticmethod
    def get_sample_idxs_to_keep(X, T):
        x_norm = np.linalg.norm(X, ord=2, axis=1)
        return x_norm <= T

    def fit(self, X, y):
        # TODO(jpgard): verify base e for log in paper.
        n = len(X)
        T = 4 * math.sqrt(math.log(self.p) / n + math.log(n) / n)
        idxs = self.get_sample_idxs_to_keep(X, T)
        X_fit = X[idxs]
        y_fit = y[idxs]

        # Add an intercept column to x
        X_fit = np.concatenate((X_fit, np.ones(len(X_fit))), axis=1)

        # TODO(jpgard): Solve the linear programming problem.
        # See https://developers.google.com/optimization/lp/lp_example
        # and https://developers.google.com/optimization/lp/stigler_diet .

        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError
