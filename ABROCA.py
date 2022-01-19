import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
from scipy.special import expit
from scipy import interpolate
from scipy import integrate
from collections import Counter
from matplotlib.colors import ListedColormap


def interpolate_roc_fun(fpr, tpr, n_grid):
    """https://github.com/VaibhavKaushik3220/abroca/blob/main/abroca/compute_abroca.py"""
    roc_approx = interpolate.interp1d(x=fpr, y=tpr)
    x_new = np.linspace(0, 1, num=n_grid)
    y_new = roc_approx(x_new)
    return x_new, y_new


def compute_abroca(fpr_0, tpr_0, fpr_1, tpr_1, n_grid=10000, lb=0, ub=1, limit=1000):
    """https://github.com/VaibhavKaushik3220/abroca/blob/main/abroca/compute_abroca.py"""
    # Compute the value of the abroca statistic.

    # compare minority to majority class; accumulate absolute difference btw ROC curves to slicing statistic
    majority_roc_x, majority_roc_y = interpolate_roc_fun(fpr_0, tpr_0, n_grid)
    minority_roc_x, minority_roc_y = interpolate_roc_fun(fpr_1, tpr_1, n_grid)

    # use function approximation to compute slice statistic via piecewise linear function
    assert list(majority_roc_x) == list(minority_roc_x), "Majority and minority FPR are different"
    f1 = interpolate.interp1d(x=majority_roc_x, y=(majority_roc_y - minority_roc_y))
    f2 = lambda x, acc: abs(f1(x))
    slice, _ = integrate.quad(f2, lb, ub, limit)

    return slice


def simulate(subgroup_n=[10000, 10000], mu_0=[0, 0], mu_1=[0, 0], sigma_0=[1, 1], sigma_1=[1, 1], eta_sd=[.1, .1], eta_mean=[0, 0], sigma_scale_factor=1, mu_scale_factor=0, label_prop=False):

    assert len(mu_1) == len(mu_0) == len(sigma_0) == len(sigma_1), 'num of covariates not consistent'
    assert len(subgroup_n) == len(eta_sd) == len(eta_mean), 'num of groups not consistent'
    if label_prop:
        mu_1 = [mu_1[0]+mu_scale_factor, mu_1[1]+mu_scale_factor]
    else:
        mu_1 = [mu_1[0]+mu_scale_factor, mu_1[1]-mu_scale_factor]
    mu = [np.array(mu_0), np.array(mu_1)]
    sigma = [np.diag(sigma_0), np.diag(np.dot(sigma_1, sigma_scale_factor))]
    X = np.array([])
    y = np.array([])

    for i in range(len(subgroup_n)):
        n_i = subgroup_n[i]
        ax = np.random.multivariate_normal(mu[i], sigma[i], size=n_i)     
        X = np.append(X.reshape((-1, len(mu_0))), ax, axis=0)
        aeta = np.random.normal(eta_mean[i], eta_sd[i], n_i).reshape((-1, 1))
        theta = np.ones((1, len(mu_0)))
        ap = expit(np.sum(theta * ax + aeta, axis=1))
        ay = np.less_equal(np.random.uniform(size=len(ap)), ap)
        y = np.append(y, ay, axis=0)
    return X, y


def ABROCAvs_plot(plot_type, vs, r=10, s=0, n=[10000, 10000]):
    """plot_type options are 'default', 'sample_size', 'label_dist', 'cov_means', 'obs_noise'"""
    np.random.seed(s)
    avg_abrocas = None

    for i in range(r):
        abrocas = [np.nan for _ in range(len(vs))]
        for j, e in enumerate(vs):
            if plot_type == 'sample_size':
                X, y = simulate(subgroup_n=[int(round(sum(n)*e)), int(round(sum(n)*(1-e)))])
            elif plot_type == 'label_dist':
                X, y = simulate(subgroup_n=n, mu_scale_factor=e, label_prop=True)
            elif plot_type == 'cov_means':
                X, y = simulate(subgroup_n=n, mu_scale_factor=e)
            elif plot_type == 'obs_noise':
                X, y = simulate(subgroup_n=n, eta_sd=[0.1,e])
            else:
                X, y = simulate(subgroup_n=n)

            X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X[:n[0]], y[:n[0]], test_size=0.2, random_state=0)
            X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X[n[0]:], y[n[0]:], test_size=0.2, random_state=0)

            X_train = np.append(X_train_0, X_train_1, axis=0)
            y_train = np.append(y_train_0, y_train_1, axis=0)
            
            if len(set(y_train)) == 1:
                continue

            # regressor = RandomForestRegressor(n_estimators = 10, random_state=0)
            # regressor = LogisticRegression()
            regressor = LogisticRegressionCV(cv=5, random_state=0)
            perm = np.random.permutation(len(y_train))
            regressor.fit(X_train[perm], y_train[perm])

            y_pred_0 = regressor.predict(X_test_0)
            y_pred_1 = regressor.predict(X_test_1)

            if len(set(y_test_0)) == 1 or len(set(y_test_1)) == 1:
                continue

            fpr_0, tpr_0, _ = roc_curve(y_test_0, y_pred_0)
            fpr_1, tpr_1, _ = roc_curve(y_test_1, y_pred_1)

            g1auc = auc(fpr_0, tpr_0)
            g2auc = auc(fpr_1, tpr_1)
            abroca = compute_abroca(fpr_0, tpr_0, fpr_1, tpr_1)
            abrocas[j] = abroca

        if avg_abrocas is None:
            avg_abrocas = np.array(abrocas).reshape((-1, 1))
        else:
            avg_abrocas = np.append(avg_abrocas, np.array(abrocas).reshape((-1, 1)), axis=1)

    errors = np.nanstd(avg_abrocas, axis=1).ravel()
    errors = np.array([i/np.sqrt(r) for i in errors])
    avg_abrocas = np.nanmean(avg_abrocas, axis=1).ravel()

    return avg_abrocas, vs, errors

if __name__ == '__main__':
    avg_abrocas, eta_sds, errors = ABROCAvs_plot('obs_noise', [10 ** i for i in np.arange(-3, 3, 0.2)], s=13,
                                                        n=[5000, 5000])
