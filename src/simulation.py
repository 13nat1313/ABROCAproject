import numpy as np
from scipy.special import expit


def set_defaults(d, mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1, eta_sd, eta_mean):
    if mu_0 is None:
        mu_0 = np.zeros(d)
    if mu_1 is None:
        mu_1 = np.zeros(d)
    if sigma_0 is None:
        sigma_0 = np.ones(d)
    if sigma_1 is None:
        sigma_1 = np.ones(d)
    if theta_0 is None:
        theta_0 = np.ones(d)
    if theta_1 is None:
        theta_1 = np.ones(d)

    assert len(mu_0) == len(mu_1) == len(sigma_0) == len(sigma_1) == len(theta_0) == len(theta_1) == d, 'number of covariates not consistent'
    assert len(eta_sd) == len(eta_mean) == 2, 'number of groups not consistent'

    return mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1


def simulate(n=10000, p_0=.5, eta_sd=np.full(2, .1), eta_mean=np.zeros(2), d=2, mu_0=None, mu_1=None, sigma_0=None, sigma_1=None, theta_0=None, theta_1=None, sigma_scale_factor=1, mu_change=0, orthog_to_boundary=False):
    mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1 = set_defaults(d, mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1, eta_sd, eta_mean)

    if d==2:
        if orthog_to_boundary:
            mu_1 = [mu_1[0]+mu_change, mu_1[1]+mu_change]
        else:
            mu_1 = [mu_1[0]+mu_change, mu_1[1]-mu_change]

    mu = [np.array(mu_0), np.array(mu_1)]
    sigma = [np.diag(sigma_0), np.diag(np.dot(sigma_1, sigma_scale_factor))]
    theta = [np.array([theta_0]), np.array([theta_1])]

    X = np.array([])
    y = np.array([])

    subgroup_n = [int(round(n * p_0)), int(round(n * (1 - p_0)))]
    for i in range(2):
        n_i = subgroup_n[i]
        ax = np.random.multivariate_normal(mu[i], sigma[i], size=n_i)
        X = np.append(X.reshape((-1, d)), ax, axis=0)
        aeta = np.random.normal(eta_mean[i], eta_sd[i], n_i).reshape((-1, 1))
        ap = expit(np.sum(np.append(theta[i] * ax, aeta, axis=1), axis=1))
        ay = np.less_equal(np.random.uniform(size=len(ap)), ap)
        y = np.append(y, ay, axis=0)
    return X, y