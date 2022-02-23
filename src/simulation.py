import numpy as np
from scipy.special import expit


class SimulationParams:
    def __init__(self, d: int = 2, n: int = 10000, p_0: float = 0.5, eta_sd: np.ndarray = np.full(2, 0.1), eta_mean: np.ndarray = np.zeros(2), mu_0: np.ndarray = None, mu_1: np.ndarray = None, sigma_0: np.ndarray = None, sigma_1: np.ndarray = None, theta_0: np.ndarray = None, theta_1: np.ndarray = None, sigma_scale_factor: float = 1, mu_change: float = 0, orthog_to_boundary: bool = False):
        self.d = d
        self.n = n
        self.p_0 = p_0
        self.eta_sd = eta_sd
        self.eta_mean = eta_mean
        self.mu_0 = np.zeros(self.d) if mu_0 is None else mu_0
        self.mu_1 = np.zeros(self.d) if mu_1 is None else mu_1
        self.sigma_0 = np.ones(self.d) if sigma_0 is None else sigma_0
        self.sigma_1 = np.ones(self.d) if sigma_1 is None else sigma_1
        self.theta_0 = np.ones(self.d) if theta_0 is None else theta_0
        self.theta_1 = np.ones(self.d) if theta_1 is None else theta_1
        self.sigma_scale_factor = sigma_scale_factor
        self.mu_change = mu_change
        self.orthog_to_boundary = orthog_to_boundary

        assert len(self.mu_0) == len(self.mu_1) == len(self.sigma_0) == len(self.sigma_1) == len(self.theta_0) == len(
            self.theta_1) == self.d, 'number of covariates not consistent'
        assert len(self.eta_sd) == len(self.eta_mean) == 2, 'number of groups not consistent'

        self.mu_0 = np.array(self.mu_0)
        if self.orthog_to_boundary:
            self.mu_1 = np.array(self.mu_1) + (np.array(self.theta_1) * self.mu_change)
        else:
            v = np.ones(len(self.theta_1) - 1)
            total = 0
            for i, v_i in enumerate(v):
                total += self.theta_1[i] * v_i
            z = total / (-self.theta_1[-1])
            v = np.append(v, z)
            self.mu_1 = np.array(self.mu_1) + (v * self.mu_change)
        self.sigma_0 = np.diag(self.sigma_0)
        self.sigma_1 = np.diag(np.dot(self.sigma_1, self.sigma_scale_factor))
        self.theta_0 = np.array([self.theta_0])
        self.theta_1 = np.array([self.theta_1])


def simulate(dflts: SimulationParams = None):
    if dflts is None:
        dflts = SimulationParams()

    mu = [dflts.mu_0, dflts.mu_1]
    sigma = [dflts.sigma_0, dflts.sigma_1]
    theta = [dflts.theta_0, dflts.theta_1]

    X = np.array([])
    y = np.array([])

    subgroup_n = [int(round(dflts.n * dflts.p_0)), int(round(dflts.n * (1 - dflts.p_0)))]
    for i in range(2):
        n_i = subgroup_n[i]
        ax = np.random.multivariate_normal(mu[i], sigma[i], size=n_i)
        X = np.append(X.reshape((-1, dflts.d)), ax, axis=0)
        aeta = np.random.normal(dflts.eta_mean[i], dflts.eta_sd[i], n_i).reshape((-1, 1))
        ap = expit(np.sum(np.append(theta[i] * ax, aeta, axis=1), axis=1))
        ay = np.less_equal(np.random.uniform(size=len(ap)), ap)
        y = np.append(y, ay, axis=0)
    return X, y