import numpy as np
from scipy.special import expit


class SimulationParams:
    """Object storing default data simulation parameters, e.g., number of covariates, subgroup sizes, covariate means, etc.

    Attributes:
        n (int): Population size.
        d (int): Number of covariates.
        p_0 (float): Size of subgroup 0 as proportion of population size.
        eta_sd (np.ndarray[float]): Standard deviation of error (η) for each subgroup.
        eta_mean (np.ndarray[float]): Mean of error (η) for each subgroup.
        mu_0 (np.ndarray[float]): Mean (μ) of each covariate of subgroup 0.
        mu_1 (np.ndarray[float]): Means (μ) of each covariate of subgroup 1.
        sigma_0 (np.ndarray[float]): Standard deviations (σ) of each covariate of subgroup 0.
        sigma_1 (np.ndarray[float]): Standard deviations (σ) of each covariate of subgroup 1.
        theta_0 (np.ndarray[float]): Weights (θ) of each covariate of subgroup 0.
        theta_1 (np.ndarray[float]): Weights (θ) of each covariate of subgroup 1.
        sigma_scale_factor (float): Value used to change 'sigma_1' using scalar multiplication ('sigma_scale_factor' * 'sigma_1').
        mu_change (float): Value used to change means of covariates of subgroup 1 in relation to classification boundary
        orthog_to_boundary (bool): If True, any change to 'mu_1' will occur othogonally to classification boundary. If False, change will occur parallel to boundary.
    """
    def __init__(self, d: int = 2, n: int = 10000, p_0: float = 0.5, eta_sd: np.ndarray = np.full(2, 0.1),
                 eta_mean: np.ndarray = np.zeros(2), mu_0: np.ndarray = None, mu_1: np.ndarray = None,
                 sigma_0: np.ndarray = None, sigma_1: np.ndarray = None, theta_0: np.ndarray = None,
                 theta_1: np.ndarray = None, sigma_scale_factor: float = 1, mu_change: float = 0,
                 orthog_to_boundary: bool = False):
        """Initializes SimulationParams with given parameters.

        Args:
            d: Number of covariates.
            n: Population size.
            p_0: Size of subgroup 0 as proportion of population size.
            eta_sd: Standard deviation of error (η) for each subgroup.
            eta_mean: Mean of error (η) for each subgroup.
            mu_0: Mean (μ) of each covariate of subgroup 0.
            mu_1: Means (μ) of each covariate of subgroup 1.
            sigma_0: Standard deviations (σ) of each covariate of subgroup 0.
            sigma_1: Standard deviations (σ) of each covariate of subgroup 1.
            theta_0: Weights (θ) of each covariate of subgroup 0.
            theta_1: Weights (θ) of each covariate of subgroup 1.
            sigma_scale_factor: Value used to change 'sigma_1' using scalar multiplication ('sigma_scale_factor' * 'sigma_1').
            mu_change: Value used to change means of covariates of subgroup 1 in relation to classification boundary
            orthog_to_boundary: If True, any change to 'mu_1' will occur othogonally to classification boundary. If False, change will occur parallel to boundary.

        Raises:
            AssertionError: If length of any of 'mu_0', 'mu_1', 'sigma_0', 'sigma_1', 'theta_0', or 'theta_1' do not equal 'd'.
            AssertionError: If length of either of 'eta_sd' or 'eta_mean' does not equal 2.
        """
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
        self.sigma_1 = np.diag(self.sigma_scale_factor*np.array(self.sigma_1))
        self.theta_0 = np.array([self.theta_0])
        self.theta_1 = np.array([self.theta_1])


def simulate(defaults: SimulationParams = None) -> (np.ndarray, np.ndarray):
    """Generate simulated data and matching labels.

    Args:
        defaults: Object storing default data simulation parameters, e.g., number of covariates, subgroup sizes, covariate means, etc.

    Returns:
        A tuple (X, y), where 'X' is an array of data of shape (NxM) where N equals population size and M equals number of covariates and 'y' is an array binary labels (coresponding to 'X') of shape (Nx1) where N equals population size.
    """
    if defaults is None:
        defaults = SimulationParams()

    mu = [defaults.mu_0, defaults.mu_1]
    sigma = [defaults.sigma_0, defaults.sigma_1]
    theta = [defaults.theta_0, defaults.theta_1]

    X = np.array([])
    y = np.array([])

    subgroup_n = [int(round(defaults.n * defaults.p_0)), int(round(defaults.n * (1 - defaults.p_0)))]
    for i in range(2):
        n_i = subgroup_n[i]
        ax = np.random.multivariate_normal(mu[i], sigma[i], size=n_i)
        X = np.append(X.reshape((-1, defaults.d)), ax, axis=0)
        aeta = np.random.normal(defaults.eta_mean[i], defaults.eta_sd[i], n_i).reshape((-1, 1))
        ap = expit(np.sum(np.append(theta[i] * ax, aeta, axis=1), axis=1))
        ay = np.less_equal(np.random.uniform(size=len(ap)), ap)
        y = np.append(y, ay, axis=0)
    return X, y
