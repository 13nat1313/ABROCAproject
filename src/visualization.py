import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from scipy.stats import sem

from src.abroca import compute_abroca
from src.simulation import set_defaults, simulate


def regress(X, y, n, p_0):
    n_0 = int(round(n * p_0))
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X[:n_0], y[:n_0], test_size=0.2, random_state=0)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X[n_0:], y[n_0:], test_size=0.2, random_state=0)

    X_train = np.append(X_train_0, X_train_1, axis=0)
    y_train = np.append(y_train_0, y_train_1, axis=0)

    if len(set(y_train)) == 1:
        return

    # regressor = RandomForestRegressor(n_estimators = 10, random_state=0)
    regressor = LogisticRegression()
    # regressor = LogisticRegressionCV(cv=5)
    perm = np.random.permutation(len(y_train))
    regressor.fit(X_train[perm], y_train[perm])

    y_pred_0 = regressor.predict(X_test_0)
    y_pred_1 = regressor.predict(X_test_1)

    if len(set(y_test_0)) == 1 or len(set(y_test_1)) == 1:
        return

    fpr_0, tpr_0, _ = roc_curve(y_test_0, y_pred_0)
    fpr_1, tpr_1, _ = roc_curve(y_test_1, y_pred_1)

    g0auc = auc(fpr_0, tpr_0)
    g1auc = auc(fpr_1, tpr_1)
    abroca = compute_abroca(fpr_0, tpr_0, fpr_1, tpr_1)

    return fpr_0, tpr_0, fpr_1, tpr_1, g0auc, g1auc, abroca


def ABROCAvs_plot(plot_type, versus, r=10, s=0, n=10000, p_0=.5, eta_sd=np.full(2, .1), eta_mean=np.zeros(2), d=2,
                  mu_0=None, mu_1=None, sigma_0=None, sigma_1=None, theta_0=None, theta_1=None, sigma_scale_factor=1,
                  mu_change=0, orthog_to_boundary=False):
    """plot_type options are 'n', 'p_0', 'mu_orthogonal', 'mu_parallel', 'obs_noise', 'theta_diff'"""

    np.random.seed(s)
    avg_abrocas = None

    for i in range(r):
        abrocas = [np.nan for _ in range(len(versus))]
        for j, e in enumerate(versus):
            if plot_type == 'p_0':
                p_0 = e
            elif plot_type == 'mu_orthogonal':
                mu_change, orthog_to_boundary = e, True
            elif plot_type == 'mu_parallel':
                mu_change = e
            elif plot_type == 'obs_noise':
                eta_sd = [0.1, e]
            elif plot_type == 'theta_diff':
                theta_1 = [1, 1 * e]
            elif plot_type == 'n':
                n = e

            X, y = simulate(n=n, p_0=p_0, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd,
                            eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor,
                            mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)

            reg = regress(X, y, n, p_0)
            if reg is None:
                continue
            else:
                fpr_0, tpr_0, fpr_1, tpr_1, g0auc, g1auc, abroca = reg

            abrocas[j] = abroca

        if avg_abrocas is None:
            avg_abrocas = np.array(abrocas).reshape((-1, 1))
        else:
            avg_abrocas = np.append(avg_abrocas, np.array(abrocas).reshape((-1, 1)), axis=1)

    errors = sem(avg_abrocas, axis=1, nan_policy='omit').reshape(-1)
    avg_abrocas = np.nanmean(avg_abrocas, axis=1).ravel()

    return avg_abrocas, errors


def two_way_plot(big_x, big_y, plot_type, s=0, figsize=(20, 20), n=10000, p_0=.5, eta_sd=np.full(2, .1),
                 eta_mean=np.zeros(2), d=2, mu_0=None, mu_1=None, sigma_0=None, sigma_1=None, theta_0=None,
                 theta_1=None, sigma_scale_factor=1, mu_change=0, orthog_to_boundary=False):
    """options = 'p_0 vs mu_orthogonal', 'p_0 vs mu_parallel', 'p_0 vs obs_noise', 'p_0 vs theta_diff', 'p_0 vs n', 'mu_orthogonal vs obs_noise', 'mu_orthogonal vs theta_diff', 'mu_orthogonal vs n', 'mu_parallel vs obs_noise', 'mu_parallel vs theta_diff', 'mu_parallel vs n', 'obs_noise vs theta_diff', 'obs_noise vs n', 'theta_diff vs n'"""
    mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1 = set_defaults(d, mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1,
                                                                  eta_sd, eta_mean)

    np.random.seed(s)
    fig, axs = plt.subplots(len(big_x), len(big_y), figsize=figsize)

    for row, a_x in enumerate(big_x):
        for col, a_y in enumerate(big_y):
            if plot_type == 'p_0 vs mu_orthogonal':
                p_0, mu_change, orthog_to_boundary, xaxis, yaxis = a_x, a_y, True, 'p_0', 'mu_orthogonal'
            elif plot_type == 'p_0 vs mu_parallel':
                p_0, mu_change, xaxis, yaxis = a_x, a_y, 'p_0', 'mu_parallel'
            elif plot_type == 'p_0 vs obs_noise':
                p_0, eta_sd, xaxis, yaxis = a_x, [0.1, a_y], 'p_0', 'obs_noise'
            elif plot_type == 'p_0 vs theta_diff':
                p_0, theta_1, xaxis, yaxis = a_x, [1, 1 * a_y], 'p_0', 'theta_diff'
            elif plot_type == 'p_0 vs n':
                n, p_0, xaxis, yaxis = a_y, a_x, 'p_0', 'n'
            elif plot_type == 'mu_orthogonal vs obs_noise':
                mu_change, orthog_to_boundary, eta_sd, xaxis, yaxis = a_x, True, [0.1,
                                                                                  a_y], 'mu_orthogonal', 'obs_noise'
            elif plot_type == 'mu_orthogonal vs theta_diff':
                mu_change, orthog_to_boundary, theta_1, xaxis, yaxis = a_x, True, [1,
                                                                                   1 * a_y], 'mu_orthogonal', 'theta_diff'
            elif plot_type == 'mu_orthogonal vs n':
                n, mu_change, orthog_to_boundary, xaxis, yaxis = a_y, a_x, True, 'mu_orthogonal', 'n'
            elif plot_type == 'mu_parallel vs obs_noise':
                mu_change, eta_sd, xaxis, yaxis = a_x, [0.1, a_y], 'mu_parallel', 'obs_noise'
            elif plot_type == 'mu_parallel vs theta_diff':
                mu_change, theta_1, xaxis, yaxis = a_x, [1, 1 * a_y], 'mu_parallel', 'theta_diff'
            elif plot_type == 'mu_parallel vs n':
                n, mu_change, xaxis, yaxis = a_y, a_x, 'mu_parallel', 'n'
            elif plot_type == 'obs_noise vs theta_diff':
                eta_sd, theta_1, xaxis, yaxis = [0.1, a_x], [1, 1 * a_y], 'obs_noise', 'theta_diff'
            elif plot_type == 'obs_noise vs n':
                n, eta_sd, xaxis, yaxis = a_y, [0.1, a_x], 'obs_noise', 'n'
            elif plot_type == 'theta_diff vs n':
                n, theta_1, xaxis, yaxis = a_y, [1, 1 * a_x], 'theta_diff', 'n'

            X, y = simulate(n=n, p_0=p_0, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd,
                            eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor,
                            mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)

            reg = regress(X, y, n, p_0)
            if reg is None:
                continue
            else:
                fpr_0, tpr_0, fpr_1, tpr_1, g0auc, g1auc, abroca = reg

            axs[row, col].plot(fpr_0, tpr_0, label='Group 0')
            axs[row, col].plot(fpr_1, tpr_1, label='Group 1')
            axs[row, col].legend()
            axs[row, col].set_title(
                f"{xaxis}={a_x}, {yaxis}={a_y}\n abroca={abroca:.3f}, g0auc={g0auc:.3f}, g1auc={g1auc:.3f}")
    fig.tight_layout(pad=3.0)
    return fig


def visualize_data(point_size=.5, s=0, n=10000, p_0=.5, eta_sd=np.full(2, .1), eta_mean=np.zeros(2), d=2, mu_0=None,
                   mu_1=None, sigma_0=None, sigma_1=None, theta_0=None, theta_1=None, sigma_scale_factor=1, mu_change=0,
                   orthog_to_boundary=False):
    mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1 = set_defaults(d, mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1)
    assert len(mu_0) == len(mu_1) == len(sigma_0) == len(sigma_1) == len(theta_0) == len(
        theta_1), 'number of covariates not consistent'
    assert len(eta_sd) == len(eta_mean), 'number of groups not consistent'

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    np.random.seed(s)
    X, y = simulate(n=n, p_0=p_0, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd,
                    eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor,
                    mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)
    n_0 = int(round(n * p_0))
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X[:n_0], y[:n_0], test_size=0.2, random_state=0)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X[n_0:], y[n_0:], test_size=0.2, random_state=0)

    X_train = np.append(X_train_0, X_train_1, axis=0)
    y_train = np.append(y_train_0, y_train_1, axis=0)

    perm = np.random.permutation(len(y_train))

    regressor = LogisticRegressionCV(cv=5)
    regressor.fit(X_train[perm], y_train[perm])

    y_pred_0 = regressor.predict(X_test_0)
    y_pred_1 = regressor.predict(X_test_1)

    fpr_0, tpr_0, _ = roc_curve(y_test_0, y_pred_0)
    fpr_1, tpr_1, _ = roc_curve(y_test_1, y_pred_1)
    g0auc = auc(fpr_0, tpr_0)
    g1auc = auc(fpr_1, tpr_1)
    abroca = compute_abroca(fpr_0, tpr_0, fpr_1, tpr_1)

    axs[0, 0].plot(fpr_0, tpr_0)
    axs[0, 0].set_title(f'Group 0 ROC\n auc = {g0auc}')
    axs[0, 1].plot(fpr_1, tpr_1)
    axs[0, 1].set_title(f'Group 1 ROC\n auc = {g1auc}')

    plot_x0_0 = [a for a, b in zip(X_train_0[:, 0], y_train_0) if b == 0]
    plot_y0_0 = [a for a, b in zip(X_train_0[:, 1], y_train_0) if b == 0]
    axs[1, 0].scatter(plot_x0_0, plot_y0_0, color='blue', s=point_size, label=f'0 label ({len(plot_x0_0)} points)')
    plot_x0_1 = [a for a, b in zip(X_train_0[:, 0], y_train_0) if b == 1]
    plot_y0_1 = [a for a, b in zip(X_train_0[:, 1], y_train_0) if b == 1]
    axs[1, 0].scatter(plot_x0_1, plot_y0_1, color='orange', s=point_size, label=f'1 label ({len(plot_x0_1)} points)')
    axs[1, 0].legend()
    axs[1, 0].set_title('Group 0')

    plot_x1_0 = [a for a, b in zip(X_train_1[:, 0], y_train_1) if b == 0]
    plot_y1_0 = [a for a, b in zip(X_train_1[:, 1], y_train_1) if b == 0]
    axs[1, 1].scatter(plot_x1_0, plot_y1_0, color='blue', s=point_size, label=f'0 label ({len(plot_x1_0)} points)')
    plot_x1_1 = [a for a, b in zip(X_train_1[:, 0], y_train_1) if b == 1]
    plot_y1_1 = [a for a, b in zip(X_train_1[:, 1], y_train_1) if b == 1]
    axs[1, 1].scatter(plot_x1_1, plot_y1_1, color='orange', s=point_size, label=f'1 label ({len(plot_x1_1)} points)')
    axs[1, 1].legend()
    axs[1, 1].set_title('Group 1')

    xlim = [int(min(plot_x0_0 + plot_x0_1 + plot_x1_0 + plot_x1_1)) - 1,
            int(max(plot_x0_0 + plot_x0_1 + plot_x1_0 + plot_x1_1)) + 1]
    ylim = [int(min(plot_y0_0 + plot_y0_1 + plot_y1_0 + plot_y1_1)) - 1,
            int(max(plot_y0_0 + plot_y0_1 + plot_y1_0 + plot_y1_1)) + 1]
    axs[1, 0].set_xlim(xlim)
    axs[1, 1].set_xlim(xlim)
    axs[1, 0].set_ylim(ylim)
    axs[1, 1].set_ylim(ylim)

    fig.suptitle(f'abroca = {abroca}')
    fig.tight_layout(pad=10.0)
    return fig
