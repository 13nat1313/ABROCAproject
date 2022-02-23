import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from scipy.stats import sem

from src.abroca import compute_abroca, abroca_from_predictions
from src.simulation import simulate, SimulationParams


def regress(X, y, n, p_0):
    n_0 = int(round(n * p_0))
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X[:n_0], y[:n_0], test_size=0.2, random_state=0)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X[n_0:], y[n_0:], test_size=0.2, random_state=0)

    if len(set(y_test_0)) == 1 or len(set(y_test_1)) == 1:
        return

    X_train = np.append(X_train_0, X_train_1, axis=0)
    y_train = np.append(y_train_0, y_train_1, axis=0)
    y_test = np.append(y_test_0, y_test_1, axis=0)

    if len(set(y_train)) == 1:
        return

    # regressor = RandomForestRegressor(n_estimators = 10, random_state=0)
    regressor = LogisticRegression(solver='lbfgs')
    # regressor = LogisticRegressionCV(cv=5)
    perm = np.random.permutation(len(y_train))
    regressor.fit(X_train[perm], y_train[perm])

    y_pred_0 = regressor.predict(X_test_0)
    y_pred_1 = regressor.predict(X_test_1)
    y_pred = np.append(y_pred_0, y_pred_1, axis=0)

    fpr_0, tpr_0, _ = roc_curve(y_test_0, y_pred_0)
    fpr_1, tpr_1, _ = roc_curve(y_test_1, y_pred_1)

    g0auc = auc(fpr_0, tpr_0)
    g1auc = auc(fpr_1, tpr_1)
    abroca = compute_abroca(fpr_0, tpr_0, fpr_1, tpr_1)

    return fpr_0, tpr_0, fpr_1, tpr_1, g0auc, g1auc, abroca


def ABROCAvs_plot(plot_type, versus, r=10, s=0, **kwargs):
    """plot_type options are 'n', 'p_0', 'mu_orthogonal', 'mu_parallel', 'obs_noise', 'theta_diff'"""

    np.random.seed(s)
    avg_abrocas = None

    for i in range(r):
        abrocas = [np.nan for _ in range(len(versus))]
        for j, e in enumerate(versus):
            if plot_type == 'p_0':
                kwargs['p_0'] = e
            elif plot_type == 'mu_orthogonal':
                kwargs['mu_change'], kwargs['orthog_to_boundary'] = e, True
            elif plot_type == 'mu_parallel':
                kwargs['mu_change'] = e
            elif plot_type == 'obs_noise':
                kwargs['eta_sd'] = [.1, e]
            elif plot_type == 'theta_diff':
                kwargs['theta_1'] = [1, 1 * e]
            elif plot_type == 'n':
                kwargs['n'] = e

            dflts = SimulationParams(**kwargs)
            X, y = simulate(dflts)

            reg = regress(X, y, dflts.n, dflts.p_0)
            if reg is None:
                continue
            else:
                _, _, _, _, _, _, abroca = reg

            abrocas[j] = abroca

        if avg_abrocas is None:
            avg_abrocas = np.array(abrocas).reshape((-1, 1))
        else:
            avg_abrocas = np.append(avg_abrocas, np.array(abrocas).reshape((-1, 1)), axis=1)

    errors = sem(avg_abrocas, axis=1, nan_policy='omit').reshape(-1)
    avg_abrocas = np.nanmean(avg_abrocas, axis=1).ravel()

    return avg_abrocas, errors


def two_way_plot(big_x, big_y, plot_type, s=0, figsize=(20, 20), **kwargs):
    """options = 'p_0 vs mu_orthogonal', 'p_0 vs mu_parallel', 'p_0 vs obs_noise', 'p_0 vs theta_diff', 'p_0 vs n', 'mu_orthogonal vs obs_noise', 'mu_orthogonal vs theta_diff', 'mu_orthogonal vs n', 'mu_parallel vs obs_noise', 'mu_parallel vs theta_diff', 'mu_parallel vs n', 'obs_noise vs theta_diff', 'obs_noise vs n', 'theta_diff vs n'"""

    np.random.seed(s)
    fig, axs = plt.subplots(len(big_x), len(big_y), figsize=figsize)

    for row, a_x in enumerate(big_x):
        for col, a_y in enumerate(big_y):
            if plot_type == 'p_0 vs mu_orthogonal':
                kwargs['p_0'], kwargs['mu_change'], kwargs['orthog_to_boundary'], xaxis, yaxis = a_x, a_y, True, 'p_0', 'mu_orthogonal'
            elif plot_type == 'p_0 vs mu_parallel':
                kwargs['p_0'], kwargs['mu_change'], xaxis, yaxis = a_x, a_y, 'p_0', 'mu_parallel'
            elif plot_type == 'p_0 vs obs_noise':
                kwargs['p_0'], kwargs['eta_sd'], xaxis, yaxis = a_x, [0.1, a_y], 'p_0', 'obs_noise'
            elif plot_type == 'p_0 vs theta_diff':
                kwargs['p_0'], kwargs['theta_1'], xaxis, yaxis = a_x, [1, 1 * a_y], 'p_0', 'theta_diff'
            elif plot_type == 'p_0 vs n':
                kwargs['n'], kwargs['p_0'], xaxis, yaxis = a_y, a_x, 'p_0', 'n'
            elif plot_type == 'mu_orthogonal vs obs_noise':
                kwargs['mu_change'], kwargs['orthog_to_boundary'], kwargs['eta_sd'], xaxis, yaxis = a_x, True, [0.1,
                                                                                  a_y], 'mu_orthogonal', 'obs_noise'
            elif plot_type == 'mu_orthogonal vs theta_diff':
                kwargs['mu_change'], kwargs['orthog_to_boundary'], kwargs['theta_1'], xaxis, yaxis = a_x, True, [1,
                                                                                   1 * a_y], 'mu_orthogonal', 'theta_diff'
            elif plot_type == 'mu_orthogonal vs n':
                kwargs['n'], kwargs['mu_change'], kwargs['orthog_to_boundary'], xaxis, yaxis = a_y, a_x, True, 'mu_orthogonal', 'n'
            elif plot_type == 'mu_parallel vs obs_noise':
                kwargs['mu_change'], kwargs['eta_sd'], xaxis, yaxis = a_x, [0.1, a_y], 'mu_parallel', 'obs_noise'
            elif plot_type == 'mu_parallel vs theta_diff':
                kwargs['mu_change'], kwargs['theta_1'], xaxis, yaxis = a_x, [1, 1 * a_y], 'mu_parallel', 'theta_diff'
            elif plot_type == 'mu_parallel vs n':
                kwargs['n'], kwargs['mu_change'], xaxis, yaxis = a_y, a_x, 'mu_parallel', 'n'
            elif plot_type == 'obs_noise vs theta_diff':
                kwargs['eta_sd'], kwargs['theta_1'], xaxis, yaxis = [0.1, a_x], [1, 1 * a_y], 'obs_noise', 'theta_diff'
            elif plot_type == 'obs_noise vs n':
                kwargs['n'], kwargs['eta_sd'], xaxis, yaxis = a_y, [0.1, a_x], 'obs_noise', 'n'
            elif plot_type == 'theta_diff vs n':
                kwargs['n'], kwargs['theta_1'], xaxis, yaxis = a_y, [1, 1 * a_x], 'theta_diff', 'n'

            dflts = SimulationParams(**kwargs)
            X, y = simulate(dflts)

            reg = regress(X, y, dflts.n, dflts.p_0)
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


def visualize_data_3d(plot_vars):
    colors = ['0 label' for _ in plot_vars[0]] + ['1 label' for _ in plot_vars[3]]
    fig0 = px.scatter_3d(x=plot_vars[0]+plot_vars[3],y=plot_vars[1]+plot_vars[4],z=plot_vars[2]+plot_vars[5], color=colors, size=[.02 for _ in plot_vars[0]+plot_vars[3]], title='Group 0')

    colors = ['0 label' for _ in plot_vars[6]] + ['1 label' for _ in plot_vars[9]]
    fig1 = px.scatter_3d(x=plot_vars[6]+plot_vars[9],y=plot_vars[7]+plot_vars[10],z=plot_vars[8]+plot_vars[11], color=colors, size=[1 for _ in plot_vars[6]+plot_vars[9]], title='Group 1')
    return fig0, fig1


def visualize_data(point_size=.5, s=0, figsize=(20, 20), **kwargs):
    dflts = SimulationParams(**kwargs)
    np.random.seed(s)

    X, y = simulate(dflts)
    n_0 = int(round(dflts.n * dflts.p_0))
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

    plot_x0_0 = [a for a, b in zip(X_train_0[:, 0], y_train_0) if b == 0]
    plot_y0_0 = [a for a, b in zip(X_train_0[:, 1], y_train_0) if b == 0]
    plot_x0_1 = [a for a, b in zip(X_train_0[:, 0], y_train_0) if b == 1]
    plot_y0_1 = [a for a, b in zip(X_train_0[:, 1], y_train_0) if b == 1]
    plot_x1_0 = [a for a, b in zip(X_train_1[:, 0], y_train_1) if b == 0]
    plot_y1_0 = [a for a, b in zip(X_train_1[:, 1], y_train_1) if b == 0]
    plot_x1_1 = [a for a, b in zip(X_train_1[:, 0], y_train_1) if b == 1]
    plot_y1_1 = [a for a, b in zip(X_train_1[:, 1], y_train_1) if b == 1]

    if dflts.d != 2:
        plot_z0_0 = [a for a, b in zip(X_train_0[:, 2], y_train_0) if b == 0]
        plot_z0_1 = [a for a, b in zip(X_train_0[:, 2], y_train_0) if b == 1]
        plot_z1_0 = [a for a, b in zip(X_train_1[:, 2], y_train_1) if b == 0]
        plot_z1_1 = [a for a, b in zip(X_train_1[:, 2], y_train_1) if b == 1]
        plot_vars = (plot_x0_0, plot_y0_0, plot_z0_0, plot_x0_1, plot_y0_1, plot_z0_1, plot_x1_0, plot_y1_0, plot_z1_0, plot_x1_1, plot_y1_1, plot_z1_1)
        return visualize_data_3d(plot_vars)
    else:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs[1, 0].plot(fpr_0, tpr_0)
        axs[1, 0].set_title(f'Group 0 ROC\n auc = {g0auc}')
        axs[1, 1].plot(fpr_1, tpr_1)
        axs[1, 1].set_title(f'Group 1 ROC\n auc = {g1auc}')

        axs[0, 0].scatter(plot_x0_0, plot_y0_0, color='blue', s=point_size, label=f'0 label ({len(plot_x0_0)} points)')
        axs[0, 0].scatter(plot_x0_1, plot_y0_1, color='orange', s=point_size, label=f'1 label ({len(plot_x0_1)} points)')
        axs[0, 0].legend()
        axs[0, 0].set_title('Group 0')

        axs[0, 1].scatter(plot_x1_0, plot_y1_0, color='blue', s=point_size, label=f'0 label ({len(plot_x1_0)} points)')

        axs[0, 1].scatter(plot_x1_1, plot_y1_1, color='orange', s=point_size, label=f'1 label ({len(plot_x1_1)} points)')
        axs[0, 1].legend()
        axs[0, 1].set_title('Group 1')

        xlim = [int(min(plot_x0_0 + plot_x0_1 + plot_x1_0 + plot_x1_1)) - 1,
                int(max(plot_x0_0 + plot_x0_1 + plot_x1_0 + plot_x1_1)) + 1]
        ylim = [int(min(plot_y0_0 + plot_y0_1 + plot_y1_0 + plot_y1_1)) - 1,
                int(max(plot_y0_0 + plot_y0_1 + plot_y1_0 + plot_y1_1)) + 1]
        axs[0, 0].set_xlim(xlim)
        axs[0, 1].set_xlim(xlim)
        axs[0, 0].set_ylim(ylim)
        axs[0, 1].set_ylim(ylim)

        fig.suptitle(f'abroca = {abroca}')
        fig.tight_layout(pad=10.0)
        return fig