import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from src.abroca import compute_abroca
from src.simulation import set_defaults, simulate


def ABROCAvs_plot(plot_type, versus, r=10, s=0, n=[10000, 10000], eta_sd=[.1, .1], eta_mean=[0, 0], d=2, mu_0=None, mu_1=None, sigma_0=None, sigma_1=None, theta_0=None, theta_1=None, sigma_scale_factor=1, mu_change=0, orthog_to_boundary=False):
    """plot_type options are 'default', 'sample_prop', 'label_dist', 'cov_means', 'obs_noise', 'theta_diff'"""
    mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1 = set_defaults(d, mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1)
    assert len(mu_0) == len(mu_1) == len(sigma_0) == len(sigma_1) == len(theta_0) == len(theta_1), 'number of covariates not consistent'
    assert len(n) == len(eta_sd) == len(eta_mean), 'num of groups not consistent'

    np.random.seed(s)
    avg_abrocas = None
    g1_label_props_avg = None

    for i in range(r):
        g1_label_props = [np.nan for _ in range(len(versus))]
        abrocas = [np.nan for _ in range(len(versus))]
        for j, e in enumerate(versus):
            if plot_type == 'sample_prop':
                n = [int(round(sum(n)*e)), int(round(sum(n)*(1-e)))]
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'label_dist':
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=e, orthog_to_boundary=True)
            elif plot_type == 'cov_means':
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=e, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'obs_noise':
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=[0.1, e], eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'theta_diff':
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=[1, 1 * e], sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)
            else:
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)

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

            g1_label0 = len([a for a,b in zip(X_train_1[:,0], y_train_1) if b==0])
            g1_label1 = len([a for a,b in zip(X_train_1[:,0], y_train_1) if b==1])
            g1_label_prop = g1_label0/(g1_label0+g1_label1)
            g1_label_props[j] = g1_label_prop

        if g1_label_props_avg is None:
            g1_label_props_avg = np.array(g1_label_props).reshape((-1, 1))
        else:
            g1_label_props_avg = np.append(g1_label_props_avg, np.array(g1_label_props).reshape((-1, 1)), axis=1)
        if avg_abrocas is None:
            avg_abrocas = np.array(abrocas).reshape((-1, 1))
        else:
            avg_abrocas = np.append(avg_abrocas, np.array(abrocas).reshape((-1, 1)), axis=1)

    errors = np.nanstd(avg_abrocas, axis=1).ravel()
    errors = np.array([i/np.sqrt(r) for i in errors])
    avg_abrocas = np.nanmean(avg_abrocas, axis=1).ravel()
    g1_label_props_avg = np.nanmean(g1_label_props_avg, axis=1).ravel()

    return avg_abrocas, errors, g1_label_props_avg


def two_way_plot(big_x, big_y, plot_type, s=0, figsize=(20,20), n=[10000, 10000], eta_sd=[.1, .1], eta_mean=[0, 0], d=2, mu_0=None, mu_1=None, sigma_0=None, sigma_1=None, theta_0=None, theta_1=None, sigma_scale_factor=1, mu_change=0, orthog_to_boundary=False):
    """options = 'sample_prop vs label_dist', 'sample_prop vs cov_means', 'sample_prop vs obs_noise', 'sample_prop vs theta_diff', 'sample_prop vs sample_size', 'label_dist vs obs_noise', 'label_dist vs theta_diff', 'label_dist vs sample_size', 'cov_means vs obs_noise', 'cov_means vs theta_diff', 'cov_means vs sample_size', 'obs_noise vs theta_diff', 'obs_noise vs sample_size', 'theta_diff vs sample_size'"""
    mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1 = set_defaults(d, mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1)
    assert len(mu_0) == len(mu_1) == len(sigma_0) == len(sigma_1) == len(theta_0) == len(theta_1), 'number of covariates not consistent'
    assert len(n) == len(eta_sd) == len(eta_mean), 'num of groups not consistent'

    np.random.seed(s)
    fig, axs = plt.subplots(len(big_x), len(big_y), figsize=figsize)

    avg_auc = []
    for row, a_x in enumerate(big_x):
        for col, a_y in enumerate(big_y):
            if plot_type == 'sample_prop vs label_dist':
                xaxis, yaxis = 'sample_prop', 'label_dist'
                n = [int(round(sum(n)*a_x)), int(round(sum(n)*(1-a_x)))]
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=a_y, orthog_to_boundary=True)
            elif plot_type == 'sample_prop vs cov_means':
                xaxis, yaxis = 'sample_prop', 'cov_means'
                n = [int(round(sum(n) * a_x)), int(round(sum(n) * (1 - a_x)))]
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=a_y, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'sample_prop vs obs_noise':
                xaxis, yaxis = 'sample_prop', 'obs_noise'
                n = [int(round(sum(n) * a_x)), int(round(sum(n) * (1 - a_x)))]
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=[0.1, a_y], eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'sample_prop vs theta_diff':
                xaxis, yaxis = 'sample_prop', 'theta_diff'
                n = [int(round(sum(n) * a_x)), int(round(sum(n) * (1 - a_x)))]
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=[1, 1 * a_y], sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'sample_prop vs sample_size':
                xaxis, yaxis = 'sample_prop', 'sample_size'
                n = [int(round(a_y * a_x)), int(round(a_y * (1 - a_x)))]
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'label_dist vs obs_noise':
                xaxis, yaxis = 'label_dist', 'obs_noise'
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=[0.1, a_y], eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=a_x, orthog_to_boundary=True)
            elif plot_type == 'label_dist vs theta_diff':
                xaxis, yaxis = 'label_dist', 'theta_diff'
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=[1, 1 * a_y], sigma_scale_factor=sigma_scale_factor, mu_change=a_x, orthog_to_boundary=True)
            elif plot_type == 'label_dist vs sample_size':
                xaxis, yaxis = 'label_dist', 'sample_size'
                n = [a_y // 2, a_y // 2]
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=a_x, orthog_to_boundary=True)
            elif plot_type == 'cov_means vs obs_noise':
                xaxis, yaxis = 'cov_means', 'obs_noise'
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=[0.1, a_y], eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=a_x, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'cov_means vs theta_diff':
                xaxis, yaxis = 'cov_means', 'theta_diff'
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=[1, 1 * a_y], sigma_scale_factor=sigma_scale_factor, mu_change=a_x, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'cov_means vs sample_size':
                xaxis, yaxis = 'cov_means', 'sample_size'
                n = [a_y // 2, a_y // 2]
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=a_x, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'obs_noise vs theta_diff':
                xaxis, yaxis = 'obs_noise', 'theta_diff'
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=[0.1, a_x], eta_mean=eta_mean, theta_0=theta_0, theta_1=[1, 1 * a_y], sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'obs_noise vs sample_size':
                xaxis, yaxis = 'obs_noise', 'sample_size'
                n = [a_y // 2, a_y // 2]
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=[0.1, a_x], eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)
            elif plot_type == 'theta_diff vs sample_size':
                xaxis, yaxis = 'theta_diff', 'sample_size'
                n = [a_y // 2, a_y // 2]
                X, y = simulate(n=n, d=d, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=[1, 1 * a_x], sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)


            X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X[:n[0]], y[:n[0]], test_size=0.2,
                                                                        random_state=0)
            X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X[n[0]:], y[n[0]:], test_size=0.2,
                                                                        random_state=0)

            X_train = np.append(X_train_0, X_train_1, axis=0)
            y_train = np.append(y_train_0, y_train_1, axis=0)

            if len(set(y_train)) == 1:
                continue

            # regressor = RandomForestRegressor(n_estimators = 10, random_state=0)
            # regressor = LogisticRegression() #find regularized logisticRegression - cv logistic regression
            perm = np.random.permutation(len(y_train))

            regressor = LogisticRegressionCV(cv=5, random_state=0)
            regressor.fit(X_train[perm], y_train[perm])

            y_pred_0 = regressor.predict(X_test_0)
            y_pred_1 = regressor.predict(X_test_1)

            if len(set(y_test_0)) == 1 or len(set(y_test_1)) == 1:
                continue

            fpr_0, tpr_0, _ = roc_curve(y_test_0, y_pred_0)
            fpr_1, tpr_1, _ = roc_curve(y_test_1, y_pred_1)

            g0auc = auc(fpr_0, tpr_0)
            g1auc = auc(fpr_1, tpr_1)
            avg_auc += [g0auc, g1auc]
            abroca = compute_abroca(fpr_0, tpr_0, fpr_1, tpr_1)

            axs[row, col].plot(fpr_0, tpr_0, label='Group 0')
            axs[row, col].plot(fpr_1, tpr_1, label='Group 1')
            axs[row, col].legend()
            axs[row, col].set_title(
                f"{xaxis}={a_x}, {yaxis}={a_y}\n abroca={abroca:.3f}, g0auc={g0auc:.3f}, g1auc={g1auc:.3f}")
    fig.tight_layout(pad=3.0)
    print_text = f"average auc score: {np.nanmean(avg_auc)}"
    return fig, print_text


def visualize_data(point_size=.5, s=0, n=[10000, 10000], eta_sd=[.1, .1], eta_mean=[0, 0], d=2, mu_0=None, mu_1=None, sigma_0=None, sigma_1=None, theta_0=None, theta_1=None, sigma_scale_factor=1, mu_change=0, orthog_to_boundary=False):
    mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1 = set_defaults(d, mu_0, mu_1, sigma_0, sigma_1, theta_0, theta_1)
    assert len(mu_0) == len(mu_1) == len(sigma_0) == len(sigma_1) == len(theta_0) == len(theta_1), 'number of covariates not consistent'
    assert len(n) == len(eta_sd) == len(eta_mean), 'num of groups not consistent'

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    np.random.seed(s)
    X, y = simulate(n=n, mu_0=mu_0, mu_1=mu_1, sigma_0=sigma_0, sigma_1=sigma_1, eta_sd=eta_sd, eta_mean=eta_mean, theta_0=theta_0, theta_1=theta_1, sigma_scale_factor=sigma_scale_factor, mu_change=mu_change, orthog_to_boundary=orthog_to_boundary)

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X[:n[0]], y[:n[0]], test_size=0.2, random_state=0)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X[n[0]:], y[n[0]:], test_size=0.2, random_state=0)

    X_train = np.append(X_train_0, X_train_1, axis=0)
    y_train = np.append(y_train_0, y_train_1, axis=0)

    perm = np.random.permutation(len(y_train))

    regressor = LogisticRegressionCV(cv=5, random_state=0)
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