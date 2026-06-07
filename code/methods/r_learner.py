import numpy as np
import pandas as pd
import math
from numpy.linalg import solve as npsolve


def _fit_logistic_regression(X, y):
    from sklearn.linear_model import LogisticRegression
    for kwargs in [
        {"solver": "lbfgs", "penalty": None, "max_iter": 1000},
        {"solver": "lbfgs", "penalty": "none", "max_iter": 1000},
        {"solver": "lbfgs", "max_iter": 1000},
    ]:
        try:
            model = LogisticRegression(**kwargs)
            model.fit(X, y)
            return model
        except Exception:
            pass
    raise RuntimeError("Logistic regression failed.")


def r_learner(S: pd.DataFrame, T: pd.DataFrame, X_new: np.ndarray, *, Kxx, Kxy):
    """
    R-learner benchmark.
    """
    d_ = X_new.shape[1]
    ns = S.shape[0]

    if ns <= 1:
        max_power = 0
    else:
        max_power = int(math.ceil(math.log2(10 * (ns / 2.0))))
    lambdas = [(2.0 ** p) / (10.0) / (ns / 2.0) for p in range(max_power + 1)]
    if len(lambdas) == 0:
        lambdas = [1e-6]
    lambdas = np.array(lambdas)

    indices = np.random.permutation(ns)
    n1 = int(math.ceil(ns / 2.0))
    D1 = S.iloc[indices[:n1], :].copy()
    D2 = S.iloc[indices[n1:], :].copy()
    S_split = [D1, D2]

    perms = [[0, 1], [1, 0]]
    est_list = []
    eps = 1e-3

    for perm in perms:
        # Nuisance fold
        S_nc = S_split[perm[0]].copy()

        # Internal split for selecting lambda for m(z)
        idx_part = np.random.permutation(S_nc.shape[0])
        n_nc1 = int(math.ceil(S_nc.shape[0] / 2.0))
        S_nc1 = S_nc.iloc[idx_part[:n_nc1], :].copy()
        S_nc2 = S_nc.iloc[idx_part[n_nc1:], :].copy()

        X_nc1 = S_nc1.iloc[:, :d_].values
        Y_nc1 = S_nc1["y"].values
        X_nc2 = S_nc2.iloc[:, :d_].values
        Y_nc2 = S_nc2["y"].values

        K_nc1 = Kxx(X_nc1)
        n_train_m = X_nc1.shape[0]

        sse_m_list = []
        for lam in lambdas:
            inv_m = npsolve(K_nc1 + n_train_m * lam * np.eye(n_train_m), Y_nc1)
            pred_m = inv_m @ Kxy(X_nc1, X_nc2)
            sse_m_list.append(np.sum((pred_m - Y_nc2) ** 2))
        bestlambda_m = lambdas[np.argmin(sse_m_list)]

        # Fit m(z) on the full nuisance fold
        X_nc = S_nc.iloc[:, :d_].values
        Y_nc = S_nc["y"].values
        A_nc = S_nc["a"].values
        K_nc = Kxx(X_nc)
        inv_m_full = npsolve(
            K_nc + X_nc.shape[0] * bestlambda_m * np.eye(X_nc.shape[0]),
            Y_nc,
        )

        # Fit pi(z) on the full nuisance fold
        model_ps = _fit_logistic_regression(X_nc, A_nc)

        # Target-parameter fold, split into training/validation
        S_tg = S_split[perm[1]].copy()
        idx_tg = np.random.permutation(S_tg.shape[0])
        n_tg1 = int(math.ceil(S_tg.shape[0] / 2.0))
        S_tg1 = S_tg.iloc[idx_tg[:n_tg1], :].copy()
        S_tg2 = S_tg.iloc[idx_tg[n_tg1:], :].copy()

        X_tg1 = S_tg1.iloc[:, :d_].values
        X_tg2 = S_tg2.iloc[:, :d_].values
        A_tg1 = S_tg1["a"].values.astype(float)
        A_tg2 = S_tg2["a"].values.astype(float)
        Y_tg1 = S_tg1["y"].values.astype(float)
        Y_tg2 = S_tg2["y"].values.astype(float)

        # Residuals on training part
        mhat_tg1 = inv_m_full @ Kxy(X_nc, X_tg1)
        pihat_tg1 = np.clip(model_ps.predict_proba(X_tg1)[:, 1], eps, 1.0 - eps)
        yres_tg1 = Y_tg1 - mhat_tg1
        ares_tg1 = A_tg1 - pihat_tg1

        # Residuals on validation part
        mhat_tg2 = inv_m_full @ Kxy(X_nc, X_tg2)
        pihat_tg2 = np.clip(model_ps.predict_proba(X_tg2)[:, 1], eps, 1.0 - eps)
        yres_tg2 = Y_tg2 - mhat_tg2
        ares_tg2 = A_tg2 - pihat_tg2

        # Final R-loss KRR tuning
        Kmat_tg1 = Kxx(X_tg1)
        D_tg1 = np.diag(ares_tg1 ** 2)
        rhs_tg1 = ares_tg1 * yres_tg1

        rloss_list = []
        for lam in lambdas:
            inv_tau = npsolve(
                D_tg1 @ Kmat_tg1 + X_tg1.shape[0] * lam * np.eye(X_tg1.shape[0]),
                rhs_tg1,
            )
            pred_tau_tg2 = inv_tau @ Kxy(X_tg1, X_tg2)
            rloss_list.append(np.sum((yres_tg2 - ares_tg2 * pred_tau_tg2) ** 2))
        bestlambda_tau = lambdas[np.argmin(rloss_list)]

        inv_tau_best = npsolve(
            D_tg1 @ Kmat_tg1 + X_tg1.shape[0] * bestlambda_tau * np.eye(X_tg1.shape[0]),
            rhs_tg1,
        )
        est_list.append(inv_tau_best @ Kxy(X_tg1, X_new))

    return np.mean(est_list, axis=0)