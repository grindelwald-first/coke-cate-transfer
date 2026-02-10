import numpy as np
import pandas as pd
import math
from numpy.linalg import solve as npsolve

def dr_cate(S: pd.DataFrame, T: pd.DataFrame, X_new: np.ndarray, *, Kxx, Kxy):
    """
    DR-CATE benchmark method.
    """
    from sklearn.linear_model import LogisticRegression

    d_ = X_new.shape[1]
    ns = S.shape[0]

    if ns <= 1:
        max_power = 0
    else:
        max_power = int(math.ceil(math.log2(10 * (ns / 2))))
    lambdas = [(2.0 ** p) / (10.0) / (ns / 2.0) for p in range(max_power + 1)]
    if len(lambdas) == 0:
        lambdas = [1e-6]
    lambdas = np.array(lambdas)

    # split S
    idx = np.random.permutation(ns)
    n1 = int(math.ceil(ns / 2.0))
    D1 = S.iloc[idx[:n1], :].copy()
    D2 = S.iloc[idx[n1:], :].copy()
    S_split = [D1, D2]

    perms = [[0, 1], [1, 0]]
    est_list = []

    for perm in perms:
        S1 = S_split[perm[0]]
        S2 = S_split[perm[1]]

        # cross-fitting partition in S1
        idx_part = np.random.choice(S1.index, size=int(math.ceil(len(S1) / 2)), replace=False)
        S1_1 = S1.loc[idx_part, :]
        S1_2 = S1.drop(idx_part)

        X1_1or1 = S1_1[S1_1["a"] == 1].iloc[:, :d_].values
        Y1_1or1 = S1_1[S1_1["a"] == 1]["y"].values
        X1_1or0 = S1_1[S1_1["a"] == 0].iloc[:, :d_].values
        Y1_1or0 = S1_1[S1_1["a"] == 0]["y"].values

        X1_2or1 = S1_2[S1_2["a"] == 1].iloc[:, :d_].values
        Y1_2or1 = S1_2[S1_2["a"] == 1]["y"].values
        X1_2or0 = S1_2[S1_2["a"] == 0].iloc[:, :d_].values
        Y1_2or0 = S1_2[S1_2["a"] == 0]["y"].values

        K_1or1_1 = Kxx(X1_1or1)
        K_1or0_1 = Kxx(X1_1or0)
        n1_1or1 = len(Y1_1or1)
        n1_1or0 = len(Y1_1or0)

        # Propensity with logistic
        lr_ps = LogisticRegression(solver="lbfgs")
        X_S1 = S1.iloc[:, :d_].values
        A_S1 = S1["a"].values
        lr_ps.fit(X_S1, A_S1)

        # choose best lambda for mu0
        ssemu0_list = []
        for lam in lambdas:
            inv_ = npsolve(K_1or0_1 + n1_1or0 * lam * np.eye(n1_1or0), Y1_1or0)
            pred_ = inv_ @ Kxy(X1_1or0, X1_2or0)
            ssemu0_list.append(np.sum((pred_ - Y1_2or0) ** 2))
        bestlambda_mu0 = lambdas[np.argmin(ssemu0_list)]

        # choose best lambda for mu1
        ssemu1_list = []
        for lam in lambdas:
            inv_ = npsolve(K_1or1_1 + n1_1or1 * lam * np.eye(n1_1or1), Y1_1or1)
            pred_ = inv_ @ Kxy(X1_1or1, X1_2or1)
            ssemu1_list.append(np.sum((pred_ - Y1_2or1) ** 2))
        bestlambda_mu1 = lambdas[np.argmin(ssemu1_list)]

        # final mu0/mu1 on entire S1
        X1_or1 = S1[S1["a"] == 1].iloc[:, :d_].values
        Y1_or1 = S1[S1["a"] == 1]["y"].values
        X1_or0 = S1[S1["a"] == 0].iloc[:, :d_].values
        Y1_or0 = S1[S1["a"] == 0]["y"].values

        K1_or1 = Kxx(X1_or1)
        K1_or0 = Kxx(X1_or0)

        inv_or0 = npsolve(K1_or0 + X1_or0.shape[0] * bestlambda_mu0 * np.eye(X1_or0.shape[0]), Y1_or0)
        inv_or1 = npsolve(K1_or1 + X1_or1.shape[0] * bestlambda_mu1 * np.eye(X1_or1.shape[0]), Y1_or1)

        K_10 = Kxy(X1_or0, X_S1)
        K_11 = Kxy(X1_or1, X_S1)

        mu0hat_S1 = inv_or0 @ K_10
        mu1hat_S1 = inv_or1 @ K_11

        pihat_S1 = lr_ps.predict_proba(X_S1)[:, 1]
        Y_S1 = S1["y"].values

        phihat_S1 = (mu1hat_S1 - mu0hat_S1) \
                    + A_S1 * (Y_S1 - mu1hat_S1) / pihat_S1 \
                    - (1.0 - A_S1) * (Y_S1 - mu0hat_S1) / (1.0 - pihat_S1)

        # Evaluate on S2
        X_S2 = S2.iloc[:, :d_].values
        A_S2 = S2["a"].values
        Y_S2 = S2["y"].values

        K_2_0 = Kxy(X1_or0, X_S2)
        K_2_1 = Kxy(X1_or1, X_S2)

        mu0hat_S2 = inv_or0 @ K_2_0
        mu1hat_S2 = inv_or1 @ K_2_1

        pihat_S2 = lr_ps.predict_proba(X_S2)[:, 1]
        phihat_S2 = (mu1hat_S2 - mu0hat_S2) \
                    + A_S2 * (Y_S2 - mu1hat_S2) / pihat_S2 \
                    - (1.0 - A_S2) * (Y_S2 - mu0hat_S2) / (1.0 - pihat_S2)

        # pick best lam for target param
        K1_full = Kxx(X_S1)
        K_12 = Kxy(X_S1, X_S2)

        esttg_list = []
        for lam in lambdas:
            inv_ = npsolve(K1_full + X_S1.shape[0] * lam * np.eye(X_S1.shape[0]), phihat_S1)
            pred_ = inv_ @ K_12
            esttg_list.append(pred_)
        esttg_list = np.array(esttg_list)

        ssetg_ = np.sum((esttg_list - phihat_S2) ** 2, axis=1)
        bestlambda_tg = lambdas[np.argmin(ssetg_)]

        # final on X_new
        K_1new = Kxy(X_S1, X_new)
        inv_ = npsolve(K1_full + X_S1.shape[0] * bestlambda_tg * np.eye(X_S1.shape[0]), phihat_S1)
        est_list.append(inv_ @ K_1new)

    return np.mean(est_list, axis=0)
