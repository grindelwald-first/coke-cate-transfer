import numpy as np
import pandas as pd
import math
from numpy.linalg import solve as npsolve


def dr_cate(S: pd.DataFrame, T: pd.DataFrame, X_new: np.ndarray, *, Kxx, Kxy):
    """
    DR-CATE benchmark.
    """
    from sklearn.linear_model import LogisticRegression

    d_ = X_new.shape[1]
    ns = S.shape[0]

    if ns <= 1:
        max_power = 0
    else:
        max_power = int(math.ceil(math.log2(10 * (ns / 2.0))))

    lambdas = np.array([
        (2.0 ** p) / 10.0 / (ns / 2.0)
        for p in range(max_power + 1)
    ])

    if len(lambdas) == 0:
        lambdas = np.array([1e-6])

    # Split S into two parts
    indices = np.random.permutation(ns)
    n1 = int(math.ceil(ns / 2.0))
    D1 = S.iloc[indices[:n1], :].copy()
    D2 = S.iloc[indices[n1:], :].copy()
    S_split = [D1, D2]

    perms = [[0, 1], [1, 0]]
    est_list = []

    for perm in perms:
        # Split into nuisance sample and target sample
        S_nc = S_split[perm[0]].copy()
        S_tg = S_split[perm[1]].copy()

        # Nuisance estimation on S_nc
        n_nc = S_nc.shape[0]

        indices_nc = np.random.choice(
            S_nc.index,
            size=int(math.ceil(n_nc / 2.0)),
            replace=False
        )

        S_nc1 = S_nc.loc[indices_nc, :].copy()
        S_nc2 = S_nc.drop(indices_nc).copy()

        X_nc1_or1 = S_nc1[S_nc1["a"] == 1].iloc[:, :d_].values
        Y_nc1_or1 = S_nc1[S_nc1["a"] == 1]["y"].values

        X_nc2_or1 = S_nc2[S_nc2["a"] == 1].iloc[:, :d_].values
        Y_nc2_or1 = S_nc2[S_nc2["a"] == 1]["y"].values

        X_nc1_or0 = S_nc1[S_nc1["a"] == 0].iloc[:, :d_].values
        Y_nc1_or0 = S_nc1[S_nc1["a"] == 0]["y"].values

        X_nc2_or0 = S_nc2[S_nc2["a"] == 0].iloc[:, :d_].values
        Y_nc2_or0 = S_nc2[S_nc2["a"] == 0]["y"].values

        n_nc1_or1 = len(Y_nc1_or1)
        n_nc1_or0 = len(Y_nc1_or0)

        # Full nuisance training data by treatment group
        X_ncor1 = S_nc[S_nc["a"] == 1].iloc[:, :d_].values
        Y_ncor1 = S_nc[S_nc["a"] == 1]["y"].values

        X_ncor0 = S_nc[S_nc["a"] == 0].iloc[:, :d_].values
        Y_ncor0 = S_nc[S_nc["a"] == 0]["y"].values

        Kmat_ncor1 = Kxx(X_ncor1)
        Kmat_ncor0 = Kxx(X_ncor0)

        # Propensity score estimation on S_nc
        X_S_nc = S_nc.iloc[:, :d_].values
        A_S_nc = S_nc["a"].values

        model_ps = LogisticRegression(solver="lbfgs")
        model_ps.fit(X_S_nc, A_S_nc)

        # Choose lambda for mu0 by validation within S_nc
        sse_mu0 = []

        for lam in lambdas:
            if len(X_nc1_or0) == 0:
                est = np.zeros(len(Y_nc2_or0))
            else:
                K_temp = Kxx(X_nc1_or0)
                A = K_temp + n_nc1_or0 * lam * np.eye(len(X_nc1_or0))
                B = Kxy(X_nc1_or0, X_nc2_or0)
                sol = npsolve(A, B)
                est = np.dot(Y_nc1_or0, sol)

            sse_mu0.append(np.sum((est - Y_nc2_or0) ** 2))

        bestlambda_mu0 = lambdas[np.argmin(np.array(sse_mu0))]

        # Choose lambda for mu1 by validation within S_nc
        sse_mu1 = []

        for lam in lambdas:
            if len(X_nc1_or1) == 0:
                est = np.zeros(len(Y_nc2_or1))
            else:
                K_temp = Kxx(X_nc1_or1)
                A = K_temp + n_nc1_or1 * lam * np.eye(len(X_nc1_or1))
                B = Kxy(X_nc1_or1, X_nc2_or1)
                sol = npsolve(A, B)
                est = np.dot(Y_nc1_or1, sol)

            sse_mu1.append(np.sum((est - Y_nc2_or1) ** 2))

        bestlambda_mu1 = lambdas[np.argmin(np.array(sse_mu1))]

        # Split S_tg into target-training and target-validation
        n_tg = S_tg.shape[0]

        indices_tg = np.random.choice(
            S_tg.index,
            size=int(math.ceil(n_tg / 2.0)),
            replace=False
        )

        S_tg1 = S_tg.loc[indices_tg, :].copy()
        S_tg2 = S_tg.drop(indices_tg).copy()

        X_tg1 = S_tg1.iloc[:, :d_].values
        X_tg2 = S_tg2.iloc[:, :d_].values

        Kmat_tg1 = Kxx(X_tg1)
        K_tg1_tg2 = Kxy(X_tg1, X_tg2)

        # Compute DR pseudo-outcome on S_tg1
        pihat_tg1 = model_ps.predict_proba(X_tg1)[:, 1]

        if len(X_ncor0) == 0:
            mu0hat_tg1 = np.zeros(X_tg1.shape[0])
        else:
            A0 = Kmat_ncor0 + len(X_ncor0) * bestlambda_mu0 * np.eye(len(X_ncor0))
            B0 = Kxy(X_ncor0, X_tg1)
            mu0hat_tg1 = np.dot(Y_ncor0, npsolve(A0, B0))

        if len(X_ncor1) == 0:
            mu1hat_tg1 = np.zeros(X_tg1.shape[0])
        else:
            A1 = Kmat_ncor1 + len(X_ncor1) * bestlambda_mu1 * np.eye(len(X_ncor1))
            B1 = Kxy(X_ncor1, X_tg1)
            mu1hat_tg1 = np.dot(Y_ncor1, npsolve(A1, B1))

        a_tg1 = S_tg1["a"].values
        y_tg1 = S_tg1["y"].values

        phihat_tg1 = (
            mu1hat_tg1 - mu0hat_tg1
            + a_tg1 * (y_tg1 - mu1hat_tg1) / pihat_tg1
            - (1.0 - a_tg1) * (y_tg1 - mu0hat_tg1) / (1.0 - pihat_tg1)
        )

        # Compute DR pseudo-outcome on S_tg2
        pihat_tg2 = model_ps.predict_proba(X_tg2)[:, 1]

        if len(X_ncor0) == 0:
            mu0hat_tg2 = np.zeros(X_tg2.shape[0])
        else:
            B0_tg2 = Kxy(X_ncor0, X_tg2)
            mu0hat_tg2 = np.dot(Y_ncor0, npsolve(A0, B0_tg2))

        if len(X_ncor1) == 0:
            mu1hat_tg2 = np.zeros(X_tg2.shape[0])
        else:
            B1_tg2 = Kxy(X_ncor1, X_tg2)
            mu1hat_tg2 = np.dot(Y_ncor1, npsolve(A1, B1_tg2))

        a_tg2 = S_tg2["a"].values
        y_tg2 = S_tg2["y"].values

        phihat_tg2 = (
            mu1hat_tg2 - mu0hat_tg2
            + a_tg2 * (y_tg2 - mu1hat_tg2) / pihat_tg2
            - (1.0 - a_tg2) * (y_tg2 - mu0hat_tg2) / (1.0 - pihat_tg2)
        )

        # Choose lambda for target CATE regression
        sse_tg = []

        for lam in lambdas:
            A_tg = Kmat_tg1 + X_tg1.shape[0] * lam * np.eye(X_tg1.shape[0])
            sol_tg = npsolve(A_tg, K_tg1_tg2)
            est_tg = np.dot(phihat_tg1, sol_tg)
            sse_tg.append(np.sum((est_tg - phihat_tg2) ** 2))

        bestlambda_tg = lambdas[np.argmin(np.array(sse_tg))]

        # Final prediction on X_new
        K_tg1_new = Kxy(X_tg1, X_new)
        A_final = Kmat_tg1 + X_tg1.shape[0] * bestlambda_tg * np.eye(X_tg1.shape[0])
        sol_final = npsolve(A_final, K_tg1_new)

        est_perm = np.dot(phihat_tg1, sol_final)
        est_list.append(est_perm)

    return np.mean(est_list, axis=0)