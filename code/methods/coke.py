import numpy as np
import pandas as pd
import math
from numpy.linalg import solve as npsolve

def coke(S: pd.DataFrame, T: pd.DataFrame, X_new: np.ndarray, *, Kxx, Kxy):
    """
    COKE (Proposed Method).
    Inputs:
      S, T: DataFrames with columns [x1,...,xd, a, y]
      X_new: (nnew x d) array
      Kxx, Kxy: kernel functions passed from the main script
    Output:
      A vector of length nnew (the estimated CATE).
    """
    d_ = X_new.shape[1]
    ns = S.shape[0]

    # candidate lambdas
    if ns <= 1:
        max_power = 0
    else:
        max_power = int(math.ceil(math.log2(10 * (ns / 2))))
    lambdas = [(2.0 ** p) / (10.0) / (ns / 2.0) for p in range(max_power + 1)]
    if len(lambdas) == 0:
        lambdas = [1e-6]
    lambdas = np.array(lambdas)

    # split S into two parts for cross-fitting
    idx = np.random.permutation(ns)
    n1 = int(math.ceil(ns / 2.0))
    D1 = S.iloc[idx[:n1], :].copy()
    D2 = S.iloc[idx[n1:], :].copy()
    S_split = [D1, D2]

    perms = [[0, 1], [1, 0]]
    est_list = []

    XT = T.iloc[:, :d_].values  # T as array (only x columns)

    for perm in perms:
        S1 = S_split[perm[0]]
        S2 = S_split[perm[1]]

        lambda_min = lambdas.min()

        X1_or1 = S1[S1["a"] == 1].iloc[:, :d_].values
        Y1_or1 = S1[S1["a"] == 1]["y"].values
        X1_or0 = S1[S1["a"] == 0].iloc[:, :d_].values
        Y1_or0 = S1[S1["a"] == 0]["y"].values

        K_or0 = Kxx(X1_or0)
        K_or1 = Kxx(X1_or1)

        inv_or0 = npsolve(K_or0 + X1_or0.shape[0] * lambda_min * np.eye(X1_or0.shape[0]), Y1_or0)
        inv_or1 = npsolve(K_or1 + X1_or1.shape[0] * lambda_min * np.eye(X1_or1.shape[0]), Y1_or1)

        # pseudo outcome on S1
        X1_ = S1.iloc[:, :d_].values
        K_0_1 = Kxy(X1_or0, X1_)
        K_1_1 = Kxy(X1_or1, X1_)

        mu0hat = inv_or0 @ K_0_1
        mu1hat = inv_or1 @ K_1_1

        A_S1 = S1["a"].values
        Y_S1 = S1["y"].values
        phihat = np.where(A_S1 == 1, Y_S1 - mu0hat, mu1hat - Y_S1)

        # pseudo label on T using S2
        X2_or1 = S2[S2["a"] == 1].iloc[:, :d_].values
        Y2_or1 = S2[S2["a"] == 1]["y"].values
        X2_or0 = S2[S2["a"] == 0].iloc[:, :d_].values
        Y2_or0 = S2[S2["a"] == 0]["y"].values

        K2_or1 = Kxx(X2_or1)
        K2_or0 = Kxx(X2_or0)

        inv2_or1 = npsolve(K2_or1 + X2_or1.shape[0] * lambda_min * np.eye(X2_or1.shape[0]), Y2_or1)
        inv2_or0 = npsolve(K2_or0 + X2_or0.shape[0] * lambda_min * np.eye(X2_or0.shape[0]), Y2_or0)

        K_2T_or1 = Kxy(X2_or1, XT)
        K_2T_or0 = Kxy(X2_or0, XT)

        pseudo_T = (inv2_or1 @ K_2T_or1) - (inv2_or0 @ K_2T_or0)

        # pick best lambda for T
        K_1_ = Kxx(X1_)
        K_1T = Kxy(X1_, XT)

        esttg_lams = []
        for lam in lambdas:
            inv_ = npsolve(K_1_ + X1_.shape[0] * lam * np.eye(X1_.shape[0]), phihat)
            est_ = inv_ @ K_1T
            esttg_lams.append(est_)
        esttg_lams = np.array(esttg_lams)

        sse_list = np.sum((esttg_lams - pseudo_T) ** 2, axis=1)
        best_lambda = lambdas[np.argmin(sse_list)]

        # final on X_new
        K_1Xnew = Kxy(X1_, X_new)
        inv_best = npsolve(K_1_ + X1_.shape[0] * best_lambda * np.eye(X1_.shape[0]), phihat)
        est_final_perm = inv_best @ K_1Xnew

        est_list.append(est_final_perm)

    return np.mean(est_list, axis=0)
