import numpy as np
import pandas as pd
import math
from numpy.linalg import solve as npsolve

def separate_regression(S: pd.DataFrame, T: pd.DataFrame, X_new: np.ndarray, *, Kxx, Kxy):
    """
    Separate Regression (Benchmark).
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

    idx_split = np.random.choice(S.index, size=int(math.ceil(ns / 2.0)), replace=False)
    S1 = S.loc[idx_split, :]
    S2 = S.drop(idx_split)

    X1or1 = S1[S1["a"] == 1].iloc[:, :d_].values
    Y1or1 = S1[S1["a"] == 1]["y"].values
    X1or0 = S1[S1["a"] == 0].iloc[:, :d_].values
    Y1or0 = S1[S1["a"] == 0]["y"].values

    X2or1 = S2[S2["a"] == 1].iloc[:, :d_].values
    Y2or1 = S2[S2["a"] == 1]["y"].values
    X2or0 = S2[S2["a"] == 0].iloc[:, :d_].values
    Y2or0 = S2[S2["a"] == 0]["y"].values

    K1or1 = Kxx(X1or1)
    K1or0 = Kxx(X1or0)

    # choose lambda f0
    ssemu0_list = []
    for lam in lambdas:
        inv_ = npsolve(K1or0 + len(Y1or0) * lam * np.eye(len(Y1or0)), Y1or0)
        pred_ = inv_ @ Kxy(X1or0, X2or0)
        ssemu0_list.append(np.sum((pred_ - Y2or0) ** 2))
    bestlambda_mu0 = lambdas[np.argmin(ssemu0_list)]

    # choose lambda f1
    ssemu1_list = []
    for lam in lambdas:
        inv_ = npsolve(K1or1 + len(Y1or1) * lam * np.eye(len(Y1or1)), Y1or1)
        pred_ = inv_ @ Kxy(X1or1, X2or1)
        ssemu1_list.append(np.sum((pred_ - Y2or1) ** 2))
    bestlambda_mu1 = lambdas[np.argmin(ssemu1_list)]

    inv0_best = npsolve(K1or0 + len(Y1or0) * bestlambda_mu0 * np.eye(len(Y1or0)), Y1or0)
    inv1_best = npsolve(K1or1 + len(Y1or1) * bestlambda_mu1 * np.eye(len(Y1or1)), Y1or1)

    mu0_new = inv0_best @ Kxy(X1or0, X_new)
    mu1_new = inv1_best @ Kxy(X1or1, X_new)

    return mu1_new - mu0_new
