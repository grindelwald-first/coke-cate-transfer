import numpy as np
import pandas as pd
import math
from numpy.linalg import solve as npsolve

def acw_cate(S: pd.DataFrame, T: pd.DataFrame, X_new: np.ndarray, *, Kxx, Kxy):
    """
    ACW-CATE (Benchmark).
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

    idx = np.random.permutation(ns)
    n1 = int(math.ceil(ns / 2.0))
    D1 = S.iloc[idx[:n1], :].copy()
    D2 = S.iloc[idx[n1:], :].copy()
    S_split = [D1, D2]

    perms = [[0, 1], [1, 0]]
    est_list = []
    XT = T.iloc[:, :d_].values
    nt = T.shape[0]

    for perm in perms:
        S1 = S_split[perm[0]]
        S2 = S_split[perm[1]]
        n_1 = S1.shape[0]

        Sbind = pd.concat([S1, S2], ignore_index=True)

        partS1 = np.random.choice(S1.index, size=int(math.ceil(len(S1) / 2.0)), replace=False)
        S1_1 = S1.loc[partS1, :]
        S1_2 = S1.drop(partS1)

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

        # propensity on S1
        lr_ps = LogisticRegression(solver="lbfgs")
        X_S1 = S1.iloc[:, :d_].values
        A_S1 = S1["a"].values
        lr_ps.fit(X_S1, A_S1)

        # outcome regression f0
        ssemu0_list = []
        for lam in lambdas:
            inv_ = npsolve(K_1or0_1 + len(Y1_1or0) * lam * np.eye(len(Y1_1or0)), Y1_1or0)
            pred_ = inv_ @ Kxy(X1_1or0, X1_2or0)
            ssemu0_list.append(np.sum((pred_ - Y1_2or0) ** 2))
        bestlambda_mu0 = lambdas[np.argmin(ssemu0_list)]

        # outcome regression f1
        ssemu1_list = []
        for lam in lambdas:
            inv_ = npsolve(K_1or1_1 + len(Y1_1or1) * lam * np.eye(len(Y1_1or1)), Y1_1or1)
            pred_ = inv_ @ Kxy(X1_1or1, X1_2or1)
            ssemu1_list.append(np.sum((pred_ - Y1_2or1) ** 2))
        bestlambda_mu1 = lambdas[np.argmin(ssemu1_list)]

        # density ratio estimation
        Stemp = S1.iloc[:, :d_].copy()
        Stemp["k"] = 0
        Ttemp = T.copy()
        Ttemp["k"] = 1

        all_ = pd.concat([Stemp, Ttemp], ignore_index=True)
        X_all_ = all_.iloc[:, :d_].values
        k_all_ = all_["k"].values

        lr_dens = LogisticRegression(solver="lbfgs")
        lr_dens.fit(X_all_, k_all_)

        X_Sbind = Sbind.iloc[:, :d_].values
        p_k1 = lr_dens.predict_proba(X_Sbind)[:, 1]

        weight = (p_k1 / (1.0 - p_k1)) * (ns / float(nt))
        weight = weight * ns / weight.sum()
        weight = np.minimum(weight, 20.0)
        weight = weight * ns / weight.sum()

        Sbind_ = Sbind.copy()
        Sbind_["what"] = weight

        pihat_Sbind = lr_ps.predict_proba(X_Sbind)[:, 1]
        Sbind_["pihat"] = pihat_Sbind

        # mu0hat, mu1hat for Sbind
        X1_or0_all = S1[S1["a"] == 0].iloc[:, :d_].values
        Y1_or0_all = S1[S1["a"] == 0]["y"].values
        K1_or0_all = Kxx(X1_or0_all)

        X1_or1_all = S1[S1["a"] == 1].iloc[:, :d_].values
        Y1_or1_all = S1[S1["a"] == 1]["y"].values
        K1_or1_all = Kxx(X1_or1_all)

        inv_or0 = npsolve(
            K1_or0_all + X1_or0_all.shape[0] * bestlambda_mu0 * np.eye(X1_or0_all.shape[0]),
            Y1_or0_all,
        )
        inv_or1 = npsolve(
            K1_or1_all + X1_or1_all.shape[0] * bestlambda_mu1 * np.eye(X1_or1_all.shape[0]),
            Y1_or1_all,
        )

        mu0hat_Sbind = inv_or0 @ Kxy(X1_or0_all, X_Sbind)
        mu1hat_Sbind = inv_or1 @ Kxy(X1_or1_all, X_Sbind)

        Sbind_["mu0hat"] = mu0hat_Sbind
        Sbind_["mu1hat"] = mu1hat_Sbind

        A_Sbind = Sbind_["a"].values
        Y_Sbind = Sbind_["y"].values

        Sbind_["phihat"] = Sbind_["what"] * (ns + nt) / float(ns) * (
            A_Sbind * (Y_Sbind - Sbind_["mu1hat"]) / Sbind_["pihat"]
            - (1.0 - A_Sbind) * (Y_Sbind - Sbind_["mu0hat"]) / (1.0 - Sbind_["pihat"])
        )

        # T_hat
        T_hat = T.copy()
        X_T = T_hat.iloc[:, :d_].values

        mu0hat_T = inv_or0 @ Kxy(X1_or0_all, X_T)
        mu1hat_T = inv_or1 @ Kxy(X1_or1_all, X_T)

        T_hat["mu0hat"] = mu0hat_T
        T_hat["mu1hat"] = mu1hat_T
        T_hat["phihat"] = (ns + nt) / float(nt) * (T_hat["mu1hat"] - T_hat["mu0hat"])

        # split T into T1, T2
        idxT = np.random.choice(T_hat.index, size=int(math.ceil(nt / 2.0)), replace=False)
        T1 = T_hat.loc[idxT, :]
        T2 = T_hat.drop(idxT)

        mix1 = pd.concat([Sbind_.iloc[:n_1, :], T1], ignore_index=True)
        mix2 = pd.concat([Sbind_.iloc[n_1:, :], T2], ignore_index=True)

        X_mix1 = mix1.iloc[:, :d_].values
        X_mix2 = mix2.iloc[:, :d_].values

        K_mix1 = Kxx(X_mix1)
        phihat_mix1 = mix1["phihat"].values
        phihat_mix2 = mix2["phihat"].values

        estmix_list = []
        for lam in lambdas:
            inv_ = npsolve(K_mix1 + X_mix1.shape[0] * lam * np.eye(X_mix1.shape[0]), phihat_mix1)
            pred_ = inv_ @ Kxy(X_mix1, X_mix2)
            estmix_list.append(pred_)
        estmix_list = np.array(estmix_list)

        sse_mix = np.sum((estmix_list - phihat_mix2) ** 2, axis=1)
        bestlambda_mix = lambdas[np.argmin(sse_mix)]

        K_mix1_new = Kxy(X_mix1, X_new)
        inv_ = npsolve(K_mix1 + X_mix1.shape[0] * bestlambda_mix * np.eye(X_mix1.shape[0]), phihat_mix1)
        est_list.append(inv_ @ K_mix1_new)

    return np.mean(est_list, axis=0)
