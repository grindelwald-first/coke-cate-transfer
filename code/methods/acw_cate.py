def acw_cate(S: pd.DataFrame, T: pd.DataFrame, X_new: np.ndarray, *, Kxx, Kxy):
    """
    ACW-CATE benchmark.
    """
    from sklearn.linear_model import LogisticRegression

    d_ = X_new.shape[1]
    ns = S.shape[0]
    nt = T.shape[0]

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

        # Choose lambda for mu0
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

        # Choose lambda for mu1
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

        # Density ratio estimation using S_nc versus T
        Stemp = S_nc.iloc[:, :d_].copy()
        Stemp["k"] = 0

        Ttemp = T.iloc[:, :d_].copy()
        Ttemp["k"] = 1

        all_data = pd.concat([Stemp, Ttemp], ignore_index=True)

        X_all = all_data.iloc[:, :d_].values
        y_all = all_data["k"].values

        model_dens = LogisticRegression(solver="lbfgs")
        model_dens.fit(X_all, y_all)

        # Compute density-ratio weights for S_tg
        X_tg = S_tg.iloc[:, :d_].values

        p_k1 = model_dens.predict_proba(X_tg)[:, 1]

        weight = (p_k1 / (1.0 - p_k1)) * (ns / float(nt))
        weight = weight * (len(S_tg) / np.sum(weight))
        weight = np.minimum(weight, 20.0)
        weight = weight * (len(S_tg) / np.sum(weight))

        # Compute ACW source pseudo-outcome on S_tg
        S_tg = S_tg.copy()
        S_tg["what"] = weight

        pihat = model_ps.predict_proba(X_tg)[:, 1]
        S_tg["pihat"] = pihat

        if len(X_ncor0) == 0:
            mu0hat = np.zeros(X_tg.shape[0])
        else:
            A0 = Kmat_ncor0 + len(X_ncor0) * bestlambda_mu0 * np.eye(len(X_ncor0))
            B0 = Kxy(X_ncor0, X_tg)
            mu0hat = np.dot(Y_ncor0, npsolve(A0, B0))

        if len(X_ncor1) == 0:
            mu1hat = np.zeros(X_tg.shape[0])
        else:
            A1 = Kmat_ncor1 + len(X_ncor1) * bestlambda_mu1 * np.eye(len(X_ncor1))
            B1 = Kxy(X_ncor1, X_tg)
            mu1hat = np.dot(Y_ncor1, npsolve(A1, B1))

        S_tg["mu0hat"] = mu0hat
        S_tg["mu1hat"] = mu1hat

        a_tg = S_tg["a"].values
        y_tg = S_tg["y"].values

        phihat = (
            weight
            * ((len(S_tg) + nt) / float(len(S_tg)))
            * (
                a_tg * (y_tg - mu1hat) / pihat
                - (1.0 - a_tg) * (y_tg - mu0hat) / (1.0 - pihat)
            )
        )

        S_tg["phihat"] = phihat

        # Keep only x columns and phihat
        S_tg.drop(
            columns=["a", "y", "what", "pihat", "mu0hat", "mu1hat"],
            inplace=True
        )

        # Split S_tg into S_tg1 and S_tg2
        n_tg = S_tg.shape[0]

        indices_tg = np.random.choice(
            S_tg.index,
            size=int(math.ceil(n_tg / 2.0)),
            replace=False
        )

        S_tg1 = S_tg.loc[indices_tg, :].copy()
        S_tg2 = S_tg.drop(indices_tg).copy()

        # Construct T_hat
        T_hat = T.copy()
        X_T = T.iloc[:, :d_].values

        if len(X_ncor0) == 0:
            mu0hat_T = np.zeros(X_T.shape[0])
        else:
            B0_T = Kxy(X_ncor0, X_T)
            mu0hat_T = np.dot(Y_ncor0, npsolve(A0, B0_T))

        if len(X_ncor1) == 0:
            mu1hat_T = np.zeros(X_T.shape[0])
        else:
            B1_T = Kxy(X_ncor1, X_T)
            mu1hat_T = np.dot(Y_ncor1, npsolve(A1, B1_T))

        T_hat["mu0hat"] = mu0hat_T
        T_hat["mu1hat"] = mu1hat_T

        T_hat["phihat"] = (
            ((len(S_tg) + nt) / float(nt))
            * (T_hat["mu1hat"] - T_hat["mu0hat"])
        )

        # Split T_hat into T1 and T2
        indices_T = np.random.choice(
            T_hat.index,
            size=int(math.ceil(nt / 2.0)),
            replace=False
        )

        T1 = T_hat.loc[indices_T, :].copy()
        T2 = T_hat.drop(indices_T).copy()

        # 12) Target regression on mix1, validate on mix2
        mix1 = pd.concat([S_tg1, T1], ignore_index=True)
        mix2 = pd.concat([S_tg2, T2], ignore_index=True)

        X_mix1 = mix1.iloc[:, :d_].values
        X_mix2 = mix2.iloc[:, :d_].values

        Kmat_mix1 = Kxx(X_mix1)

        phihat_mix1 = mix1["phihat"].values
        phihat_mix2 = mix2["phihat"].values

        sse_mix = []

        for lam in lambdas:
            A_mix = Kmat_mix1 + X_mix1.shape[0] * lam * np.eye(X_mix1.shape[0])
            sol_mix = npsolve(A_mix, Kxy(X_mix1, X_mix2))
            est_mix = np.dot(phihat_mix1, sol_mix)
            sse_mix.append(np.sum((est_mix - phihat_mix2) ** 2))

        bestlambda_mix = lambdas[np.argmin(np.array(sse_mix))]

        # Final prediction on X_new
        K_mix1_new = Kxy(X_mix1, X_new)
        A_final = Kmat_mix1 + X_mix1.shape[0] * bestlambda_mix * np.eye(X_mix1.shape[0])
        sol_final = npsolve(A_final, K_mix1_new)

        est_perm = np.dot(phihat_mix1, sol_final)
        est_list.append(est_perm)

    return np.mean(est_list, axis=0)