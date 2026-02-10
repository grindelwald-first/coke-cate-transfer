###################################################
# 0) Imports + Path setup
###################################################
from pathlib import Path
import numpy as np
import pandas as pd
import math
from numpy.linalg import solve as npsolve
from scipy.spatial.distance import pdist, cdist, squareform

# Output path: repo_root/output/changeB_CF.csv
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "changeB_CF.csv"


###################################################
# 1) Define Key Parameters
###################################################
B_vals = [1, 5, 10, 15, 20, 25]
R = 2
c = 1
d = 4
beta = 1
sd = 0.5
sim = 100
nnew = 10000


###################################################
# 2) Helper Functions
###################################################
def expit(x):
    return 1.0 / (1.0 + np.exp(-x))

def ps(x):
    return expit(np.sum(x[0:4]) / 8.0 * R)

def or0(x):
    part1 = (
        2.0 * (abs(x[0]) - np.pi / 4.0) * (abs(x[0]) >= np.pi / 2.0)
        + abs(x[0]) * (abs(x[0]) < np.pi / 2.0)
    )
    return c * part1 - 0.5 * math.sin(x[0])

def or1(x):
    part1 = (
        2.0 * (abs(x[0]) - np.pi / 4.0) * (abs(x[0]) >= np.pi / 2.0)
        + abs(x[0]) * (abs(x[0]) < np.pi / 2.0)
    )
    return c * part1 + 0.5 * math.sin(x[0])


###################################################
# 3) Matern Kernel
###################################################
rho = 5

def matern_kernel_kappa2(u, rho):
    u = u + 1e-100
    return 4.0 * np.exp(-2.0 * math.sqrt(2.0) * u / rho) / ((math.pi ** 0.5) * rho)

def Kxx(x):
    dist_xx = squareform(pdist(x, metric="euclidean"))
    return matern_kernel_kappa2(dist_xx, rho)

def Kxy(x, y):
    dist_xy = cdist(x, y, metric="euclidean")
    return matern_kernel_kappa2(dist_xy, rho)


###################################################
# 4) Main (COKE CF experiment, perm1/perm2/all)
###################################################
def main():
    results_df = pd.DataFrame(columns=["B", "type", "risk"])

    mse_all = np.empty(sim)
    mse_perm1 = np.empty(sim)
    mse_perm2 = np.empty(sim)

    for B_ in B_vals:
        nt = int(round((70 * math.sqrt(B_) + 12 * R + 5) * 5))
        ns = nt * 4
        p = 1.0 / (1.0 + B_ ** (1.0 / beta))

        for time_ in range(sim):
            np.random.seed(time_ + 1000)

            ############################################
            # 4.1 Generate Source Data (S)
            ############################################
            S_array = np.zeros((ns, d))

            for col in range(beta, d):
                S_array[:, col] = np.random.uniform(-math.pi, math.pi, size=ns)

            for col in range(beta):
                for i in range(ns):
                    while True:
                        x_candi = np.random.uniform(-math.pi, math.pi)
                        if x_candi <= 0:
                            if np.random.rand() > p:
                                S_array[i, col] = x_candi
                                break
                        else:
                            if np.random.rand() > (1.0 - p):
                                S_array[i, col] = x_candi
                                break

            ps_vals = np.apply_along_axis(ps, 1, S_array)
            A_vals = (np.random.rand(ns) < ps_vals).astype(int)

            or_vals = np.zeros(ns)
            for i in range(ns):
                or_vals[i] = or0(S_array[i, :]) if A_vals[i] == 0 else or1(S_array[i, :])

            Y_vals = np.random.normal(loc=or_vals, scale=sd, size=ns)

            colnames = [f"x{j+1}" for j in range(d)]
            S_df = pd.DataFrame(S_array, columns=colnames)
            S_df["a"] = A_vals
            S_df["y"] = Y_vals

            ############################################
            # 4.2 Generate Target Data (T)
            ############################################
            T_array = np.zeros((nt, d))

            for col in range(beta, d):
                T_array[:, col] = np.random.uniform(-math.pi, math.pi, size=nt)

            for col in range(beta):
                for i in range(nt):
                    while True:
                        x_candi = np.random.uniform(-math.pi, math.pi)
                        if x_candi <= 0:
                            if np.random.rand() > (1.0 - p):
                                T_array[i, col] = x_candi
                                break
                        else:
                            if np.random.rand() > p:
                                T_array[i, col] = x_candi
                                break

            T_df = pd.DataFrame(T_array, columns=colnames)

            ############################################
            # 4.3 Generate X_new
            ############################################
            X_new = np.zeros((nnew, d))

            for col in range(beta, d):
                X_new[:, col] = np.random.uniform(-math.pi, math.pi, size=nnew)

            for col in range(beta):
                for i in range(nnew):
                    while True:
                        x_candi = np.random.uniform(-math.pi, math.pi)
                        if x_candi <= 0:
                            if np.random.rand() > (1.0 - p):
                                X_new[i, col] = x_candi
                                break
                        else:
                            if np.random.rand() > p:
                                X_new[i, col] = x_candi
                                break

            true_cate = np.array([or1(X_new[i, :]) - or0(X_new[i, :]) for i in range(nnew)])

            ############################################
            # 4.4 COKE with 2 permutations (exact logic)
            ############################################
            if ns < 2:
                max_power = 0
            else:
                max_power = int(math.ceil(math.log2(10 * (ns / 2.0))))
            lambdas = [(2.0 ** p_) / (10.0) / (ns / 2.0) for p_ in range(max_power + 1)]
            lambdas = np.array(lambdas) if len(lambdas) > 0 else np.array([1e-6])

            idx_perm = np.random.permutation(ns)
            n1 = int(math.ceil(ns / 2.0))
            D1 = S_df.iloc[idx_perm[:n1], :].copy()
            D2 = S_df.iloc[idx_perm[n1:], :].copy()
            S_split = [D1, D2]

            est_single = []

            XT = T_df.values

            perms = [[0, 1], [1, 0]]
            for pm in perms:
                S1 = S_split[pm[0]]
                S2 = S_split[pm[1]]

                lambda_min = lambdas.min()

                X1_or1 = S1[S1["a"] == 1].iloc[:, :d].values
                Y1_or1 = S1[S1["a"] == 1]["y"].values
                X1_or0 = S1[S1["a"] == 0].iloc[:, :d].values
                Y1_or0 = S1[S1["a"] == 0]["y"].values

                K_or0 = Kxx(X1_or0)
                K_or1 = Kxx(X1_or1)

                inv_or0 = npsolve(
                    K_or0 + X1_or0.shape[0] * lambda_min * np.eye(X1_or0.shape[0]),
                    Y1_or0,
                )
                inv_or1 = npsolve(
                    K_or1 + X1_or1.shape[0] * lambda_min * np.eye(X1_or1.shape[0]),
                    Y1_or1,
                )

                X1_full = S1.iloc[:, :d].values
                K_0_1 = Kxy(X1_or0, X1_full)
                K_1_1 = Kxy(X1_or1, X1_full)

                mu0hat = inv_or0 @ K_0_1
                mu1hat = inv_or1 @ K_1_1

                phihat = np.where(
                    S1["a"].values == 1,
                    S1["y"].values - mu0hat,
                    mu1hat - S1["y"].values,
                )

                X2_or1 = S2[S2["a"] == 1].iloc[:, :d].values
                Y2_or1 = S2[S2["a"] == 1]["y"].values
                X2_or0 = S2[S2["a"] == 0].iloc[:, :d].values
                Y2_or0 = S2[S2["a"] == 0]["y"].values

                K2_or1 = Kxx(X2_or1)
                K2_or0 = Kxx(X2_or0)

                inv2_or1 = npsolve(
                    K2_or1 + X2_or1.shape[0] * lambda_min * np.eye(X2_or1.shape[0]),
                    Y2_or1,
                )
                inv2_or0 = npsolve(
                    K2_or0 + X2_or0.shape[0] * lambda_min * np.eye(X2_or0.shape[0]),
                    Y2_or0,
                )

                K_2T_or1 = Kxy(X2_or1, XT)
                K_2T_or0 = Kxy(X2_or0, XT)

                pseudo_T = (inv2_or1 @ K_2T_or1) - (inv2_or0 @ K_2T_or0)

                K1_full = Kxx(X1_full)
                K_1T = Kxy(X1_full, XT)

                esttg_lambdas = []
                for lam in lambdas:
                    inv_ = npsolve(
                        K1_full + X1_full.shape[0] * lam * np.eye(X1_full.shape[0]),
                        phihat,
                    )
                    est_ = inv_ @ K_1T
                    esttg_lambdas.append(est_)
                esttg_lambdas = np.array(esttg_lambdas)

                ssetg_list = np.sum((esttg_lambdas - pseudo_T) ** 2, axis=1)
                bestlambda_tg = lambdas[np.argmin(ssetg_list)]

                K_1Xnew = Kxy(X1_full, X_new)
                inv_best = npsolve(
                    K1_full + X1_full.shape[0] * bestlambda_tg * np.eye(X1_full.shape[0]),
                    phihat,
                )
                est_final_perm = inv_best @ K_1Xnew

                est_single.append(est_final_perm)

            est_all = 0.5 * (est_single[0] + est_single[1])

            mse_perm1[time_] = np.mean((est_single[0] - true_cate) ** 2)
            mse_perm2[time_] = np.mean((est_single[1] - true_cate) ** 2)
            mse_all[time_] = np.mean((est_all - true_cate) ** 2)

        # store rows
        for i_sim in range(sim):
            results_df.loc[len(results_df)] = [B_, "perm1", mse_perm1[i_sim]]
            results_df.loc[len(results_df)] = [B_, "perm2", mse_perm2[i_sim]]
            results_df.loc[len(results_df)] = [B_, "all", mse_all[i_sim]]

    ###################################################
    # 5) Save
    ###################################################
    print("Head of results:\n", results_df.head(), "\n")
    print("Tail of results:\n", results_df.tail(), "\n")

    results_df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
