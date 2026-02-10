############################################################
# 0) Imports + Path setup
############################################################
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import pdist, cdist, squareform

# Add methods folder to import path (avoids needing __init__.py)
THIS_DIR = Path(__file__).resolve().parent
METHODS_DIR = THIS_DIR / "methods"
sys.path.insert(0, str(METHODS_DIR))

from separate_regression import separate_regression
from coke import coke
from dr_cate import dr_cate
from acw_cate import acw_cate

# Output path: repo_root/output/changeC_seed.csv
REPO_ROOT = THIS_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "changeC_seed.csv"


############################################################
# 1) Global parameters
############################################################
B = 10                 # fixed
R = 2                  # fixed
d = 4                  # dimension
beta = 1               # number of covariates with shift
sd = 0.5               # sd of Y|A,X
sim = 300              # number of simulation replications
nnew = 10000           # number of new samples from T for MSE
c_values = [0.5, 1.0, 1.5, 2.0, 2.5]


############################################################
# 2) Kernel
############################################################
rho = 5

def matern_kernel_kappa2(u, rho):
    """
    Matern kernel, kappa=2:
      out = 4 * exp(-2*sqrt(2)*u/rho) / (sqrt(pi)*rho).
    """
    u = u + 1e-100
    return 4.0 * np.exp(-2.0 * math.sqrt(2.0) * u / rho) / ((math.pi ** 0.5) * rho)

def Kxx(x):
    dist_xx = squareform(pdist(x, metric="euclidean"))
    return matern_kernel_kappa2(dist_xx, rho)

def Kxy(x, y):
    dist_xy = cdist(x, y, metric="euclidean")
    return matern_kernel_kappa2(dist_xy, rho)


############################################################
# 3) Helper functions (ps fixed; or0/or1 depend on c_val)
############################################################
def expit(x):
    return 1.0 / (1.0 + np.exp(-x))

def ps(x):
    # propensity score = expit( sum(x[0:4]) / 8 * R )
    return expit(np.sum(x[0:4]) / 8.0 * R)

def or0(x, c_val):
    part1 = (
        2.0 * (abs(x[0]) - math.pi / 4.0) * (abs(x[0]) >= math.pi / 2.0)
        + abs(x[0]) * (abs(x[0]) < math.pi / 2.0)
    )
    return c_val * part1 - 0.5 * math.sin(x[0])

def or1(x, c_val):
    part1 = (
        2.0 * (abs(x[0]) - math.pi / 4.0) * (abs(x[0]) >= math.pi / 2.0)
        + abs(x[0]) * (abs(x[0]) < math.pi / 2.0)
    )
    return c_val * part1 + 0.5 * math.sin(x[0])


############################################################
# 4) Main simulation loop (vary c)
############################################################
def main():
    results_df = pd.DataFrame(columns=["B", "R", "c", "method", "risk"])

    mse_sr = np.zeros(sim)
    mse_coke = np.zeros(sim)
    mse_dr = np.zeros(sim)
    mse_acw = np.zeros(sim)

    # define nt, ns (fixed because B,R fixed)
    nt = int(round((70 * math.sqrt(B) + 12 * R + 5) * 5))
    ns = nt * 4

    # p controls covariate shift
    p = 1.0 / (1.0 + B ** (1.0 / beta))

    for c_val in c_values:
        for rep_ in range(sim):
            np.random.seed(rep_ + 1000)

            ############################################
            # Generate Source Data (S)
            ############################################
            S_array = np.zeros((ns, d))

            for col in range(beta, d):
                S_array[:, col] = np.random.uniform(-math.pi, math.pi, ns)

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

            # generate A, Y in S
            ps_vals = np.apply_along_axis(ps, 1, S_array)
            A_vals = (np.random.rand(ns) < ps_vals).astype(int)

            or_vals = np.zeros(ns)
            for i in range(ns):
                or_vals[i] = or0(S_array[i, :], c_val) if A_vals[i] == 0 else or1(S_array[i, :], c_val)

            Y_vals = np.random.normal(loc=or_vals, scale=sd, size=ns)

            colnames = [f"x{k+1}" for k in range(d)]
            S_df = pd.DataFrame(S_array, columns=colnames)
            S_df["a"] = A_vals
            S_df["y"] = Y_vals

            ############################################
            # Generate Target Data (T)
            ############################################
            T_array = np.zeros((nt, d))

            for col in range(beta, d):
                T_array[:, col] = np.random.uniform(-math.pi, math.pi, nt)

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
            # Generate X_new (for MSE)
            ############################################
            X_new = np.zeros((nnew, d))

            for col in range(beta, d):
                X_new[:, col] = np.random.uniform(-math.pi, math.pi, nnew)

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

            true_cate = np.array([or1(X_new[i, :], c_val) - or0(X_new[i, :], c_val) for i in range(nnew)])

            ############################################
            # Estimate w/ SR, COKE, DR, ACW
            ############################################
            est_sr_ = separate_regression(S_df, T_df, X_new, Kxx=Kxx, Kxy=Kxy)
            est_coke_ = coke(S_df, T_df, X_new, Kxx=Kxx, Kxy=Kxy)
            est_dr_ = dr_cate(S_df, T_df, X_new, Kxx=Kxx, Kxy=Kxy)
            est_acw_ = acw_cate(S_df, T_df, X_new, Kxx=Kxx, Kxy=Kxy)

            mse_sr[rep_] = np.mean((est_sr_ - true_cate) ** 2)
            mse_coke[rep_] = np.mean((est_coke_ - true_cate) ** 2)
            mse_dr[rep_] = np.mean((est_dr_ - true_cate) ** 2)
            mse_acw[rep_] = np.mean((est_acw_ - true_cate) ** 2)

            results_df.loc[len(results_df)] = [B, R, c_val, "SR", mse_sr[rep_]]
            results_df.loc[len(results_df)] = [B, R, c_val, "DR", mse_dr[rep_]]
            results_df.loc[len(results_df)] = [B, R, c_val, "ACW", mse_acw[rep_]]
            results_df.loc[len(results_df)] = [B, R, c_val, "COKE", mse_coke[rep_]]

    ############################################################
    # Save Results
    ############################################################
    print("Head of results_df:\n", results_df.head(), "\n")
    print("Tail of results_df:\n", results_df.tail(), "\n")

    results_df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
