############################################################
# 0) Imports + Path setup
############################################################
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import math
from numpy.linalg import solve as npsolve
from scipy.spatial.distance import pdist, cdist, squareform

# Add methods folder to import path (avoids needing __init__.py)
THIS_DIR = Path(__file__).resolve().parent
METHODS_DIR = THIS_DIR / "methods"
sys.path.insert(0, str(METHODS_DIR))

from separate_regression import separate_regression
from coke import coke
from dr_cate import dr_cate
from acw_cate import acw_cate

# Output path: repo_root/output/changeB_seed.csv
REPO_ROOT = THIS_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "changeB_seed.csv"


############################################################
# 1) Global parameters
############################################################
B_values = [1, 5, 10, 15, 20]  # vary S_B
R = 2                          # degree of covariate shift (treatment/control)
c_val = 1                      # complexity of m_a(x)
d = 4                          # dimension of X
beta = 1                       # number of covariates that have covariate shift
sd = 0.5                       # sd of Y|A,X
sim = 100                      # simulation times
nnew = 10000                   # number of new samples from T for MSE estimation


############################################################
# 2) Helper Functions: expit, ps, or0, or1
############################################################
def expit(x):
    return 1.0 / (1.0 + np.exp(-x))

def ps(x):
    """
    Propensity score = expit( sum(x[0:4])/8 * R ).
    """
    return expit(np.sum(x[0:4]) / 8.0 * R)

def or0(x):
    """
    or0(x):
      c * [2*(|x[0]| - pi/4)*(|x[0]| >= pi/2) + |x[0]|*(|x[0]|<pi/2)]
      - 0.5*sin(x[0])
    """
    part1 = (
        2.0 * (abs(x[0]) - (math.pi / 4.0)) * (abs(x[0]) >= (math.pi / 2.0))
        + abs(x[0]) * (abs(x[0]) < (math.pi / 2.0))
    )
    return c_val * part1 - 0.5 * math.sin(x[0])

def or1(x):
    """
    or1(x):
      c * [2*(|x[0]| - pi/4)*(|x[0]|>=pi/2) + |x[0]|*(|x[0]|<pi/2)]
      + 0.5*sin(x[0])
    """
    part1 = (
        2.0 * (abs(x[0]) - (math.pi / 4.0)) * (abs(x[0]) >= (math.pi / 2.0))
        + abs(x[0]) * (abs(x[0]) < (math.pi / 2.0))
    )
    return c_val * part1 + 0.5 * math.sin(x[0])


############################################################
# 3) Matern Kernel (kappa=2)
############################################################
rho = 5  # parameter rho in Matern kernel

def matern_kernel_kappa2(u, rho):
    """
    Matern kernel, kappa=2:
      out = 4 * exp(-2 * sqrt(2)*u / rho) / (sqrt(pi)*rho).
    """
    u = u + 1e-100  # to avoid 0-dist
    return 4.0 * np.exp(-2.0 * math.sqrt(2.0) * u / rho) / ((math.pi ** 0.5) * rho)

def Kxx(x):
    """
    Kxx(x) => Matern kernel dist(x, x)
    """
    dist_xx = squareform(pdist(x, metric="euclidean"))
    return matern_kernel_kappa2(dist_xx, rho)

def Kxy(x, y):
    """
    Kxy(x, y) => Matern kernel dist(x, y)
    """
    dist_xy = cdist(x, y, metric="euclidean")
    return matern_kernel_kappa2(dist_xy, rho)


############################################################
# 4) Main Loop (methods are imported)
############################################################
def main():
    results_df = pd.DataFrame(columns=["B", "R", "c", "method", "risk"])

    mse_coke = np.zeros(sim)
    mse_dr   = np.zeros(sim)
    mse_acw  = np.zeros(sim)
    mse_sr   = np.zeros(sim)

    for B_ in B_values:
        # define nt, ns
        nt = int(round((70 * math.sqrt(B_) + 12 * R + 5) * 5))
        ns = nt * 4
        p = 1.0 / (1.0 + B_ ** (1.0 / beta))

        for rep_ in range(sim):
            np.random.seed(rep_ + 1000)
            print(f"Current simulation: {rep_}, B = {B_}")

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

            ps_vals = np.apply_along_axis(ps, 1, S_array)
            A_vals = (np.random.rand(ns) < ps_vals).astype(int)

            or_vals = np.zeros(ns)
            for i in range(ns):
                or_vals[i] = or0(S_array[i, :]) if A_vals[i] == 0 else or1(S_array[i, :])

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
            # Generate X_new for MSE
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

            true_cate = np.array([or1(X_new[i, :]) - or0(X_new[i, :]) for i in range(nnew)])

            ############################################
            # Estimate with 4 Methods
            ############################################
            est_sr_   = separate_regression(S_df, T_df, X_new, Kxx=Kxx, Kxy=Kxy)
            est_coke_ = coke(S_df, T_df, X_new, Kxx=Kxx, Kxy=Kxy)
            est_dr_   = dr_cate(S_df, T_df, X_new, Kxx=Kxx, Kxy=Kxy)
            est_acw_  = acw_cate(S_df, T_df, X_new, Kxx=Kxx, Kxy=Kxy)

            ############################################
            # Compute MSE
            ############################################
            mse_sr[rep_]   = np.mean((est_sr_   - true_cate) ** 2)
            mse_coke[rep_] = np.mean((est_coke_ - true_cate) ** 2)
            mse_dr[rep_]   = np.mean((est_dr_   - true_cate) ** 2)
            mse_acw[rep_]  = np.mean((est_acw_  - true_cate) ** 2)

            # Add rows to results_df
            results_df.loc[len(results_df)] = [B_, R, c_val, "SR",   mse_sr[rep_]]
            results_df.loc[len(results_df)] = [B_, R, c_val, "DR",   mse_dr[rep_]]
            results_df.loc[len(results_df)] = [B_, R, c_val, "ACW",  mse_acw[rep_]]
            results_df.loc[len(results_df)] = [B_, R, c_val, "COKE", mse_coke[rep_]]

    ############################################################
    # 5) Save Results
    ############################################################
    print("Head of results_df:\n", results_df.head())
    print("\nTail of results_df:\n", results_df.tail())

    results_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

if __name__ == "__main__":
    main()
