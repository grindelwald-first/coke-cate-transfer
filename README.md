# Transfer Learning of CATE with Kernel Ridge Regression

This repository contains the reproducibility materials accompanying the manuscript  
**“Transfer Learning of CATE with Kernel Ridge Regression.”**

The repository includes implementations of the proposed method **COKE**, benchmark methods, simulation studies, and real-world data analyses (401(k) and NHANES). Numerical results and figures reported in the manuscript can be reproduced using the materials provided here.


## Repository Structure

```text
coke-cate-transfer/
├── code/
│   ├── methods/
│   │   ├── separate_regression.py
│   │   ├── coke.py
│   │   ├── dr_cate.py
│   │   └── acw_cate.py
│   ├── changeB.py
│   ├── changeR.py
│   ├── changeC.py
│   ├── changeN.py
│   ├── changeB_2dim.py
│   ├── changeB_CF.py
│   ├── simulation_figures.R
│   └── realexample/
│       ├── 401k_main_analysis.R
│       ├── nhanes2001_data_clean.R
│       ├── nhanes2015_data_clean.R
│       ├── nhanes_main_analysis.R
│       └── real_data_main_function.R
├── data/
│   ├── 401k/
│   └── NHANES/
├── output/
│   └── (saved simulation outputs)
├── README.md
└── LICENSE
````


## Methods

The following Python scripts implement the proposed method and benchmark methods:

* `code/methods/separate_regression.py`: Separate Regression (**SR**).
* `code/methods/coke.py`: Proposed method **COKE**.
* `code/methods/dr_cate.py`: Doubly Robust learner for CATE (**DR-CATE**).
* `code/methods/acw_cate.py`: ACW estimator tailored for CATE estimation (**ACW-CATE**).

Each file contains a self-contained implementation of the corresponding method.


## Simulation Studies

Simulation studies reported in the paper are driven by the following Python scripts located in `code/`:

* `changeB.py`: Vary $S_B$ with other parameters fixed under $q = 1$.
* `changeR.py`: Vary $S_R$ with other parameters fixed.
* `changeC.py`: Vary $c$ with other parameters fixed.
* `changeN.py`: Vary $n_{\mathcal{T}} = n/4$ with other parameters fixed.
* `changeB_2dim.py`: Vary $S_B$ with other parameters fixed under $q = 2$.
* `changeB_CF.py`: Compare the cross-fitting version of **COKE** with the original Algorithm 3.

Simulation outputs are saved to the `output/` directory.

### Simulation Figures

* `code/simulation_figures.R` generates Figure 1 and Figure A1 using the saved simulation outputs.


## Real-World Examples

Two real-world datasets are analyzed in the manuscript:

### 1. 401(k) Dataset

* Data located in: `data/401k/`
* Main analysis script: `code/realexample/401k_main_analysis.R`

### 2. NHANES Dataset

* Data located in: `data/NHANES/`
* Data cleaning scripts:
  * `code/realexample/nhanes2001_data_clean.R`
  * `code/realexample/nhanes2015_data_clean.R`
* Main analysis script:
  * `code/realexample/nhanes_main_analysis.R`

### Real-Data Functions

* `code/realexample/real_data_main_function.R` contains functions implementing the four methods for real-data analyses.


## Reproducibility Workflow

### A. Reproducing Simulation Results

1. Run the simulation scripts in `code/`.
2. Simulation results will be saved in `output/`.
3. Run `code/simulation_figures.R` to generate the simulation figures reported in the manuscript.

Due to the computational cost of the simulation studies, these scripts were executed using Google Colab. Running locally may require substantially longer runtimes.

### B. Reproducing Real-World Analyses

1. For NHANES, run the data cleaning scripts:
   * `code/realexample/nhanes2001_data_clean.R`
   * `code/realexample/nhanes2015_data_clean.R`
2. Run the main analysis scripts:
   * `code/realexample/401k_main_analysis.R`
   * `code/realexample/nhanes_main_analysis.R`


## Software Requirements and Packages

### Python 3.9

* numpy
* math
* pandas
* scipy
* scikit-learn

### R version 4.3.2

* tidyverse
* latex2exp
* gridExtra
* grid
* glmnet
* splines
* dplyr
* haven
* here


## License

This project is released under the MIT License.
