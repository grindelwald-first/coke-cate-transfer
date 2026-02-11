# Transfer Learning of CATE with Kernel Ridge Regression

This repository contains the reproducibility materials accompanying the manuscript  
**“Transfer Learning of CATE with Kernel Ridge Regression.”**

The repository provides implementations of the proposed method COKE, benchmark methods, simulation studies, real-world data analyses, and scripts for generating the figures reported in the paper.

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
│   └── simulation_figures.R
├── output/
│   └── (saved simulation outputs)
├── README.md
└── LICENSE
````


## Methods

The following Python scripts implement the proposed method and competing estimators:

* `code/methods/separate_regression.py`: Separate Regression (**SR**).
* `code/methods/coke.py`: Proposed method **COKE**.
* `code/methods/dr_cate.py`: Doubly Robust learner for CATE (**DR-CATE**).
* `code/methods/acw_cate.py`: ACW estimator tailored for CATE estimation (**ACW-CATE**).

Each file contains a self-contained implementation of the corresponding method.


## Simulation Scripts

Simulation studies reported in the paper are driven by the following Python scripts located in `code/`:

* `changeB.py`: Vary $S_B$ with other parameters fixed under $q = 1$.
* `changeR.py`: Vary $S_R$ with other parameters fixed.
* `changeC.py`: Vary $c$ with other parameters fixed.
* `changeN.py`: Vary $n_{\mathcal{T}} = n/4$ with other parameters fixed.
* `changeB_2dim.py`: Vary $S_B$ with other parameters fixed under $q = 2$.
* `changeB_CF.py`: Compare the cross-fitting version of **COKE** with the original Algorithm 3.

Each script imports method implementations from `code/methods/` and generates numerical outputs for the corresponding simulation setting.


## Output

The `output/` directory contains saved numerical results from all simulation settings.
These files are used as inputs for figure generation and allow reproduction of the figures without rerunning the full simulation studies.


## Plotting

* `code/simulation_figures.R`:
  Generates **Figure 1** and **Figure A1** in the paper using the saved simulation outputs.


## Reproducibility Workflow

1. Run the simulation scripts in `code/` to generate numerical outputs.
2. Simulation results are saved to the `output/` directory.
3. Run `code/simulation_figures.R` to generate the figures reported in the manuscript.

Due to the computational cost of the simulation studies, the Python scripts were executed using Google Colab.
The code can be run locally on a standard desktop machine, though substantially longer runtimes should be expected.


## Software Requirements and Packages

### Python ≥ 3.9

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


## License

This project is released under the MIT License.
