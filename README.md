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
│   │   ├── acw_cate.py
│   │   └── r_learner.py
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
│       ├── real_data_main_function.R
│       └── realdata_figures.R
├── data/
│   ├── 401k/
│   └── NHANES/
├── output/
│   └── (saved simulation outputs)
└── README.md
````


## Methods

The following Python scripts implement the proposed method and benchmark methods:

* `code/methods/separate_regression.py`: Separate Regression (**SR**).
* `code/methods/coke.py`: Proposed method **COKE**.
* `code/methods/dr_cate.py`: Doubly Robust learner for CATE (**DR-CATE**).
* `code/methods/acw_cate.py`: ACW estimator tailored for CATE estimation (**ACW-CATE**).
* `code/methods/r_learner.py`: R-learner (**R-Learner**).

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

* `code/simulation_figures.R` generates Figure 1 and Figure S1 using the saved simulation outputs.

## Real-World Examples

Two real-world datasets are analyzed in the manuscript: the 401(k) pension dataset and NHANES.

### 1. 401(k) Dataset

The 401(k) analysis uses the `pension` dataset from the R package `hdm`.

Official documentation:

* CRAN package page: https://CRAN.R-project.org/package=hdm
* `hdm::pension` documentation: https://search.r-project.org/CRAN/refmans/hdm/html/pension.html

In R, the documentation can be accessed by `help("pension", package = "hdm")` or `?hdm::pension`.

The data file used in this repository corresponds to the following 12-variable subset: `net_tfa, e401, p401, age, inc, fsize, educ, db, marr, twoearn, pira, hown`.

The subsequent preprocessing, source/target split, and variable selection are implemented in: `code/realexample/401k_main_analysis.R`.

The saved `data/401k/401k_data.rda` may have a different row order from the `hdm::pension` subset. The analysis is unaffected because the real-data analysis does not rely on the original row ordering.

### 2. NHANES Dataset

The NHANES analysis uses NHANES 2001--2002 as the source sample and NHANES 2015--2016 as the target sample.

The official CDC NHANES website is: https://wwwn.cdc.gov/nchs/nhanes/Default.aspx

#### NHANES 2001--2002 Source Sample

Download the following files from the CDC NHANES 2001--2002 pages and place them in:
```text
data/NHANES/2001/
```

Required files:

| File           | Component                      | CDC page                                                                                         | Direct documentation link                                                | Direct data link                                                         |
| -------------- | ------------------------------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `DEMO_B.XPT`   | Demographics                   | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&CycleBeginYear=2001 | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DEMO_B.htm   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DEMO_B.XPT   |
| `BMX_B.XPT`    | Body Measures                  | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2001  | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BMX_B.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BMX_B.XPT    |
| `BPX_B.XPT`    | Blood Pressure                 | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2001  | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BPX_B.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BPX_B.XPT    |
| `DRXTOT_B.XPT` | Dietary Total Nutrient Intakes | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&CycleBeginYear=2001      | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DRXTOT_B.htm | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DRXTOT_B.XPT |
| `SMQ_B.XPT`    | Smoking Questionnaire          | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2001-2002    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/SMQ_B.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/SMQ_B.XPT    |

On each CDC page, click the corresponding `Doc` link for the codebook and the `Data [XPT]` link for the raw data file.

#### NHANES 2015--2016 Target Sample

Download the following files from the CDC NHANES 2015--2016 pages and place them in:
```text
data/NHANES/2015/
```

Required files:

| File           | Component                                  | CDC page                                                                                      | Direct documentation link                                                | Direct data link                                                         |
| -------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `DEMO_I.XPT`   | Demographics                               | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=2015-2016  | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.htm   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.XPT   |
| `BMX_I.XPT`    | Body Measures                              | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2015-2016   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.XPT    |
| `BPX_I.XPT`    | Blood Pressure                             | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2015-2016   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.XPT    |
| `DR1TOT_I.XPT` | Dietary Total Nutrient Intakes, First Day  | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&CycleBeginYear=2015   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1TOT_I.htm | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1TOT_I.XPT |
| `DR2TOT_I.XPT` | Dietary Total Nutrient Intakes, Second Day | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&CycleBeginYear=2015   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR2TOT_I.htm | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR2TOT_I.XPT |
| `SMQ_I.XPT`    | Smoking Questionnaire                      | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2015-2016 | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.XPT    |

On each CDC page, click the corresponding `Doc` link for the codebook and the `Data [XPT]` link for the raw data file.

The NHANES data cleaning scripts are:

```text
code/realexample/nhanes2001_data_clean.R
code/realexample/nhanes2015_data_clean.R
```

The NHANES main analysis script is:

```text
code/realexample/nhanes_main_analysis.R
```

### Real-Data Functions and Figures

* `code/realexample/real_data_main_function.R` contains functions implementing the methods for real-data analyses.
* `code/realexample/401k_main_analysis.R` runs the 401(k) real-data analysis.
* `code/realexample/nhanes_main_analysis.R` runs the NHANES real-data analysis.
* `code/realexample/realdata_figures.R` generates the real-data density-ratio plots reported as Figure S2 and Figure S3.

## Data Redistribution

The repository provides code and instructions for obtaining the raw public datasets from their official sources.

NHANES public-use data files should be downloaded directly from the CDC NHANES website. Users should comply with the NCHS Data User Agreement:

https://www.cdc.gov/nchs/policy/data-user-agreement.html

The 401(k) data can be obtained directly from the R package `hdm` using:

```r
data("pension", package = "hdm")
```

Therefore, the repository does not require redistributing raw NHANES XPT files. The real-data analyses are reproducible from the official source files and the provided cleaning scripts.

## Reproducibility Workflow

### A. Reproducing Simulation Results

1. Run the simulation scripts in `code/`.
2. Simulation results will be saved in `output/`.
3. Run `code/simulation_figures.R` to generate the simulation figures reported in the manuscript.

Due to the computational cost of the simulation studies, these scripts were executed using Google Colab. Running locally may require substantially longer runtimes.

### B. Reproducing Real-World Analyses

For the 401(k) analysis, run `code/realexample/401k_main_analysis.R`.

For the NHANES analysis, first run the data cleaning scripts:
* `code/realexample/nhanes2001_data_clean.R`
* `code/realexample/nhanes2015_data_clean.R`

Then run the main NHANES analysis script: `code/realexample/nhanes_main_analysis.R`.

To reproduce the real-data density-ratio plots, run: `code/realexample/realdata_figures.R`.


## Software Requirements and Packages

### Python: 3.12.13

* numpy: 2.0.2
* pandas: 2.2.2
* scipy: 1.16.3
* scikit-learn: 1.6.1

### R: 4.3.2

* tidyverse: 2.0.0
* latex2exp: 0.9.6
* gridExtra: 2.3
* grid: 4.3.2
* glmnet: 4.1.8
* splines: 4.3.2
* dplyr: 1.1.4
* haven: 2.5.4
* here: 1.0.2
* hdm: **record the installed version used for the 401(k) data source**
