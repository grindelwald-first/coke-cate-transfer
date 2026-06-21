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


### 2. NHANES Dataset

The NHANES analysis uses NHANES 2001--2002 as the source sample and NHANES 2015--2016 as the target sample.

The NHANES public-use XPT files used in the analysis are included in this repository under:

```text
data/NHANES/2001/
data/NHANES/2015/
```

We also provide the official CDC NHANES links below so that users can access the corresponding documentation and download the original data files directly from CDC. The official CDC NHANES website is: https://wwwn.cdc.gov/nchs/nhanes/Default.aspx

#### NHANES 2001--2002 Source Sample

The repository includes the following files in `data/NHANES/2001/`:

| File in repository | Component                       | CDC page                                                                                         | Documentation                                                            | Data file                                                                |
| ------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `DEMO_B.xpt`       | Demographics                    | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&CycleBeginYear=2001 | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DEMO_B.htm   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DEMO_B.XPT   |
| `BMX_B.xpt`        | Body Measures                   | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2001  | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BMX_B.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BMX_B.XPT    |
| `BPX_B.xpt`        | Blood Pressure                  | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2001  | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BPX_B.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BPX_B.XPT    |
| `DRXTOT_B.xpt`     | Dietary Total Nutrient Intakes  | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&CycleBeginYear=2001      | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DRXTOT_B.htm | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DRXTOT_B.XPT |
| `PAQ_B.xpt`        | Physical Activity Questionnaire | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2001-2002    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/PAQ_B.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/PAQ_B.XPT    |
| `SMQ_B.xpt`        | Smoking Questionnaire           | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2001-2002    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/SMQ_B.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/SMQ_B.XPT    |

#### NHANES 2015--2016 Target Sample

The repository includes the following files in `data/NHANES/2015/`:

| File in repository | Component                                  | CDC page                                                                                      | Documentation                                                            | Data file                                                                |
| ------------------ | ------------------------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `DEMO_I.xpt`       | Demographics                               | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=2015-2016  | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.htm   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.XPT   |
| `BMX_I.xpt`        | Body Measures                              | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2015-2016   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.XPT    |
| `BPX_I.xpt`        | Blood Pressure                             | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2015-2016   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.XPT    |
| `DR1TOT_I.xpt`     | Dietary Total Nutrient Intakes, First Day  | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&CycleBeginYear=2015   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1TOT_I.htm | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1TOT_I.XPT |
| `DR2TOT_I.xpt`     | Dietary Total Nutrient Intakes, Second Day | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&CycleBeginYear=2015   | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR2TOT_I.htm | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR2TOT_I.XPT |
| `PAQ_I.xpt`        | Physical Activity Questionnaire            | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2015-2016 | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/PAQ_I.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/PAQ_I.XPT    |
| `SLQ_I.xpt`        | Sleep Disorders Questionnaire              | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2015-2016 | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.XPT    |
| `SMQ_I.xpt`        | Smoking Questionnaire                      | https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2015-2016 | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.htm    | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.XPT    |

On each CDC NHANES page, the `Doc` link provides the codebook/documentation and the `Data [XPT]` link provides the raw data file. Users should comply with the NCHS Data User Agreement when using NHANES data.

The NHANES data cleaning scripts are:

```text
code/realexample/nhanes2001_data_clean.R
code/realexample/nhanes2015_data_clean.R
```

The NHANES main analysis script is:

```text
code/realexample/nhanes_main_analysis.R
```

In the NHANES analysis:

* Source sample: NHANES 2001--2002
* Target sample: NHANES 2015--2016
* Outcome: `MeanSY`, the average of three systolic blood pressure measurements
* Treatment: `Ti = 1{fat1 / energy1 > 0.4 / 9}`
* Covariates: `sex`, `age`, `smoke`, `education`, `alcohol1`

The NHANES variables used in the analysis are:

| Analysis variable | Raw variable in 2001--2002           | Raw variable in 2015--2016           | Codebook/file          |
| ----------------- | ------------------------------------ | ------------------------------------ | ---------------------- |
| `sex`             | `RIAGENDR`                           | `RIAGENDR`                           | `DEMO_B`, `DEMO_I`     |
| `age`             | `RIDAGEYR`                           | `RIDAGEYR`                           | `DEMO_B`, `DEMO_I`     |
| `race`            | `RIDRETH1`                           | `RIDRETH3`                           | `DEMO_B`, `DEMO_I`     |
| `education`       | `DMDEDUC2`                           | `DMDEDUC2`                           | `DEMO_B`, `DEMO_I`     |
| `bmi`             | `BMXBMI`                             | `BMXBMI`                             | `BMX_B`, `BMX_I`       |
| `smoke`           | `SMQ040`                             | `SMQ040`                             | `SMQ_B`, `SMQ_I`       |
| `SY1`             | `BPXSY1`                             | `BPXSY1`                             | `BPX_B`, `BPX_I`       |
| `SY2`             | `BPXSY2`                             | `BPXSY2`                             | `BPX_B`, `BPX_I`       |
| `SY3`             | `BPXSY3`                             | `BPXSY3`                             | `BPX_B`, `BPX_I`       |
| `MeanSY`          | constructed from `SY1`, `SY2`, `SY3` | constructed from `SY1`, `SY2`, `SY3` | constructed outcome    |
| `fat1`            | `DRXTTFAT`                           | `DR1TTFAT`                           | `DRXTOT_B`, `DR1TOT_I` |
| `energy1`         | `DRXTKCAL`                           | `DR1TKCAL`                           | `DRXTOT_B`, `DR1TOT_I` |
| `sugar1`          | `DRXTSUGR`                           | `DR1TSUGR`                           | `DRXTOT_B`, `DR1TOT_I` |
| `protein1`        | `DRXTPROT`                           | `DR1TPROT`                           | `DRXTOT_B`, `DR1TOT_I` |
| `alcohol1`        | `DRXTALCO`                           | `DR1TALCO`                           | `DRXTOT_B`, `DR1TOT_I` |
| `fat2`            | not used                             | `DR2TTFAT`                           | `DR2TOT_I`             |
| `Ti`              | constructed from `fat1 / energy1`    | constructed from `fat1 / energy1`    | constructed treatment  |

### Real-Data Functions, Figures and Tables

* `code/realexample/real_data_main_function.R` contains functions implementing the methods for real-data analyses.
* `code/realexample/401k_main_analysis.R` runs the 401(k) real-data analysis and computes the results reported in Table 2, Table S3, and Figure S3. In this script, `tab` stores the Pearson and Spearman correlations over repeated runs, and `tab_se` stores the corresponding bootstrap standard errors. The reported table entries are obtained from `colMeans(tab)` and `colMeans(tab_se)`.
* `code/realexample/nhanes_main_analysis.R` runs the NHANES real-data analysis and computes the results reported in Table 1, Table S4, and Figure S5. In this script, `tab` stores the Pearson and Spearman correlations over repeated runs, and `tab_se` stores the corresponding bootstrap standard errors. The reported table entries are obtained from `colMeans(tab)` and `colMeans(tab_se)`.
* `code/realexample/realdata_figures.R` generates the real-data density-ratio plots reported as Figure S2 and Figure S4.


## Data Redistribution

The repository includes the public-use NHANES XPT files and the 401(k) data file used in the real-data analyses for reproducibility. We also provide official source links and documentation above so that users can obtain the original datasets directly from their official sources.

For NHANES, users should comply with the NCHS Data User Agreement: https://www.cdc.gov/nchs/policy/data-user-agreement.html

For the 401(k) analysis, the data are based on the `pension` dataset from the R package `hdm`. The original dataset can be obtained in R using `data("pension", package = "hdm")`. The corresponding documentation can be accessed by `help("pension", package = "hdm")`.


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
* ggplot2: 3.5.2
* patchwork: 1.3.2
* ranger: 0.17.0
* proxy: 0.4.27
* hdm: 0.3.2
* scales: 1.4.0