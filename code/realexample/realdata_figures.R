library(tidyverse)
library(here)
library(haven)
library(ggplot2)
library(ranger)

### 1. 401(k)

load(here("data", "401k", "401k_data.rda"))

# Preprocessing
data$age <- data$age / sd(data$age)
data$inc <- data$inc / sd(data$inc)
data$educ <- data$educ / sd(data$educ)
data$fsize <- data$fsize / sd(data$fsize)

# Source and target split
set_S <- which(data$marr == 1)
set_T <- which(data$marr == 0)

# Covariates
set_var <- c("age", "inc", "educ", "fsize", "db", "pira", "hown")

X_S <- as.data.frame(data[set_S, set_var])
X_T <- as.data.frame(data[set_T, set_var])
X_combined <- rbind(X_S, X_T)
nS <- nrow(X_S)
nT <- nrow(X_T)
labels <- c(rep(0, nS), rep(1, nT))

# Convert matrix to data frame for logistic regression
data_combined <- as.data.frame(X_combined)
data_combined$label <- labels

# Fit a logistic regression model to estimate density ratio
logistic_model <- glm(label ~ ., data = data_combined, family = "binomial")

# Predict probabilities (p_hat) from the logistic model
data_combined$p_hat <- predict(logistic_model, newdata = data_combined, type = "response")

# Compute the density ratio: DR = p_hat / (1 - p_hat)
data_combined$density_ratio <- nS*data_combined$p_hat / (1 - data_combined$p_hat)/nT

data_combined$source <- ifelse(data_combined$label == 0, "Source", "Target")

### Figure S2: density-ratio plot for the 401(k) study

ggplot(data_combined, aes(x = density_ratio, fill = source)) +
  geom_histogram(position = "identity", aes(y = after_stat(density)), alpha = 0.6, bins = 30, color = "black") +
  scale_x_log10() +  # Apply log10 transformation to the x-axis
  labs(
    x = "Density Ratio",
    y = "Density",
    fill = "") +
  theme_minimal() +
  scale_x_log10(limits = c(0.00005, 450)) +  # Set x-axis range to [0.1, 10]
  
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )

data_combined %>% 
  group_by(source) %>% 
  summarize(mean_density_ratio = mean(log10(density_ratio)),
            sd_density_ratio = sd(log10(density_ratio)))

source_sample <- subset(data_combined, label == 0)
weights <- source_sample$density_ratio
ess <- (sum(weights)^2) / sum(weights^2)
ess

rm(list = ls())



### 2. NHANES

read_xpt_any <- function(folder, filename_no_ext) {
  candidates <- file.path(
    folder,
    c(paste0(filename_no_ext, ".xpt"),
      paste0(filename_no_ext, ".XPT"))
  )
  path <- candidates[file.exists(candidates)][1]
  read_xpt(path)
}

source_dir <- here("data", "NHANES", "2001")
target_dir <- here("data", "NHANES", "2015")

df  <- read_xpt_any(target_dir, "DEMO_I")
df2 <- read_xpt_any(target_dir, "BMX_I")
df3 <- read_xpt_any(target_dir, "BPX_I")
df4 <- read_xpt_any(target_dir, "DR1TOT_I")
df5 <- read_xpt_any(target_dir, "DR2TOT_I")
df6 <- read_xpt_any(target_dir, "SMQ_I")

df_T <- merge(df, df2, by = "SEQN", all.x = TRUE)
df_T <- merge(df_T, df3, by = "SEQN", all.x = TRUE)
df_T <- merge(df_T, df4, by = "SEQN", all.x = TRUE)
df_T <- merge(df_T, df5, by = "SEQN", all.x = TRUE)
df_T <- merge(df_T, df6, by = "SEQN", all.x = TRUE)

data_T <- df_T[, c(
  "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2",
  "BMXBMI", "SMQ040",
  "BPXSY1", "BPXDI1",
  "BPXSY2", "BPXDI2",
  "BPXSY3", "BPXDI3",
  "BPXSY4", "BPXDI4",
  "DR1TTFAT", "DR1TKCAL", "DR1TSUGR", "DR1TPROT", "DR1TALCO",
  "DR2TTFAT"
)]

colnames(data_T) <- c(
  "sex", "age", "race", "education",
  "bmi", "smoke",
  "SY1", "DI1",
  "SY2", "DI2",
  "SY3", "DI3",
  "SY4", "DI4",
  "fat1", "energy1", "sugar1", "protein1", "alcohol1",
  "fat2"
)

data_T <- data_T[
  !is.na(data_T$SY1) & !is.na(data_T$DI1) &
    !is.na(data_T$SY2) & !is.na(data_T$DI2) &
    !is.na(data_T$SY3) & !is.na(data_T$DI3) &
    is.na(data_T$SY4) & is.na(data_T$DI4),
]

data_T <- data_T[
  !is.na(data_T$fat1) & !is.na(data_T$fat2),
]

data_T$MeanSY <- (data_T$SY1 + data_T$SY2 + data_T$SY3) / 3

# Clean NHANES 2001 data: source data
df  <- read_xpt_any(source_dir, "DEMO_B")
df2 <- read_xpt_any(source_dir, "BMX_B")
df3 <- read_xpt_any(source_dir, "BPX_B")
df4 <- read_xpt_any(source_dir, "DRXTOT_B")
df6 <- read_xpt_any(source_dir, "SMQ_B")

df_S <- merge(df, df2, by = "SEQN", all.x = TRUE)
df_S <- merge(df_S, df3, by = "SEQN", all.x = TRUE)
df_S <- merge(df_S, df4, by = "SEQN", all.x = TRUE)
df_S <- merge(df_S, df6, by = "SEQN", all.x = TRUE)

data_S <- df_S[, c(
  "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2",
  "BMXBMI", "SMQ040",
  "BPXSY1", "BPXSY2", "BPXSY3",
  "DRXTTFAT", "DRXTKCAL", "DRXTSUGR", "DRXTPROT", "DRXTALCO"
)]

colnames(data_S) <- c(
  "sex", "age", "race", "education",
  "bmi", "smoke",
  "SY1", "SY2", "SY3",
  "fat1", "energy1", "sugar1", "protein1", "alcohol1"
)

data_S$MeanSY <- (data_S$SY1 + data_S$SY2 + data_S$SY3) / 3

var_set <- c(
  "sex", "age", "education", "race", "smoke", "bmi",
  "alcohol1", "fat1", "energy1", "sugar1", "protein1", "MeanSY"
)

data_T <- data_T[, var_set]
data_S <- data_S[, var_set]

data_T$smoke <- ifelse(is.na(data_T$smoke), 0, 1)
data_S$smoke <- ifelse(is.na(data_S$smoke), 0, 1)

data_T <- data_T[complete.cases(data_T), ]
data_S <- data_S[complete.cases(data_S), ]

# Ti = 1 if fat / total energy > 0.4 / 9
data_T$Ti <- ifelse((data_T$fat1 / data_T$energy1) > (0.4 / 9), 1, 0)
data_S$Ti <- ifelse((data_S$fat1 / data_S$energy1) > (0.4 / 9), 1, 0)

data_T$fat1 <- NULL
data_S$fat1 <- NULL

# Log-transform dietary variables
data_T$energy1  <- log(data_T$energy1 + 1)
data_S$energy1  <- log(data_S$energy1 + 1)
data_T$alcohol1 <- log(data_T$alcohol1 + 1)
data_S$alcohol1 <- log(data_S$alcohol1 + 1)
data_T$sugar1   <- log(data_T$sugar1 + 1)
data_S$sugar1   <- log(data_S$sugar1 + 1)

# Standardization and rescaling
data_S$age <- data_S$age / 10
data_T$age <- data_T$age / 10
data_S$education <- data_S$education / 5
data_T$education <- data_T$education / 5
data_S$bmi <- data_S$bmi / sd(data_T$bmi)
data_T$bmi <- data_T$bmi / sd(data_T$bmi)
data_S$energy1 <- data_S$energy1 / sd(data_T$energy1)
data_T$energy1 <- data_T$energy1 / sd(data_T$energy1)
data_S$alcohol1 <- data_S$alcohol1 / sd(data_T$alcohol1)
data_T$alcohol1 <- data_T$alcohol1 / sd(data_T$alcohol1)

var_X <- c("sex", "age", "smoke", "education", "alcohol1")

Y_S <- data_S$MeanSY
A_S <- data_S$Ti
X_S <- as.matrix(data_S[, var_X])

Y_T <- data_T$MeanSY
A_T <- data_T$Ti
X_T <- as.matrix(data_T[, var_X])

# Density-ratio plot: target 2015 vs source 2001
X_combined <- rbind(X_S, X_T)

nS <- nrow(X_S)
nT <- nrow(X_T)

labels <- c(rep(0, nS), rep(1, nT))

data_combined <- as.data.frame(X_combined)
data_combined$label <- labels
# Treat binary variables as factors in the density-ratio model.
data_combined$sex_f <- factor(data_combined$sex)
data_combined$smoke_f <- factor(data_combined$smoke)

# random forest
set.seed(123)
make_folds <- function(n, K = 2) {
  sample(rep(1:K, length.out = n))
}

estimate_dr_rf_crossfit <- function(X_S, X_T, K = 2, num.trees = 500,
                                    min.node.size = 20, truncate_at = Inf) {
  nS <- nrow(X_S)
  nT <- nrow(X_T)
  
  dat <- rbind(
    data.frame(X_S, label = 0, source = "Source: NHANES 2001"),
    data.frame(X_T, label = 1, source = "Target: NHANES 2015")
  )
  
  dat$sex <- factor(dat$sex)
  dat$smoke <- factor(dat$smoke)
  
  fold_S <- make_folds(nS, K)
  fold_T <- make_folds(nT, K)
  
  fold <- c(fold_S, fold_T)
  
  p_hat <- rep(NA, nrow(dat))
  
  for (k in 1:K) {
    train_id <- which(fold != k)
    test_id  <- which(fold == k)
    
    train_dat <- dat[train_id, ]
    test_dat  <- dat[test_id, ]
    
    fit <- ranger(
      factor(label) ~ sex + age + smoke + education + alcohol1,
      data = train_dat,
      probability = TRUE,
      num.trees = num.trees,
      min.node.size = min.node.size
    )
    
    pred <- predict(fit, data = test_dat)$predictions[, "1"]
    p_hat[test_id] <- pred
  }
  
  p_hat <- pmin(pmax(p_hat, 1e-6), 1 - 1e-6)
  
  dat$p_hat <- p_hat
  dat$density_ratio <- (nS / nT) * p_hat / (1 - p_hat)
  
  if (is.finite(truncate_at)) {
    dat$density_ratio <- pmin(dat$density_ratio, truncate_at)
  }
  
  return(dat)
}

dr_rf_cf <- estimate_dr_rf_crossfit(
  X_S = as.data.frame(X_S),
  X_T = as.data.frame(X_T),
  K = 2,
  num.trees = 500,
  min.node.size = 20,
  truncate_at = Inf
)

### Figure S4: density-ratio plot in the NHANES study

p_rf_cf <- ggplot(dr_rf_cf, aes(x = density_ratio, fill = source)) +
  geom_histogram(
    position = "identity",
    aes(y = after_stat(density)),
    alpha = 0.6,
    binwidth = 0.07,
    color = "black"
  ) +
  scale_x_log10() +
  labs(
    x = "Estimated Density Ratio",
    y = "Density",
    fill = ""
  ) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    legend.position = "bottom"
  )

print(p_rf_cf)

# effective sample size
source_sample <- subset(dr_rf_cf, label == 0)
weights <- source_sample$density_ratio

ess <- (sum(weights)^2) / sum(weights^2)

cat("Source sample size nS =", nrow(source_sample), "\n")
cat("Effective sample size ESS =", ess, "\n")
cat("ESS / nS =", ess / nrow(source_sample), "\n")