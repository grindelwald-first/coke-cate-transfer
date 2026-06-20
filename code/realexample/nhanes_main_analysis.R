rm(list = ls())
library(dplyr)

library(splines)
library(glmnet)
source('code/realexample/real_data_main_function.R')

## Read 2015 data as the target data

source('code/realexample/nhanes2015_data_clean.R')

data_T <- final_without_na
data_T$MeanSY <- (data_T$SY1 + data_T$SY2 + data_T$SY3) / 3

var_set <- c("sex", "age", "education", 'race', 'smoke',
             "bmi", "alcohol1", "fat1", "energy1", 'sugar1', 'protein1', "MeanSY")

# Preprocessing:
data_T <- data_T[,var_set]
data_T$smoke <- ifelse(is.na(data_T$smoke), 0, 1)
data_T <- data_T[which(complete.cases(data_T)),]

# The exposure is defined as the indicator of "fat / total energy > 0.4"
# Here, the denominator 9 is used to convert (from col to kg) and unify the unit of two variables.
data_T$Ti <- ifelse((data_T$fat1 / data_T$energy1) > (0.4 / 9), 1, 0)
data_T$fat1 <- NULL
data_T$energy1 <- log(data_T$energy1 + 1)
data_T$alcohol1 <- log(data_T$alcohol1 + 1)
data_T$sugar1 <- log(data_T$sugar1 + 1)


## Read 2001 data as the source data
## Process the data in the same way as above.

source('code/realexample/nhanes2001_data_clean.R')

data_S <- final
data_S$MeanSY <- (data_S$SY1 + data_S$SY2 + data_S$SY3) / 3
data_S <- data_S[,var_set]
data_S$smoke <- ifelse(is.na(data_S$smoke), 0, 1)
data_S <- data_S[which(complete.cases(data_S)),]

data_S$Ti <- ifelse((data_S$fat1 /data_S$energy1) > (0.4 / 9), 1, 0)
data_S$fat1 <- NULL
data_S$energy1 <- log(data_S$energy1 + 1)
data_S$alcohol1 <- log(data_S$alcohol1 + 1)
data_S$sugar1 <- log(data_S$sugar1 + 1)


# Standardization and Rescaling:

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


# Specify the covariates X as well as the source and target data sets.

var_X <- c("sex", "age", 'smoke', "education", "alcohol1")
Y_S <- data_S$MeanSY
A_S <- data_S$Ti
X_S <- as.matrix(data_S[,var_X])

Y_T <- data_T$MeanSY
A_T <- data_T$Ti
X_T <- as.matrix(data_T[,var_X])


## Train and evaluate the methods. 
## Since the methods involve random data splitting procedures, we repeat for 30 times and average over them.
## This procedure is time-consuming and recommended to be paralleled.

# Tables for the mean and standard error of the evaluation metrics.
tab <- c()
tab_se <- c()


for(seed in 1:30){
  set.seed(seed)
  result <- TL_CATE(X_S, A_S, Y_S, X_T, rho = 5)
  
  #file.nm = paste("~/Desktop/Research/Transfer_CATE/nhanes_results/", "result_01to15_rho5_seed", seed, ".Rdata", sep="")
  #save(X_S, A_S, Y_S, X_T, A_T, Y_T, result, file = file.nm)
  #file.nm = paste("~/Desktop/Research/Transfer_CATE/nhanes_results/", "result_01to15_seed", seed, ".Rdata", sep="")
  #load(file = file.nm)
  
  ## Cross-fitted version (corresponding to Table 1 in the main paper): 
  
  est_indep <- result$SR
  est_ra <- result$COKE
  est_dr <- result$DR
  est_acw <- result$ACW
  est_rle <- result$RLearner
  
  ## Data-splitting version without cross-fitting (corresponding to Table S4 in the Supplement Material): 
  
  #est_indep <- result$SR1
  #est_ra <- result$COKE1
  #est_dr <- result$DR1
  #est_acw <- result$ACW1
  #est_rle <- result$RLearner1
  
  ### Generated the empirical gold standard using the target label through the DR-learner (with generalized additive models): 
  
  # Fit nuisance models:
  model.fit <- glmnet(as.matrix(X_T), A_T, lambda = 0, family = 'binomial')
  A_pred <- predict(model.fit, as.matrix(X_T), type = 'response')
  model.fit <- glmnet(as.matrix(X_T), Y_T, lambda = 0)
  Y_pred <- predict(model.fit, as.matrix(X_T))
  cate_pred <- (Y_T - Y_pred) * (A_T - A_pred) / (A_pred * (1 - A_pred))
  
  Ti_adj <- A_T - A_pred
  
  # Fit the empirical gold standard CATE model:
  X_T_spline <- X_T
  for (j in c(2,4,5)) {
    X_T_spline <- cbind(X_T_spline, (X_T[,j])^2)
  }
  model <- lm(Y_T ~ Ti_adj + X_T_spline + Ti_adj:X_T_spline)
  summary(model)
  pred <- X_T_spline %*% as.vector(model$coefficients)[c((ncol(X_T_spline)+3):length(model$coefficients))]
  
  ## Evaluate and save the results:  

  vec <- c(cor(est_indep, pred), cor(est_ra, pred),
           cor(est_dr, pred), cor(est_acw, pred), cor(est_rle, pred),
           cor(pred, est_indep, method = 'spearman'),
           cor(pred, est_ra, method = 'spearman'),
           cor(pred, est_dr, method = 'spearman'),
           cor(pred, est_acw, method = 'spearman'),
           cor(pred, est_rle, method = 'spearman'))
  
  sd_vec <- c(bootstrap_corr_se(est_indep, pred),
              bootstrap_corr_se(est_ra, pred),
              bootstrap_corr_se(est_dr, pred),
              bootstrap_corr_se(est_acw, pred),
              bootstrap_corr_se(est_rle, pred),
              bootstrap_corr_se(est_indep, pred, method = 'spearman'),
              bootstrap_corr_se(est_ra, pred, method = 'spearman'),
              bootstrap_corr_se(est_dr, pred, method = 'spearman'),
              bootstrap_corr_se(est_acw, pred, method = 'spearman'),
              bootstrap_corr_se(est_rle, pred, method = 'spearman'))
  tab <- rbind(tab, vec)
  tab_se <- rbind(tab_se, sd_vec)
  
  print(vec)
  print(seed)
  
}

## Output the results for Table 1 in the main paper and the Table S4 in the Supplement Material,
## (depending on whether cross-fitting is used in the above loop):

colMeans(tab)
colMeans(tab_se)



##### Scatter plots for the empirical gold standard CATE vs COKE (Figure S4) #####

library(ggplot2)
library(patchwork)

# ---------------------------------------------------------
# Linear Alignment between coke_cate and gold_standard.
# ---------------------------------------------------------
# cate_pred and est_ra are outputs from one seed in the previous training and evaluating procedures. 

pred <- cate_pred
model <- lm(est_ra ~ pred)
x <- model$fitted.values
results_df <- cbind(x, est_ra)
colnames(results_df) <- c('gold_standard', 'coke_cate')
results_df <- as.data.frame(results_df)

# ---------------------------------------------------------
# Create the Main Scatter Plot
# ---------------------------------------------------------
p_main <- ggplot(results_df, aes(x = gold_standard, y = coke_cate)) +
  # Scatter points with transparency to highlight the tails
  geom_point(alpha = 0.3, color = "#2c3e50", size = 1.5) +
  
  # 2D Density contours for the joint distribution/copula effect
  geom_density_2d(color = "#3498db", alpha = 0.8, linewidth = 0.8) + 
  
  # 45-degree reference line for perfect concordance
  #geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#e74c3c", linewidth = 1) + 
  
  # Best fit line
  #geom_smooth(method = "lm", se = FALSE, color = "#27ae60", linetype = "dotted", linewidth = 1) + 
  
  theme_minimal(base_size = 14) +
  labs(
    x = expression(paste("Empirical Gold Standard (", hat(s)[0], ")")),
    y = expression(paste("COKE Estimate (", hat(h)[COKE], ")"))
  ) +
  theme(panel.grid.minor = element_blank())


ggsave("CATE_Joint_Distribution_nhanse.pdf", plot = p_main, width = 7, height = 7)




