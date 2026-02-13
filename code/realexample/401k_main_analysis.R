rm(list = ls())

library(splines)
library(dplyr)
library(glmnet)
source('code/realexample/real_data_main_function.R')
library(splines)

load("data/401k/401k_data.rda")

# Data preprocessing:
data$net_tfa <- data$net_tfa / 1000
data$age <- data$age / sd(data$age)
data$inc <- data$inc / sd(data$inc)
data$educ <- data$educ / sd(data$educ)
data$fsize <- data$fsize / sd(data$fsize)

# Extract source and target data sets
set_S <- which(data$marr == 1)
set_T <- which(data$marr == 0)
set_var <- c('age', 'inc', 'educ', 'fsize', 'db', 'pira', 'hown')

Y_S <- data$net_tfa[set_S]
A_S <- data$e401[set_S]
X_S <- as.matrix(data[set_S, set_var])

Y_T <- data$net_tfa[set_T]
A_T <- data$e401[set_T]
X_T <- data[set_T, set_var]



## Train and evaluate the methods. 
## Since the methods involve random data splitting procedures, we repeat for 30 times and average over them.
## This procedure is time-consuming and recommended to be paralleled.

# Tables for the mean and standard error of the evaluation metrics.
tab <- c()
tab_se <- c()
for (seedi in 1:30) {
  
  set.seed(seedi)
  result <- TL_CATE(X_S, A_S, Y_S, X_T, rho = 10)
  
  ## Cross-fitted version: 
  
  est_indep <- result$SR
  est_ra <- result$COKE
  est_dr <- result$DR
  est_acw <- result$ACW
  
  ## Data-splitting version without cross-fitting: 
  #est_indep <- result$SR1
  #est_ra <- result$COKE1
  #est_dr <- result$DR1
  #est_acw <- result$ACW1

  ### Generated the empirical gold standard using the target label through the DR-learner (with generalized additive models): 
  
  # Fit nuisance models:
  cv.fit <- cv.glmnet(as.matrix(X_T), A_T, family = 'binomial')
  model.fit <- glmnet(as.matrix(X_T), A_T, lambda = cv.fit$lambda.min, family = 'binomial')
  A_pred <- expit(predict(model.fit, as.matrix(X_T)))
  cv.fit <- cv.glmnet(as.matrix(X_T), Y_T)
  model.fit <- glmnet(as.matrix(X_T), Y_T, lambda = cv.fit$lambda.min)
  
  Y_pred <- predict(model.fit, as.matrix(X_T))
  cate_pred <- (Y_T - Y_pred) * (A_T - A_pred)
  
  # Fit the empirical gold standard CATE model:
  X_T_spline <- as.matrix(X_T)
  for (j in 1:4) {
    X_T_spline <- cbind(X_T_spline, bs(X_T[,j], df = 3))
  }
  model.fit <- glmnet(as.matrix( X_T_spline), as.matrix( cate_pred), alpha = 0, lambda = 0.02)
  cate_pred <- predict(model.fit, as.matrix(X_T_spline))
  
  ## Evaluate and save the results:  
  
  vec <- c(cor(cate_pred, est_indep),
           cor(cate_pred, est_ra),
           cor(cate_pred, est_dr),
           cor(cate_pred, est_acw),
           cor(cate_pred, est_indep, method = 'spearman'),
           cor(cate_pred, est_ra, method = 'spearman'),
           cor(cate_pred, est_dr, method = 'spearman'),
           cor(cate_pred, est_acw, method = 'spearman'))
  
  sd_vec <- c(bootstrap_corr_se(est_indep, cate_pred),
              bootstrap_corr_se(est_ra, cate_pred),
              bootstrap_corr_se(est_dr, cate_pred),
              bootstrap_corr_se(est_acw, cate_pred),
              bootstrap_corr_se(est_indep, cate_pred, method = 'spearman'),
              bootstrap_corr_se(est_ra, cate_pred, method = 'spearman'),
              bootstrap_corr_se(est_dr, cate_pred, method = 'spearman'),
              bootstrap_corr_se(est_acw, cate_pred, method = 'spearman'))
  
  tab <- rbind(tab, vec)
  tab_se <- rbind(tab_se, sd_vec)
  print(seedi)
  
}


colMeans(tab)
colMeans(tab_se)

