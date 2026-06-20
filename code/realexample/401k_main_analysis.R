rm(list = ls())

library(splines)
library(dplyr)
library(glmnet)
source('code/realexample/real_data_main_function.R')

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
  result <- TL_CATE(X_S, A_S, Y_S, X_T, rho = 5)
  
  ## Cross-fitted version (corresponding to Table 2 in the main paper): 
  
  est_indep <- result$SR
  est_ra <- result$COKE
  est_dr <- result$DR
  est_acw <- result$ACW
  est_rle <- result$RLearner
  
  ## Data-splitting version without cross-fitting (corresponding to Table S3 in the Supplement Material):
  #est_indep <- result$SR1
  #est_ra <- result$COKE1
  #est_dr <- result$DR1
  #est_acw <- result$ACW1
  #est_rle <- result$RLearner1

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
           cor(cate_pred, est_rle),
           cor(cate_pred, est_indep, method = 'spearman'),
           cor(cate_pred, est_ra, method = 'spearman'),
           cor(cate_pred, est_dr, method = 'spearman'),
           cor(cate_pred, est_acw, method = 'spearman'),
           cor(cate_pred, est_rle, method = 'spearman'))
  
  sd_vec <- c(bootstrap_corr_se(est_indep, cate_pred),
              bootstrap_corr_se(est_ra, cate_pred),
              bootstrap_corr_se(est_dr, cate_pred),
              bootstrap_corr_se(est_acw, cate_pred),
              bootstrap_corr_se(est_rle, cate_pred),
              bootstrap_corr_se(est_indep, cate_pred, method = 'spearman'),
              bootstrap_corr_se(est_ra, cate_pred, method = 'spearman'),
              bootstrap_corr_se(est_dr, cate_pred, method = 'spearman'),
              bootstrap_corr_se(est_acw, cate_pred, method = 'spearman'),
              bootstrap_corr_se(est_rle, cate_pred, method = 'spearman'))
  
  tab <- rbind(tab, vec)
  tab_se <- rbind(tab_se, sd_vec)
  print(seedi)
  
}

## Output the results for Table 2 in the main paper and the Table S3 in the Supplement Material,
## (depending on whether cross-fitting is used in the above loop):

colMeans(tab)
colMeans(tab_se)



##### Scatter plots for the empirical gold standard CATE vs COKE (Figure S3) #####

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


ggsave("CATE_Joint_Distribution_401.pdf", plot = p_main, width = 7, height = 7)




