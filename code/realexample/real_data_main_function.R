############################################################
####### Main function for running the methods in R ########
############################################################

TL_CATE <- function(X_S, A_S, Y_S, X_T, rho = 5){
  
  #----------------------
  
  ### Setup
  ns = length(A_S) # number of samples in S
  nt = nrow(X_T) # number of samples in T
  d = ncol(X_S) # dimension of X
  k = 10 # smallest lambda = 1/nrow(training)/k. also for pseudo-labeling
  
  kapp = 2
  
  Kxx = function(x) matern.kernel_kappa2(u = as.matrix(dist(x)), rho = rho)
  Kxy = function(x, y) matern.kernel_kappa2(u = proxy::dist(x, y), rho = rho)
  
  lambdas = 2^(0:ceiling(log2(k*(ns/2))))/k/(ns/2) 
  n_lambdas = length(lambdas)
  lambdatilde = 1/k/(ns/2)
  
  
  # source data S
  S = data.frame(X_S)
  colnames(S) = paste0("x", 1:d)
  S$a = A_S
  S$y = Y_S
  
  # target data T
  T = data.frame(X_T)
  colnames(T) = paste0("x", 1:d)
  
  # new data drawn from T for excess risk estimation
  X_new = X_T
  
  ### Methods
  ## (Only Method without cross-fitting:)
  ## 1. independent regression (benchmark)
  # Split S into 2 parts
  
  split_idx = sample(1:ns, ceiling(ns/2))
  S1 = S[split_idx, ]
  S2 = S[-split_idx, ]
  X1or1 = as.matrix(S1[S1$a == 1, 1:d])
  X1or0 = as.matrix(S1[S1$a == 0, 1:d])
  Y1or1 = S1$y[S1$a == 1]
  Y1or0 = S1$y[S1$a == 0]
  n1or1 = length(Y1or1)
  n1or0 = length(Y1or0)
  X2or1 = as.matrix(S2[S2$a == 1, 1:d])
  X2or0 = as.matrix(S2[S2$a == 0, 1:d])
  Y2or1 = S2$y[S2$a == 1]
  Y2or0 = S2$y[S2$a == 0]
  # Compute kernel matrices
  Kmat1or1 = Kxx(X1or1)
  Kmat1or0 = Kxx(X1or0)
  
  # Choose lambda for f0 by cross???validation
  estmu0_lambdas = sapply(lambdas, function(lambda) {
    ((Y1or0 %*% solve(Kmat1or0 + n1or0 * lambda * diag(n1or0))) %*% Kxy(X1or0, X2or0))[,]
  })
  ssemu0_lambdas = apply(estmu0_lambdas, 2, function(est) {
    sum((est - Y2or0)^2)
  })
  bestlambda_mu0 = lambdas[which.min(ssemu0_lambdas)]
  
  # Choose lambda for f1
  estmu1_lambdas = sapply(lambdas, function(lambda) {
    ((Y1or1 %*% solve(Kmat1or1 + n1or1 * lambda * diag(n1or1))) %*% Kxy(X1or1, X2or1))[,]
  })
  ssemu1_lambdas = apply(estmu1_lambdas, 2, function(est) {
    sum((est - Y2or1)^2)
  })
  bestlambda_mu1 = lambdas[which.min(ssemu1_lambdas)]
  
  # Final estimation on X_new
  est_mu1_new = ((Y1or1 %*% solve(Kmat1or1 + n1or1 * bestlambda_mu1 * diag(n1or1))) %*% Kxy(X1or1, X_new))[,]
  est_mu0_new = ((Y1or0 %*% solve(Kmat1or0 + n1or0 * bestlambda_mu0 * diag(n1or0))) %*% Kxy(X1or0, X_new))[,]
  est_sr = est_mu1_new - est_mu0_new

  
  ## (Methods with cross-fitting:)
  ## 2. RA-Learner (proposed method)
  # split S into 2 parts for the cross-fitting methods
  
  indices = sample(1:ns)
  n1 = ceiling(ns/2)
  D1 = S[indices[1:n1], ]
  D2 = S[indices[(n1 + 1):ns], ]
  S_split = list(D1, D2)
  
  # define the 2 cyclic permutations
  perms = list(c(1, 2), c(2, 1))
  est_coke_list = est_dr_list = est_acw_list = list()
  XT = as.matrix(T)
  
  
  for (i in seq_along(perms)) {
    perm = perms[[i]]
    
    # nuisance parameter estimation on S1 (using small lambdas)
    S1 = S_split[[perm[1]]]
    X1_or1 = as.matrix(S1[S1$a == 1,1:d])
    X1_or0 = as.matrix(S1[S1$a == 0,1:d])
    
    # create pseudo-outcome on S1
    X1 = as.matrix(S1[,1:d])
    S1$mu0hat = (S1$y[S1$a == 0] %*% solve(Kxx(X1_or0) + nrow(X1_or0) * min(lambdas) * diag(nrow(X1_or0))) %*% Kxy(X1_or0, X1))[,]
    S1$mu1hat = (S1$y[S1$a == 1] %*% solve(Kxx(X1_or1) + nrow(X1_or1) * min(lambdas) * diag(nrow(X1_or1))) %*% Kxy(X1_or1, X1))[,]
    S1 = S1 %>% 
      mutate(phihat = (a == 1) * (y - mu0hat) + (a == 0) * (mu1hat - y))
    
    # target parameter estimation using:
    ##S1 (candidate models)
    ##S2 (imputation model using a small lambda)
    ##T (pseudo label)
    S2 = S_split[[perm[2]]]
    Kmat1 = Kxx(X1)
    X2_or1 = as.matrix(S2[S2$a == 1,1:d])
    X2_or0 = as.matrix(S2[S2$a == 0,1:d])
    
    # create pseudo label on XT
    pseudo = (S2$y[S2$a == 1] %*% solve(Kxx(X2_or1) + nrow(X2_or1) * min(lambdas) * diag(nrow(X2_or1))) %*% Kxy(X2_or1, XT))[,] -
      (S2$y[S2$a == 0] %*% solve(Kxx(X2_or0) + nrow(X2_or0) * min(lambdas) * diag(nrow(X2_or0))) %*% Kxy(X2_or0, XT))[,]
    esttg_lambdas = sapply(lambdas, function(lambda) {
      ((S1$phihat %*% solve(Kmat1 + nrow(Kmat1) * lambda * diag(nrow(Kmat1)))) %*% Kxy(X1, XT))[,]
    })
    ssetg_lambdas = apply(esttg_lambdas, 2, function(est) {
      sum((est - pseudo)^2)
    })
    bestlambda_tg = lambdas[which.min(ssetg_lambdas)]
    
    # final estimation on X_new
    est_coke_list[[i]] = (S1$phihat %*% solve(Kmat1 + nrow(Kmat1) * bestlambda_tg * diag(nrow(Kmat1))) %*% Kxy(X1, X_new))[,]
  }
  
  # return the average estimate over the three permutations
  est_coke = Reduce("+", est_coke_list) / length(est_coke_list)  
  
  ## 3. DR-Learner (benchmark)
  variables = paste0("x", 1:d, collapse = " + ")
  
  for (i in seq_along(perms)) {
    perm = perms[[i]]
    
    # nuisance parameter estimation on S_nc
    S_nc = S_split[[perm[1]]]
    partS_nc = sample(1:nrow(S_nc), ceiling(nrow(S_nc)/2))
    S_nc1 = S_nc[partS_nc,]
    S_nc2 = S_nc[-partS_nc,]
    X_nc1 = as.matrix(S_nc1[,1:d])
    X_nc2 = as.matrix(S_nc2[,1:d])
    X_nc1or1 = as.matrix(S_nc1[S_nc1$a == 1,1:d])
    X_nc2or1 = as.matrix(S_nc2[S_nc2$a == 1,1:d])
    X_nc1or0 = as.matrix(S_nc1[S_nc1$a == 0,1:d])
    X_nc2or0 = as.matrix(S_nc2[S_nc2$a == 0,1:d])
    Y_nc1or1 = S_nc1$y[S_nc1$a == 1]
    Y_nc2or1 = S_nc2$y[S_nc2$a == 1]
    Y_nc1or0 = S_nc1$y[S_nc1$a == 0]
    Y_nc2or0 = S_nc2$y[S_nc2$a == 0]
    n_nc1or1 = length(Y_nc1or1)
    n_nc1or0 = length(Y_nc1or0)
    X_ncor1 = as.matrix(S_nc[S_nc$a == 1,1:d])
    X_ncor0 = as.matrix(S_nc[S_nc$a == 0,1:d])
    Kmat_ncor1 = Kxx(X_ncor1)
    Kmat_ncor0 = Kxx(X_ncor0)
    
    # propensity score estimation
    formula = as.formula(paste("a ~", variables))
    model_ps = glm(formula, data = S_nc, family = binomial)
    
    # outcome regression estimation
    # f0
    estmu0_lambdas = sapply(lambdas, function(lambda) {
      ((Y_nc1or0 %*% solve(Kxx(X_nc1or0) + n_nc1or0 * lambda * diag(n_nc1or0))) %*% Kxy(X_nc1or0, X_nc2or0))[,]
    })
    ssemu0_lambdas = apply(estmu0_lambdas, 2, function(est) {
      sum((est - Y_nc2or0)^2)
    })
    bestlambda_mu0 = lambdas[which.min(ssemu0_lambdas)]
    # f1
    estmu1_lambdas = sapply(lambdas, function(lambda) {
      ((Y_nc1or1 %*% solve(Kxx(X_nc1or1) + n_nc1or1 * lambda * diag(n_nc1or1))) %*% Kxy(X_nc1or1, X_nc2or1))[,]
    })
    ssemu1_lambdas = apply(estmu1_lambdas, 2, function(est) {
      sum((est - Y_nc2or1)^2)
    })
    bestlambda_mu1 = lambdas[which.min(ssemu1_lambdas)]
    
    # target parameter estimation using:
    ##S_tg1 (training data)
    ##S_tg2 (test data)
    S_tg = S_split[[perm[2]]]
    partS_tg = sample(1:nrow(S_tg), ceiling(nrow(S_tg)/2))
    S_tg1 = S_tg[partS_tg,]
    S_tg2 = S_tg[-partS_tg,]
    X_tg1 = as.matrix(S_tg1[,1:d])
    X_tg2 = as.matrix(S_tg2[,1:d])
    Kmat_tg1 = Kxx(X_tg1)
    K_tg1tg2 = Kxy(X_tg1, X_tg2)
    
    # pseudo-outcome phihat
    S_tg1$pihat = predict(model_ps, newdata = S_tg1, type = "response")
    S_tg1$mu0hat = (S_nc$y[S_nc$a == 0] %*% solve(Kmat_ncor0 + nrow(X_ncor0) * bestlambda_mu0 * diag(nrow(X_ncor0))) %*% Kxy(X_ncor0, X_tg1))[,]
    S_tg1$mu1hat = (S_nc$y[S_nc$a == 1] %*% solve(Kmat_ncor1 + nrow(X_ncor1) * bestlambda_mu1 * diag(nrow(X_ncor1))) %*% Kxy(X_ncor1, X_tg1))[,]
    S_tg1 = S_tg1 %>% 
      mutate(phihat = mu1hat - mu0hat + a * (y - mu1hat) / pihat - (1 - a) * (y - mu0hat) / (1 - pihat))
    
    S_tg2$pihat = predict(model_ps, newdata = S_tg2, type = "response")
    S_tg2$mu0hat = (S_nc$y[S_nc$a == 0] %*% solve(Kmat_ncor0 + nrow(X_ncor0) * bestlambda_mu0 * diag(nrow(X_ncor0))) %*% Kxy(X_ncor0, X_tg2))[,]
    S_tg2$mu1hat = (S_nc$y[S_nc$a == 1] %*% solve(Kmat_ncor1 + nrow(X_ncor1) * bestlambda_mu1 * diag(nrow(X_ncor1))) %*% Kxy(X_ncor1, X_tg2))[,]
    S_tg2 = S_tg2 %>% 
      mutate(phihat = mu1hat - mu0hat + a * (y - mu1hat) / pihat - (1 - a) * (y - mu0hat) / (1 - pihat))
    
    esttg_lambdas = sapply(lambdas, function(lambda) {
      ((S_tg1$phihat %*% solve(Kmat_tg1 + nrow(Kmat_tg1) * lambda * diag(nrow(Kmat_tg1)))) %*% K_tg1tg2)[,]
    })
    ssetg_lambdas = apply(esttg_lambdas, 2, function(est) {
      sum((est - S_tg2$phihat)^2)
    })
    bestlambda_tg = lambdas[which.min(ssetg_lambdas)]
    
    # final estimation on X_new
    est_perm = (S_tg1$phihat %*% solve(Kmat_tg1 + nrow(Kmat_tg1) * bestlambda_tg * diag(nrow(Kmat_tg1))) %*% Kxy(X_tg1, X_new))[,]
    est_dr_list[[i]] = est_perm
    
    
    ### 4. ACW estimator
    perm = perms[[i]]
    
    # nuisance parameter estimation
    S_nc = S_split[[perm[1]]]
    partS_nc = sample(1:nrow(S_nc), ceiling(nrow(S_nc)/2))
    S_nc1 = S_nc[partS_nc,]
    S_nc2 = S_nc[-partS_nc,]
    X_nc1 = as.matrix(S_nc1[,1:d])
    X_nc1or1 = as.matrix(S_nc1[S_nc1$a == 1,1:d])
    X_nc2or1 = as.matrix(S_nc2[S_nc2$a == 1,1:d])
    X_nc1or0 = as.matrix(S_nc1[S_nc1$a == 0,1:d])
    X_nc2or0 = as.matrix(S_nc2[S_nc2$a == 0,1:d])
    Y_nc1or1 = S_nc1$y[S_nc1$a == 1]
    Y_nc2or1 = S_nc2$y[S_nc2$a == 1]
    Y_nc1or0 = S_nc1$y[S_nc1$a == 0]
    Y_nc2or0 = S_nc2$y[S_nc2$a == 0]
    n_nc1or1 = length(Y_nc1or1)
    n_nc1or0 = length(Y_nc1or0)
    X_ncor1 = as.matrix(S_nc[S_nc$a == 1,1:d])
    X_ncor0 = as.matrix(S_nc[S_nc$a == 0,1:d])
    Kmat_ncor1 = Kxx(X_ncor1)
    Kmat_ncor0 = Kxx(X_ncor0)
    
    # propensity score estimation
    variables = paste0("x", 1:d, collapse = " + ")
    formula = as.formula(paste("a ~", variables))
    model_ps = glm(formula, data = S_nc, family = binomial)
    
    # outcome regression estimation
    # f0
    estmu0_lambdas = sapply(lambdas, function(lambda) {
      ((Y_nc1or0 %*% solve(Kxx(X_nc1or0) + n_nc1or0 * lambda * diag(n_nc1or0))) %*% Kxy(X_nc1or0, X_nc2or0))[,]
    })
    ssemu0_lambdas = apply(estmu0_lambdas, 2, function(est) {
      sum((est - Y_nc2or0)^2)
    })
    bestlambda_mu0 = lambdas[which.min(ssemu0_lambdas)]
    # f1
    estmu1_lambdas = sapply(lambdas, function(lambda) {
      ((Y_nc1or1 %*% solve(Kxx(X_nc1or1) + n_nc1or1 * lambda * diag(n_nc1or1))) %*% Kxy(X_nc1or1, X_nc2or1))[,]
    })
    ssemu1_lambdas = apply(estmu1_lambdas, 2, function(est) {
      sum((est - Y_nc2or1)^2)
    })
    bestlambda_mu1 = lambdas[which.min(ssemu1_lambdas)]
    
    # density ratio estimation
    Stemp = S_nc[, 1:d]
    Ttemp = T
    Stemp$k = 0
    Ttemp$k = 1
    all = rbind(Stemp, Ttemp)
    formula = as.formula(paste("k ~", variables))
    model_dens = glm(formula, data = all, family = binomial)
    
    S_tg = S_split[[perm[2]]]
    weight = predict(model_dens, newdata = S_tg, type = "response") /
      (1 - predict(model_dens, newdata = S_tg, type = "response"))*ns/nt
    weight = weight * nrow(S_tg)/sum(weight)
    weight = pmin(weight, 20) # truncation
    weight = weight * nrow(S_tg)/sum(weight)
    
    # pseudo-outcome phihat
    X_tg = as.matrix(S_tg[,1:d])
    S_tg$what = weight
    S_tg$pihat = predict(model_ps, newdata = S_tg, type = "response")
    S_tg$mu0hat = (S_nc$y[S_nc$a == 0] %*% solve(Kmat_ncor0 + nrow(X_ncor0) * bestlambda_mu0 * diag(nrow(X_ncor0))) %*% Kxy(X_ncor0, X_tg))[,]
    S_tg$mu1hat = (S_nc$y[S_nc$a == 1] %*% solve(Kmat_ncor1 + nrow(X_ncor1) * bestlambda_mu1 * diag(nrow(X_ncor1))) %*% Kxy(X_ncor1, X_tg))[,]
    S_tg = S_tg %>% 
      mutate(phihat = what * (nrow(S_tg) + nt) / nrow(S_tg) * (a * (y - mu1hat) / pihat - (1 - a) * (y - mu0hat) / (1 - pihat))) %>% 
      select(-a, -y, -what, -pihat, -mu0hat, -mu1hat)
    partS_tg = sample(1:nrow(S_tg), ceiling(nrow(S_tg)/2))
    S_tg1 = S_tg[partS_tg,]
    S_tg2 = S_tg[-partS_tg,]
    
    T_hat = T
    T_hat$mu0hat = (S_nc$y[S_nc$a == 0] %*% solve(Kmat_ncor0 + nrow(X_ncor0) * bestlambda_mu0 * diag(nrow(X_ncor0))) %*% Kxy(X_ncor0, XT))[,]
    T_hat$mu1hat = (S_nc$y[S_nc$a == 1] %*% solve(Kmat_ncor1 + nrow(X_ncor1) * bestlambda_mu1 * diag(nrow(X_ncor1))) %*% Kxy(X_ncor1, XT))[,]
    T_hat = T_hat %>% 
      mutate(phihat = (nrow(S_tg) + nt) / nt * (mu1hat - mu0hat)) %>% 
      select(-mu0hat, -mu1hat)
    partT = sample(1:nt, ceiling(nt/2))
    T1 = T_hat[partT,]
    T2 = T_hat[-partT,]
    
    # target parameter estimation
    ## S_tg1, T1: training data
    ## S_tg2, T2: test data
    mix1 = rbind(S_tg1, T1)
    mix2 = rbind(S_tg2, T2)
    X_mix1 = as.matrix(mix1[,1:d])
    X_mix2 = as.matrix(mix2[,1:d])
    Kmat_mix1 = Kxx(X_mix1)
    
    estmix_lambdas = sapply(lambdas, function(lambda) {
      ((mix1$phihat %*% solve(Kmat_mix1 + nrow(mix1) * lambda * diag(nrow(mix1)))) %*% Kxy(X_mix1, X_mix2))[,]
    })
    ssemix_lambdas = apply(estmix_lambdas, 2, function(est) {
      sum((est - mix2$phihat)^2)
    })
    bestlambda_mix = lambdas[which.min(ssemix_lambdas)]
    
    # final estimation on X_new
    est_perm = (mix1$phihat %*% solve(Kmat_mix1 + nrow(mix1) * bestlambda_mix * diag(nrow(mix1))) %*% Kxy(X_mix1, X_new))[,]
    est_acw_list[[i]] = est_perm
  }
  
  # return the average estimate over the three permutations
  est_dr = Reduce("+", est_dr_list) / length(est_dr_list)
  est_acw = Reduce("+", est_acw_list) / length(est_acw_list)  
  
  return(list(COKE = est_coke, DR = est_dr, SR = est_sr, ACW = est_acw,
              COKE1 = est_coke_list[[1]], DR1 = est_dr_list[[1]], SR1 = est_sr, ACW1 = est_acw_list[[1]]))
  
}

#################################
####### Kernel functuon ########
#################################

matern.kernel_kappa2 = function(u, rho) 
{
  u <- u + 1e-100
  out <- 4 * exp(-2 * sqrt(2) * u/rho)/((pi^0.5) * rho)
  out
}

#################################
####### Expit function ########
#################################

expit <- function(x){
  exp(x) / (1 + exp(x))
}


########################################################################################
####### Bootstrap to calculate the standard error of the evaluation metrics ########
############################################################################################

bootstrap_corr_se <- function(x, y, B = 1000, method = "pearson", seed = NULL) {
  # Input validation
  if (!is.null(seed)) set.seed(seed)
  if (length(x) != length(y)) {
    stop("Vectors x and y must have the same length")
  }
  if (sum(complete.cases(x, y)) < 3) {
    stop("Need at least 3 complete observations to calculate correlation")
  }
  
  n <- length(x)
  boot_cor <- numeric(B)
  
  # Bootstrap resampling
  for (i in 1:B) {
    indices <- sample(1:n, n, replace = TRUE)
    x_boot <- x[indices]
    y_boot <- y[indices]
    
    # Calculate correlation for bootstrap sample
    boot_cor[i] <- cor(x_boot, y_boot, method = method, use = "complete.obs")
  }
  
  # Calculate standard error (std dev of bootstrap estimates)
  se <- sd(boot_cor, na.rm = TRUE)
  return(se)
}




