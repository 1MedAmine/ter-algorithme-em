# Charger le dataset
data <- read.csv("Iris.csv")
# Vérifier les premières lignes
head(data)
# Voir la structure
str(data)
K = 3 
n = 150
X = data$PetalWidthCm
## Fonctions EM
gauss = function(x, mu, var) {
  dnorm(x, mean = mu, sd = sqrt(var))
}
etape_E = function(X, alpha, mu, var) {
  gamma = matrix(0, nrow = length(X), ncol = K)
  for (k in 1:K) {
    gamma[, k] = alpha[k] * gauss(X, mu[k], var[k])
  }
  for (i in 1:length(X)) {
    gamma[i, ] = gamma[i, ] / sum(gamma[i, ])
  }
  return(gamma)
}
etape_M = function(X, gamma) {
  
  n = length(X)
  alpha = numeric(K)
  mu = numeric(K)
  var = numeric(K)
  
  for (k in 1:K) {
    Nk = sum(gamma[, k])
    alpha[k] = Nk / n
    mu[k] = sum(gamma[, k] * X) / Nk
    var[k] = sum(gamma[, k] * (X - mu[k])^2) / Nk
  }
  
  return(list(alpha = alpha, mu = mu, var = var))
}
## Boucle EM
alpha_est = rep(1/K, K)
mu_est = c(0.2, 1.3, 2)
var_est = rep(0.1, K)
for (iter in 1:200) {
  gamma = etape_E(X, alpha_est, mu_est, var_est)
  params = etape_M(X, gamma)
  alpha_old = alpha_est
  alpha_est = params$alpha
  mu_est = params$mu
  var_est = params$var
  if (iter > 1 && max(abs(alpha_est - alpha_old)) < 1e-6) {
    cat("Convergence à l'itération", iter, "\n")
    break
  }
}
for (k in 1:K) {
  cat("Espèce", k, ":\n")
  cat("  alpha estimé =", round(alpha_est[k], 3))
  cat("  mu estimé =", round(mu_est[k],    3) )
  cat("  var estimé =", round(var_est[k],   3))
  cat("\n")
}