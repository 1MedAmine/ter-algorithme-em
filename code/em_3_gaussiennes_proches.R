# ALGORITHME EM — Mélange de 3 gaussiennes (composantes proches)
# alpha=(0.4, 0.4, 0.2)  mu=(0.5, 1, 1.5)  var=(0.3, 0.2, 0.2)

K = 3
n = 1000
alpha_vrai = c(0.4, 0.4, 0.2)
mu_vrai = c(0.5, 1, 1.5)
var_vrai = c(0.3, 0.2, 0.2)


## Simulation des données

estm_obs = function(n) {
  X = numeric(n)
  for (i in 1:n) {
    u = runif(1)
    if (u < 0.4) {
      X[i] = rnorm(1, mean = 0.5, sd = sqrt(0.3))
    } else if (u < 0.8) {
      X[i] = rnorm(1, mean = 1, sd = sqrt(0.2))
    } else {
      X[i] = rnorm(1, mean = 1.5, sd = sqrt(0.2))
    }
  }
  return(X)
}

X = estm_obs(n)


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
mu_est = c(0, 1, 2)

var_est = rep(0.5, K)

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
  cat("  alpha  : estimé =", round(alpha_est[k], 3), " | vrai =", alpha_vrai[k], "\n")
  cat("  mu     : estimé =", round(mu_est[k],    3), " | vrai =", mu_vrai[k],    "\n")
  cat("  var    : estimé =", round(var_est[k],   3), " | vrai =", var_vrai[k],   "\n")
  cat("\n")
}

## Graphique
x_seq = seq(-2, 6, by = 0.01)
couleurs = c("blue", "red", "green")

# On initialise le graphique vide
plot(x_seq, gauss(x_seq, mu_vrai[1], var_vrai[1]),
     type = "l", lty = 2, col = couleurs[1], lwd = 2,
     xlab = "Taille des nids", ylab = "Densité",
     main = "Vrai vs Estimé par EM",
     ylim = c(0, 1))

# On ajoute les courbes vraies (pointillé)
for (k in 2:K) {
  lines(x_seq, gauss(x_seq, mu_vrai[k], var_vrai[k]),
        lty = 2, col = couleurs[k], lwd = 2)
}

# On ajoute les courbes estimées (trait plein)
for (k in 1:K) {
  lines(x_seq, gauss(x_seq, mu_est[k], var_est[k]),
        lty = 1, col = couleurs[k], lwd = 2)
}

# Légende
legend("topright",legend = c("Espèce 1 vrai", "Espèce 2 vrai", "Espèce 3 vrai",
                             "Espèce 1 EM",   "Espèce 2 EM",   "Espèce 3 EM"),
       col = c(couleurs, couleurs),lty    = c(2, 2, 2, 1, 1, 1),lwd = 2)



## Test de Kolmogorov-Smirnov sur le mélange estimé
# CDF du mélange : F(x) = sum_k alpha_k * Phi((x - mu_k) / sqrt(var_k))

X = estm_obs(n)

cdf_melange = function(x, alpha, mu, var) {
  sapply(x, function(xi) {
    sum(alpha * pnorm(xi, mean = mu, sd = sqrt(var)))
  })
}

# On passe la CDF comme fonction de distribution empirique à tester
ks_result = ks.test(X, function(x) cdf_melange(x, alpha_est, mu_est, var_est))
print(ks_result)



