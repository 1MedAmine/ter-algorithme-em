# ALGORITHME EM — Mélange de 5 gaussiennes
# alpha=(0.25, 0.15, 0.25, 0.20, 0.15)  mu=(-6, -3, 0, 3, 6)  var=(0.3, 0.4, 0.3, 0.2, 0.3)

## Paramètres vrais
K = 5
n = 1000
alpha_vrai = c(0.25, 0.15, 0.25, 0.20, 0.15)
mu_vrai = c(-6, -3, 0, 3, 6)
var_vrai = c(0.3, 0.4, 0.3, 0.2, 0.3)


## Simulation des données
estm_obs = function(n) {
  X = numeric(n)
  for (i in 1:n) {
    u = runif(1)
    if (u < 0.25) {
      X[i] = rnorm(1, mean = -6, sd = sqrt(0.3))
    } else if (u < 0.40) {
      X[i] = rnorm(1, mean = -3, sd = sqrt(0.4))
    } else if (u < 0.65) {
      X[i] = rnorm(1, mean = 0, sd = sqrt(0.3))
    } else if (u < 0.85) {
      X[i] = rnorm(1, mean = 3, sd = sqrt(0.2))
    } else {
      X[i] = rnorm(1, mean = 6, sd = sqrt(0.3))
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
mu_est = c(-7, -4, 0.5, 2, 5)
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
x_seq = seq(-10, 10, by = 0.01)
couleurs = c("blue", "red", "green", "purple", "orange")

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
legend("topright",legend = c(paste("Espèce", 1:K, "vrai"), paste("Espèce", 1:K, "EM")),
       col = c(couleurs, couleurs),lty = c(rep(2, K), rep(1, K)),lwd = 2, cex = 0.7)



## Test de Kolmogorov-Smirnov sur le mélange estimé
# CDF du mélange : F(x) = sum_k alpha_k * Phi((x - mu_k) / sqrt(var_k))
cdf_melange = function(x, alpha, mu, var) {
  sapply(x, function(xi) {
    sum(alpha * pnorm(xi, mean = mu, sd = sqrt(var)))
  })
}

# On passe la CDF comme fonction de distribution empirique à tester

X_test = estm_obs(n)

ks_result = ks.test(X_test, function(x) cdf_melange(x, alpha_est, mu_est, var_est))
print(ks_result)
