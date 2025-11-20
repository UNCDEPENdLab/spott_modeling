# Analytical solution for fixed-ratio schedules under the average-reward
# framework (translation of V_Solve_FR.m).

V_Solve_FR <- function(ratio,
                       k_v,
                       rho,
                       Utility,
                       Nactions,
                       r_avg,
                       EatTime) {
  ratio <- as.integer(ratio)
  V <- array(0, dim = c(ratio, Nactions, 2))

  if (r_avg == 0) {
    if (EatTime > 0) {
      p1 <- Utility
      p2 <- ratio
      p3 <- rho[1]
      p4 <- rho[Nactions]
      p5 <- k_v[1, 1]
      p6 <- k_v[1, Nactions]
      p7 <- k_v[Nactions, 1]
      p8 <- EatTime
      objective <- function(x) {
        num <- (p1 + p2 * p3 + p4 -
                  (p2 - 1) * sqrt(p5 * x) -
                  sqrt(p7 * x) -
                  sqrt(p6 * x))
        denom <- (p8 +
                    (p2 - 1) * sqrt(p5 / x) +
                    sqrt(p7 / x) +
                    sqrt(p6 / x))
        num / denom - x
      }
      a <- EatTime
      b <- 2 * ((ratio - 1) * sqrt(k_v[1, 1]) +
                  sqrt(k_v[1, Nactions]) +
                  sqrt(k_v[Nactions, 1]))
      c <- -(Utility + ratio * rho[1] + rho[Nactions])
      disc <- max(b^2 - 4 * a * c, .Machine$double.eps)
      hint <- ((-b + sqrt(disc)) / (2 * a))^2
      upper <- max(hint * 10, 1)
      lower <- 1e-8
      f_lower <- objective(lower)
      f_upper <- objective(upper)
      iter <- 0
      while (f_lower * f_upper > 0 && iter < 25) {
        upper <- upper * 2
        f_upper <- objective(upper)
        iter <- iter + 1
      }
      if (f_lower * f_upper > 0) {
        stop("Failed to bracket root for r_avg in V_Solve_FR")
      }
      r_avg <- uniroot(objective, c(lower, upper))$root
      cat(sprintf("Solved for the average reward numerically: %3.4f\n", r_avg))
    } else {
      num <- Utility + ratio * rho[1] + rho[Nactions]
      denom <- 2 * ((ratio - 1) * sqrt(k_v[1, 1]) +
                      sqrt(k_v[1, Nactions]) +
                      sqrt(k_v[Nactions, 1]))
      r_avg <- (num / denom)^2
    }
  }

  K <- 2 * sqrt(r_avg * k_v)
  V[1, Nactions, 1] <- 0
  V[2, 1, 1] <- K[Nactions, 1] - rho[1]
  if (ratio >= 3) {
    for (i in 3:ratio) {
      V[i, 1, 1] <- K[1, 1] - rho[1] + V[i - 1, 1, 1]
    }
  }
  V[1, 1, 2] <- K[1, 1] - rho[1] + V[ratio, 1, 1]
  cat(sprintf("Sanity check (should be zero): %3.4f\n",
              K[1, Nactions] - rho[Nactions] + V[1, 1, 2] - Utility + r_avg * EatTime))

  if (ratio >= 3) {
    V[2:(ratio - 1), Nactions, 1] <- rho[1] - K[Nactions, 1] + V[3:ratio, 1, 1]
  }
  if (ratio >= 2) {
    V[ratio, Nactions, 1] <- rho[1] - K[Nactions, 1] + V[1, 1, 2]
  }
  V[, 1, 2] <- rho[Nactions] - K[1, Nactions] + Utility + V[, Nactions, 1] - r_avg * EatTime
  if (Nactions > 2) {
    for (a_prev in 2:(Nactions - 1)) {
      if (ratio >= 2) {
        V[1:(ratio - 1), a_prev, 1] <- rho[1] - K[a_prev, 1] + V[2:ratio, 1, 1]
      }
      V[ratio, a_prev, 1] <- rho[1] - K[a_prev, 1] + V[1, 1, 2]
      V[, a_prev, 2] <- rho[Nactions] - K[a_prev, Nactions] +
        V[, Nactions, 1] + Utility - r_avg * EatTime
    }
  }

  V <- V - V[1, 1, 2]
  V[, Nactions, 2] <- 0
  V[1, 1, 1] <- 0

  list(V = V, r_avg = r_avg)
}
