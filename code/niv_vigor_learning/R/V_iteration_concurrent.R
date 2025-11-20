# Value iteration for concurrent schedules where each lever's elapsed time
# is part of the state space. Translated from V_iteration_concurrent.m.

V_iteration_concurrent <- function(V,
                                   rho,
                                   k_v,
                                   Utility,
                                   Temp,
                                   dt,
                                   times,
                                   p_r,
                                   Nactions,
                                   Nstates,
                                   EatTime,
                                   Beta,
                                   r_avg) {
  Ntimes <- length(times)
  if (is.null(dim(V)) || any(dim(V) != c(Ntimes + 1, Ntimes + 1, Nactions, 2))) {
    V <- array(0, dim = c(Ntimes + 1, Ntimes + 1, Nactions, 2))
    cat("Starting with new Vs\n")
  }

  if (Beta > 0) {
    discount <- exp(-Beta * times) %o% rep(1, Nactions)
  } else {
    discount <- NULL
  }

  MaxChange <- 0.01
  change <- MaxChange + 1
  Niter <- 0

  while (abs(change) > MaxChange) {
    Niter <- Niter + 1
    V_new <- array(0, dim = c(Ntimes + 1, Ntimes + 1, Nactions, 2))

    for (i_r in 1:2) {
      for (prev_action in 1:Nactions) {
        max_t_lp <- rep(Ntimes + 1, 2)
        if (prev_action < 3) {
          max_t_lp[prev_action] <- 1
        }
        if (i_r == 2 && prev_action == Nactions) {
          next
        }
        for (t_lp1 in seq_len(max_t_lp[1])) {
          for (t_lp2 in seq_len(max_t_lp[2])) {
            Q <- matrix(
              rep(rho, times = Ntimes),
              nrow = Ntimes, ncol = Nactions, byrow = TRUE
            ) - outer(1 / times, k_v[prev_action, ])

            t_index1 <- pmin(t_lp1 + seq_len(Ntimes), Ntimes + 1)
            t_index2 <- pmin(t_lp2 + seq_len(Ntimes), Ntimes + 1)

            prob_lp1 <- p_r[, t_lp1, 1]
            prob_lp2 <- p_r[, t_lp2, 2]

            Q[, 1] <- Q[, 1] +
              prob_lp1 * V[1, t_index2, 1, 2] +
              (1 - prob_lp1) * V[1, t_index2, 1, i_r]
            Q[, 2] <- Q[, 2] +
              prob_lp2 * V[t_index1, 1, 2, 2] +
              (1 - prob_lp2) * V[t_index1, 1, 2, i_r]

            if (Nactions > 3) {
              for (row in seq_len(Ntimes)) {
                Q[row, 3:(Nactions - 1)] <- Q[row, 3:(Nactions - 1)] +
                  V[t_index1[row], t_index2[row], 3:(Nactions - 1), i_r]
              }
            }

            if (i_r == 2) {
              t_idx1_np <- pmin(t_index1 + EatTime * dt, Ntimes + 1)
              t_idx2_np <- pmin(t_index2 + EatTime * dt, Ntimes + 1)
              for (row in seq_len(Ntimes)) {
                Q[row, Nactions] <- Q[row, Nactions] +
                  V[t_idx1_np[row], t_idx2_np[row], Nactions, 1] + Utility
              }
            } else {
              for (row in seq_len(Ntimes)) {
                Q[row, Nactions] <- Q[row, Nactions] +
                  V[t_index1[row], t_index2[row], Nactions, 1]
              }
            }

            if (Beta > 0) {
              Q <- discount * Q
            }

            V_new[t_lp1, t_lp2, prev_action, i_r] <- max(Q)
          }
        }
      }
    }

    change <- sum(V - V_new)
    V <- V_new
  }

  cat(sprintf("V iteration (concurrent) ended after %d iterations, last change %5.4f\n",
              Niter, change))
  list(V = V, r_avg = r_avg)
}
