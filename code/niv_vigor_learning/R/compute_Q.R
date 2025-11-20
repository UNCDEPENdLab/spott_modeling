# Compute state-action differential values (Q) for a given state.
# This is a direct translation of compute_Q.m and follows the formulation
# in Niv (2007, Eq. 2.8) where actions are parameterized by their latency.

compute_Q <- function(t_s,
                      prev_action,
                      i_r,
                      V,
                      Utility,
                      EatTime,
                      times,
                      Ntimes,
                      rho,
                      k_v,
                      p_r,
                      p_non,
                      p_bsr,
                      r_avg,
                      Nactions,
                      Beta,
                      discount,
                      schedule_type) {
  dt <- 1 / (times[2] - times[1])
  Q_base <- matrix(rep(rho, times = Ntimes), nrow = Ntimes, ncol = Nactions, byrow = TRUE)
  transition_costs <- outer(1 / times, k_v[prev_action, ])
  bsr_matrix <- matrix(rep(times, times = Nactions), nrow = Ntimes, ncol = Nactions) * p_bsr
  Q <- Q_base - transition_costs + bsr_matrix

  if (tolower(schedule_type) == "fixed") {
    if (i_r == 1 || Beta > 0) {
      if (t_s == p_r) {
        Q[, 1] <- Q[, 1] + V[1, 1, 2]
      } else {
        Q[, 1] <- Q[, 1] + V[t_s + 1, 1, i_r]
      }
    } else {
      Q[, 1] <- Q[, 1] + V[t_s, 1, i_r]
    }
    if (Nactions > 2) {
      Q[, 2:(Nactions - 1)] <- Q[, 2:(Nactions - 1)] +
        matrix(rep(V[t_s, 2:(Nactions - 1), i_r], times = Ntimes),
               nrow = Ntimes, byrow = TRUE)
    }
    Q[, Nactions] <- Q[, Nactions] + V[t_s, Nactions, 1] + (i_r - 1) * Utility
  } else {
    t_index <- pmin(t_s + seq_len(Ntimes), Ntimes + 1)
    p_non_r <- 1 - exp(-p_non * times)

    if (tolower(schedule_type) == "random") {
      Q[, 1] <- Q[, 1] +
        (p_non_r + p_r[, t_s]) * V[1, 1, 2] +
        (1 - p_r[, t_s] - p_non_r) * V[1, 1, i_r]
    } else {
      Q[, 1] <- Q[, 1] +
        (p_non_r + p_r[, t_s]) * V[1, 1, 2] +
        (1 - p_r[, t_s] - p_non_r) * V[t_index, 1, i_r]
    }

    if (Nactions > 2) {
      p_non_mat <- matrix(rep(p_non_r, times = Nactions - 2),
                          nrow = Ntimes, ncol = Nactions - 2)
      survival_mat <- matrix(rep(1 - p_non_r, times = Nactions - 2),
                             nrow = Ntimes, ncol = Nactions - 2)
      Q[, 2:(Nactions - 1)] <- Q[, 2:(Nactions - 1)] +
        p_non_mat * V[t_index, 2:(Nactions - 1), 2, drop = FALSE] +
        survival_mat * V[t_index, 2:(Nactions - 1), i_r, drop = FALSE]
    }

    if (i_r == 2) {
      t_index_np <- pmin(t_index + EatTime * dt, Ntimes + 1)
      p_non_np <- 1 - exp(-p_non * (times + EatTime))
      Q[, Nactions] <- Q[, Nactions] +
        p_non_np * V[t_index_np, Nactions, 2] +
        (1 - p_non_np) * V[t_index_np, Nactions, 1] +
        Utility + p_bsr * EatTime
    } else {
      Q[, Nactions] <- Q[, Nactions] +
        p_non_r * V[t_index, Nactions, 2] +
        (1 - p_non_r) * V[t_index, Nactions, 1]
    }
  }

  if (Beta == 0) {
    avg_matrix <- matrix(rep(times * r_avg, times = Nactions),
                         nrow = Ntimes, ncol = Nactions)
    Q <- Q - avg_matrix
    if (i_r == 2) {
      Q[, Nactions] <- Q[, Nactions] - r_avg * EatTime
    }
  } else {
    Q <- discount * Q
  }

  Q
}
