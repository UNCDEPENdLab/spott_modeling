# Value iteration for the average-reward semi-Markov decision process described
# in Niv (2007). Returns optimal differential state values and (if applicable)
# the corresponding steady-state average reward rate.

V_iteration <- function(V,
                        rho,
                        k_v,
                        Utility,
                        Temp,
                        dt,
                        times,
                        p_r,
                        p_non,
                        p_bsr,
                        Nactions,
                        Nstates,
                        EatTime,
                        Beta,
                        r_avg,
                        schedule_type) {
  Ntimes <- length(times)
  state_dim <- as.integer(Nstates / (2 * Nactions))

  if (is.null(dim(V)) || any(dim(V) != c(state_dim, Nactions, 2))) {
    V <- array(0, dim = c(state_dim, Nactions, 2))
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
    V_new <- array(0, dim = c(state_dim, Nactions, 2))

    for (i_r in 1:2) {
      for (prev_action in 1:Nactions) {
        max_t_s <- Ntimes + 1
        if (prev_action == 1 && tolower(schedule_type) == "random") {
          max_t_s <- 1
        }
        if (tolower(schedule_type) == "fixed") {
          max_t_s <- p_r
        }
        max_t_s_int <- as.integer(max_t_s)
        if (max_t_s_int < 1) {
          next
        }
        for (t_s in seq_len(max_t_s_int)) {
          Q_next <- compute_Q(
            t_s = t_s,
            prev_action = prev_action,
            i_r = i_r,
            V = V,
            Utility = Utility,
            EatTime = EatTime,
            times = times,
            Ntimes = Ntimes,
            rho = rho,
            k_v = k_v,
            p_r = p_r,
            p_non = p_non,
            p_bsr = p_bsr,
            r_avg = r_avg,
            Nactions = Nactions,
            Beta = Beta,
            discount = discount,
            schedule_type = schedule_type
          )
          V_new[t_s, prev_action, i_r] <- max(Q_next)
        }
      }
    }

    if (Beta == 0) {
      V_new <- V_new - V_new[1, 1, 2]
      if (tolower(schedule_type) == "random" && dim(V_new)[1] > 1) {
        V_new[2:dim(V_new)[1], 1, ] <- 0
      }
      V0 <- compute_Q(
        t_s = 1,
        prev_action = 1,
        i_r = 2,
        V = V_new,
        Utility = Utility,
        EatTime = EatTime,
        times = times,
        Ntimes = Ntimes,
        rho = rho,
        k_v = k_v,
        p_r = p_r,
        p_non = p_non,
        p_bsr = p_bsr,
        r_avg = 0,
        Nactions = Nactions,
        Beta = Beta,
        discount = discount,
        schedule_type = schedule_type
      )
      denom <- matrix(rep(times, times = Nactions),
                      nrow = Ntimes, ncol = Nactions)
      eat_vec <- c(rep(0, Nactions - 1), EatTime)
      denom <- denom + matrix(rep(eat_vec, each = Ntimes),
                              nrow = Ntimes, ncol = Nactions)
      r_avg <- max(V0 / denom)
    }

    change <- sum(V - V_new)
    V <- V_new
  }

  cat(sprintf("V iteration ended after %d iterations, last change was %5.4f\n",
              Niter, change))
  if (Beta == 0) {
    cat(sprintf("\tThe optimal average reward is %5.4f\n", r_avg))
  }

  list(V = V, r_avg = r_avg)
}
