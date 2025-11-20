# Legacy driver script (translated from driver.m) for running value iteration
# under different interval/ratio schedules. The original MATLAB version
# depends on GenerateData.m, which is not available in this directory.

run_driver <- function(V = NULL, r_avg = 0) {
  actions <- c("LP", "NP")
  Nactions <- length(actions)

  EatTime <- 3
  rho <- c(-0.15, -0.05)

  s <- 0.5
  d <- 1.0
  k_v <- matrix(c(s, d,
                  d, s),
                nrow = Nactions, byrow = TRUE)

  if (length(rho) != Nactions || nrow(k_v) != Nactions) {
    stop("DRIVER ERROR: Adjust rho or k_v to match the number of actions.")
  }

  Beta <- 0.05
  Temp <- 0.1
  Softmax <- 1
  figures <- 1

  schedule_type <- "Variable"
  schedule <- "Interval"
  interval <- c(30)
  ratio <- c(1, 3, 9, 27)
  Utility <- 40

  S_lp <- matrix(0, nrow = length(ratio), ncol = 2)
  Latencies <- matrix(0, nrow = length(ratio), ncol = 2)

  dt <- 10
  times <- seq(1 / dt, 30, by = 1 / dt)
  Ntimes <- length(times)
  Nstates <- (Ntimes + 1) * Nactions * 2

  p_r <- NULL
  if (tolower(schedule_type) == "variable") {
    startRew <- 15 * dt + 1
    endRew <- 45 * dt + 1
    p_r <- matrix(0, nrow = Ntimes, ncol = Ntimes + 1)
    for (tau in seq_len(Ntimes)) {
      for (t_r in seq_len(Ntimes + 1)) {
        if ((t_r + tau) <= startRew) {
          p_r[tau, t_r] <- 0
        } else if ((t_r + tau) >= endRew) {
          p_r[tau, t_r] <- 1
        } else {
          p_r[tau, t_r] <- (tau + min(t_r - startRew - 1, 0)) /
            ((endRew - startRew) - max(t_r - startRew - 1, 0))
        }
      }
    }
  }

  if (is.null(V)) {
    V <- array(0, dim = c(Ntimes + 1, Nactions, 2))
    cat("Starting with new Vs\n")
  }

  Reward_times <- 0
  Ndata <- 10000
  Nsessions <- 200
  rates <- numeric(length(interval))

  for (i in seq_along(interval)) {
    cat(sprintf("\n%s %s %3.2f\n", schedule_type, schedule, interval[i]))
    if (tolower(schedule_type) == "fixed") {
      Nstates <- ratio[i] * Nactions * 2
    }

    if (tolower(schedule_type) == "random" && tolower(schedule) == "interval") {
      p_r <- matrix(0, nrow = Ntimes, ncol = Ntimes + 1)
      for (tau in seq_len(Ntimes)) {
        for (t_lp in seq_len(Ntimes + 1)) {
          p_r[tau, t_lp] <- 1 - exp(-((tau + t_lp - 1) / (interval[i] * dt)))
        }
      }
    }
    if (tolower(schedule_type) == "random" && tolower(schedule) == "ratio") {
      p_r <- matrix(1 / ratio[i], nrow = Ntimes, ncol = Ntimes + 1)
    }
    if (tolower(schedule_type) == "fixed" && tolower(schedule) == "ratio") {
      p_r <- ratio[i]
    }

    if (tolower(schedule_type) == "fixed" && Beta == 0) {
      result <- V_Solve_FR(ratio[i], k_v, rho, Utility, Nactions, r_avg, EatTime)
      V <- result$V
      r_avg <- result$r_avg / dt
    } else {
      iter_result <- V_iteration(
        V = V,
        rho = rho,
        k_v = k_v,
        Utility = Utility,
        Temp = Temp,
        dt = dt,
        times = times,
        p_r = p_r,
        p_non = 0,
        p_bsr = 0,
        Nactions = Nactions,
        Nstates = Nstates,
        EatTime = EatTime,
        Beta = Beta,
        r_avg = r_avg,
        schedule_type = schedule_type
      )
      V <- iter_result$V
      r_avg <- iter_result$r_avg
    }

    rates[i] <- r_avg
    cat("Now generating data...\n")
    if (exists("GenerateData")) {
      GenerateData(
        V, r_avg, rho, k_v, Utility, 1 / interval[i], times, p_r,
        EatTime, Nactions, Nstates, dt, Temp, Beta,
        Ndata, Nsessions, Softmax, figures, Reward_times,
        schedule, schedule_type
      )
    } else {
      warning("GenerateData function is not available; skipping simulation.")
    }
  }

  list(V = V, r_avg = r_avg, rates = rates)
}
