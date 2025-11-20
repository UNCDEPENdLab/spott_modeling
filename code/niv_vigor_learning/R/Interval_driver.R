# Driver for interval schedules (translation of Interval_driver.m).

run_interval_driver <- function(Utility = c(60),
                                V = NULL,
                                r_avg = 0,
                                p_non = 0,
                                p_bsr = 0) {
  actions <- c("LP", "Other", "NP")
  Nactions <- length(actions)

  EatTime <- 6
  rho <- c(-0.15, 0.15, -0.15)

  s <- 0.5
  d <- 1.5
  k_v <- matrix(c(s, d, d,
                  d, s, d,
                  d, d, s),
                nrow = Nactions, byrow = TRUE)

  if (length(rho) != Nactions || nrow(k_v) != Nactions) {
    stop("DRIVER ERROR: Adjust rho or k_v to match the number of actions.")
  }

  Beta <- 0
  Temp <- 0.25
  Softmax <- 1
  figures <- 0

  schedule_type <- "Random"
  schedule <- "Interval"
  interval <- 30

  Reward_times <- 0
  Ndata <- 0
  Nsessions <- 600
  max_actions <- 500

  dt <- 10
  times <- seq(1 / dt, 20, by = 1 / dt)
  Ntimes <- length(times)
  Nstates <- (Ntimes + 1) * Nactions * 2

  if (tolower(schedule_type) == "variable") {
    startRew <- 15 * dt + 1
    endRew <- 45 * dt + 1
    p_r <- array(0, dim = c(Ntimes, Ntimes + 1))
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
  } else {
    p_r <- NULL
  }

  if (is.null(V)) {
    V <- array(0, dim = c(Ntimes + 1, Nactions, 2))
    cat("Starting with new Vs\n")
  }

  rates <- numeric(length(Utility))
  summaries <- vector("list", length(Utility))

  for (i in seq_along(Utility)) {
    cat(sprintf("\n%s %s %3.2f\n", schedule_type, schedule, interval))

    if (tolower(schedule_type) == "random" && tolower(schedule) == "interval") {
      p_r <- matrix(0, nrow = Ntimes, ncol = Ntimes + 1)
      for (tau in seq_len(Ntimes)) {
        for (t_lp in seq_len(Ntimes + 1)) {
          p_r[tau, t_lp] <- 1 - exp(-((tau + t_lp - 1) / (interval * dt)))
        }
      }
    }

    iter_result <- V_iteration(
      V = V,
      rho = rho,
      k_v = k_v,
      Utility = Utility[i],
      Temp = Temp,
      dt = dt,
      times = times,
      p_r = p_r,
      p_non = p_non,
      p_bsr = p_bsr,
      Nactions = Nactions,
      Nstates = Nstates,
      EatTime = EatTime,
      Beta = Beta,
      r_avg = r_avg,
      schedule_type = schedule_type
    )
    V <- iter_result$V
    r_avg <- iter_result$r_avg

    rates[i] <- r_avg

    if (exists("GenerateData")) {
      summaries[[i]] <- GenerateData(
        V = V,
        r_avg = r_avg,
        rho = rho,
        k_v = k_v,
        Utility = Utility[i],
        bait_rate = 1 / interval,
        times = times,
        p_r = p_r,
        p_non = p_non,
        p_bsr = p_bsr,
        EatTime = EatTime,
        Nactions = Nactions,
        Nstates = Nstates,
        dt = dt,
        Temp = Temp,
        Beta = Beta,
        Ndata = Ndata,
        Nsessions = Nsessions,
        max_actions = max_actions,
        Softmax = Softmax,
        figures = figures,
        Reward_times = Reward_times,
        schedule = schedule,
        schedule_type = schedule_type
      )
    } else {
      warning("GenerateData function is not available; skipping simulation.")
    }
  }

  list(V = V, r_avg = r_avg, rates = rates, summaries = summaries)
}
