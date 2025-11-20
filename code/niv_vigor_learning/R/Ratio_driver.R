# Driver for ratio schedules (translation of Ratio_driver.m).

run_ratio_driver <- function(ratio = c(10),
                             Utility = 10,
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
  schedule <- "Ratio"

  Reward_times <- 0
  Ndata <- 0
  Nsessions <- 300
  max_actions <- 500

  dt <- 20
  times <- seq(1 / dt, 10, by = 1 / dt)
  Ntimes <- length(times)
  Nstates <- (Ntimes + 1) * Nactions * 2

  if (is.null(V)) {
    V <- array(0, dim = c(Ntimes + 1, Nactions, 2))
    cat("Starting with new Vs\n")
  }

  results <- vector("list", length(ratio))
  rates <- numeric(length(ratio))

  for (i in seq_along(ratio)) {
    Utility_i <- if (length(Utility) >= i) Utility[i] else Utility[length(Utility)]
    cat(sprintf("\n%s %s %d\n", schedule_type, schedule, ratio[i]))
    local_Nstates <- Nstates
    p_r <- NULL
    if (tolower(schedule_type) == "fixed") {
      local_Nstates <- ratio[i] * Nactions * 2
      p_r <- ratio[i]
    }

    if (tolower(schedule_type) == "random" && tolower(schedule) == "ratio") {
      p_r <- matrix(1 / ratio[i], nrow = Ntimes, ncol = Ntimes + 1)
    }

    if (tolower(schedule_type) == "fixed" && Beta == 0) {
      cat("Finding values by solving the cyclic equations...\n")
      sol <- V_Solve_FR(ratio[i], k_v, rho, Utility_i, Nactions, r_avg, EatTime)
      V <- sol$V
      r_avg <- sol$r_avg
      cat(sprintf("Average reward rate: %3.4f\n", r_avg))
    } else {
      cat("Finding values by value iteration...\n")
      iter_result <- V_iteration(
        V = V,
        rho = rho,
        k_v = k_v,
        Utility = Utility_i,
        Temp = Temp,
        dt = dt,
        times = times,
        p_r = p_r,
        p_non = p_non,
        p_bsr = p_bsr,
        Nactions = Nactions,
        Nstates = local_Nstates,
        EatTime = EatTime,
        Beta = Beta,
        r_avg = r_avg,
        schedule_type = schedule_type
      )
      V <- iter_result$V
      r_avg <- iter_result$r_avg
    }

    cat("Now generating data...\n")
    if (exists("GenerateData")) {
      results[[i]] <- GenerateData(
        V = V,
        r_avg = r_avg,
        rho = rho,
        k_v = k_v,
        Utility = Utility_i,
        bait_rate = 1 / ratio[i],
        times = times,
        p_r = p_r,
        p_non = p_non,
        p_bsr = p_bsr,
        EatTime = EatTime,
        Nactions = Nactions,
        Nstates = local_Nstates,
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
    rates[i] <- r_avg
  }

  list(V = V, r_avg = r_avg, rates = rates, summaries = results)
}
