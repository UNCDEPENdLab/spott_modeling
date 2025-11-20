#' Simulate observable behavior under the optimal policy.
#'
#' This function mirrors the MATLAB GenerateData.m helper that accompanied the
#' drivers in Niv (2007). Given an optimal state-value function produced by
#' value iteration, it simulates sequences of actions, latencies, and rewards on
#' ratio/interval schedules and returns descriptive summaries.
GenerateData <- function(V,
                         r_avg,
                         rho,
                         k_v,
                         Utility,
                         bait_rate,
                         times,
                         p_r = NULL,
                         p_non = 0,
                         p_bsr = 0,
                         EatTime,
                         Nactions,
                         Nstates,
                         dt,
                         Temp,
                         Beta,
                         Ndata,
                         Nsessions,
                         max_actions = 500,
                         Softmax = TRUE,
                         figures = 0,
                         Reward_times = NULL,
                         schedule = c("Interval", "Ratio"),
                         schedule_type = c("Random", "Fixed", "Variable")) {
  schedule <- match.arg(tolower(schedule), c("interval", "ratio"))
  schedule_type <- match.arg(tolower(schedule_type), c("random", "fixed", "variable"))
  Softmax <- as.logical(Softmax)
  Temp <- max(Temp, 1e-8)
  Ntimes <- length(times)
  state_dim <- dim(V)[1]
  if (state_dim < 1) {
    stop("V must contain at least one discretized state.")
  }

  if (Beta > 0) {
    discount <- exp(-Beta * times) %o% rep(1, Nactions)
  } else {
    discount <- NULL
  }

  Ntrials <- ifelse(Ndata > 0, as.integer(Ndata), as.integer(max(1, Nsessions)))
  if (!is.finite(Ntrials) || Ntrials <= 0) {
    Ntrials <- 1L
  }
  max_actions <- max(1L, as.integer(max_actions))
  p_non <- max(0, p_non)
  p_bsr <- max(0, p_bsr)

  responses <- integer()
  latencies <- numeric()
  cumulative_times <- numeric()
  action_trial <- integer()
  action_state <- list()
  cache_Q <- new.env(parent = emptyenv())

  trial_lp <- integer(Ntrials)
  trial_np <- integer(Ntrials)
  trial_action_counts <- integer(Ntrials)
  first_lp_latency <- rep(NA_real_, Ntrials)
  irt_lp <- numeric()
  consumption_durations <- numeric()
  reward_times <- rep(NA_real_, Ntrials)
  reward_latency <- rep(NA_real_, Ntrials)

  total_time <- 0
  press_counter <- 0
  last_lp_time <- NA_real_
  last_bait_time <- NA_real_
  bait_intervals <- numeric()

  choose_action <- function(Q) {
    flat_Q <- as.vector(Q)
    if (Softmax) {
      logits <- flat_Q / Temp
      logits <- logits - max(logits)
      probs <- exp(logits)
      probs_sum <- sum(probs)
      if (!is.finite(probs_sum) || probs_sum <= 0) {
        probs <- rep(1 / length(flat_Q), length(flat_Q))
      } else {
        probs <- probs / probs_sum
      }
      choice <- sample.int(length(probs), size = 1, prob = probs)
    } else {
      choice <- which.max(flat_Q)
    }
    lat_idx <- as.integer(((choice - 1) %% Ntimes) + 1L)
    action <- as.integer(((choice - 1) %/% Ntimes) + 1L)
    list(lat_idx = lat_idx, action = action)
  }

  get_Q <- function(t_s, prev_action, i_r) {
    key <- paste(t_s, prev_action, i_r, sep = "::")
    if (!is.null(cache_Q[[key]])) {
      return(cache_Q[[key]])
    }
    Q <- compute_Q(
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
    cache_Q[[key]] <- Q
    Q
  }

  t_s <- 1
  prev_action <- Nactions
  i_r <- 1

  for (trial in seq_len(Ntrials)) {
    rewarded <- FALSE
    actions_this_trial <- 0
    lp_first_recorded <- FALSE
    trial_start_time <- total_time

    while (!rewarded && actions_this_trial < max_actions) {
      Q_state <- get_Q(t_s, prev_action, i_r)
      choice <- choose_action(Q_state)
      lat_idx <- choice$lat_idx
      action <- choice$action
      latency <- times[lat_idx]

      pr_sched <- 0
      if (!is.null(p_r) && length(dim(p_r)) == 2) {
        if (action == 1 && nrow(p_r) >= lat_idx) {
          t_col <- min(t_s, ncol(p_r))
          pr_sched <- p_r[lat_idx, t_col]
        }
      }
      reward_available <- FALSE
      reward_source <- NA_character_
      if (schedule_type == "fixed" && action == 1 && is.numeric(p_r) && length(p_r) == 1) {
        press_counter <- press_counter + 1
        if (press_counter >= p_r) {
          reward_available <- TRUE
          reward_source <- "fixed"
          press_counter <- 0
        }
      }

      p_non_prob <- 1 - exp(-p_non * latency)
      if (action == 1 && reward_source != "fixed") {
        p_total <- min(1, pr_sched + p_non_prob)
        if (runif(1) < p_total) {
          reward_available <- TRUE
          alloc_prob <- if (p_total > 0) pr_sched / p_total else 0
          reward_source <- if (runif(1) < alloc_prob) "schedule" else "noncontingent"
        }
      } else if (action != Nactions) {
        if (runif(1) < p_non_prob) {
          reward_available <- TRUE
          reward_source <- "noncontingent"
        }
      }

      prev_time <- total_time
      time_spent <- latency
      new_t_s <- min(t_s + lat_idx, state_dim)
      new_i_r <- if (reward_available) 2 else i_r

      if (!is.null(reward_source)) {
        bait_ts <- prev_time + latency
        if (is.na(last_bait_time)) {
          last_bait_time <- bait_ts
        } else {
          bait_intervals <- c(bait_intervals, bait_ts - last_bait_time)
          last_bait_time <- bait_ts
        }
      }

      if (action == 1) {
        trial_lp[trial] <- trial_lp[trial] + 1L
        if (!lp_first_recorded) {
          first_lp_latency[trial] <- (prev_time + latency) - trial_start_time
          lp_first_recorded <- TRUE
        }
        lp_time <- prev_time + latency
        if (!is.na(last_lp_time)) {
          irt_lp <- c(irt_lp, lp_time - last_lp_time)
        }
        last_lp_time <- lp_time
        new_t_s <- 1
      } else if (action == Nactions) {
        trial_np[trial] <- trial_np[trial] + 1L
        if (i_r == 2) {
          time_spent <- latency + EatTime
          new_t_s <- min(t_s + lat_idx + EatTime * dt, state_dim)
          new_i_r <- 1
          rewarded <- TRUE
          consumption_durations <- c(consumption_durations, EatTime)
          reward_times[trial] <- prev_time + time_spent
          reward_latency[trial] <- reward_times[trial] - trial_start_time
        }
      }

      actions_this_trial <- actions_this_trial + 1L
      trial_action_counts[trial] <- actions_this_trial
      total_time <- total_time + time_spent

      responses <- c(responses, action)
      latencies <- c(latencies, latency)
      cumulative_times <- c(cumulative_times, total_time)
      action_trial <- c(action_trial, trial)
      action_state[[length(action_state) + 1L]] <- list(t_s = t_s, prev_action = prev_action, reward_state = i_r, lat_idx = lat_idx)

      t_s <- new_t_s
      prev_action <- action
      i_r <- new_i_r
    }
    if (!rewarded) {
      reward_times[trial] <- NA_real_
    }

    t_s <- 1
    prev_action <- Nactions
    i_r <- 1
  }

  total_lp <- sum(trial_lp)
  total_time_obs <- max(1e-8, tail(cumulative_times, 1))
  lever_rate <- total_lp / total_time_obs
  reward_rate <- sum(!is.na(reward_times)) / total_time_obs

  Session_lp <- rep(0, max(1, Nsessions))
  if (Ntrials > 0) {
    session_id <- ceiling(seq_len(Ntrials) * max(1, Nsessions) / Ntrials)
    for (i in seq_along(session_id)) {
      Session_lp[session_id[i]] <- Session_lp[session_id[i]] + trial_lp[i]
    }
  }

  CacheIndex <- do.call(rbind, lapply(seq_along(action_state), function(i) {
    state <- action_state[[i]]
    c(event = i, trial = action_trial[i], t_s = state$t_s,
      prev_action = state$prev_action, reward_state = state$reward_state,
      lat_idx = state$lat_idx)
  }))
  if (is.null(CacheIndex)) {
    CacheIndex <- matrix(0, nrow = 0, ncol = 6)
  } else if (nrow(CacheIndex) > 0) {
    colnames(CacheIndex) <- c("event", "trial", "t_s", "prev_action", "reward_state", "lat_idx")
  }

  cache_names <- ls(envir = cache_Q, all.names = TRUE)
  Q_cache_list <- lapply(cache_names, function(nm) cache_Q[[nm]])
  names(Q_cache_list) <- cache_names

  first_lp_summary <- c(
    mean = mean(first_lp_latency, na.rm = TRUE),
    sd = sd(first_lp_latency, na.rm = TRUE),
    n = sum(!is.na(first_lp_latency))
  )
  reward_latency_summary <- c(
    mean = mean(reward_latency, na.rm = TRUE),
    sd = sd(reward_latency, na.rm = TRUE),
    n = sum(!is.na(reward_latency))
  )
  irt_summary <- c(
    mean = mean(irt_lp, na.rm = TRUE),
    sd = sd(irt_lp, na.rm = TRUE),
    n = length(irt_lp)
  )
  consump_summary <- c(
    mean = mean(consumption_durations, na.rm = TRUE),
    sd = sd(consumption_durations, na.rm = TRUE),
    n = length(consumption_durations)
  )

  bait_interval <- if (length(bait_intervals)) mean(bait_intervals) else if (!is.null(bait_rate) && length(bait_rate) == 1 && bait_rate > 0) 1 / bait_rate else NA_real_

  out_summary <- c(
    mean_first_lp = first_lp_summary["mean"],
    lever_rate = lever_rate,
    reward_rate = reward_rate,
    mean_actions = mean(trial_action_counts),
    mean_reward_latency = reward_latency_summary["mean"]
  )

  list(
    FirstLP_summary = first_lp_summary,
    IRT_LP_summary = irt_summary,
    Responses = responses,
    Latencies = latencies,
    Times = cumulative_times,
    R_summary = reward_latency_summary,
    Consump_summary = consump_summary,
    Nacts_per_trial = trial_action_counts,
    LPs = trial_lp,
    NPs = trial_np,
    Session_lp = Session_lp,
    CacheQ = Q_cache_list,
    CacheIndex = CacheIndex,
    Rewards = reward_times,
    L_lp = mean(latencies[responses == 1], na.rm = TRUE),
    L_np = mean(latencies[responses == Nactions], na.rm = TRUE),
    Bait_interval = bait_interval,
    LPrate = lever_rate,
    OutSummary = out_summary,
    Reward_times_input = Reward_times,
    Meta = list(
      schedule = schedule,
      schedule_type = schedule_type,
      bait_rate = bait_rate,
      total_time = total_time_obs
    )
  )
}
