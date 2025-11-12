# Free-operant TD(λ) with Dutch traces, average-reward baseline, vigor + sticky softmax
# Time is discretized in bins of dt seconds (default 0.05 = 50 ms).

softmax <- function(m) {
  m <- m - max(m)
  e <- exp(m)
  e / sum(e)
}

simulate_free_operant <- function(
    n_bins             = 4000,        # e.g., 200 s at dt=0.05
    dt                 = 0.05,        # 50 ms bins
    seed               = NULL,
    # --- Learning & traces ---
    alpha              = 0.10,        # value learning rate (per bin)
    eta                = 0.01,        # average-reward learning rate (slow)
    tau_e              = 0.50,        # eligibility time constant (seconds)
    # --- Action selection: whether & which ---
    beta0_vigor        = -4.0,        # intercept (logit scale) for "respond?"
    beta_t_vigor       = 0.025,       # effect of time-since-last-response (per second)
    beta_q_vigor       = 0.50,        # effect of total value (Qx+Qy)
    refractory_s       = 0.10,        # hard refractory (seconds) after a response
    kappa              = 4.0,         # inverse temperature for softmax over Q
    omega_sticky       = 0.50,        # stickiness bonus to last action
    # --- Environment: response-triggered delayed rewards ---
    env_type           = c("response_triggered"),
    p_trigger          = c(x = 0.50, y = 0.50), # prob a press starts a reward job
    delay_mean_s       = c(x = 0.50, y = 0.50), # mean delay (seconds) for reward arrival
    reward_amount      = c(x = 1.0,  y = 1.0)   # reward magnitude
) {
  if (!is.null(seed)) set.seed(seed)
  env_type <- match.arg(env_type)
  
  # Derived params
  lambda_bin <- exp(-dt / tau_e)     # per-bin eligibility decay
  refractory_bins <- ceiling(refractory_s / dt)
  
  # State
  Q <- c(x = 0, y = 0)               # action values
  z <- c(x = 0, y = 0)               # Dutch eligibility trace
  Rbar <- 0                          # average reward
  last_action <- NA_character_
  last_resp_bin <- -Inf
  
  # Pending reward queue: list of (due_bin, amount)
  # Keep two independent queues if you want to tag by action; here we just deliver scalar r_t
  reward_due_bins <- integer(0)
  reward_amounts  <- numeric(0)
  
  # Storage
  out <- data.frame(
    bin      = seq_len(n_bins),
    time_s   = (seq_len(n_bins) - 1L) * dt,
    responded= FALSE,
    action   = NA_character_,
    r        = 0.0,
    delta    = 0.0,
    Qx       = NA_real_,
    Qy       = NA_real_,
    zx       = NA_real_,
    zy       = NA_real_,
    Rbar     = NA_real_,
    p_respond= NA_real_,
    p_x      = NA_real_,
    p_y      = NA_real_,
    t_since_s= NA_real_
  )
  
  for (b in seq_len(n_bins)) {
    # 1) WHETHER to respond: logistic in time-since-last + total value
    t_since_bins <- b - last_resp_bin
    t_since_s <- max((t_since_bins - 1L), 0L) * dt
    in_refractory <- t_since_bins <= refractory_bins
    
    Qsum <- sum(Q)
    logit_p <- beta0_vigor + beta_t_vigor * t_since_s + beta_q_vigor * Qsum
    p_respond <- if (in_refractory) 0 else 1 / (1 + exp(-logit_p))
    respond <- runif(1) < p_respond
    
    # 2) WHICH action (if responding): sticky softmax over Q
    action <- NA_character_
    p_a <- c(x = NA_real_, y = NA_real_)
    if (respond) {
      m <- kappa * Q + if (is.na(last_action)) 0 else omega_sticky * as.numeric(names(Q) == last_action)
      p_a <- softmax(m)
      action <- sample(c("x", "y"), size = 1, prob = p_a)
      last_resp_bin <- b
      last_action <- action
    }
    
    # 3) Dutch trace update (bump only if an action emitted)
    x_t <- c(x = 0, y = 0)
    if (!is.na(action)) x_t[action] <- 1
    z <- lambda_bin * z + (1 - alpha * lambda_bin * sum(z * x_t)) * x_t
    
    # 4) Environment: schedule rewards from responses, deliver any due now
    r_t <- 0.0
    if (!is.na(action) && env_type == "response_triggered") {
      if (runif(1) < p_trigger[action]) {
        delay <- rexp(1, rate = 1 / delay_mean_s[action])
        due_bin <- b + max(1L, ceiling(delay / dt))  # at least next bin
        reward_due_bins <- c(reward_due_bins, due_bin)
        reward_amounts  <- c(reward_amounts,  reward_amount[action])
      }
    }
    # Deliver all rewards due at this bin
    if (length(reward_due_bins)) {
      idx <- which(reward_due_bins == b)
      if (length(idx)) {
        r_t <- sum(reward_amounts[idx])
        # remove delivered
        keep <- setdiff(seq_along(reward_due_bins), idx)
        reward_due_bins <- reward_due_bins[keep]
        reward_amounts  <- reward_amounts[keep]
      }
    }
    
    # 5) TD error (average-reward differential) and Q update via traces
    delta <- r_t - Rbar
    Q <- Q + alpha * delta * z
    
    # 6) Update average-reward baseline
    Rbar <- (1 - eta) * Rbar + eta * r_t
    
    # 7) Log
    out$responded[b] <- respond
    out$action[b]    <- if (is.na(action)) NA_character_ else action
    out$r[b]         <- r_t
    out$delta[b]     <- delta
    out$Qx[b]        <- Q["x"];  out$Qy[b] <- Q["y"]
    out$zx[b]        <- z["x"];  out$zy[b] <- z["y"]
    out$Rbar[b]      <- Rbar
    out$p_respond[b] <- p_respond
    out$p_x[b]       <- if (all(is.na(p_a))) NA_real_ else p_a["x"]
    out$p_y[b]       <- if (all(is.na(p_a))) NA_real_ else p_a["y"]
    out$t_since_s[b] <- t_since_s
  }
  
  out
}

# ----------------------------
# Small helper to plot dynamics
# ----------------------------
plot_dynamics <- function(sim_df, title = NULL, caption = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting. Install it via install.packages('ggplot2').")
  }
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("Package 'patchwork' is required for combined plots. Install it via install.packages('patchwork').")
  }
  
  reward_events <- sim_df[sim_df$r > 0, c("time_s", "r"), drop = FALSE]
  response_events <- sim_df[sim_df$responded, c("time_s"), drop = FALSE]
  
  reward_plot <- ggplot2::ggplot(sim_df, ggplot2::aes(time_s, r)) +
    ggplot2::geom_segment(
      data = reward_events,
      ggplot2::aes(x = time_s, xend = time_s, y = 0, yend = r),
      linewidth = 0.5, color = "steelblue"
    ) +
    ggplot2::geom_rug(
      data = response_events, ggplot2::aes(x = time_s), inherit.aes = FALSE,
      color = "gray40", sides = "b"
    ) +
    ggplot2::labs(title = "Reward deliveries", x = "Time (s)", y = "Reward") +
    ggplot2::theme_minimal(base_size = 12)
  
  q_long <- rbind(
    data.frame(time_s = sim_df$time_s, series = "Qx", value = sim_df$Qx),
    data.frame(time_s = sim_df$time_s, series = "Qy", value = sim_df$Qy)
  )
  
  q_plot <- ggplot2::ggplot(q_long, ggplot2::aes(time_s, value, color = series)) +
    ggplot2::geom_line(linewidth = 0.7, na.rm = TRUE) +
    ggplot2::labs(title = "Action values (Q)", x = "Time (s)", y = "Value", color = NULL) +
    ggplot2::scale_color_manual(values = c("Qx" = "#1b9e77", "Qy" = "#d95f02")) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(legend.position = "top")
  
  rbar_plot <- ggplot2::ggplot(sim_df, ggplot2::aes(time_s, Rbar)) +
    ggplot2::geom_line(color = "#7570b3", linewidth = 0.7, na.rm = TRUE) +
    ggplot2::labs(title = "Average reward (Rbar)", x = "Time (s)", y = "Rbar") +
    ggplot2::theme_minimal(base_size = 12)
  
  delta_plot <- ggplot2::ggplot(sim_df, ggplot2::aes(time_s, delta)) +
    ggplot2::geom_line(color = "gray40", linewidth = 0.7, na.rm = TRUE) +
    ggplot2::labs(title = "TD error (delta)", x = "Time (s)", y = "Delta") +
    ggplot2::theme_minimal(base_size = 12)
  
  combined_plot <- reward_plot / q_plot / rbar_plot / delta_plot
  if (!is.null(title) || !is.null(caption)) {
    combined_plot <- combined_plot + patchwork::plot_annotation(
      title = title, caption = caption
    )
  }
  print(combined_plot)
  
  invisible(list(
    combined = combined_plot,
    reward = reward_plot,
    q_values = q_plot,
    average_reward = rbar_plot,
    delta = delta_plot
  ))
}

summarize_dynamics <- function(sim_df) {
  dt <- if (nrow(sim_df) > 1) sim_df$time_s[2] - sim_df$time_s[1] else NA_real_
  total_time <- if (!is.na(dt)) nrow(sim_df) * dt else NA_real_
  action_mask <- !is.na(sim_df$action)
  
  responses_per_s <- if (!is.na(dt) && dt > 0) mean(sim_df$responded) / dt else NA_real_
  choice_x <- if (any(action_mask)) mean(sim_df$action[action_mask] == "x") else NA_real_
  reward_rate <- if (!is.na(total_time) && total_time > 0) sum(sim_df$r) / total_time else NA_real_
  avg_delta <- mean(sim_df$delta)
  
  c(
    responses_per_s = responses_per_s,
    choice_x = choice_x,
    reward_rate = reward_rate,
    avg_delta = avg_delta
  )
}

run_example_scenarios <- function(show_plots = interactive()) {
  base_args <- list(n_bins = 4000, dt = 0.05)
  scenarios <- get_example_scenarios()
  
  summaries <- lapply(scenarios, function(cfg) {
    sim_args <- modifyList(base_args, cfg$args)
    sim <- do.call(simulate_free_operant, sim_args)
    
    if (isTRUE(show_plots)) {
      message("\nScenario: ", cfg$name, " - ", cfg$desc)
      plot_dynamics(sim, title = cfg$name, caption = cfg$desc)
    }
    
    metrics <- summarize_dynamics(sim)
    data.frame(
      scenario = cfg$name,
      description = cfg$desc,
      responses_per_s = metrics["responses_per_s"],
      choice_x = metrics["choice_x"],
      reward_rate = metrics["reward_rate"],
      avg_delta = metrics["avg_delta"],
      row.names = NULL,
      check.names = FALSE
    )
  })
  
  summary_df <- do.call(rbind, summaries)
  rownames(summary_df) <- NULL
  summary_df
}

get_example_scenarios <- function() {
  list(
    list(
      name = "Baseline symmetric",
      desc = "Balanced environment with default parameters.",
      args = list(seed = 1230)
    ),
    list(
      name = "Pure exploration",
      desc = "Low kappa without stickiness keeps choices balanced under symmetry.",
      args = list(seed = 1235, kappa = 1.2, omega_sticky = 0.0, beta_t_vigor = 0.035)
    ),
    list(
      name = "Sticky streaks",
      desc = "Low kappa plus strong stickiness produces long streaks on the same action.",
      args = list(seed = 1231, kappa = 1.5, omega_sticky = 0.9, beta_t_vigor = 0.04)
    ),
    list(
      name = "Deterministic symmetric",
      desc = "High kappa with zero stickiness in a symmetric task locks in after early noise.",
      args = list(seed = 1236, kappa = 12.0, omega_sticky = 0.0)
    ),
    list(
      name = "Asymmetric payoffs",
      desc = "High kappa plus richer X rewards yields strong exploitation.",
      args = list(
        seed = 1232, kappa = 9.0, omega_sticky = 0.0,
        p_trigger = c(x = 0.85, y = 0.45),
        reward_amount = c(x = 1.0, y = 0.3)
      )
    ),
    list(
      name = "Delayed rewards, short trace",
      desc = "Long reward delays with short eligibility (tau_e) hinder credit assignment.",
      args = list(
        seed = 1237, tau_e = 0.2,
        delay_mean_s = c(x = 1.2, y = 1.2),
        alpha = 0.08
      )
    ),
    list(
      name = "Long eligibility",
      desc = "Longer traces (tau_e) and slower learning spread credit over time.",
      args = list(seed = 1233, tau_e = 1.2, alpha = 0.05, eta = 0.003)
    ),
    list(
      name = "Fast baseline",
      desc = "Large eta lets the average-reward baseline track payoffs quickly.",
      args = list(seed = 1238, eta = 0.05)
    ),
    list(
      name = "Slow baseline",
      desc = "Tiny eta keeps the baseline sluggish, yielding larger delta swings.",
      args = list(seed = 1239, eta = 0.001)
    ),
    list(
      name = "High vigor",
      desc = "Elevated vigor parameters yield denser responding despite refractory limits.",
      args = list(seed = 1240, beta0_vigor = -2.5, beta_t_vigor = 0.06, refractory_s = 0.05)
    ),
    list(
      name = "Hard refractory",
      desc = "Half-second refractory windows create pronounced pauses between responses.",
      args = list(seed = 1241, refractory_s = 0.5)
    ),
    list(
      name = "Weak vigor",
      desc = "Lower vigor intercept/slope suppresses response initiation.",
      args = list(seed = 1234, beta0_vigor = -5.8, beta_t_vigor = 0.01, beta_q_vigor = 0.4)
    )
  )
}
