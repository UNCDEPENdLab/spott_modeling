#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(rstan)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
})

source("eligibility_test.R")

n_sessions <- 5L
n_bins_per_session <- 4000L
dt <- 0.05
true_params <- list(
  seed = 24680,
  n_bins = n_bins_per_session,
  dt = dt,
  alpha = 0.08,
  eta = 0.005,
  tau_e = 0.6,
  beta0_vigor = -4.0,
  beta_t_vigor = 0.03,
  beta_q_vigor = 0.6,
  refractory_s = 0.10,
  kappa = 9.0,
  omega_sticky = 0.0,
  p_trigger = c(x = 0.85, y = 0.45),
  delay_mean_s = c(x = 0.4, y = 0.4),
  reward_amount = c(x = 1.0, y = 0.3)
)

sim_sessions <- lapply(seq_len(n_sessions), function(s) {
  sim_args <- modifyList(true_params, list(seed = true_params$seed + s, n_bins = n_bins_per_session))
  sim <- do.call(simulate_free_operant, sim_args)
  sim$session <- s
  sim
})
sim_data <- bind_rows(sim_sessions)
action_idx <- ifelse(
  is.na(sim_data$action), 0L,
  ifelse(sim_data$action == "x", 1L, 2L)
)

stan_data <- list(
  B = nrow(sim_data),
  S = n_sessions,
  dt = dt,
  refractory_bins = ceiling(true_params$refractory_s / dt),
  reward = sim_data$r,
  responded = as.integer(sim_data$responded),
  action_idx = action_idx,
  session_id = sim_data$session
)

stan_file <- "free_operant_fit.stan"
rstan_options(auto_write = TRUE)
options(mc.cores = 1L)
sm <- stan_model(stan_file)

fit <- sampling(
  sm,
  data = stan_data,
  seed = 54321,
  chains = 2,
  iter = 1000,
  warmup = 400,
  refresh = 200,
  control = list(adapt_delta = 0.9)
)

param_names <- c(
  "alpha", "eta", "tau_e", "beta0_vigor",
  "beta_t_vigor", "beta_q_vigor", "kappa", "omega_sticky"
)

stan_summary <- summary(fit, pars = param_names)$summary
posterior_summary <- as.data.frame(stan_summary) %>%
  mutate(variable = rownames(stan_summary)) %>%
  select(variable, mean, sd, `2.5%`, `97.5%`, n_eff, Rhat)

truth_tbl <- tibble(
  variable = param_names,
  true_value = c(
    true_params$alpha,
    true_params$eta,
    true_params$tau_e,
    true_params$beta0_vigor,
    true_params$beta_t_vigor,
    true_params$beta_q_vigor,
    true_params$kappa,
    true_params$omega_sticky
  )
)

comparison <- posterior_summary %>%
  left_join(truth_tbl, by = "variable") %>%
  mutate(
    error = mean - true_value,
    rel_error = if_else(true_value == 0, NA_real_, error / true_value)
  ) %>%
  as_tibble()

write_csv(comparison, "stan_fit_parameter_recovery.csv")
print(comparison, n = nrow(comparison))

saveRDS(list(
  fit = fit,
  posterior_summary = comparison,
  sim_data = sim_data,
  true_params = true_params
), file = "stan_fit_results.rds")
