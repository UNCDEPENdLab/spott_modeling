# Utilities for fitting the direct-actor policy (Algorithm 5.1 in Niv, 2007)
# to empirical event logs via maximum likelihood.

softmax_vec <- function(logits) {
  shift <- logits - max(logits)
  exp_shift <- exp(shift)
  exp_shift / sum(exp_shift)
}

pack_params <- function(M, Alpha, Beta) {
  c(as.vector(M), as.vector(Alpha), as.vector(Beta))
}

unpack_params <- function(vec, Nactions, Nstates) {
  block <- Nactions * Nstates
  M <- matrix(vec[seq_len(block)], nrow = Nactions, ncol = Nstates)
  Alpha <- matrix(vec[(block + 1):(2 * block)], nrow = Nactions, ncol = Nstates)
  Beta <- matrix(vec[(2 * block + 1):(3 * block)], nrow = Nactions, ncol = Nstates)
  list(M = M, Alpha = Alpha, Beta = Beta)
}

compute_loglik_grad <- function(M,
                                Alpha,
                                Beta,
                                Nactions,
                                Nstates,
                                states,
                                actions,
                                latencies,
                                minA,
                                minB,
                                includeR,
                                R_avg) {
  loglik <- 0
  grad_M <- matrix(0, nrow = Nactions, ncol = Nstates)
  grad_A <- matrix(0, nrow = Nactions, ncol = Nstates)
  grad_B <- matrix(0, nrow = Nactions, ncol = Nstates)
  scale_factor <- if (includeR) sqrt(R_avg) else 1

  state_levels <- sort(unique(states))
  for (s in state_levels) {
    idx_state <- which(states == s)
    if (!length(idx_state)) next
    logits <- M[, s]
    probs <- softmax_vec(logits)

    acts_state <- actions[idx_state]
    loglik <- loglik + sum(log(probs[acts_state]))
    counts <- tabulate(acts_state, nbins = Nactions)
    grad_M[, s] <- grad_M[, s] + counts - length(idx_state) * probs

    unique_actions <- which(counts > 0)
    for (a in unique_actions) {
      idx_action <- idx_state[acts_state == a]
      lat_vec <- pmax(latencies[idx_action], 1e-8)
      shape <- exp(Alpha[a, s]) + minA
      scale_raw <- exp(Beta[a, s]) + minB
      scale <- scale_raw / scale_factor

      loglik <- loglik + sum(dgamma(lat_vec, shape = shape, scale = scale, log = TRUE))
      d_shape <- sum(log(lat_vec) - log(scale) - digamma(shape))
      grad_A[a, s] <- grad_A[a, s] + exp(Alpha[a, s]) * d_shape

      d_scale <- sum((lat_vec - shape * scale) / (scale^2))
      grad_B[a, s] <- grad_B[a, s] + (exp(Beta[a, s]) / scale_factor) * d_scale
    }
  }

  grad <- c(as.vector(grad_M), as.vector(grad_A), as.vector(grad_B))
  list(loglik = loglik, grad = grad)
}

#' Fit the direct-actor policy parameters by maximum likelihood.
#'
#' @param data Data frame with columns state (1..Nstates), action (1..Nactions), latency (seconds).
#' @param Nstates Number of states in the discretized task.
#' @param Actions Character vector of action labels (length defines Nactions).
#' @param minA, minB Constants used in the original policy parameterization.
#' @param includeR Whether to divide gamma scales by sqrt(R_avg) as in LearningDriver.
#' @param R_avg Average reward baseline used when includeR = TRUE (must be >0).
#' @param init Optional list with elements M, Alpha, Beta (each Nactions x Nstates) for initialization.
#' @param optim_control List of arguments passed to optim().
#' @return List containing fitted parameter matrices and optimization details.
fit_learning_driver_mle <- function(data,
                                    Nstates,
                                    Actions = c("LP", "Other", "NP"),
                                    minA = 1.05,
                                    minB = 0.05,
                                    includeR = TRUE,
                                    R_avg = 1,
                                    init = NULL,
                                    optim_control = list(maxit = 200)) {
  if (!all(c("state", "action", "latency") %in% names(data))) {
    stop("Data must contain columns state, action, and latency (in seconds).")
  }
  Nactions <- length(Actions)
  states <- data$state
  actions <- data$action
  latencies <- data$latency

  if (any(states < 1) || any(states > Nstates)) {
    stop("State indices must be within 1..Nstates.")
  }
  if (any(actions < 1) || any(actions > Nactions)) {
    stop("Action indices must be within 1..Nactions.")
  }
  if (any(latencies <= 0)) {
    warning("Latencies <= 0 detected; they will be clamped to 1e-8 seconds in the likelihood.")
  }
  if (includeR && R_avg <= 0) {
    stop("R_avg must be positive when includeR = TRUE.")
  }

  if (is.null(init)) {
    init <- list(
      M = matrix(0, nrow = Nactions, ncol = Nstates),
      Alpha = matrix(0, nrow = Nactions, ncol = Nstates),
      Beta = matrix(0, nrow = Nactions, ncol = Nstates)
    )
  }

  theta0 <- pack_params(init$M, init$Alpha, init$Beta)
  cache <- new.env(parent = emptyenv())

  ll_grad <- function(theta) {
    params <- unpack_params(theta, Nactions, Nstates)
    compute_loglik_grad(
      params$M,
      params$Alpha,
      params$Beta,
      Nactions,
      Nstates,
      states,
      actions,
      latencies,
      minA = minA,
      minB = minB,
      includeR = includeR,
      R_avg = R_avg
    )
  }

  fn <- function(theta) {
    stats <- ll_grad(theta)
    cache$grad <- stats$grad
    cache$theta <- theta
    -stats$loglik
  }

  gr <- function(theta) {
    if (!is.null(cache$theta) && length(cache$theta) == length(theta) &&
        all(cache$theta == theta)) {
      -cache$grad
    } else {
      -ll_grad(theta)$grad
    }
  }

  opt <- optim(theta0, fn = fn, gr = gr, method = "BFGS", control = optim_control)
  params_hat <- unpack_params(opt$par, Nactions, Nstates)

  list(
    params = params_hat,
    logLik = -opt$value,
    optim = opt
  )
}
