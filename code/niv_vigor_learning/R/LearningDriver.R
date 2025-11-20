# ---------------------------------------------------
# driver for running online RL for rates of behavior
# ---------------------------------------------------
#
# This is a faithful line-by-line translation of LearningDriver.m.
# Where the MATLAB logic diverges from the dissertation description
# (see niv_dissertation.txt:2063-2103 for the narrative about
# lever presses followed by nose pokes), explicit comments below
# flag those inconsistencies.

run_learning_driver <- function(
    Actions = c("LP", "Other", "NP"),
    EatTime = 6,
    rho = c(-0.15, 0.15, -0.15),
    s = 0.5,
    d = 1.5,
    schedule_type = "Random",
    schedule = "Interval",
    ratio = 10,
    interval = 30,
    Utility = 90,
    LearnMode = "DirectActor",
    constrained = 0,
    includeR = 1,
    Mlearning = 1,
    learn = 1,
    Nsmooth = 5,
    Reward_times = 0,
    Ndata = 2000,
    Nsessions = 1000,
    max_actions = 200,
    figures = 1
) {
  Nactions <- length(Actions)
  ratio_val <- if (length(ratio) == 0) 1 else ratio[1]
  k_v <- matrix(d, nrow = Nactions, ncol = Nactions)
  diag(k_v) <- s

  if (length(rho) != Nactions || nrow(k_v) != Nactions) {
    stop("DRIVER ERROR: Update rho or k_v to match the number of actions.")
  }

  cat(sprintf("\n The utility of the reward in this run is %d", Utility))

  if (tolower(schedule) == "interval") {
    dt <- 10
    times <- seq(1 / dt, 30, by = 1 / dt)
  } else {
    dt <- 20
    times <- seq(1 / dt, 5, by = 1 / dt)
  }
  Ntimes <- length(times)

  # Build the state index tensor (faithful to LearningDriver.m)
  max_dim <- if (tolower(schedule) == "ratio") max(ratio_val, 1) else Ntimes
  Sindex <- array(NA_integer_, dim = c(max_dim, Nactions, 2))
  n <- 0
  if (tolower(schedule) == "ratio") {
    if (tolower(schedule_type) == "random") {
      for (i in seq_len(Nactions)) {
        for (j in 1:2) {
          n <- n + 1
          Sindex[1, i, j] <- n
        }
      }
      Sindex[1, Nactions, 2] <- NA
      n <- n - 1
      Slog <- Sindex[1, 1, 1]
    } else {
      for (i in seq_len(Nactions)) {
        for (j in 1:2) {
          for (k in seq_len(ratio_val)) {
            n <- n + 1
            Sindex[k, i, j] <- n
          }
        }
      }
      for (k in seq_len(ratio_val)) {
        Sindex[k, Nactions, 2] <- NA
        n <- n - 1
      }
      Slog <- Sindex[min(2, ratio_val), 1, 1]
    }
  } else {
    for (i in seq_len(Nactions)) {
      for (j in 1:2) {
        for (k in seq_len(Ntimes)) {
          n <- n + 1
          Sindex[k, i, j] <- n
        }
      }
    }
    for (k in seq_len(Ntimes)) {
      Sindex[k, Nactions, 2] <- NA
      n <- n - 1
    }
    Slog <- Sindex[1, 1, 1]
  }
  Nstates <- n

  # Pre-sample baiting times
  Bait <- rep(0, Ndata)
  if (tolower(schedule) == "interval") {
    if (length(Reward_times) == Ndata) {
      Bait <- Reward_times
    } else if (tolower(schedule_type) == "random") {
      for (i in seq_len(Ndata)) {
        a <- runif(1, 0, 10 * interval)
        while (runif(1) > (exp(-a / interval) / interval)) {
          a <- runif(1, 0, 10 * interval)
        }
        Bait[i] <- a
      }
    } else {
      Bait <- c(
        rep(1 / dt, 30),
        runif(30, 1, 3),
        runif(60, 7.5, 22.5),
        runif(max(Ndata - 120, 0), 15, 45)
      )
    }
  }

  cat(sprintf("\n%s %s\n", schedule_type, schedule))

  i_r <- 1
  a_prev <- Nactions
  t_s <- 1

  Temp <- 7
  minTemp <- 0.5
  etaQ <- 0.05
  etaV <- 0.05
  etaM <- 0.01
  etaA <- 0.01
  etaB <- 0.01
  etaR <- 0.0005
  logEtaR <- log(etaR)
  minA <- 1.05
  minB <- 0.05
  Rmin <- 0.01
  TempDecayRate <- 0.9997
  EtaDecayRate <- 0.999
  Mdecay <- 1
  N_R <- 10000

  if (learn) {
    V <- rep(0, Nstates)
    dV <- rep(0, Nstates)
    R <- Rmin
    if (tolower(LearnMode) != "directactor") {
      Q <- array(0, dim = c(Ntimes, Nactions, Nstates))
      dQ <- array(0, dim = c(Ntimes, Nactions, Nstates))
      L_Q <- Ntimes * Nactions
    } else {
      if (Mlearning) {
        M <- matrix(1, nrow = Nactions, ncol = Nstates)
      } else {
        Q <- matrix(0, nrow = Nactions, ncol = Nstates)
        dQ <- matrix(0, nrow = Nactions, ncol = Nstates)
      }
      Alpha <- matrix(max(times) / 3, nrow = Nactions, ncol = Nstates)
      Beta <- matrix(1, nrow = Nactions, ncol = Nstates)
      if (constrained == 0 && Mlearning) {
        M <- log(M)
      }
      if (constrained == 0) {
        Alpha <- matrix(1, nrow = Nactions, ncol = Nstates)
        Beta <- matrix(-1.5, nrow = Nactions, ncol = Nstates)
      }
    }
  }

  cum_count <- 0
  Reward_flag <- 0
  Nresponses <- 0
  t <- 0
  T <- 0
  N_session <- 1
  Nhalfway <- 0
  DM <- rep(0, Nactions)
  DAlpha <- 0
  DBeta <- 0

  initlength <- if (tolower(schedule) == "interval") Ndata * 20 else Ndata * ratio_val * 2
  Responses <- rep(0, initlength)
  Times <- rep(0, initlength)
  Latencies <- rep(0, initlength)
  EatTimes <- rep(0, initlength)
  EatTrials <- rep(0, initlength)
  Deltas <- rep(0, initlength)
  Rewards <- rep(0, Ndata)
  Tconsume <- rep(0, Ndata)
  rate_bins <- max(1, as.integer(max(times) * 2 * dt / 5))
  RateLP <- rep(0, rate_bins)
  BaitNorm <- pmin(Bait, max(times) * 2)
  Session_lp <- rep(0, Nsessions + 1)
  R_log <- rep(0, initlength)
  R_log_real <- c()
  Rreal <- 0
  LogBeta <- NULL
  LogAlpha <- NULL
  LogM <- NULL
  if (tolower(LearnMode) == "directactor") {
    LogBeta <- rep(0, initlength)
    LogAlpha <- rep(0, initlength)
    LogM <- matrix(0, nrow = Nactions, ncol = initlength)
    AlphaInt <- 1
    BetaInt <- 1
  }
  LogS <- rep(0, initlength)
  if (tolower(schedule_type) == "random" && tolower(schedule) == "ratio") {
    LogV <- matrix(0, nrow = Nstates, ncol = initlength)
  } else {
    LogV <- rep(0, initlength)
  }

  if (tolower(LearnMode) == "onpolicyq") {
    S <- Sindex[t_s, a_prev, i_r]
    Qtemp <- as.numeric(Q[, , S])
    Qtemp <- Qtemp - max(Qtemp)
    SoftQtemp <- exp(Qtemp / Temp) / sum(exp(Qtemp / Temp))
  }

  Wsmooth <- 1 - (1:Nsmooth) / (Nsmooth + 1)
  Wsmooth <- c(rev(Wsmooth), 1, Wsmooth)

  RunAborted <- FALSE

  histc <- function(x, bins) {
    cut_idx <- findInterval(x, bins, rightmost.closed = TRUE, left.open = FALSE)
    tab <- tabulate(cut_idx, nbins = length(bins))
    tab
  }

  for (trial in seq_len(Ndata)) {
    if (trial %% 1000 == 0) {
      cat(sprintf("\n ---> working... already at trial %d", trial))
    }

    if (tolower(LearnMode) == "directactor" && learn) {
      V <- V + etaV * dV
      dV <- dV * 0
      if (!Mlearning) {
        Q <- Q + etaQ * dQ
        dQ <- dQ * 0
        Q <- Q - max(Q)
      }
    }

    if (tolower(LearnMode) == "hybridsarsa") {
      V <- V + etaV * dV
      g <- max(V)
      V <- V - g
      Q <- Q - g
      Q <- Q + etaQ * dQ
      dV <- dV * 0
      dQ <- dQ * 0
    }

    if (trial == Ndata / 2) {
      Nhalfway <- Nresponses
    }

    rewarded <- FALSE
    t <- 0
    num_actions <- 0

    while (!rewarded && num_actions < max_actions) {
      idx_t <- min(t_s, dim(Sindex)[1])
      S <- Sindex[idx_t, a_prev, i_r]
      if (is.na(S)) S <- Slog
      i_prev <- i_r
      t_prev <- t_s

      if (tolower(LearnMode) == "directactor") {
        if (constrained) {
          probs <- M[, S] / sum(M[, S])
          cum <- cumsum(probs)
          action <- which(runif(1) <= cum)[1]
          if (is.na(action)) action <- which.max(probs)
          tau <- rgamma(1, shape = Alpha[action, S], rate = 1 / Beta[action, S])
          gammean <- Alpha[action, S] * Beta[action, S]
          gamstd <- sqrt(Alpha[action, S]) * Beta[action, S]
        } else {
          if (Mlearning) {
            pref <- exp(M[, S] - max(M[, S]))
            probs <- pref / sum(pref)
            action <- which(runif(1) <= cumsum(probs))[1]
            if (is.na(action)) action <- which.max(probs)
          } else {
            Qtemp <- exp((Q[, S] - max(Q[, S])) / Temp)
            probs <- Qtemp / sum(Qtemp)
            action <- which(runif(1) <= cumsum(probs))[1]
            if (is.na(action)) action <- which.max(probs)
          }
          Atilda <- exp(Alpha[action, S]) + minA
          Btilda <- exp(Beta[action, S]) + minB
          if (includeR) {
            Btilda <- Btilda / sqrt(max(R, .Machine$double.eps))
          }
          tau <- rgamma(1, shape = Atilda, scale = Btilda)
          gammean <- Atilda * Btilda
          gamstd <- sqrt(Atilda) * Btilda
        }
        tau <- max(tau, 1 / dt)
        tau <- min(tau, max(3 * gamstd + gammean, 200))
        if (tau > 200) {
          cat(sprintf("\nStep %d: Exceptionally long tau (%3.2f) for action %d and state (%d,%d,%d)",
                      Nresponses + 1, tau, action, t_s, a_prev, i_r))
        }
        tau <- tau * dt
      } else {
        if (tolower(LearnMode) != "onpolicyq") {
          Qtemp <- as.numeric(Q[, , S])
          Qtemp <- Qtemp - max(Qtemp)
          probs <- exp(Qtemp / Temp)
          probs <- probs / sum(probs)
          index <- which(runif(1) <= cumsum(probs))[1]
        } else {
          index <- which(runif(1) <= cumsum(SoftQtemp))[1]
        }
        action <- ceiling(index / Ntimes)
        tau <- index - (action - 1) * Ntimes
      }

      t <- t + tau / dt
      T <- T + tau / dt

      Reward <- 0

      if (i_r == 2) {
        if (action == Nactions) {
          Reward <- 1
          i_r <- 1
        }
        if (tolower(schedule) == "interval") {
          if (tolower(schedule_type) == "variable") {
            t_s <- min(t_s + ceiling(tau), Ntimes)
          } else if (action != 1) {
            t_s <- min(t_s + ceiling(tau), Ntimes)
          }
        }
      } else {
        if (action == 1) {
          if (tolower(schedule_type) %in% c("random", "variable")) {
            if (tolower(schedule) == "ratio") {
              cum_count <- cum_count + 1
              if ((runif(1) < 1 / ratio_val) || cum_count == 3 * ratio_val) {
                i_r <- 2
                rewarded <- TRUE
                Bait[trial] <- t
                cum_count <- 0
              }
            } else {
              if (t > Bait[trial]) {
                i_r <- 2
                rewarded <- TRUE
                # NOTE: dissertation states that a nose poke should follow the lever press
                # to consume the reinforcer (niv_dissertation.txt:2063-2103). However,
                # the original MATLAB sets rewarded=TRUE immediately when the interval
                # condition is satisfied, before any NP occurs. This R port keeps that
                # logic to remain faithful to the code despite the conceptual mismatch.
              }
              t_s <- 1
            }
          } else if (tolower(schedule_type) == "fixed") {
            if (t_s == ratio_val) {
              i_r <- 2
              rewarded <- TRUE
              Bait[trial] <- t
              t_s <- 1
            } else {
              t_s <- t_s + 1
            }
          } else {
            if (t > Bait[trial]) {
              i_r <- 2
              rewarded <- TRUE
              t_s <- 1
            } else {
              t_s <- min(t_s + ceiling(tau), Ntimes)
            }
          }
        } else if (tolower(schedule) == "interval") {
          t_s <- min(t_s + ceiling(tau), Ntimes)
        }
      }

      Nresponses <- Nresponses + 1
      Responses[Nresponses] <- action
      Latencies[Nresponses] <- tau / dt
      Times[Nresponses] <- t
      num_actions <- num_actions + 1
      EatTrials[Nresponses] <- 0
      EatTimes[Nresponses] <- 0

      if (Reward) {
        EatTrials[Nresponses] <- 1
        Eat <- EatTime
        EatTimes[Nresponses] <- Eat
        if (trial > 1) {
          Tconsume[trial - 1] <- t
        }
        t <- t + Eat
        T <- T + Eat
        if (tolower(schedule) == "interval") {
          t_s <- min(t_s + round(Eat * dt), Ntimes)
        }
      }

      if (action == 1) {
        Session_lp[N_session] <- Session_lp[N_session] + 1
        if (tolower(schedule) == "interval" && t < BaitNorm[trial] && Nhalfway) {
          idx <- ceiling(t * dt / 5)
          if (idx >= 1 && idx <= length(RateLP)) {
            RateLP[idx] <- RateLP[idx] + 1
          }
        }
      }

      if (rewarded) {
        Rewards[trial] <- t
      }

      if (learn) {
        Cost <- Reward * Utility + rho[action] - k_v[a_prev, action] / (tau / dt)
        timeInt <- exp(log(1 - exp(-etaR * (tau / dt + EatTimes[Nresponses]))) - logEtaR)

        if (tolower(LearnMode) == "directactor") {
          Snext <- Sindex[min(t_s, dim(Sindex)[1]), action, i_r]
          if (is.na(Snext)) Snext <- Slog
          delta <- Cost - timeInt * R + V[Snext] - V[S]
          if (abs(delta) > 2 * Utility) {
            cat(sprintf("\nUnusually large PE (%3.2f) at trial %d timestep %d",
                        delta, trial, Nresponses))
          }
          if (constrained) {
            DAlpha <- (log(tau / (dt * Beta[action, S])) - digamma(Alpha[action, S]))
            DBeta <- 1 / Beta[action, S] * (tau / (dt * Beta[action, S]) - Alpha[action, S])
            Alpha[action, S] <- Alpha[action, S] + etaA * delta * DAlpha
            Beta[action, S] <- Beta[action, S] + etaB * delta * DBeta
            for (i in seq_len(Nactions)) {
              DM[i] <- (i == action) / M[action, S] - 1 / sum(M[, S])
            }
            M[, S] <- pmax(M[, S] + etaM * delta * DM, 0)
            Alpha[action, S] <- max(Alpha[action, S], 1.05)
            Beta[action, S] <- max(Beta[action, S], 0.05)
          } else {
            Atilda <- exp(Alpha[action, S]) + minA
            Btilda <- exp(Beta[action, S]) + minB
            if (includeR) {
              Btilda <- Btilda / sqrt(max(R, .Machine$double.eps))
            }
            Stabilize <- -abs(Alpha[action, S]) - abs(Beta[action, S])
            AlphaInt <- exp(Alpha[action, S] + Stabilize)
            BetaInt <- exp(Beta[action, S] + Stabilize)
            if (includeR) {
              BetaInt <- BetaInt / sqrt(max(R, .Machine$double.eps))
            }
            DAlpha <- (log(tau / (dt * Btilda)) - digamma(Atilda))
            DBeta <- 1 / Btilda * (tau / (dt * Btilda) - Atilda)
            Alpha[action, S] <- Alpha[action, S] + etaA * AlphaInt * delta * DAlpha
            Beta[action, S] <- Beta[action, S] + etaB * BetaInt * delta * DBeta
            if (Mlearning) {
              DM <- (seq_len(Nactions) == action) - exp(M[, S]) / sum(exp(M[, S]))
              M[, S] <- M[, S] + etaM * delta * DM
              if (any(M[, S] > 700) || Alpha[action, S] > 700 || Beta[action, S] > 700) {
                RunAborted <- TRUE
                cat("\nRun aborted to prevent overflow.\n")
                break
              }
            }
          }
          if (!is.null(LogS)) {
            LogS[Nresponses] <- S
          }
          if (!is.null(LogM)) {
            if (Mlearning) {
              LogM[, Nresponses] <- M[, S]
            } else {
              LogM[, Nresponses] <- Q[, S]
            }
          }
          if (!is.null(LogBeta)) {
            LogBeta[Nresponses] <- Beta[1, S]
          }
          if (!is.null(LogAlpha)) {
            LogAlpha[Nresponses] <- Alpha[1, S]
          }
          if (tolower(schedule_type) == "random" && tolower(schedule) == "ratio") {
            LogV[, Nresponses] <- V
          } else {
            LogV[Nresponses] <- V[Slog]
          }
        }

        if (tolower(LearnMode) == "hybridsarsa") {
          Snext <- Sindex[min(t_s, dim(Sindex)[1]), action, i_r]
          if (is.na(Snext)) Snext <- Slog
          delta <- Cost - timeInt * R + V[Snext]
          dV[S] <- dV[S] + delta - V[S]
          if (tau < Nsmooth + 1 || (tau + Nsmooth) > Ntimes) {
            dQ[tau, action, S] <- dQ[tau, action, S] + delta - Q[tau, action, S]
          } else {
            idx1 <- max(tau - Nsmooth, 1)
            idx2 <- min(tau + Nsmooth, Ntimes)
            dQ[idx1:idx2, action, S] <- dQ[idx1:idx2, action, S] +
              (delta - Q[tau, action, S]) * Wsmooth[1:(idx2 - idx1 + 1)]
          }
          V[S] <- (1 - etaV) * V[S] + etaV * delta
          Deltas[Nresponses] <- delta - Q[tau, action, S]
        }

        if (tolower(LearnMode) == "offpolicyq") {
          Snext <- Sindex[min(t_s, dim(Sindex)[1]), action, i_r]
          if (is.na(Snext)) Snext <- Slog
          delta <- -Q[tau, action, S] + Cost -
            (tau / dt + EatTimes[Nresponses]) * R +
            max(Q[, , Snext])
          Q[tau, action, S] <- Q[tau, action, S] + etaQ * delta
          Q <- Q - max(Q)
          Deltas[Nresponses] <- delta
        }

        if (tolower(LearnMode) == "onpolicyq") {
          Snext <- Sindex[min(t_s, dim(Sindex)[1]), action, i_r]
          if (is.na(Snext)) Snext <- Slog
          Qtemp <- as.numeric(Q[, , Snext])
          Qtemp2 <- Qtemp - max(Qtemp)
          SoftQtemp <- exp(Qtemp2 / Temp) / sum(exp(Qtemp2 / Temp))
          delta <- -Q[tau, action, S] + Cost -
            (tau / dt + EatTimes[Nresponses]) * R +
            sum(SoftQtemp * Qtemp)
          Q[tau, action, S] <- Q[tau, action, S] + etaQ * delta
          Q <- Q - max(Q)
          Deltas[Nresponses] <- delta
        }

        Rold <- R
        R <- R * exp(-etaR * (Latencies[Nresponses] + EatTimes[Nresponses])) + etaR * Cost
        if (abs(R - Rold) > 0.2) {
          cat(sprintf("\n Trial %d: latency %3.4f cost %3.4f, R change %3.2f",
                      trial, tau / dt, Cost, R - Rold))
        }
        R <- max(R, Rmin)
        R_log[Nresponses] <- R
      }

      a_prev <- action

      if (T > 30 * 60) {
        N_session <- N_session + 1
        T <- 0
      }

      if (RunAborted) break
    }

    if (RunAborted) break

    if (N_session > Nsessions) {
      Rewards <- Rewards[seq_len(trial)]
      Tconsume <- Tconsume[seq_len(trial)]
      Ndata <- trial
      break
    }

    if (num_actions == max_actions) {
      cat(sprintf("\nRun stopped after %d trials because of too many (%d) actions and no reward",
                  trial, Nresponses))
      RunAborted <- TRUE
      break
    }

    if (tolower(LearnMode) != "directactor") {
      Temp <- max(Temp * TempDecayRate, minTemp)
      etaQ <- etaQ * EtaDecayRate
      etaV <- etaV * EtaDecayRate
    }
  }

  Halfdata <- ceiling(Ndata / 2)
  Reward_summary <- c(
    mean(Rewards[Halfdata:(Ndata - 1)]),
    sd(Rewards[Halfdata:(Ndata - 1)])
  )
  Consump_summary <- c(
    mean(Tconsume[Halfdata:(Ndata - 1)]),
    sd(Tconsume[Halfdata:(Ndata - 1)])
  )
  Nacts_per_trial <- (Nresponses - Nhalfway) / (trial / 2)
  second_range <- if (Nresponses > Nhalfway) ((Nhalfway + 1):Nresponses) else integer(0)
  resp_second <- if (length(second_range)) Responses[second_range] else integer(0)
  lat_second <- if (length(second_range)) Latencies[second_range] else numeric(0)
  time_second <- if (length(second_range)) Times[second_range] else numeric(0)
  eat_second <- if (length(second_range)) EatTrials[second_range] else integer(0)
  L_lp <- if (length(resp_second)) lat_second[resp_second == 1] else numeric(0)
  L_np <- if (length(eat_second)) lat_second[eat_second == 1] else numeric(0)
  L_np_extra <- if (length(resp_second)) lat_second[resp_second == Nactions & eat_second == 0] else numeric(0)
  bait_interval <- mean(Bait[seq_len(Ndata)])

  bins <- seq(
    0,
    ceiling((Reward_summary[1] + 2 * Reward_summary[2]) / dt) * dt,
    by = 1 / (dt / 5)
  )
  bait_hist <- histc(Bait[(Halfdata + 1):Ndata], bins)
  if (length(bait_hist) == 0) bait_hist <- rep(0, length(bins))
  nrm <- Halfdata - cumsum(bait_hist)
  if (length(nrm) > 0) {
    nrm <- nrm[-length(nrm)]
  }
  bin_centers <- if (length(bins) > 1) bins[-length(bins)] + 0.5 else numeric(0)
  len <- min(length(nrm), length(bin_centers))
  nrm <- if (len) nrm[seq_len(len)] else numeric(0)
  bin_centers <- if (len) bin_centers[seq_len(len)] else numeric(0)
  valid <- nrm > 0
  nrm <- nrm[valid]
  bin_centers <- bin_centers[valid]
  if (length(nrm)) {
    RateLP <- RateLP[seq_len(length(nrm))]
  } else {
    RateLP <- numeric(0)
  }
  if (length(nrm) == 0) {
    LPs <- NPs <- Others <- numeric(0)
  } else if (tolower(schedule) == "ratio") {
    idx <- if (length(resp_second)) which(resp_second == 1) + Nhalfway else integer(0)
    LP_hist <- if (length(idx)) histc(Times[idx], bins) else rep(0, length(bins))
    Nvalid <- sum(nrm > Ndata / 5)
    Nvalid <- min(Nvalid, length(nrm))
    LPs <- if (Nvalid)
      LP_hist[seq_len(Nvalid)] * 60 * dt * 0.2 / nrm[seq_len(Nvalid)] else numeric(0)
  } else {
    LPs <- (RateLP[seq_along(nrm)] * 60 * dt * 0.2) / nrm
  }
  Others <- NULL
  if (Nactions > 2 && any(resp_second == (Nactions - 1))) {
    idx <- which(resp_second == (Nactions - 1)) + Nhalfway
    other_hist <- histc(Times[idx], bins) * 60 * dt * 0.2
    Others <- other_hist[seq_along(nrm)] / nrm
  }
  idx_np <- if (length(resp_second)) which(resp_second == Nactions) + Nhalfway else integer(0)
  np_hist <- if (length(idx_np)) histc(Times[idx_np], bins) else rep(0, length(bins))
  NPs <- np_hist[seq_along(nrm)] * 60 * dt * 0.2 / nrm

  if (!is.null(LogBeta)) {
    LogBeta <- LogBeta[seq_len(Nresponses)]
    LogAlpha <- LogAlpha[seq_len(Nresponses)]
    LogM <- LogM[, seq_len(Nresponses), drop = FALSE]
  }
  LogS <- LogS[seq_len(Nresponses)]
  Responses <- Responses[seq_len(Nresponses)]
  Times <- Times[seq_len(Nresponses)]
  Latencies <- Latencies[seq_len(Nresponses)]
  Deltas <- Deltas[seq_len(Nresponses)]
  R_log <- R_log[seq_len(Nresponses)]
  EatTimes <- EatTimes[seq_len(Nresponses)]
  EatTrials <- EatTrials[seq_len(Nresponses)]
  if (is.matrix(LogV)) {
    LogV <- LogV[, seq_len(Nresponses), drop = FALSE]
  } else {
    LogV <- LogV[seq_len(Nresponses)]
  }

  if (figures) {
    default_par <- graphics::par(no.readonly = TRUE)
    on.exit(graphics::par(default_par), add = TRUE)
    plot_colors <- rep("gray40", Nactions)
    if (Nactions >= 1) plot_colors[1] <- "blue"
    if (Nactions >= 2) plot_colors[Nactions] <- "red"
    if (Nactions >= 3) plot_colors[Nactions - 1] <- "darkgreen"

    reset_par <- function() {
      graphics::par(default_par)
    }

    # ---- Figure 1: Response rates over time ----
    valid_rates <- c(LPs, NPs, if (!is.null(Others)) Others else numeric(0))
    if (length(bin_centers) && any(is.finite(valid_rates))) {
      y_range <- range(valid_rates[is.finite(valid_rates)])
      x_range <- if (diff(range(bin_centers)) == 0) {
        rep(bin_centers[1], 2) + c(-1, 1)
      } else {
        range(bin_centers)
      }
      if (diff(y_range) == 0) {
        y_range <- y_range + c(-0.5, 0.5)
      }
      reset_par()
      plot(x_range, y_range, type = "n",
           xlab = "time (sec) after reward",
           ylab = "response rate per min in 2nd half of training",
           main = sprintf("R_avg = %s, <Trial length> = %3.2f sec",
                          if (exists("R")) sprintf("%3.2f", R) else "NA",
                          Reward_summary[1]))
      if (length(LPs)) {
        graphics::lines(bin_centers[seq_along(LPs)], LPs,
                        col = plot_colors[1], lwd = 2, pch = 8, type = "o")
      }
      if (length(NPs)) {
        graphics::lines(bin_centers[seq_along(NPs)], NPs,
                        col = plot_colors[Nactions], lwd = 2, lty = 2, pch = 8, type = "o")
      }
      if (!is.null(Others) && length(Others)) {
        idx <- seq_along(Others)
        graphics::lines(bin_centers[idx], Others,
                        col = plot_colors[max(1, Nactions - 1)], lwd = 2, pch = 16, type = "o")
      }
      legend_labels <- c("LP", "NP")
      legend_cols <- c(plot_colors[1], plot_colors[Nactions])
      if (!is.null(Others) && length(Others)) {
        legend_labels <- c("LP", "NP", "Other")
        legend_cols <- c(plot_colors[1], plot_colors[Nactions], plot_colors[max(1, Nactions - 1)])
      }
      graphics::legend("topright", legend = legend_labels,
                       col = legend_cols, lwd = 2, pch = c(8, 8, 16)[seq_along(legend_labels)], bty = "n")
    } else {
      message("Not enough actions in the back half of training to plot response rates.")
    }

    # ---- Figure 2: Policy/Q visualization ----
    if (tolower(LearnMode) == "directactor" && exists("Alpha") && exists("V")) {
      states_to_plot <- integer(0)
      if (tolower(schedule) == "interval") {
        states_to_plot <- c(Sindex[1, 1, 1], Sindex[1, 1, 2])
      } else if (tolower(schedule_type) == "fixed") {
        idx_seq <- if (dim(Sindex)[1] >= 2) as.integer(Sindex[2:dim(Sindex)[1], 1, 1]) else integer(0)
        states_to_plot <- c(Sindex[1, min(2, Nactions), 1], idx_seq, Sindex[1, 1, 2])
      } else {
        states_to_plot <- seq_len(Nstates)
      }
      states_to_plot <- unique(states_to_plot[!is.na(states_to_plot)])
      if (length(states_to_plot)) {
        n_panels <- if (length(states_to_plot) < 5) c(length(states_to_plot), 1) else c(ceiling(length(states_to_plot) / 3), 3)
        reset_par()
        graphics::par(mfrow = n_panels)
        for (s in states_to_plot) {
          idx_info <- which(Sindex == s, arr.ind = TRUE)
          if (!nrow(idx_info)) next
          idx <- idx_info[1, ]
          Atilda <- Btilda <- rep(NA_real_, Nactions)
          Mtilda <- rep(1 / Nactions, Nactions)
          if (constrained) {
            Atilda <- Alpha[, s]
            Btilda <- Beta[, s]
            if (exists("M")) {
              Mtilda <- M[, s]
            } else if (exists("Q")) {
              Mtilda <- Q[, s]
            }
            Mtilda <- Mtilda / sum(Mtilda)
          } else {
            Atilda <- exp(Alpha[, s]) + minA
            Btilda <- exp(Beta[, s]) + minB
            if (includeR) {
              Btilda <- Btilda / sqrt(max(R, .Machine$double.eps))
            }
            if (Mlearning && exists("M")) {
              pref <- exp(M[, s] - max(M[, s]))
              Mtilda <- pref / sum(pref)
            } else if (exists("Q")) {
              Mtilda <- Q[, states_to_plot[1]]
            }
          }
          densities <- sapply(seq_len(Nactions), function(i) {
            stats::dgamma(times, shape = Atilda[i], scale = Btilda[i])
          })
          densities <- sweep(densities, 2, Mtilda, `*`)
          ylim <- range(densities, finite = TRUE)
          if (!all(is.finite(ylim))) {
            ylim <- c(0, 1)
          }
          plot(times, densities[, 1], type = "l", col = plot_colors[1], lwd = 2,
               ylim = ylim, xlab = "Time", ylab = "Probability density")
          if (Nactions > 1) {
            graphics::lines(times, densities[, Nactions], col = plot_colors[Nactions], lwd = 2)
          }
          if (Nactions > 2) {
            graphics::lines(times, densities[, Nactions - 1], col = plot_colors[Nactions - 1], lwd = 2)
          }
          if (Nactions > 2) {
            legend_idx <- c(1, Nactions, Nactions - 1)
            legend_labels <- c("LP", "NP", "Other")
          } else {
            legend_idx <- c(1, Nactions)
            legend_labels <- c("LP", "NP")
          }
          graphics::legend("topright", legend = legend_labels,
                           col = plot_colors[legend_idx], lwd = 2, bty = "n")
          title(sprintf("Policy for (t_s=%d,a_prev=%d,i_r=%d), V=%.4f",
                        idx[1], idx[2], idx[3], V[s]))
        }
        reset_par()
      }
    } else if (exists("Q")) {
      q_states <- list(
        list(idx = c(1, 1, 1), title = "Q values after an unrewarded LP"),
        list(idx = c(1, 1, 2), title = "Q values after a rewarded LP")
      )
      if (tolower(schedule) == "ratio") {
        q_states <- c(q_states, list(list(idx = c(1, min(2, dim(Sindex)[2]), 1),
                                          title = "Q values after a NP")))
      }
      q_states <- Filter(function(x) {
        idx <- x$idx
        idx[1] <- min(idx[1], dim(Sindex)[1])
        idx[2] <- min(idx[2], dim(Sindex)[2])
        !is.na(Sindex[idx[1], idx[2], idx[3]])
      }, q_states)
      if (length(q_states)) {
        reset_par()
        graphics::par(mfrow = c(length(q_states), 1))
        for (qs in q_states) {
          idx <- qs$idx
          idx[1] <- min(idx[1], dim(Sindex)[1])
          s <- Sindex[idx[1], idx[2], idx[3]]
          q_mat <- Q[, , s]
          ylim <- range(q_mat, finite = TRUE)
          if (!all(is.finite(ylim))) ylim <- c(0, 1)
          plot(times, q_mat[, 1], type = "l", col = plot_colors[1], lwd = 2,
               ylim = ylim, xlab = "Time", ylab = "Q value", main = qs$title)
          for (a in seq_len(Nactions)[-1]) {
            graphics::lines(times, q_mat[, a], col = plot_colors[a], lwd = 2)
          }
          q_vec <- as.numeric(q_mat)
          q_soft <- exp((q_vec - max(q_vec)) / Temp)
          q_soft <- matrix(q_soft / sum(q_soft), nrow = Ntimes, ncol = Nactions)
          scale_factor <- 5 * diff(ylim)
          if (!is.finite(scale_factor) || scale_factor == 0) scale_factor <- 1
          for (a in seq_len(Nactions)) {
            graphics::lines(times, scale_factor * q_soft[, a] + ylim[1],
                            col = plot_colors[a], lwd = 1, lty = 2)
          }
          graphics::legend("topright", legend = Actions,
                           col = plot_colors[seq_along(Actions)], lwd = 2, bty = "n")
        }
        reset_par()
      }
    }

    if (learn) {
      # ---- Figure 3: Average reward ----
      if (length(R_log)) {
        reset_par()
        plot(seq_along(R_log), R_log, type = "l", lwd = 2, col = "blue",
             xlab = "actions", ylab = "estimated average reward",
             main = "Average reward estimates over actions")
        graphics::abline(h = Rmin, lty = 3)
      }

      # ---- Figure 4: Prediction errors at reward ----
      y_reward <- Deltas[EatTrials > 0]
      y_reward <- c(y_reward, 0)
      cut_len <- floor(length(y_reward) / 100) * 100
      if (cut_len >= 100) {
        y_reward <- y_reward[seq_len(cut_len)]
        rolling <- colMeans(matrix(y_reward, nrow = 100))
        reset_par()
        plot(seq_len(cut_len), y_reward, type = "l", col = "blue", lwd = 1.5,
             xlab = "actions", ylab = "prediction error",
             main = "Prediction errors at reward delivery")
        graphics::lines(seq(51, cut_len, by = 100), rolling, col = "red", lwd = 2)
        graphics::abline(h = 0, lty = 2)
      }

      # ---- Figure 5: Prediction errors between rewards ----
      if (Nresponses > Nhalfway) {
        idx <- Nhalfway + which(EatTrials[(Nhalfway + 1):Nresponses] == 0)
        y_between <- c(Deltas[idx], 0)
        cut_len <- floor(length(y_between) / 100) * 100
        if (cut_len >= 100) {
          y_between <- tail(y_between, cut_len)
          rolling <- colMeans(matrix(y_between, nrow = 100))
          reset_par()
          plot(seq_len(cut_len), y_between, type = "l", col = "blue", lwd = 1.5,
               xlab = "actions", ylab = "prediction error",
               main = "Prediction errors between rewards (2nd half)")
          graphics::lines(seq(51, cut_len, by = 100), rolling, col = "red", lwd = 2)
          graphics::abline(h = 0, lty = 2)
        }
      }

      # ---- Figure 6: Policy parameter trajectories ----
      if (tolower(LearnMode) == "directactor" && !is.null(LogAlpha)) {
        idx_slog <- which(Responses == 1 & LogS == Slog)
        if (length(idx_slog) > 1) {
          if (constrained) {
            alpha_vals <- LogAlpha[idx_slog]
            beta_vals <- LogBeta[idx_slog]
            mix_vals <- LogM[, idx_slog, drop = FALSE]
          } else {
            alpha_vals <- exp(LogAlpha[idx_slog]) + minA
            beta_vals <- exp(LogBeta[idx_slog]) + minB
            if (includeR) {
              beta_vals <- beta_vals / sqrt(pmax(R_log[idx_slog], .Machine$double.eps))
            }
            mix_vals <- if (Mlearning) exp(LogM[, idx_slog, drop = FALSE]) else LogM[, idx_slog, drop = FALSE]
          }
          layout_matrix <- matrix(c(1, 2, 3, 1, 4, 5), nrow = 2, byrow = TRUE)
          reset_par()
          graphics::layout(layout_matrix)
          plot(alpha_vals, beta_vals, type = "l", col = "darkgreen", lwd = 1.5,
               xlab = "Alpha(LP,1)", ylab = "Beta(LP,1)")
          points(alpha_vals[1], beta_vals[1], col = "red", pch = 8)
          plot(beta_vals, type = "l", col = "black",
               main = "Beta(LP,1) over the trials", xlab = "trials", ylab = "Beta")
          plot(alpha_vals, type = "l", col = "black",
               main = "Alpha(LP,1) over the trials", xlab = "trials", ylab = "Alpha")
          mix_idx <- if (Nactions > 2) c(1, Nactions - 1, Nactions) else c(1, Nactions)
          matplot(t(mix_vals[mix_idx, , drop = FALSE]), type = "l", lwd = 1.5, lty = 1,
                  main = if (Mlearning) "M(:,1) over the trials" else "Q(:,1) over the trials",
                  xlab = "trials", ylab = "value",
                  col = plot_colors[mix_idx])
          graphics::legend("topright",
                           legend = Actions[mix_idx],
                           col = plot_colors[mix_idx], lwd = 1, bty = "n")
          if (is.matrix(LogV) && ncol(LogV)) {
            matplot(t(LogV), type = "l", lwd = 1.5,
                    main = "V over the trials", xlab = "actions", ylab = "Value")
          } else if (!is.matrix(LogV) && length(LogV)) {
            plot(LogV, type = "l", lwd = 1.5,
                 main = "V over the trials", xlab = "actions", ylab = "Value")
          } else {
            plot.new()
            title("V over the trials")
            text(0.5, 0.5, "No V logs recorded")
          }
          reset_par()
        }
      }
    }
  }

  list(
    Responses = Responses,
    Times = Times,
    Latencies = Latencies,
    Rewards = Rewards,
    Summary = list(
      Reward = Reward_summary,
      Consumption = Consump_summary,
      BaitInterval = bait_interval,
      LP_rates = LPs,
      NP_rates = NPs,
      Other_rates = Others
    ),
    Logs = list(
      LogM = LogM,
      LogAlpha = LogAlpha,
      LogBeta = LogBeta,
      LogS = LogS,
      Deltas = Deltas,
      R_log = R_log
    ),
    RunAborted = RunAborted
  )
}
