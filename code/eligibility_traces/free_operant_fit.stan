data {
  int<lower=1> B;                                 // total number of bins across sessions
  int<lower=1> S;                                 // number of sessions
  real<lower=0> dt;                               // bin width (s)
  int<lower=0> refractory_bins;                   // refractory period in bins
  vector[B] reward;                               // reward received per bin
  array[B] int<lower=0, upper=1> responded;       // whether an action was emitted
  array[B] int<lower=0, upper=2> action_idx;      // action index (0 = none, 1 = x, 2 = y)
  array[B] int<lower=1, upper=S> session_id;      // session identifier per bin
}

parameters {
  real<lower=0, upper=1> alpha;                   // value learning rate
  real<lower=0, upper=1> eta;                     // baseline learning rate
  real<lower=1e-3> tau_e;                         // eligibility time constant (s)
  real beta0_vigor;
  real beta_t_vigor;
  real beta_q_vigor;
  real<lower=0> kappa;
  real omega_sticky;
}

transformed parameters {
  real lambda_bin = exp(-dt / tau_e);
}

model {
  // Priors (centered around plausible values from the simulator)
  alpha ~ beta(2, 2);
  eta ~ beta(2, 20);
  tau_e ~ lognormal(log(0.5), 0.4);
  beta0_vigor ~ normal(-4, 2);
  beta_t_vigor ~ normal(0.03, 0.02);
  beta_q_vigor ~ normal(0.5, 0.3);
  kappa ~ lognormal(log(6), 0.4);
  omega_sticky ~ normal(0, 1);
  
  // Deterministic state evolution
  real Qx = 0;
  real Qy = 0;
  real zx = 0;
  real zy = 0;
  real Rbar = 0;
  int last_resp_bin = -refractory_bins - 10;
  int last_action = 0;
  int current_session = session_id[1];
  int bin_in_session = 0;
  
  for (b in 1:B) {
    if (b == 1 || session_id[b] != current_session) {
      current_session = session_id[b];
      Qx = 0;
      Qy = 0;
      zx = 0;
      zy = 0;
      Rbar = 0;
      last_resp_bin = -refractory_bins - 10;
      last_action = 0;
      bin_in_session = 0;
    }
    
    bin_in_session += 1;
    int t_since_bins = bin_in_session - last_resp_bin;
    real t_since_s = fmax(t_since_bins - 1, 0) * dt;
    
    if (t_since_bins > refractory_bins) {
      real logit_p = beta0_vigor + beta_t_vigor * t_since_s + beta_q_vigor * (Qx + Qy);
      target += bernoulli_logit_lpmf(responded[b] | logit_p);
    } else if (responded[b] == 1) {
      reject("Observed response during refractory period at bin ", b);
    }
    
    if (responded[b] == 1) {
      vector[2] m;
      m[1] = kappa * Qx + (last_action == 1 ? omega_sticky : 0);
      m[2] = kappa * Qy + (last_action == 2 ? omega_sticky : 0);
      target += log_softmax(m)[action_idx[b]];
      last_resp_bin = bin_in_session;
      last_action = action_idx[b];
    }
    
    real z_dot = action_idx[b] == 1 ? zx : (action_idx[b] == 2 ? zy : 0);
    real new_zx = lambda_bin * zx;
    real new_zy = lambda_bin * zy;
    if (action_idx[b] == 1) {
      new_zx += 1 - alpha * lambda_bin * z_dot;
    }
    if (action_idx[b] == 2) {
      new_zy += 1 - alpha * lambda_bin * z_dot;
    }
    zx = new_zx;
    zy = new_zy;
    
    real delta = reward[b] - Rbar;
    Qx += alpha * delta * zx;
    Qy += alpha * delta * zy;
    Rbar = (1 - eta) * Rbar + eta * reward[b];
  }
}

generated quantities {
  vector[B] log_lik;
  real Qx = 0;
  real Qy = 0;
  real zx = 0;
  real zy = 0;
  real Rbar = 0;
  int last_resp_bin = -refractory_bins - 10;
  int last_action = 0;
  int current_session = session_id[1];
  int bin_in_session = 0;
  
  for (b in 1:B) {
    if (b == 1 || session_id[b] != current_session) {
      current_session = session_id[b];
      Qx = 0;
      Qy = 0;
      zx = 0;
      zy = 0;
      Rbar = 0;
      last_resp_bin = -refractory_bins - 10;
      last_action = 0;
      bin_in_session = 0;
    }
    
    bin_in_session += 1;
    int t_since_bins = bin_in_session - last_resp_bin;
    real t_since_s = fmax(t_since_bins - 1, 0) * dt;
    real ll = 0;
    
    if (t_since_bins > refractory_bins) {
      real logit_p = beta0_vigor + beta_t_vigor * t_since_s + beta_q_vigor * (Qx + Qy);
      ll += bernoulli_logit_lpmf(responded[b] | logit_p);
    }
    
    if (responded[b] == 1) {
      vector[2] m;
      m[1] = kappa * Qx + (last_action == 1 ? omega_sticky : 0);
      m[2] = kappa * Qy + (last_action == 2 ? omega_sticky : 0);
      ll += log_softmax(m)[action_idx[b]];
      last_resp_bin = b;
      last_action = action_idx[b];
    }
    
    real z_dot = action_idx[b] == 1 ? zx : (action_idx[b] == 2 ? zy : 0);
    real new_zx = lambda_bin * zx;
    real new_zy = lambda_bin * zy;
    if (action_idx[b] == 1) new_zx += 1 - alpha * lambda_bin * z_dot;
    if (action_idx[b] == 2) new_zy += 1 - alpha * lambda_bin * z_dot;
    zx = new_zx;
    zy = new_zy;
    
    real delta = reward[b] - Rbar;
    Qx += alpha * delta * zx;
    Qy += alpha * delta * zy;
    Rbar = (1 - eta) * Rbar + eta * reward[b];
    
    log_lik[b] = ll;
  }
}
