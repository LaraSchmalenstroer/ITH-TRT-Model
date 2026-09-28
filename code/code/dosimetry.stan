data {
  int<lower=1> N;

  vector[N] t;
  vector[N] y;

  real<lower=0> TIA_exp;
  real<lower=0> TIA_sd;

  // prediction grid
  int<lower=1> N_pred;
  vector[N_pred] t_pred;
}

transformed data {
  real lambda_phys = log(2) / 159.528;
}

parameters {
  real<lower=0> C0;
  real<lower=0> k_bio;
  real<lower=0> sigma;
}

transformed parameters {
  real k_eff;
  real TIA_model;
  vector[N] mu;
  k_eff = k_bio + lambda_phys;
  for (i in 1:N) {
    mu[i] = C0 * exp(-k_eff * t[i]);
  }
  TIA_model = C0 / k_eff;
}

model {
  C0 ~ lognormal(log(max(y)), 1);
  k_bio ~ lognormal(-2, 1);
  sigma ~ exponential(1);
  y ~ lognormal(log(mu), sigma);
  TIA_exp ~ normal(TIA_model, TIA_sd);
}

generated quantities {
  vector[N_pred] y_pred;
  vector[N] y_rep;
  real half_life_eff;
  real half_life_bio;
  half_life_eff = log(2) / k_eff;
  half_life_bio = log(2) / k_bio;
  for (i in 1:N_pred) {
    real mu_pred;
    mu_pred = C0 * exp(-k_eff * t_pred[i]);
    y_pred[i] = lognormal_rng(log(mu_pred), sigma);
  }
  
  for (i in 1:N) {
    y_rep[i] = lognormal_rng(log(mu[i]), sigma);
  }
}

