library(readr)
library(tidyr)
library(rstan)
library(patchwork)
library(ggplot2)

tumor_percentIA_g <- read_csv("tumor_activity.csv")
tumor_volumes <- read_delim("tumor_volume.csv")

tumor_percentIA_g <- tumor_percentIA_g[complete.cases(tumor_volumes), ]
tumor_volumes <- tumor_volumes[complete.cases(tumor_volumes), ]

IA = 0.5

tumor_activities <- data.frame(
  activity_MBq_g = tumor_percentIA_g$percent_IA_perg * 0.01 * IA,
  id = tumor_percentIA_g$replicate
)

tumor_activities <- tumor_activities[tumor_activities$time_h > 6, ]

y <- tumor_activities$activity_MBq_g
t <- tumor_activities$time_h
N <- length(t) 
TIA_exp <- 191.80
TIA_sd <- 25.59
N_pred <- 200
t_pred <- seq(
  from = 0,
  to = max(t) * 1.5,
  length.out = N_pred
)

# compile 
m <- rstan::stan_model(file = "dosimetry.stan")

fit <- rstan::sampling(object = m,
                       data = list(y = y,
                                   t = array(t),
                                   N = N, 
                                   TIA_exp = TIA_exp,
                                   TIA_sd = TIA_sd,
                                   N_pred = N_pred,
                                   t_pred = t_pred),
                       chains = 3, cores = 3, 
                       iter = 1500, warmup = 500,
                       control = list(adapt_delta = 0.99))

plot(fit, par=c("C0"), show_density = TRUE, ci_level = 0.95, fill_color = "purple")
plot(fit, par=c("k_bio"), show_density = TRUE, ci_level = 0.95, fill_color = "purple")
plot(fit, par=c("k_eff"), show_density = TRUE, ci_level = 0.95, fill_color = "purple")
plot(fit, par=c("TIA_model"), show_density = TRUE, ci_level = 0.95, fill_color = "purple")


summary(fit, par = c("C0"))$summary
summary(fit, par = c("k_bio"))$summary
summary(fit, par = c("k_eff"))$summary
summary(fit, par = c("TIA_model"))$summary

df_ypred = data.frame(summary(fit, par = c("y_pred"))$summary)
df_ypred$time  = t_pred

ggplot(df_ypred, aes(x = t_pred, y = mean)) +
  # error bars for credible intervals
  geom_errorbar(aes(ymin = X2.5., ymax = X97.5.), width = 0.4, color = "blue") +
  
  # posterior mean points
  geom_point(color = "blue", size=1) +
  
  
  labs(x = "Time", y = "Value",
       title = "Posterior Predictive Check") +
  theme_minimal()


posterior <- extract(fit)

posterior_df <- data.frame(C0=posterior$C0, k_eff=posterior$k_eff, k_bio=posterior$k_bio)
write.csv(posterior_df, "posterior_samples.csv", row.names=FALSE)

y_rep <- posterior$y_rep
ppc_dens_overlay(
  y = y,
  yrep = y_rep[1:100, ]
)




# Summary of all parameters
fit_summary <- summary(fit)$summary

# Convert to data frame
fit_summary_df <- as.data.frame(fit_summary)

# Add parameter names as a column
fit_summary_df$Parameter <- rownames(fit_summary_df)

# Move parameter column to the front
fit_summary_df <- fit_summary_df[, c("Parameter", setdiff(names(fit_summary_df), "Parameter"))]

# Save to CSV
write.csv(fit_summary_df,
          file = "biodistribution_params.csv",
          row.names = FALSE)