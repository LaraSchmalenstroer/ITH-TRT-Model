"""
This script computes the time-integrated activity (TIA) in the tumor from biodistribution data.
It reads in the fitted parameters of the fitted activity concentration function and 
plots the mean and 95% credible interval of the fitted function, as well as the 95% posterior 
predictive interval. It computes the initial dose rates in the low-dose and high-dose treatment groups using the
fitted parameters and the S-values for different tumor volumes. It saves the results to a .csv file.

It is used to create the following figures:

- Figure S3: 1) Histogram of bootstrap TIA samples with mean and 95% CI
- Figure S4: 3) Fitted mean, 95% credible interval, and 95% posterior predictive interval of tumor activity concentration over time
- Figure S5a,b: 4), 5) S-values vs tumor volume with log-log axes and linear axes

It is also used to create the following tables:
- Table S2: Initial dose rates in low-dose treatment group (18.5 MBq)
- Table S3: Initial dose rates in high-dose treatment group (29.6 MBq)
"""

using DataFrames
using Statistics
using Random
using StatsBase
using CSV 
using StatsPlots
using Statistics
using LinearAlgebra
using Random
using Distributions

## load data
path = "/home/lara/Documents/GRK2762/Projects/pde-model/Data/"
file_name = "biod_PDX2_EBRGD_processed.csv"
injected_activity_MBq = 0.5 # MBq

## define function to get activities from biodistribution data
function get_activities(df_biodb, injected_activity_MBq)
    rename!(df_biodb, :weight_mg => :organ_weight_mg)
    rename!(df_biodb, "%IA/g" => :percent_IA_perg)
    df_biodb[!, :activity_perg_MBq] = df_biodb.percent_IA_perg .* 0.01 .* injected_activity_MBq
    df_biodb[!, :activity_MBq] = df_biodb.activity_perg_MBq .* df_biodb.organ_weight_mg .* 0.001
    df_biodb = DataFrames.transform(groupby(df_biodb, [:time_h, :organ]), 
    [:activity_MBq => mean => :activity_MBq_mean, :activity_MBq => std => :activity_MBq_std])
    df_biodb = df_biodb[df_biodb.organ .== "tumor", :]
    return df_biodb
end

## define functions to compute TIA using trapezoidal rule and tail integral
function trapezoidal_auc(t, y)
  auc = 0.0
  for i in 1:(length(t)-1)
    dt = t[i+1] - t[i]
    auc += 0.5 * dt * (y[i] + y[i+1])
  end
  return auc
end

function tail_integral(t, y)
  act = y[end]
  # effective decay constant
  k = log(2) / 159.528
  return act / k
end

function compute_tia(t, y)
  auc_trap = trapezoidal_auc(t, y)
  auc_tail = tail_integral(t, y)
  return auc_trap + auc_tail
end

## define function to bootstrap TIA
function bootstrap_tia(df; n_boot=5000, rng=Random.default_rng())
  unique_times = sort(unique(df.time_h))
  tia_samples = zeros(n_boot)
  for b in 1:n_boot
    mean_activities = Float64[]
    for tt in unique_times
      subdf = df[df.time_h .== tt, :]
      # bootstrap mice at this time point
      inds = sample(
        rng,
        1:nrow(subdf),
        nrow(subdf),
        replace=true
      )
      sampled = subdf.activity_perg_MBq[inds]
      push!(mean_activities, mean(sampled))
    end
    tia_samples[b] = compute_tia(unique_times, mean_activities)
  end
  return tia_samples
end

## read in data
df_acts = CSV.read(path * file_name, DataFrame)
df_acts = get_activities(df_acts, injected_activity_MBq)

df_volumes = df_acts[!, [:time_h, :organ_weight_mg, :replicate]]
df_acts_sel = df_acts[!, [:time_h, :replicate, :percent_IA_perg, :activity_perg_MBq]]

ts = unique(df_acts_sel.time_h)
acts = [mean(df_acts_sel[df_acts_sel.time_h .== t, :activity_perg_MBq]) for t in ts]

## compute TIA using trapezoidal rule and tail integral
compute_tia(ts, acts)

tia_samples = bootstrap_tia(df_acts_sel, n_boot=10000)
TIA_exp = mean(tia_samples)
TIA_sd = std(tia_samples)

println("Bootstrap TIA mean = ", TIA_exp) # 8.426
println("Bootstrap TIA SD   = ", TIA_sd) # 1.074

CSV.write(path*"tumor_activity_chen.csv", df_acts_sel)
CSV.write(path*"tumor_volume_chen.csv", df_volumes)

## plot histogram of bootstrap TIA samples with mean and 95% CI
TIA_low = quantile(tia_samples, 0.025)
TIA_high = quantile(tia_samples, 0.975)

## 1) plot histogram of bootstrap TIA samples with mean and 95% CI
histogram(tia_samples,bins=60,xlabel="Time-integrated activity [MBq·h/g]",ylabel="Frequency",
    label="Bootstrap distribution",alpha=0.6,color=:gray,linewidth=0.8)
vline!([TIA_exp],label="Mean = $(round(TIA_exp, digits=2))",linewidth=1.5,color=:blue)
vline!([TIA_low, TIA_high],label="95% bootstrap interval",linestyle=:dash,linewidth=1.5,
    color=:red)
savefig("/home/lara/Documents/GRK2762/phd-thesis/Figures/tumor_activity_bootstrap_chen.svg")

## read biodistribution parameters from CSV file and define mean activity function

df_biodb_params = CSV.read("/home/lara/Documents/GRK2762/Projects/pde-model/Data/biodistribution_params_chen.csv", DataFrame)
A0_mean = df_biodb_params[1,2]
k_eff_mean = df_biodb_params[4,2]
act_mean(t) = A0_mean * exp(-k_eff_mean * t)

## get posterior samples from CSV file

df_posterior = CSV.read("/home/lara/Documents/GRK2762/Projects/pde-model/Data/posterior_samples_chen.csv", DataFrame)

N = 2000
t = range(0, stop=200, length=500)
inds = rand(1:nrow(df_posterior), N)

A0_samples = df_posterior.C0[inds]
k_samples  = df_posterior.k_eff[inds]

A_samples = [
    A0_samples[i] .* exp.(-k_samples[i].*t)
    for i in 1:N
]

A_mat = hcat(A_samples...)

A_low_curve  = mapslices(x -> quantile(x,0.025),A_mat,dims=2)[:]
A_high_curve = mapslices(x -> quantile(x,0.975),A_mat,dims=2)[:]

## plot mean and 95% credible interval of tumor activity over time

## 2) plot fitted mean and 95% credible interval of tumor activity concentration over time
plot(df_acts_sel.time_h, df_acts_sel.activity_perg_MBq, st=:scatter, xlabel="Time [h]", 
ylabel="Tumor Activity [MBq/g]", label="Data", color=:gray, alpha=0.3)
plot!(ts, acts, st=:scatter, label="Mean", color=:gray)
plot!(act_mean, 0, 200, label="Fit", color=:blue)
plot!(t, A_low_curve, fillrange=A_high_curve, alpha=0.15, label="95% interval", 
fill=:blue)
savefig("/home/lara/Documents/GRK2762/phd-thesis/Figures/tumor_activity_fit_chen_with_CI.svg")


## get posterior predictive samples of tumor activity over time

df_posterior = CSV.read("/home/lara/Documents/GRK2762/Projects/pde-model/Data/posterior_samples_chen.csv", DataFrame)

N = 2000
t = range(0, stop=200, length=500)
inds = rand(1:nrow(df_posterior), N)

A0_samples = df_posterior.C0[inds]
k_samples  = df_posterior.k_eff[inds]
sigma_samples = df_posterior.sigma[inds]

A_pred_samples = [
    exp.(log.(A0_samples[i] .* exp.(-k_samples[i] .* t)) .+ 
         sigma_samples[i] .* randn(length(t)))
    for i in 1:N
]

A_pred_mat = hcat(A_pred_samples...)

A_pred_low_curve  = mapslices(x -> quantile(x,0.025), A_pred_mat, dims=2)[:]
A_pred_high_curve = mapslices(x -> quantile(x,0.975), A_pred_mat, dims=2)[:]

## 3) plot fitted mean, 95% credible interval, and 95% posterior predictive interval of 
##tumor activity concentration over time
plot(t, A_pred_low_curve,
      fillrange=A_pred_high_curve,
      alpha=0.15, color=:gold2,
      label="95% posterior predictive interval")
plot!(t, A_low_curve, fillrange=A_high_curve, alpha=0.15, label="95% posterior credible interval", 
fill=:blue, linewidth=0)
plot!(df_acts_sel.time_h, df_acts_sel.activity_perg_MBq, st=:scatter, xlabel="Time [h]", 
ylabel="Tumor Activity [MBq/g]", label="Data", color=:gray, alpha=0.3)
plot!(ts, acts, st=:scatter, label="Mean", color=:gray)
plot!(act_mean, 0, 200, label="Posterior mean", color=:blue)
savefig("/home/lara/Documents/GRK2762/phd-thesis/Figures/tumor_activity_fit_chen_with_CI_PI.svg")

## interpolate S-values for tumor volumes using power law fit

vols_mg = [1,5,10,50,100,200,300,500] # data from Konijnenberg et al. 2014 (10.1186/s13550-014-0047-1)
s_vals = [17.0,3.92,2.05,0.436,0.222,0.113,0.0756,0.0457]

## 4) plot S-values vs tumor volume with log-log axes and interpolated S-value at 150 mg
plot(vols_mg, s_vals, st=:scatter, xlabel="Tumor Volume [mg]", ylabel="S-value [mGy/MBq*s]", 
marker=:xcross, color=:blue, label="", xaxis=:log, yaxis=:log)
plot!([150], [s_model(150)], st=:scatter, xaxis=:log, yaxis=:log,
marker=:xcross, color=:red, label="Interpolated S-value")
# savefig("/home/lara/Documents/GRK2762/phd-thesis/Figures/s_values_log.svg")

## 5) plot S-values vs tumor volume with linear axes and interpolated S-value at 150 mg
plot(vols_mg, s_vals, st=:scatter, xlabel="Tumor Volume [mg]", ylabel="S-value [mGy/MBq*s]", 
marker=:xcross, color=:blue, label="")
plot!([150], [s_model(150)], st=:scatter,marker=:xcross, color=:red, label="Interpolated S-value")
# savefig("/home/lara/Documents/GRK2762/phd-thesis/Figures/s_values_linear.svg")

function fit_powerlaw(V, s)
    x = log10.(V)
    y = log10.(s)

    X = hcat(x, ones(length(x)))
    a, b = X \ y

    model(Vnew) = 10 .^ b .* Vnew .^ a
    return a, b, model
end

a, b, s_model = fit_powerlaw(vols_mg, s_vals)



## compute initial dose rates for low and high dose groups

function initial_dose_rate_low(mass_mg;
                           C0_bio=A0_mean,
                           Ainj_bio=0.5,
                           Ainj_trt=18.5)

    mass_g = mass_mg * 0.001
    S_mGy_s = s_model(mass_mg) 
    S_Gy_day = S_mGy_s * 86400 / 1000

    C0 = C0_bio * Ainj_trt / Ainj_bio

    activity = C0 * mass_g

    # initial dose rate [Gy/day]
    dose_rate = activity * S_Gy_day

    return activity, dose_rate, S_Gy_day
end

path = "/home/lara/Documents/GRK2762/Projects/pde-model/Data/"
fname_trx_low = "tv_trx_low_replicates.csv"
fname_trx_high = "tv_trx_high_replicates.csv"

df_low = CSV.read(path*fname_trx_low, DataFrame)

results = DataFrame(
    replicate = String[],
    initial_volume_mm3 = Float64[],
    initial_mass_g = Float64[],
    initial_activity_MBq = Float64[],
    s_value_Gy_MBq_day = Float64[],
    initial_dose_rate_Gy_day = Float64[]
)

for rep in unique(df_low.replicate_id)

    V0 = df_low[df_low.replicate_id .== rep, :tumor_volume_smoothed][1]

    activity, dose_rate, S_Gy_day = initial_dose_rate_low(V0)

    push!(results, (
        string(rep),
        V0,
        V0 / 1000,
        activity,
        S_Gy_day,
        dose_rate
    ))
end

CSV.write("/home/lara/Documents/GRK2762/Projects/pde-model/Results/Dosimetry/2026-08_initial_dose_rates_low.csv", results)

function initial_dose_rate_high(mass_mg;
                           C0_bio=A0_mean,
                           Ainj_bio=0.5,
                           Ainj_trt=29.6)

    mass_g = mass_mg * 0.001
    S_mGy_s = s_model(mass_mg) 
    S_Gy_day = S_mGy_s * 86400 / 1000

    C0 = C0_bio * Ainj_trt / Ainj_bio

    activity = C0 * mass_g

    # initial dose rate [Gy/day]
    dose_rate = activity * S_Gy_day

    return activity, dose_rate, S_Gy_day
end

df_high = CSV.read(path*fname_trx_high, DataFrame)

results = DataFrame(
    replicate = String[],
    initial_volume_mm3 = Float64[],
    initial_mass_g = Float64[],
    initial_activity_MBq = Float64[],
    s_value_Gy_MBq_day = Float64[],
    initial_dose_rate_Gy_day = Float64[]
)

for rep in unique(df_high.replicate_id)

    V0 = df_high[df_high.replicate_id .== rep, :tumor_volume_smoothed][1]

    activity, dose_rate, S_Gy_day = initial_dose_rate_high(V0)

    push!(results, (
        string(rep),
        V0,
        V0 / 1000,
        activity,
        S_Gy_day,
        dose_rate
    ))
end

CSV.write("/home/lara/Documents/GRK2762/Projects/pde-model/Results/Dosimetry/2026-08_initial_dose_rates_high.csv", results)
