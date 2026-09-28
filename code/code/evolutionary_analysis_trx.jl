using Plots
using Measures
using LaTeXStrings
using CSV 
using DataFrames
using JLD2

"""
## Evolutionary Analysis

This code was used to create the following figures:

- Figure 2 (a-b): 1, 2

"""

function alpha(ρ, a_alpha, b_alpha, g_alpha)
    a_alpha / (1 + exp(-b_alpha * (ρ - g_alpha)))
end

function beta(ρ, a_beta, b_beta, g_beta)
    a_beta / (1 + exp(-b_beta * (ρ - g_beta)))
end

function krx(t, ρ, p)

    R0, k_dec, k_rep,
    a_alpha, b_alpha, g_alpha,
    a_beta, b_beta, g_beta = p

    α = alpha(ρ, a_alpha, b_alpha, g_alpha)
    β = beta(ρ, a_beta, b_beta, g_beta)

    R0 * exp(-k_dec*t) * α +
    (2 * β * R0^2 / (k_rep - k_dec)) *
    (exp(-2*k_dec*t) -
     exp(-(k_rep + k_dec)*t))
end


function g_treatment(t, ρ, p; kd=0.5, s=2)
    ρ - kd * ρ^s - krx(t, ρ, p)
end

plot_font = "Computer Modern"
default(fontfamily=plot_font)
scalefontsizes(1.3)

# read in data (low dose treatment)
path = "./"
file_data = "data/tv_trx_low_replicates.csv"
file_params = "data/fit_params_control_smooth.csv"
file_dose_rate = "data/2026-08_initial_dose_rates_low.csv"
file_biodb = "data/biodistribution_params.csv"

df_low = CSV.read(path*file_data, DataFrame)
df_params = CSV.read(path*file_params, DataFrame)
df_dose_rate = CSV.read(path*file_dose_rate, DataFrame)
df_biodb = CSV.read(path*file_biodb, DataFrame)

rep_id = 223

@load path*"results/"*"params_low_test_$(rep_id).jld2" params_lsqfit

mu = df_params[666,2]

R_0 = df_dose_rate[df_dose_rate.replicate .== rep_id, :initial_dose_rate_Gy_day][1] # individual initial dose rate
k_dec = df_biodb[4,2] * 24 # population-level mean decay rate from biodb data, in 1/day

s = 3.26268 # representative value from monotonically increasing trade-off 
kd = 16.1373 # representative value from monotonically increasing trade-off 

k_rep_s = params_lsqfit[1]
a_alpha = params_lsqfit[2]
b_alpha_s = params_lsqfit[3]
g_alpha_s = params_lsqfit[4]
a_beta = params_lsqfit[5]
b_beta_s = params_lsqfit[6]
g_beta_s = params_lsqfit[7]
D_s = params_lsqfit[8]
kcl_s = params_lsqfit[9]
sigma_s = params_lsqfit[10]

kcl = kcl_s * mu
D = D_s * mu^3
b_alpha = b_alpha_s / mu
g_alpha = g_alpha_s * mu
b_beta = b_beta_s / mu
g_beta = g_beta_s * mu
k_rep = k_rep_s * mu
sigma = sigma_s * mu

p = (R_0, k_dec, k_rep, a_alpha, b_alpha, g_alpha, a_beta, b_beta, g_beta)

rho_range = range(0.0, 0.2, length=500)
time_points = [0, 2, 4, 6, 8, 10, 12, 14]

# Plot 1: g(ρ,t) for different time points during treatment

plt = plot(xlabel=L"\rho", ylabel=L"g(\rho,t)", grid=true, legend=:bottomleft,
legend_columns = 2)

rho_opts = Float64[]

for idx in eachindex(time_points)
    t_i = time_points[idx]
    g_vals = [g_treatment(t_i, ρ, p) for ρ in rho_range]

    i_opt = argmax(g_vals)
    ρ_opt = rho_range[i_opt]

    push!(rho_opts, ρ_opt)

    plot!(plt, rho_range, g_vals, lw=2, label="Day $t_i", color=idx)

    scatter!(plt, [ρ_opt], [g_vals[i_opt]], marker=:star5, markersize=7, label="", color=idx)
end
display(plt)
savefig(plt, path*"results/plots/trx_exp/"*"g_rho_t.svg")

# Plot 2: Optimal ρ*(t) over time during treatment (to find evolutionary effective treatment window)

times = 0.0:0.5:20
plt = plot(xlabel="Time (days)", ylabel=L"\rho^*(t)", grid=true, legend=false)
for idx in eachindex(times)
    t_i = times[idx]
    g_vals = [g_treatment(t_i, ρ, p) for ρ in rho_range]

    i_opt = argmax(g_vals)
    ρ_opt = rho_range[i_opt]

    plot!(plt, [t_i], [ρ_opt], label="", st=:scatter, color=:blue)
end
display(plt)
savefig(plt, path*"results/plots/trx_exp/"*"evolutionary_effective_treatment_window.svg")
