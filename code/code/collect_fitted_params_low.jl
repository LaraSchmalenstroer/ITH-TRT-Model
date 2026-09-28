"""
Collect fitted parameters for low-dose experiments, dimensionalize them, and write to csv file

"""

using Plots
using ColorSchemes
using Measures
using LaTeXStrings
using Interpolations
using JLD2
using CSV
using DataFrames
using StatsBase
using Statistics
using StatsPlots

plot_font = "Computer Modern"
default(fontfamily=plot_font)
scalefontsizes(1.3)

path = "./"
file_data = "data/tv_trx_low_replicates.csv"
file_params = "data/fit_params_control_smooth.csv"
file_dose_rate = "data/2026-08_initial_dose_rates_low.csv"
file_biodb = "results/biodistribution_params.csv"

df_low = CSV.read(path*file_data, DataFrame)
df_params = CSV.read(path*file_params, DataFrame)
df_dose_rate = CSV.read(path*file_dose_rate, DataFrame)
df_biodb = CSV.read(path*file_biodb, DataFrame)

D_list = Float64[]
kcl_list = Float64[]
a_alpha_list = Float64[]
b_alpha_list = Float64[]
g_alpha_list = Float64[]
a_beta_list = Float64[]
b_beta_list = Float64[]
g_beta_list = Float64[]
krep_list = Float64[]
sigma_list = Float64[]
for idx in 1:9
    rep_id = unique(df_low.replicate_id)[idx]
    @load path*"results/"*"params_low_test_$(rep_id).jld2" params_lsqfit

    mu = df_params[666,2]

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
    krep = k_rep_s * mu
    sigma = sigma_s * mu


    push!(D_list, D)
    push!(kcl_list, kcl)
    push!(a_alpha_list, a_alpha)
    push!(b_alpha_list, b_alpha)
    push!(g_alpha_list, g_alpha)
    push!(a_beta_list, a_beta)
    push!(b_beta_list, b_beta)
    push!(g_beta_list, g_beta)
    push!(krep_list, krep)
    push!(sigma_list, sigma)
end

df_fit_params = DataFrame(replicate_id = unique(df_low.replicate_id),
kcl = kcl_list, D = D_list, a_alpha = a_alpha_list, b_alpha = b_alpha_list, g_alpha = g_alpha_list,
a_beta = a_beta_list, b_beta = b_beta_list, g_beta = g_beta_list, krep = krep_list, sigma = sigma_list)

CSV.write(path*"results/fit_params_low_test.csv", df_fit_params)
