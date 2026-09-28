"""
## This code was used to generate the PDE model fits to the low dose treatment data.

It saves the solutions for each replicate in the results folder with the name "solution_fit_times_$(replicate_id).jld2".

"""

using DataFrames 
using CSV 
using DifferentialEquations
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as Dt
using Distributed 
using MethodOfLines
using DomainSets
using ProgressMeter
using LsqFit
using Optim
using Optimization
using OptimizationBBO
using OptimizationMOI
using OptimizationNLopt
using OptimizationOptimJL
using OrdinaryDiffEq
using DiffEqParamEstim
using ForwardDiff
using LineSearches
using NonlinearSolve
using JLD2
using Plots
using ColorSchemes
using Measures
using LaTeXStrings
using Interpolations

plot_font = "Computer Modern"
default(fontfamily=plot_font)
scalefontsizes(1.3)

# read in data (low dose treatment)
path = "./"
file_data = "data/tv_trx_low_replicates.csv"
file_params = "data/fit_params_control_smooth.csv"
file_dose_rate = "data/2026-08_initial_dose_rates_low.csv"
file_biodb = "results/biodistribution_params.csv"

df_low = CSV.read(path*file_data, DataFrame)
df_params = CSV.read(path*file_params, DataFrame)
df_dose_rate = CSV.read(path*file_dose_rate, DataFrame)
df_biodb = CSV.read(path*file_biodb, DataFrame)

@variables c(..) y(..) N(..) C_v(..) C_d(..) R(..) integrand(..) x̄(..) x̄c(..) 

solutions_units = []
for idx in 1:9
    rep_id = unique(df_low.replicate_id)[idx]
    @load path*"results/"*"params_low_test_$(rep_id).jld2" params_lsqfit

    @parameters x
    @variables c(..) y(..) N(..) C_v(..) C_d(..) integrand(..) x̄(..) x̄c(..) krx(..) al(..) bet(..)

    Dxx = Differential(x)^2
    Dtt = Differential(t)^2
    Dx = Differential(x)

    mu = df_params[666,2] # 1/day, population-level mean rho from control group
    
    V_init = df_low[df_low.replicate_id .== rep_id, :tumor_volume_smoothed][1]*0.001
    times = df_low[(df_low.replicate_id .== rep_id), :time_d]

    rho_min = 0.0
    rho_max = 0.2 # 1/day

    R_0 = df_dose_rate[df_dose_rate.replicate .== rep_id, :initial_dose_rate_Gy_day][1] # individual initial dose rate
    k_dec = df_biodb[4,2] * 24 # population-level mean decay rate from biodb data, in 1/day

    s = 3.26268 # medoid from monotonically increasing trade-off 
    kd = 16.1373 # medoid from monotonically increasing trade-off 
    
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
    
    Ix = Integral(x in DomainSets.ClosedInterval(rho_min, rho_max))

    eq  = [Dt(c(t,x)) ~ D*Dxx(c(t, x)) + x*c(t,x) - kd * x^s * c(t,x) - c(t,x)*krx(t,x)
       Dt(y(t,x)) ~ c(t,x)*krx(t,x) +  kd * x^s * c(t,x) - kcl * y(t,x)
       N(t) ~ Ix(c(t,x)) + Ix(y(t,x))
       C_v(t) ~ Ix(c(t,x))
       C_d(t) ~ Ix(y(t,x))
       integrand(t,x) ~ x * c(t, x)
       x̄(t) ~ Ix(integrand(t,x))/(Ix(c(t,x)) + Ix(y(t,x)))
       x̄c(t) ~ Ix(integrand(t,x))/(Ix(c(t,x)))
       krx(t,x) ~ R_0 * exp(-(k_dec)*t) * al(t,x) + ((2*bet(t,x)*R_0^2)/(k_rep - k_dec)) * (exp(-2*k_dec*t)-exp(-(k_rep+k_dec)*t))
       al(t,x) ~ a_alpha/(1+exp(-b_alpha*(x-g_alpha)))
       bet(t,x) ~ a_beta/(1+exp(-b_beta*(x-g_beta)))]


    t0 = times[1]
    t1 = times[end]
    domains = [t ∈ (t0, t1),
    x ∈ (rho_min, rho_max)]

    bcs = [c(0,x) ~ (V_init/(sqrt(2*pi)*sigma))*exp(-((x-mu)^2)/(2*sigma^2)), 
    y(0,x) ~ 0.0, Dx(c(t, rho_min)) ~ 0.0, Dx(c(t, rho_max)) ~ 0, 
    Dx(y(t, rho_min)) ~ 0.0, Dx(y(t, rho_max)) ~ 0]

    @named pdesys = PDESystem(eq, bcs, domains, [t,x], 
        [c(t,x), y(t,x), N(t), C_v(t), C_d(t), integrand(t,x), x̄(t), x̄c(t), krx(t,x), al(t,x), bet(t,x)])
    dx=0.1
    discretization = MOLFiniteDifference([x => 120], t)

    prob = discretize(pdesys,discretization)
    sol = solve(prob, Tsit5(), saveat=t0:0.5:t1)

    push!(solutions_units, sol)

    vols_vec = df_low[(df_low.replicate_id .== rep_id), :tumor_volume_smoothed].*0.001
    times = df_low[(df_low.replicate_id .== rep_id), :time_d]

    # sol = solutions[idx]
    discrete_x = sol[x]
    discrete_t = sol[t]
    solc = sol[c(t,x)]
    solr = sol[y(t,x)]
    sol_total = sol[N(t)]
    sol_viable = sol[C_v(t)]
    sol_doomed = sol[C_d(t)]
    solx̄ = sol[x̄(t)]
    solx̄c = sol[x̄c(t)]
    sol_krx = sol[krx(t,x)]
    sol_al = sol[al(t,x)]
    sol_bet = sol[bet(t,x)]

    @save path*"results/"*"solution_fit_times_$(rep_id).jld2" discrete_x discrete_t solc sol_total sol_viable sol_doomed solx̄ solx̄c sol_krx sol_al sol_bet
end
