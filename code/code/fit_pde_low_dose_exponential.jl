"""
## Fit PDE model for treatment group (18.5 MBq)

This code was used to fit the treatment model to the low dose treatment data (18.5 MBq). 
It saves the fitted parameters for each replicate in the results folder with the name "params_low_test_$(replicate_id).jld2".

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

# read command line input
idx_str = ARGS[1]

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

# get replicate ID from index
idx = parse(Int, idx_str)
rep_id = unique(df_low.replicate_id)[idx]

println("Running fit for index $(idx) => replicate ID = $rep_id")

@parameters k_rep_s, a_alpha, b_alpha_s, g_alpha_s, a_beta, b_beta_s, g_beta_s, D_s, kcl_s, sigma_s
@variables c(..) y(..) N(..) integrand(..) x̄(..) krx(..) al(..) bet(..)

function create_radiosensitivity_problem(rep_id, p)

    @parameters x
    @parameters k_rep_s, a_alpha, b_alpha_s, g_alpha_s, a_beta, b_beta_s, g_beta_s, D_s, kcl_s, sigma_s
    @variables c(..) y(..) N(..) integrand(..) x̄(..) krx(..) al(..) bet(..)

    parameters = [k_rep_s, a_alpha, b_alpha_s, g_alpha_s, a_beta, b_beta_s, g_beta_s, D_s, kcl_s, sigma_s]

    Dxx = Differential(x)^2
    Dx = Differential(x)

    mu = df_params[666,2] # 1/day, population-level mean rho from control group

    V_init = df_low[df_low.replicate_id .== rep_id, :tumor_volume_smoothed][1]*0.001
    times_s = df_low[(df_low.replicate_id .== rep_id), :time_d].*mu

    rho_min = 0.0
    rho_max = 0.2

    mu_s = mu / mu
    V_init_s = V_init / V_init

    rho_min_s = rho_min / mu
    rho_max_s = rho_max / mu

    # sigma_s = 0.3

    R_0 = df_dose_rate[df_dose_rate.replicate .== rep_id, :initial_dose_rate_Gy_day][1] # individual initial dose rate
    k_dec = df_biodb[4,2] * 24 # population-level mean decay rate from biodb data, in 1/day

    R_0_s = R_0/mu
    k_dec_s = k_dec/mu

    s = 3.26268
    kd_s = 16.1373 / (mu^(1-s))

    Ix = Integral(x in DomainSets.ClosedInterval(rho_min_s, rho_max_s))

    eq  = [Dt(c(t,x)) ~ D_s*Dxx(c(t, x)) + x*c(t,x) - kd_s * x^s * c(t,x) - c(t,x)*krx(t,x)
       Dt(y(t,x)) ~ c(t,x)*krx(t,x) + kd_s * x^s * c(t,x) - kcl_s * y(t,x)
       N(t) ~ Ix(c(t,x)) + Ix(y(t,x))
       integrand(t,x) ~ x * c(t, x)
       x̄(t) ~ Ix(integrand(t,x))/(Ix(c(t,x)) + Ix(y(t,x)))
       krx(t,x) ~ R_0_s * exp(-(k_dec_s)*t) * al(t,x) + ((2*bet(t,x)*R_0_s^2)/(k_rep_s - k_dec_s)) * (exp(-2*k_dec_s*t)-exp(-(k_rep_s+k_dec_s)*t))
       al(t,x) ~ a_alpha/(1+exp(-b_alpha_s*(x-g_alpha_s)))
       bet(t,x) ~ a_beta/(1+exp(-b_beta_s*(x-g_beta_s)))]

    t0 = times_s[1]
    t1 = times_s[end]
    domains = [t ∈ (t0, t1),
    x ∈ (rho_min_s, rho_max_s)]

    bcs = [c(0,x) ~ (V_init_s/(sqrt(2*pi)*sigma_s))*exp(-((x-mu_s)^2)/(2*sigma_s^2)), 
    y(0,x) ~ 0.0, Dx(c(t, rho_min_s)) ~ 0.0, Dx(c(t, rho_max_s)) ~ 0, 
    Dx(y(t, rho_min_s)) ~ 0.0, Dx(y(t, rho_max_s)) ~ 0]

    @named pdesys = PDESystem(
        eq,
        bcs,
        domains,
        [t,x],
        [
            c(t,x),
            y(t,x),
            N(t),
            integrand(t,x),
            x̄(t),
            krx(t,x),
            al(t,x),
            bet(t,x)
        ],
        parameters,
        defaults = Dict(k_rep_s => p[1],a_alpha => p[2],b_alpha_s => p[3],g_alpha_s => p[4],a_beta => p[5],b_beta_s => p[6],g_beta_s => p[7],D_s => p[8],kcl_s => p[9], sigma_s=>p[10])
    )

    discretization = MOLFiniteDifference([x => 60], t)

    prob = discretize(pdesys, discretization)

    return prob, times_s
end


function fit_radiosensitivity_low_dose(prob, times_s, p)

    p1 = copy(prob.p)
    p1[1] .= p

    prob_p = remake(
        prob, p=p1
    )

    sol = solve(
        prob_p,
        Tsit5(),
        saveat=times_s
    )

    sol_total = sol[N(t)]

    return sol_total
end

mu = df_params[666,2]
p0 = [12, 0.03, 40, 0.8, 0.04, 45, 0.9, 1e-6/mu^3, 0.02, 0.3]

V_init = df_low[df_low.replicate_id .== rep_id, :tumor_volume_smoothed][1]*0.001
vols_vec = df_low[(df_low.replicate_id .== rep_id), :tumor_volume_smoothed].*0.001./V_init
data = vcat(vols_vec)

prob, times_s = create_radiosensitivity_problem(rep_id, p0)

fit = curve_fit(
    (times, p) -> fit_radiosensitivity_low_dose(prob, times_s, p),
    times_s,
    data,
    p0;
    lower=[9,0.001,0.001,0.04/mu,0.001,0.001,0.04/mu,1e-8/mu^3,0.001/mu,0.15],
    upper=[362, Inf, Inf, 0.1/mu, Inf, Inf, 0.12/mu, 1e-4/mu^3, 0.5/mu, 0.4]
)

params_lsqfit = fit.param
@save path*"results/"*"params_low_test_$(rep_id).jld2" params_lsqfit
