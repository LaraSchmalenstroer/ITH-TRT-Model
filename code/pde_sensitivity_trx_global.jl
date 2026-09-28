"""
## Global Sensitivity Analysis on scalar model outputs 

This code was used to generate Sobol indices S1 and ST on the scalar model outputs
- N_total
- N_v
- N_d 
- rho_total 
- rho_v 

at two time points: 
1. t = 10 days (right when treatment effect is decreasing)
2. t = 52 days (experimental end point)

It was used to generate the supplementary plots about GSA (S10 and S11)

"""

using DataFrames 
using CSV 
using DifferentialEquations
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as Dt
using Distributed 
using MethodOfLines
using DomainSets
using OrdinaryDiffEq
using DiffEqParamEstim
using ForwardDiff
using LineSearches
using NonlinearSolve
using Plots
using ColorSchemes
using Measures
using LaTeXStrings
using ProgressMeter
using JLD2
using GlobalSensitivity
using QuasiMonteCarlo
using Distributed

println("Threads: ", Threads.nthreads())

# read in data (low dose treatment) and parameters 

path = "./"
file_data = "data/tv_trx_low_replicates.csv"
file_params = "data/fit_params_control_smooth.csv"
file_dose_rate = "data/2026-08_initial_dose_rates_low.csv"
file_biodb = "results/biodistribution_params.csv"

df_low = CSV.read(path*file_data, DataFrame)
df_params = CSV.read(path*file_params, DataFrame)
df_dose_rate = CSV.read(path*file_dose_rate, DataFrame)
df_biodb = CSV.read(path*file_biodb, DataFrame)
 
rep_id = 223
@load path*"results/"*"params_low_test_$(rep_id).jld2" params_lsqfit

@parameters x
@parameters k_rep, a_alpha, b_alpha, g_alpha, a_beta, b_beta, g_beta, D, kcl, sigma, kd, s, V_init, mu, R0, k_dec
@variables c(..) y(..) N(..) C_v(..) C_d(..) integrand(..) x̄(..) x̄c(..) krx(..) al(..) bet(..)

Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dx = Differential(x)

V_init_p = df_low[df_low.replicate_id .== rep_id, :tumor_volume_smoothed][1]*0.001
times = df_low[(df_low.replicate_id .== rep_id), :time_d]

mu_p = df_params[666,2] # 1/day, population-level mean rho from control group

R0_p = df_dose_rate[df_dose_rate.replicate .== rep_id, :initial_dose_rate_Gy_day][1] # individual initial dose rate
k_dec_p = df_biodb[4,2] * 24 # population-level mean decay rate from biodb data, in 1/day

s_p = 3.26268 # representative value from monotonically increasing trade-off 
kd_p = 16.1373 # representative value from monotonically increasing trade-off 

# extract dimensionless parameters

k_rep_s = params_lsqfit[1]
a_alpha_p = params_lsqfit[2]
b_alpha_s = params_lsqfit[3]
g_alpha_s = params_lsqfit[4]
a_beta_p = params_lsqfit[5]
b_beta_s = params_lsqfit[6]
g_beta_s = params_lsqfit[7]
D_s = params_lsqfit[8]
kcl_s = params_lsqfit[9]
sigma_s = params_lsqfit[10]

# derive parameter values with units

kcl_p = kcl_s * mu_p
D_p = D_s * mu_p^3
b_alpha_p = b_alpha_s / mu_p
g_alpha_p = g_alpha_s * mu_p
b_beta_p = b_beta_s / mu_p
g_beta_p = g_beta_s * mu_p
k_rep_p = k_rep_s * mu_p
sigma_p = sigma_s * mu_p

function create_problem()
    parameters = [k_rep, a_alpha, b_alpha, g_alpha, a_beta, b_beta, g_beta, D, kcl, sigma, kd, s, V_init, mu, R0, D, kcl, k_dec]

    rho_min = 0.0
    rho_max = 0.2 # 1/day

    Ix = Integral(x in DomainSets.ClosedInterval(rho_min, rho_max))

    eq  = [Dt(c(t,x)) ~ D*Dxx(c(t, x)) + x*c(t,x) - kd * x^s * c(t,x) - c(t,x)*krx(t,x)
       Dt(y(t,x)) ~ c(t,x)*krx(t,x) +  kd * x^s * c(t,x) - kcl * y(t,x)
       N(t) ~ Ix(c(t,x)) + Ix(y(t,x))
       C_v(t) ~ Ix(c(t,x))
       C_d(t) ~ Ix(y(t,x))
       integrand(t,x) ~ x * c(t, x)
       x̄(t) ~ Ix(integrand(t,x))/(Ix(c(t,x)) + Ix(y(t,x)))
       x̄c(t) ~ Ix(integrand(t,x))/(Ix(c(t,x)))
       krx(t,x) ~ R0 * exp(-(k_dec)*t) * al(t,x) + ((2*bet(t,x)*R0^2)/(k_rep - k_dec)) * (exp(-2*k_dec*t)-exp(-(k_rep+k_dec)*t))
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
    [c(t,x), y(t,x), N(t), C_v(t), C_d(t), integrand(t,x), x̄(t), x̄c(t), krx(t,x), al(t,x), bet(t,x)], 
    parameters, defaults = Dict(k_rep=>k_rep_p, a_alpha=>a_alpha_p, b_alpha=>b_alpha_p, g_alpha=>g_alpha_p, a_beta=>a_beta_p, b_beta=>b_beta_p, g_beta=>g_beta_p, D=>D_p, kcl=>kcl_p, sigma=>sigma_p, kd=>kd_p, s=>s_p, V_init=>V_init_p, mu=>mu_p, R0=>R0_p, k_dec=>k_dec_p))

    discretization = MOLFiniteDifference([x => 120], t)

    prob = discretize(pdesys,discretization)
    return prob
end

function run_model(pars)
    p1 = copy(prob.p)
    p1[1] .= pars
    prob_p = remake(prob, p=p1)
    sol = solve(prob_p,Tsit5(),saveat=[10,51])
    return sol
end

function get_outputs(sol)
    N_sol = sol[N(t)]
    Cv_sol = sol[C_v(t)]
    Cd_sol = sol[C_d(t)]
    xbar_sol = sol[x̄(t)]
    xbarc_sol = sol[x̄c(t)]
    return (
        N_10 = N_sol[1],
        Cv_10 = Cv_sol[1],
        Cd_10 = Cd_sol[1],
        xbar_10 = xbar_sol[1],
        xbarc_10 = xbarc_sol[1],

        N_51 = N_sol[2],
        Cv_51 = Cv_sol[2],
        Cd_51 = Cd_sol[2],
        xbar_51 = xbar_sol[2],
        xbarc_51 = xbarc_sol[2]
    )
end

function model_outputs_batch(P)
    n = size(P,2)
    Y = Matrix{Float64}(undef,10,n)

    # Threads.@threads for i in 1:n
    for i in 1:n
        p = P[:,i]
        sol = run_model(p)
        out = get_outputs(sol)

        Y[:,i] .= (
            out.N_10,
            out.Cv_10,
            out.Cd_10,
            out.xbar_10,
            out.xbarc_10,
            out.N_51,
            out.Cv_51,
            out.Cd_51,
            out.xbar_51,
            out.xbarc_51
        )
    end

    return Y
end

# Set boundaries for parameter values 

# set lower and upper boundaries for parameter values for GSA 
lb = [0.5*k_rep_p,0.5*a_alpha_p,0.5*b_alpha_p,0.5*g_alpha_p,0.5*a_beta_p,0.5*b_beta_p,0.5*g_beta_p,
    0.5*D_p,0.5*kcl_p,0.5*sigma_p,0.5*kd_p,0.5*s_p,0.5*V_init_p,0.5*mu_p,0.5*R0_p,0.5*k_dec_p]

ub = [2*k_rep_p,2*a_alpha_p,2*b_alpha_p,2*g_alpha_p,2*a_beta_p,2*b_beta_p,2*g_beta_p,2*D_p,
    2*kcl_p,2*sigma_p,2*kd_p,2*s_p,2*V_init_p,2*mu_p,2*R0_p,2*k_dec_p]

bounds = [[lb[i], ub[i]] for i in eachindex(lb)]

# create problem, discretize once 
prob = create_problem()

# run Sobol analysis
n_samples = 2000
A, B = QuasiMonteCarlo.generate_design_matrices(
    n_samples,
    lb,
    ub,
    SobolSample()
)

sobol_result_trx = gsa(model_outputs_batch, Sobol(), A, B, batch = true)

@save path*"results/"*"gsa_results_sobol_trx_n2000.jld2" sobol_result_trx

@load path*"results/"*"gsa_results_sobol_trx_n2000.jld2" sobol_result_trx


# set parameter and output labels
param_labels = [L"\gamma", L"a_{\alpha}", L"b_{\alpha}", L"g_{\alpha}", L"a_{\beta}", L"b_{\beta}", 
L"g_{\beta}", L"D", L"k_{cl}", L"\sigma_0", L"k_d", L"s", L"N_0", L"\bar{\rho}_{total,0}", L"R_0", L"\lambda"]
output_labels = [L"N_{total}(T)", L"N_v(T)", L"N_d(T)", L"\bar{\rho}_{total}(T)", L"\bar{\rho}_{v}(T)"]

S1 = sobol_result_trx.S1
ST = sobol_result_trx.ST

S1_t1 = S1[1:5, :]
ST_t1 = ST[1:5, :]
S1_t2 = S1[6:10, :]
ST_t2 = ST[6:10, :]

# plot results 

# 1) S1 heatmap for time point 1
plt = heatmap(param_labels,output_labels,S1_t1,xlabel = "Parameter",ylabel = "Model output",
    clims = (0, 1),xrotation = 45,size = (700, 400), xticks = (1:length(param_labels), param_labels),
    colorbar_title = "S1", left_margin=2mm, bottom_margin=5mm)
savefig(plt, "results/plots/S1_2000_trx_model_t1.svg")

# 2) S1 heatmap for time point 2
plt = heatmap(param_labels,output_labels,S1_t2,xlabel = "Parameter",ylabel = "Model output",
    clims = (0, 1),xrotation = 45,size = (700, 400), xticks = (1:length(param_labels), param_labels),
    colorbar_title = "S1", left_margin=2mm, bottom_margin=5mm)
savefig(plt, "results/plots/S1_2000_trx_model_t2.svg")

# 3) ST heatmap for time point 1
plt = heatmap(param_labels,output_labels,ST_t1,xlabel = "Parameter",ylabel = "Model output",
    clims = (0, 1),xrotation = 45,size = (700, 400), xticks = (1:length(param_labels), param_labels),
    colorbar_title = "ST", left_margin=2mm, bottom_margin=5mm)
savefig(plt, "results/plots/ST_2000_trx_model_t1.svg")

# 4) ST heatmap for time point 2
p1 = heatmap(param_labels,output_labels,ST_t2,xlabel = "Parameter",ylabel = "Model output",
    clims = (0, 1),xrotation = 45,size = (700, 400), xticks = (1:length(param_labels), param_labels),
    colorbar_title = "ST", left_margin=2mm, bottom_margin=5mm)
savefig(p1, "results/plots/ST_2000_trx_model_t2.svg")
