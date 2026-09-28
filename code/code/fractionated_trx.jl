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
path = "/mnt/nfs/lara/pde_model/"
file_data = "data/tv_trx_low_replicates.csv"
file_params = "data/fit_params_control_smooth.csv"
file_dose_rate = "data/2026-08_initial_dose_rates_low.csv"
file_biodb = "data/biodistribution_params_chen.csv"

df_low = CSV.read(path*file_data, DataFrame)
df_params = CSV.read(path*file_params, DataFrame)
df_dose_rate = CSV.read(path*file_dose_rate, DataFrame)
df_biodb = CSV.read(path*file_biodb, DataFrame)


rep_id = 223
@load path*"results/"*"params_low_test_$(rep_id).jld2" params_lsqfit

@parameters x
@variables c(..) y(..) N(..) C_v(..) C_d(..) integrand(..) x̄(..) x̄c(..) krx_1(..) krx_2(..) krx_3(..) al(..) bet(..)

Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dx = Differential(x)

mu = df_params[666,2] # 1/day, population-level mean rho from control group

V_init = df_low[df_low.replicate_id .== rep_id, :tumor_volume_smoothed][1]*0.001

rho_min = 0.0
rho_max = 0.3 # 1/day

# model parameters 

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

# treatment exploration parameters 

R_0 = df_dose_rate[df_dose_rate.replicate .== rep_id, :initial_dose_rate_Gy_day][1] # 18.5 MBq
k_dec = df_biodb[4,2] * 24 # population-level mean decay rate from biodb data, in 1/day

# 12 + 12 + 6
# delta12 = 21 
# delta 23 = 26 

t_trx_1 = 0
t_trx_2 = 21
t_trx_3 = 49

dose_1 = 12
dose_2 = 6
dose_3 = 12

R_0_1 = R_0 * dose_1/18.5
R_0_2 = R_0 * dose_2/18.5 
R_0_3 = R_0 * dose_3/18.5

Ix = Integral(x in DomainSets.ClosedInterval(rho_min, rho_max))

eq  = [Dt(c(t,x)) ~ D*Dxx(c(t, x)) + x*c(t,x) - kd * x^s * c(t,x) - (t>=t_trx_1)*c(t,x)*krx_1(t,x) - (t>=t_trx_2)*c(t,x)*krx_2(t,x) - (t>=t_trx_3)*c(t,x)*krx_3(t,x)
    Dt(y(t,x)) ~ (t>=t_trx_1)*c(t,x)*krx_1(t,x) + (t>=t_trx_2)*c(t,x)*krx_2(t,x) + (t>=t_trx_3)*c(t,x)*krx_3(t,x) +  kd * x^s * c(t,x) - kcl * y(t,x)
    N(t) ~ Ix(c(t,x)) + Ix(y(t,x))
    C_v(t) ~ Ix(c(t,x))
    C_d(t) ~ Ix(y(t,x))
    integrand(t,x) ~ x * c(t, x)
    x̄(t) ~ Ix(integrand(t,x))/(Ix(c(t,x)) + Ix(y(t,x)))
    x̄c(t) ~ Ix(integrand(t,x))/(Ix(c(t,x)))
    krx_1(t,x) ~ R_0_1 * exp(-(k_dec)*(t-t_trx_1)) * al(t,x) + ((2*bet(t,x)*R_0_1^2)/(k_rep - k_dec)) * (exp(-2*k_dec*(t-t_trx_1))-exp(-(k_rep+k_dec)*(t-t_trx_1)))
    krx_2(t,x) ~ R_0_2 * exp(-(k_dec)*(t-t_trx_2)) * al(t,x) + ((2*bet(t,x)*R_0_2^2)/(k_rep - k_dec)) * (exp(-2*k_dec*(t-t_trx_2))-exp(-(k_rep+k_dec)*(t-t_trx_2)))
    krx_3(t,x) ~ R_0_3 * exp(-(k_dec)*(t-t_trx_3)) * al(t,x) + ((2*bet(t,x)*R_0_3^2)/(k_rep - k_dec)) * (exp(-2*k_dec*(t-t_trx_3))-exp(-(k_rep+k_dec)*(t-t_trx_3)))
    al(t,x) ~ a_alpha/(1+exp(-b_alpha*(x-g_alpha)))
    bet(t,x) ~ a_beta/(1+exp(-b_beta*(x-g_beta)))]


t0 = 0
t1 = 90
domains = [t ∈ (t0, t1),
x ∈ (rho_min, rho_max)]

bcs = [c(0,x) ~ (V_init/(sqrt(2*pi)*sigma))*exp(-((x-mu)^2)/(2*sigma^2)), 
y(0,x) ~ 0.0, Dx(c(t, rho_min)) ~ 0.0, Dx(c(t, rho_max)) ~ 0, 
Dx(y(t, rho_min)) ~ 0.0, Dx(y(t, rho_max)) ~ 0]

@named pdesys = PDESystem(eq, bcs, domains, [t,x], 
    [c(t,x), y(t,x), N(t), C_v(t), C_d(t), integrand(t,x), x̄(t), x̄c(t), krx_1(t,x), krx_2(t,x), krx_3(t,x), al(t,x), bet(t,x)])
dx=0.1
discretization = MOLFiniteDifference([x => 120], t)

prob = discretize(pdesys,discretization)
sol = solve(prob, Tsit5(), saveat=t0:1:t1)

discrete_x = sol[x]
discrete_t = sol[t]
solc_min = sol[c(t,x)]
sol_total_min = sol[N(t)]
solx̄_min = sol[x̄(t)]
solx̄c_min = sol[x̄c(t)]

# dose 1 = 12, dose 2 = 6, dose 3 = 12 (t2 = 27, t3 = 63)
# @save path*"results/"*"solution_frac_pareto_min_223.jld2" discrete_x discrete_t solc_min sol_total_min solx̄_min solx̄c_min
# @load path*"results/"*"solution_frac_pareto_min_223.jld2" discrete_x discrete_t solc_min sol_total_min solx̄_min solx̄c_min
# 12+6+12, 0, 34, 56
@save path*"results/"*"solution_frac_pareto_min_223_v225.jld2" discrete_x discrete_t solc_min sol_total_min solx̄_min solx̄c_min

prob = discretize(pdesys,discretization)
sol = solve(prob, Tsit5(), saveat=t0:1:t1)

discrete_x = sol[x]
discrete_t = sol[t]
solc_max = sol[c(t,x)]
sol_total_max = sol[N(t)]
solx̄_max = sol[x̄(t)]
solx̄c_max = sol[x̄c(t)]

# dose 1 = 12, dose 2 = 6, dose 3 = 12 (t2 = 21, t3 = 58)
# @save path*"results/"*"solution_frac_pareto_max_223_v225.jld2" discrete_x discrete_t solc_max sol_total_max solx̄_max solx̄c_max
# @load path*"results/"*"solution_frac_pareto_max_223_v225.jld2" discrete_x discrete_t solc_max sol_total_max solx̄_max solx̄c_max
# 12+6+12, 0, 21, 49
@save path*"results/"*"solution_frac_pareto_max_223_v225.jld2" discrete_x discrete_t solc_max sol_total_max solx̄_max solx̄c_max


plt = plot(size=(600,400), left_margin = 2mm, bottom_margin=2mm)
plot!(plt, discrete_t, sol_total_min, label="Pareto (min)", xlabel="Time [days]", 
ylabel="Volume [ml]", title="", color=:blue)
plot!(plt, discrete_t, sol_total_max, label="Pareto (max)", xlabel="Time [days]", 
ylabel="Volume [ml]", title="", color=:red)
savefig(plt,path*"results/plots/trx_exp/TV_pareto_comparison_min_max_v225.svg")
display(plt)

idx = 1:length(discrete_t)
cols = collect(get(ColorSchemes.viridis,range(0.0, 1.0, length=length(idx))))

pink = colorant"pink"
cols[1] = pink
cols[35] = pink
cols[57] = pink

plt = plot(
    size=(700, 500),
    xlabel=L"\rho\ [\mathrm{day}^{-1}]",
    ylabel=L"t\ [\mathrm{days}]",
    zlabel=L"C_v(t,\rho)",
    legend=false,
    camera=(30, 20)
)


for (i, j) in zip(reverse(idx), reverse(eachindex(idx)))
    plot!(
        plt,
        discrete_x,
        fill(discrete_t[i], length(discrete_x)),
        solc_min[i, :],
        color=cols[j],
        fillrange=0,
        fillalpha=0.25,
        fillcolor=cols[j],
        lw=2,
        xlims=(0,0.2)
    )
end
display(plt)
savefig(path*"results/plots/trx_exp/density_pareto_min_v225_scaled.svg")



idx = 1:length(discrete_t)
cols = collect(get(ColorSchemes.viridis,range(0.0, 1.0, length=length(idx))))

pink = colorant"pink"
cols[1] = pink
cols[22] = pink
cols[50] = pink

plt = plot(
    size=(700, 500),
    xlabel=L"\rho\ [\mathrm{day}^{-1}]",
    ylabel=L"t\ [\mathrm{days}]",
    zlabel=L"C_v(t,\rho)",
    legend=false,
    camera=(30, 20)
)


for (i, j) in zip(reverse(idx), reverse(eachindex(idx)))
    plot!(
        plt,
        discrete_x,
        fill(discrete_t[i], length(discrete_x)),
        solc_max[i, :],
        color=cols[j],
        fillrange=0,
        fillalpha=0.25,
        fillcolor=cols[j],
        lw=2,
        xlims=(0,0.2)
    )
end
display(plt)
savefig(path*"results/plots/trx_exp/density_pareto_max_v225_scaled.svg")


plt = plot(size=(600,400))
plot!(plt, discrete_t, solx̄c_min, xlabel="Time [days]", color=:blue, label=L"\bar{\rho}_{v}",
markersize=3, ylabel=L"\mathrm{\bar{\rho} \, [day^{-1}]}", linestyle=:solid)
plot!(plt, discrete_t, solx̄_min, xlabel="Time [days]", color=:blue, label=L"\bar{\rho}_{total}",
markersize=3, ylabel=L"\mathrm{\bar{\rho} \, [day^{-1}]}", linestyle=:dash)
plot!(plt, discrete_t, solx̄c_max, xlabel="Time [days]", color=:red, label=L"\bar{\rho}_{v}",
markersize=3, ylabel=L"\mathrm{\bar{\rho} \, [day^{-1}]}", linestyle=:solid)
plot!(plt, discrete_t, solx̄_max, xlabel="Time [days]", color=:red, label=L"\bar{\rho}_{total}",
markersize=3, ylabel=L"\mathrm{\bar{\rho} \, [day^{-1}]}", linestyle=:dash)
savefig(plt,path*"results/plots/trx_exp/rho_mean_pareto_comparison_min_max_v225.svg")
display(plt)
