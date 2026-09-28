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
using StatsPlots

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

@variables c(..) y(..) N(..) C_v(..) C_d(..) integrand(..) x̄(..) x̄c(..) krx_1(..) krx_2(..) krx_3(..) al(..) bet(..)

function create_schedule_problem(rep_id,df_params,df_low,df_dose_rate,df_biodb)
    @parameters x
    @parameters t_trx_1 t_trx_2 t_trx_3
    @parameters dose_1 dose_2 dose_3

    @variables c(..) y(..) N(..) C_v(..) C_d(..) integrand(..) x̄(..) x̄c(..) krx_1(..) krx_2(..) krx_3(..) al(..) bet(..)

    Dxx = Differential(x)^2
    Dx  = Differential(x)

    @load path*"results/"*"params_low_test_$(rep_id).jld2" params_lsqfit

    parameters = [t_trx_1, t_trx_2, t_trx_3, dose_1, dose_2, dose_3]

    mu = df_params[666, 2]  # 1/day

    V_init = df_low[df_low.replicate_id .== rep_id,:tumor_volume_smoothed][1] * 0.001

    rho_min = 0.0
    rho_max = 0.3

    s  = 3.26268
    kd = 16.1373

    k_rep_s   = params_lsqfit[1]
    a_alpha   = params_lsqfit[2]
    b_alpha_s = params_lsqfit[3]
    g_alpha_s = params_lsqfit[4]
    a_beta    = params_lsqfit[5]
    b_beta_s  = params_lsqfit[6]
    g_beta_s  = params_lsqfit[7]
    D_s       = params_lsqfit[8]
    kcl_s     = params_lsqfit[9]
    sigma_s   = params_lsqfit[10]

    # Transform dimensionless/fitted parameters
    kcl = kcl_s * mu
    D = D_s * mu^3
    b_alpha = b_alpha_s / mu
    g_alpha = g_alpha_s * mu
    b_beta = b_beta_s / mu
    g_beta = g_beta_s * mu
    k_rep = k_rep_s * mu
    sigma = sigma_s * mu

    R_0 = df_dose_rate[df_dose_rate.replicate .== rep_id,:initial_dose_rate_Gy_day][1]
    k_dec = df_biodb[4, 2] * 24  # 1/day

    # Dose rate corresponding to each administered activity.
    # dose_1, dose_2 and dose_3 are symbolic parameters.
    R_0_1 = R_0 * dose_1 / 18.5
    R_0_2 = R_0 * dose_2 / 18.5
    R_0_3 = R_0 * dose_3 / 18.5

    Ix = Integral(x in DomainSets.ClosedInterval(rho_min, rho_max))

    eq = [Dt(c(t,x)) ~ D * Dxx(c(t,x)) + x * c(t,x) - kd * x^s * c(t,x) - (t >= t_trx_1) * c(t,x) * krx_1(t,x) - (t >= t_trx_2) * c(t,x) * krx_2(t,x) - (t >= t_trx_3) * c(t,x) * krx_3(t,x)
        Dt(y(t,x)) ~ (t >= t_trx_1) * c(t,x) * krx_1(t,x) + (t >= t_trx_2) * c(t,x) * krx_2(t,x) + (t >= t_trx_3) * c(t,x) * krx_3(t,x) + kd * x^s * c(t,x) - kcl * y(t,x)
        N(t) ~ Ix(c(t,x)) + Ix(y(t,x))
        C_v(t) ~ Ix(c(t,x))
        C_d(t) ~ Ix(y(t,x))
        integrand(t,x) ~ x * c(t,x)
        x̄(t) ~ Ix(integrand(t,x))/(Ix(c(t,x)) + Ix(y(t,x)))
        x̄c(t) ~ Ix(integrand(t,x))/Ix(c(t,x))
        krx_1(t,x) ~ R_0_1 * exp(-(k_dec)*(t-t_trx_1)) * al(t,x) + ((2*bet(t,x)*R_0_1^2)/(k_rep - k_dec)) * (exp(-2*k_dec*(t-t_trx_1))-exp(-(k_rep+k_dec)*(t-t_trx_1)))
        krx_2(t,x) ~ R_0_2 * exp(-(k_dec)*(t-t_trx_2)) * al(t,x) + ((2*bet(t,x)*R_0_2^2)/(k_rep - k_dec)) * (exp(-2*k_dec*(t-t_trx_2))-exp(-(k_rep+k_dec)*(t-t_trx_2)))
        krx_3(t,x) ~ R_0_3 * exp(-(k_dec)*(t-t_trx_3)) * al(t,x) + ((2*bet(t,x)*R_0_3^2)/(k_rep - k_dec)) * (exp(-2*k_dec*(t-t_trx_3))-exp(-(k_rep+k_dec)*(t-t_trx_3)))
        al(t,x) ~ a_alpha/(1+exp(-b_alpha*(x-g_alpha)))
        bet(t,x) ~ a_beta/(1+exp(-b_beta*(x-g_beta)))]

    domains = [t ∈ (0.0, 90.0), x ∈ (rho_min, rho_max)]

    bcs = [c(0,x) ~ (V_init/(sqrt(2*pi)*sigma))*exp(-((x-mu)^2)/(2*sigma^2)), 
    y(0,x) ~ 0.0, Dx(c(t, rho_min)) ~ 0.0, Dx(c(t, rho_max)) ~ 0, 
    Dx(y(t, rho_min)) ~ 0.0, Dx(y(t, rho_max)) ~ 0]

    @named pdesys = PDESystem(eq, bcs, domains, [t,x], 
    [c(t,x), y(t,x), N(t), C_v(t), C_d(t), integrand(t,x), x̄(t), x̄c(t), krx_1(t,x), krx_2(t,x), krx_3(t,x), al(t,x), bet(t,x)],
    parameters,
    defaults = Dict(t_trx_1 => 0, t_trx_2 => 30, t_trx_3 => 60, dose_1 => 12, dose_2 => 12, dose_3 => 6))

    discretization = MOLFiniteDifference([x => 120], t)

    prob = discretize(pdesys,discretization)

    return prob
end


function evaluate_schedule(prob,Δt12,Δt23,doses)

    # Treatment times
    t1_val = 0.0
    t2_val = Δt12
    t3_val = Δt12 + Δt23

    # Dose order
    d1, d2, d3 = doses

    # New parameter values
    p_new = [t1_val, t2_val, t3_val, d1, d2, d3]

    p1 = copy(prob.p)
    p1[1] .= p_new

    prob_new = remake(prob, p=p1)

    sol = solve(
        prob_new,
        Tsit5(),
        saveat=[t1_val, t2_val, t3_val, 90.0]
    )

    V_t1 = sol[N(t)][1]
    rho_t1 = sol[x̄(t)][1]
    V_t2 = sol[N(t)][2]
    rho_t2 = sol[x̄(t)][2]
    V_t3 = sol[N(t)][3]
    rho_t3 = sol[x̄(t)][3]
    V90 = sol[N(t)][end]
    rho90 = sol[x̄(t)][end]

    return V_t1, rho_t1, V_t2, rho_t2, V_t3, rho_t3, V90, rho90
end

prob = create_schedule_problem(223,df_params,df_low,df_dose_rate,df_biodb)

orders = [(12.0, 12.0, 6.0),(6.0, 12.0, 12.0),(12.0, 6.0, 12.0)]
intervals = 20.0:1.0:40.0

results = DataFrame(
    dose_order = String[],
    dose_1 = Float64[],
    dose_2 = Float64[],
    dose_3 = Float64[],
    Δt12 = Float64[],
    Δt23 = Float64[],
    t2 = Float64[],
    t3 = Float64[],
    V_t1 = Float64[],
    rho_t1 = Float64[],
    V_t2 = Float64[],
    rho_t2 = Float64[],
    V_t3 = Float64[],
    rho_t3 = Float64[],
    V90 = Float64[],
    rho90 = Float64[]
)

for doses in orders

    order_name = "$(Int(doses[1]))+$(Int(doses[2]))+$(Int(doses[3]))"

    for Δt12 in intervals
        for Δt23 in intervals

            V_t1, rho_t1, V_t2, rho_t2, V_t3, rho_t3, V90, rho90 = evaluate_schedule(prob,Δt12,Δt23,doses)

            t2 = Δt12
            t3 = Δt12 + Δt23

            push!(results,
                (order_name,doses[1],doses[2],doses[3],Δt12,Δt23,t2,t3,V_t1, rho_t1, V_t2, rho_t2, V_t3, rho_t3, V90, rho90))
        end
    end
end

# get best results with respect to final tumor volume
best = results[argmin(results.V90), :]
println("Best schedule:")
println(best)

@save path*"/results/frac_schedule_optimization.jl2d" results
@load path*"/results/frac_schedule_optimization.jl2d" results

best50 = sort(results, [order(:V90)])[1:50, :]

best_by_order = DataFrame()
for order in unique(results.dose_order)
    df = results[results.dose_order .== order, :]
    i = argmin(df.V90)
    push!(best_by_order, df[i, :])
end

sort!(best_by_order, [order(:V90)])

best20_by_order = DataFrame()
for order in unique(results.dose_order)
    df = results[results.dose_order .== order, :]
    df_sorted = sort(df, :V90)
    best20 = df_sorted[1:min(20, nrow(df_sorted)), :]
    append!(best20_by_order, best20)
end

function pareto_optimal(df)
    n = nrow(df)
    is_pareto = trues(n)

    for i in 1:n
        for j in 1:n
            if i == j
                continue
            end
            dominates =
                df.V90[j] <= df.V90[i] &&
                df.rho90[j] <= df.rho90[i] &&
                (
                    df.V90[j] < df.V90[i] ||
                    df.rho90[j] < df.rho90[i]
                )
            if dominates
                is_pareto[i] = false
                break
            end
        end
    end
    return df[is_pareto, :]
end

pareto = pareto_optimal(results)




function pareto_opposite(df)
    n = nrow(df)
    is_pareto = trues(n)

    for i in 1:n
        for j in 1:n
            if i == j
                continue
            end

            # j dominates i if j is >= i in both objectives
            # and strictly greater in at least one
            dominates =
                df.V90[j] >= df.V90[i] &&
                df.rho90[j] >= df.rho90[i] &&
                (
                    df.V90[j] > df.V90[i] ||
                    df.rho90[j] > df.rho90[i]
                )

            if dominates
                is_pareto[i] = false
                break
            end
        end
    end

    return df[is_pareto, :]
end

pareto_back = pareto_opposite(results)



# Plot 1: all schedules with pareto front 

scatter(
    results.rho90,
    results.V90,
    group = results.dose_order,
    markersize = 3,
    markerstrokewidth = 0,
    legend = :topright
)

# Pareto front
pareto_sorted = sort(pareto, :rho90)

plot!(
    pareto_sorted.rho90,
    pareto_sorted.V90,
    color = :black,
    linewidth = 2,
    label = "Pareto front"
)

# Pareto points, retaining dose-order colors
scatter!(
    pareto.rho90,
    pareto.V90,
    group = pareto.dose_order,
    markersize = 4,
    ylabel = "Final Tumor Volume [ml]",
    xlabel = "Final "*L"\bar{\rho}_{total}"*" [1/day]",
    markerstrokecolor = :black,
    markerstrokewidth = 1.5,
    label = ""
)
savefig(path*"results/plots/trx_exp/fractionation_schemes_pareto_front.svg")


## plot 1 b pareto front and back 

scatter(
    results.rho90,
    results.V90,
    group = results.dose_order,
    markersize = 3,
    markerstrokewidth = 0,
    legend = :topright
)

# Pareto front (minimize rho90 and V90)
pareto_sorted = sort(pareto, :rho90)

plot!(
    pareto_sorted.rho90,
    pareto_sorted.V90,
    color = :black,
    linewidth = 2,
    linestyle = :solid,
    label = "Pareto front (min.)"
)

# Pareto back (maximize rho90 and V90)
pareto_back_sorted = sort(pareto_back, :rho90)

plot!(
    pareto_back_sorted.rho90,
    pareto_back_sorted.V90,
    color = :red,
    linewidth = 2,
    label = "Pareto front (max.)"
)

# Pareto-front points, retaining dose-order colors
scatter!(
    pareto.rho90,
    pareto.V90,
    group = pareto.dose_order,
    markersize = 4,
    ylabel = "Final Tumor Volume [ml]",
    xlabel = "Final "*L"\bar{\rho}_{total}"*" [1/day]",
    markerstrokecolor = :black,
    markerstrokewidth = 1.5,
    label = ""
)

# Pareto-back points, retaining dose-order colors
scatter!(
    pareto_back.rho90,
    pareto_back.V90,
    group = pareto_back.dose_order,
    markersize = 4,
    markerstrokecolor = :black,
    markerstrokewidth = 1.5,
    label = ""
)


# Highlight selected minimum-front schedule
scatter!(
    [sample_min.rho90],
    [sample_min.V90],
    markersize = 8,
    marker = :star,
    markercolor = :white,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    label = "Selected schedules"
)

# Highlight selected maximum-front schedule
scatter!(
    [sample_max.rho90],
    [sample_max.V90],
    markersize = 8,
    marker = :star,
    markercolor = :white,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    label = ""
)

savefig(path*"results/plots/trx_exp/fractionation_schemes_pareto_fronts_selected_schedules_V225.svg")


## find replicates/treatment schedules with 2.25 ml TV 

target_V90 = 2.25

# Closest point on the minimizing Pareto front
i_min = argmin(abs.(pareto.V90 .- target_V90))
sample_min = pareto[i_min, :]

# Closest point on the maximizing Pareto front
i_max = argmin(abs.(pareto_back.V90 .- target_V90))
sample_max = pareto_back[i_max, :]

println("Min Pareto front:")
println(sample_min)

println("\nMax Pareto front:")
println(sample_max)


## 1 c pareto min and max highlighted 

target_V90 = 2.25

# Select closest samples
i_min = argmin(abs.(pareto.V90 .- target_V90))
sample_min = pareto[i_min, :]

i_max = argmin(abs.(pareto_back.V90 .- target_V90))
sample_max = pareto_back[i_max, :]

# Sort Pareto fronts for connecting lines
pareto_sorted = sort(pareto, :rho90)
pareto_back_sorted = sort(pareto_back, :rho90)

# Plot all schedules
scatter(
    results.rho90,
    results.V90,
    group = results.dose_order,
    markersize = 3,
    markerstrokewidth = 0,
    alpha = 0.5,
    legend = :topright,
    xlabel = "Final "*L"\bar{\rho}_{total}"*" [1/day]",
    ylabel = "Final Tumor Volume [ml]"
)

# Minimum Pareto front
plot!(
    pareto_sorted.rho90,
    pareto_sorted.V90,
    color = :black,
    linewidth = 2,
    linestyle = :solid,
    label = "Pareto front (min.)"
)

# Maximum Pareto front
plot!(
    pareto_back_sorted.rho90,
    pareto_back_sorted.V90,
    color = :black,
    linewidth = 2,
    linestyle = :dash,
    label = "Pareto front (max.)"
)

# Highlight selected minimum-front schedule
scatter!(
    [sample_min.rho90],
    [sample_min.V90],
    markersize = 6,
    marker = :star,
    markercolor = :white,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    label = "Selected schedules"
)

# Highlight selected maximum-front schedule
scatter!(
    [sample_max.rho90],
    [sample_max.V90],
    markersize = 6,
    marker = :star,
    markercolor = :white,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    label = ""
)

savefig(path*"results/plots/trx_exp/selected_pareto_schedules.svg")





# Plot 2: best 20 tumor volumes and rho for each group 
scatter(
    best20_by_order.rho90,
    best20_by_order.V90,
    group = best20_by_order.dose_order,
    xlabel = "Final "*L"\bar{\rho}_{total}\ [\mathrm{day}^{-1}]",
    ylabel = "Final Tumor Volume [ml]",
    legend = :topright,
    markersize = 6,
    markerstrokewidth = 0,
    size = (750, 550)
)
savefig(path*"results/fractionation_optimization_best_by_order.svg")


### Pareto schedule optimization

selected_schedules = DataFrame(
    schedule = ["Pareto min", "Pareto max"],
    dose_order = [
        sample_min.dose_order,
        sample_max.dose_order
    ],
    dose_1 = [
        sample_min.dose_1,
        sample_max.dose_1
    ],
    dose_2 = [
        sample_min.dose_2,
        sample_max.dose_2
    ],
    dose_3 = [
        sample_min.dose_3,
        sample_max.dose_3
    ],
    Δt12 = [
        sample_min.Δt12,
        sample_max.Δt12
    ],
    Δt23 = [
        sample_min.Δt23,
        sample_max.Δt23
    ],
    t2 = [
        sample_min.t2,
        sample_max.t2
    ],
    t3 = [
        sample_min.t3,
        sample_max.t3
    ],
    V90 = [
        sample_min.V90,
        sample_max.V90
    ],
    rho90 = [
        sample_min.rho90,
        sample_max.rho90
    ]
)

selected_schedules




Δt23_range = 10.0:1.0:45.0

timing_results_225 = DataFrame(
    schedule = String[],
    dose_order = String[],
    Δt12 = Float64[],
    Δt23 = Float64[],
    t2 = Float64[],
    t3 = Float64[],
    V_t2 = Float64[],
    rho_t2 = Float64[],
    V_t3 = Float64[],
    rho_t3 = Float64[],
    V90 = Float64[],
    rho90 = Float64[]
)


for row in eachrow(selected_schedules)

    for Δt23 in Δt23_range

        Δt12 = row.Δt12
        doses = (row.dose_1, row.dose_2, row.dose_3)

        V_t1, rho_t1,
        V_t2, rho_t2,
        V_t3, rho_t3,
        V90, rho90 =
            evaluate_schedule(prob, Δt12, Δt23, doses)

        t2 = Δt12
        t3 = Δt12 + Δt23

        push!(
            timing_results_225,
            (
                row.schedule,
                row.dose_order,
                Δt12,
                Δt23,
                t2,
                t3,
                V_t2,
                rho_t2,
                V_t3,
                rho_t3,
                V90,
                rho90
            )
        )
    end
end
@save path*"results/fractionation_schedules_timing_sensitivity_V225.jld2" timing_results_225

plt = plot(
    xlabel = L"\Delta t_{23}\;[\mathrm{days}]",
    ylabel = "Final Tumor Volume [ml]",
    legend = :best
)

for schedule in unique(timing_results_225.schedule)

    df = timing_results_225[timing_results_225.schedule .== schedule, :]

    plot!(plt,
        df.Δt23,
        df.V90,
        linewidth = 2,
        label = schedule
    )
end
display(plt)


plt2 = plot(
    xlabel = L"\Delta t_{23}\;[\mathrm{days}]",
    ylabel = L"Final $\bar{\rho}_{total}$ [1/day]",
    legend = :best
)

for schedule in unique(timing_results_225.schedule)

    df = timing_results_225[timing_results_225.schedule .== schedule, :]

    plot!(plt2,
        df.Δt23,
        df.rho90,
        linewidth = 2,
        label = schedule
    )
end
display(plt2)


timing_results_225.ΔV90 = similar(timing_results_225.V90)
timing_results_225.Δrho90 = similar(timing_results_225.rho90)

for i in 1:nrow(timing_results_225)

    schedule = timing_results_225.schedule[i]

    baseline = selected_schedules[
        selected_schedules.schedule .== schedule,
        :
    ][1, :]

    timing_results_225.ΔV90[i] =
        timing_results_225.V90[i] / baseline.V90

    timing_results_225.Δrho90[i] =
        timing_results_225.rho90[i] / baseline.rho90
end


plot(
    xlabel = L"\Delta t_{23} - \Delta t_{23}^{\mathrm{original}}\;[\mathrm{days}]",
    ylabel = "Fold change (Volume)",
    xlims = (-10, 15),
    legend = :best,
    ylims = (0.8, 1.4)
)

for schedule in unique(timing_results_225.schedule)

    df = timing_results_225[timing_results_225.schedule .== schedule, :]

    # Original Δt23 for this schedule
    original_Δt23 = selected_schedules[
        selected_schedules.schedule .== schedule,
        :
    ].Δt23[1]

    # Shift Δt23 so that original treatment interval = 0
    shifted_Δt23 = df.Δt23 .- original_Δt23

    plot!(
        shifted_Δt23,
        df.ΔV90,
        linewidth = 2,
        label = schedule
    )

end

hline!([1], color = :black, linestyle = :dash, label = "")
vline!([0], color = :black, linestyle = :dot, linewidth = 1.5)
savefig(path*"results/plots/trx_exp/fractionation_schedules_timing_sensitivity_vol_fold_change_v225_centered.svg")




plot(
    xlabel = L"\Delta t_{23} - \Delta t_{23}^{\mathrm{original}}\;[\mathrm{days}]",
    ylabel = "Fold change "*L"\bar{\rho}_{total}",
    xlims = (-10, 15),
    legend = :best,
    ylims = (0.7, 1.2)
)

for schedule in unique(timing_results_225.schedule)

    df = timing_results_225[timing_results_225.schedule .== schedule, :]

    # Original Δt23 for this schedule
    original_Δt23 = selected_schedules[
        selected_schedules.schedule .== schedule,
        :
    ].Δt23[1]

    # Shift Δt23 so that original treatment interval = 0
    shifted_Δt23 = df.Δt23 .- original_Δt23

    plot!(
        shifted_Δt23,
        df.Δrho90,
        linewidth = 2,
        label = schedule
    )

end

hline!([1], color = :black, linestyle = :dash, label = "")
vline!([0], color = :black, linestyle = :dot, linewidth = 1.5)
savefig(path*"results/plots/trx_exp/fractionation_schedules_timing_sensitivity_rho_fold_change_v225_centered.svg")
