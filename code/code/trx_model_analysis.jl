"""
## PDE treatment model analysis

This code was used to generate the PDE model fits to the low dose treatment data (18.5 MBq) and plot 
the results. It also generates the plots for fit evaluation. 

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
using StatsPlots

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

rep_ids = unique(df_low.replicate_id)

for rep_id in rep_ids
    V_init = df_low[df_low.replicate_id .== rep_id, :tumor_volume_smoothed][1]*0.001

    @load path*"results/"*"solution_fit_times_$(rep_id).jld2" discrete_x discrete_t solc sol_total sol_viable sol_doomed solx̄ solx̄c sol_krx sol_al sol_bet

    ratio = sol_al[1,:]./sol_bet[1,:]
    print(ratio)

    plt = heatmap(discrete_t, discrete_x,sol_krx',xlabel = L"Time $t$ [days]",
    ylabel = L"Proliferation rate $\rho$ [day$^{-1}$]",colorbar_title = L"$k_{\mathrm{rx}}$",legend = false,
    colorbar = true, title="$(rep_id)", xlim=(0,20))
    # savefig(path*"results/plots/trx_exp/heatmap_krx_$(rep_id).svg")
    display(plt)

    xmin = 0.01
    xmax = 0.1

    idx = (discrete_x .>= xmin) .& (discrete_x .<= xmax)
    plt2 = plot(discrete_x[idx],ratio[idx],xlabel = L"\rho"*" [1/day]",
    ylabel = L"\mathrm{\alpha/\beta-ratio}"*" [Gy]",title = "$(rep_id)",xlim = (xmin, xmax))
    savefig(plt2, path*"results/plots/trx_exp/alpha_beta_ratio_$(rep_id).svg")
    display(plt2)


    idx = (discrete_x .>= xmin) .& (discrete_x .<= xmax)
    plt2 = plot(discrete_x[idx],sol_al[1,idx],xlabel = L"\rho"*" [1/day]", label=L"\alpha(\rho)",
    ylabel = "Radiosensitivity",title = "$(rep_id)",xlim = (xmin, xmax), color=:red)
    plot!(plt2, discrete_x[idx], sol_bet[1,idx], xlabel=L"\rho"*" [1/day]", label=L"\beta(\rho)",
    ylabel="Radiosensitivity", title="$(rep_id)", color=:blue)
    savefig(path*"results/plots/trx_exp/alpha_and_beta_$(rep_id).svg")
    display(plt2)

    plt2 = plot(discrete_x, sol_bet[1,:], xlabel=L"\rho"*" [1/day]",
    ylabel=L"\mathrm{\beta [Gy^{-2}]}", title="$(rep_id)")
    # savefig(path*"results/plots/trx_exp/beta_$(rep_id).svg")
    display(plt2)
end



rep_ids = unique(df_low.replicate_id)

for rep_id in rep_ids
    V_init = df_low[df_low.replicate_id .== rep_id, :tumor_volume_smoothed][1]*0.001
    times = df_low[df_low.replicate_id .== rep_id, :time_d]
    vols_vec = df_low[df_low.replicate_id .== rep_id, :tumor_volume_smoothed].*0.001

    @load path*"results/"*"solution_fit_times_$(rep_id).jld2" discrete_x discrete_t solc sol_total sol_viable sol_doomed solx̄ solx̄c sol_krx sol_al sol_bet

    plt = plot(size=(600,400))
    plot!(plt, times, vols_vec, seriestype=:scatter, marker=:xcross, color=:black,
    label="Data", xlabel="Time [days]", ylabel="Volume [ml]", title="")
    plot!(plt, discrete_t, sol_total, label="", xlabel="Time [days]",  zcolor=solx̄, color=palette(:vikO), msc=:white,
    ylabel="Volume [ml]", title="", st=:scatter, colorbar_title=L"\bar{\rho}_{total}")
    # savefig(plt,path*"results/plots/trx_exp/TV_low_$(rep_id).svg")
    # display(plt)

    plt = plot(size=(600,400))
    plot!(plt, discrete_t, sol_total, label=L"N_{total}", xlabel="Time [days]",  color=1,
    ylabel="Volume [ml]", title="")
    plot!(plt, discrete_t, sol_viable, label=L"N_v", xlabel="Time [days]",  color=2, 
    ylabel="Volume [ml]", title="")
    plot!(plt, discrete_t, sol_doomed, label=L"N_d", xlabel="Time [days]",  color=3, 
    ylabel="Volume [ml]", title="")
    # savefig(plt,path*"results/plots/trx_exp/TV_low_all_$(rep_id).svg")
    # display(plt)

    idx = 1:2:length(discrete_t)

    ymean = zeros(length(idx))
    for (j, i) in enumerate(idx)
        itp = LinearInterpolation(
            discrete_x,
            solc[i, :],
            extrapolation_bc = Interpolations.Flat()
        )
        ymean[j] = itp(solx̄[i])
    end

    ymean_cv = zeros(length(idx))
    for (j, i) in enumerate(idx)
        itp = LinearInterpolation(
            discrete_x,
            solc[i, :],
            extrapolation_bc = Interpolations.Flat()
        )
        ymean_cv[j] = itp(solx̄c[i])
    end

    cols = get(ColorSchemes.viridis, range(0.0, 1.0, length=length(idx)))

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
            solc[i, :],
            color=cols[j],
            fillrange=0,
            fillalpha=0.25,
            fillcolor=cols[j],
            lw=2
        )
    end

    display(plt)
    # savefig(plt, path*"results/plots/trx_exp/3d_density_short_$(rep_id).svg")

    plt = plot(
        size=(700, 500),
        xlabel=L"\rho\ [\mathrm{day}^{-1}]",
        ylabel=L"t\ [\mathrm{days}]",
        zlabel=L"C_v(t,\rho)",
        legend=false,
        camera=(45, 20)
    )

    surface!(
        plt,
        discrete_x,
        discrete_t,
        solc,
        color=:viridis,
        alpha=0.75
    )

    # display(plt)

    plt = plot(size=(400,300))
    plot!(plt, discrete_t, solx̄c, st=:scatter, msc=:white, xlabel="Time [days]", color=:orangered, 
    markersize=3, ylabel=L"\mathrm{\bar{\rho} \, [day^{-1}]}")
    plot!(plt, discrete_t, solx̄, st=:scatter, msc=:white, xlabel="Time [days]", color=:darkred,
    markersize=3, ylabel=L"\mathrm{\bar{\rho} \, [day^{-1}]}")
    # savefig(plt,path*"results/plots/trx_exp/rho_mean_low_$(rep_id).svg")
    # display(plt)
end



#Make diagnostic plots
mu = df_params[666,2]
plt_time = plot(layout=grid(3,3), size=(900,600))
plt_pred = plot(layout=grid(3,3), size=(900,600))
residuals = []
for i in 1:9
    rep_id = unique(df_low.replicate_id)[i]

    df_rep = df_low[df_low.replicate_id .== rep_id, :]

    V_init = df_rep.tumor_volume_smoothed[1] * 0.001

    data = df_rep.tumor_volume_smoothed .* 0.001 
    times = df_rep.time_d 

    prediction = solutions_units[i][N(t)]

    resids = data .- prediction

    push!(residuals, resids)
    plot!(plt_time, subplot=i, times, resids, seriestype=:scatter, color=:black,
    xlabel="Time [days]", ylabel="Residuals", title="$(rep_id)", label=false)

    scatter!(plt_pred, subplot=i, prediction,resids,xlabel = "Predicted Volume [ml]",
    ylabel = "Residuals",label = false, hline = [0], color=:black, title="$(rep_id)")
end
# savefig(plt_time,path*"results/plots/trx_exp/residuals_time_low_dose.svg")
display(plt_time)
# savefig(plt_pred,path*"results/plots/trx_exp/residuals_prediction_low_dose.svg")
display(plt_pred)

