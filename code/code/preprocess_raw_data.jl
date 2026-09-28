"""
This file reads in the tumor volumes from the Zhao et al. (2020) paper. 
It creates a dataframe from the .xlsx files and saves it as a .csv file. It smoothes 
the tumor volumes using a spline and saves the smoothed data to a .csv file.

The code was used to create the following figures:
- Figure S1: Tumor volumes in low-dose treatment group (18.5 MBq) with smoothed data
- Figure S2: Tumor volumes in high-dose treatment group (29.6 MBq) with smoothed data

"""

using XLSX
using CSV 
using DataFrames
using Statistics
using Plots
using StatsPlots
using StatsBase
using Pipe
using Dierckx
using Measures
using LaTeXStrings

### READ IN DATA AND CREATE DF FOR TREATMENT GROUPS

path = "/home/lara/Documents/GRK2762/Projects/explorations/Chen_data/"
fname_trx = "pdx2_trx_replicates_processed.xlsx"

## Treatment Groups 

xf_trx = XLSX.readxlsx(path*fname_trx)
data_trx = xf_trx[XLSX.sheetnames(xf_trx)[1]]

times_trx = [0,2,4,6,8,10,12,14,16,18,20,22,24,27,29,31,33,35,37,39,41,43,45,47,49,51]

## High-dose group
ids_high = string.(vec(data_trx["A"][4:10, :]))
measurements_high = data_trx["B:DA"][4:10, :]
cols_trx = collect(3:4:103) # select only column with tumor volumes
measurements_high = measurements_high[:, cols_trx]
measurements_high = permutedims(measurements_high)
df_trx_high = DataFrame(measurements_high, :auto)
rename!(df_trx_high, ids_high)
df_trx_high[!, "time_d"] = times_trx
df_trx_high_replicates = stack(df_trx_high, Not("time_d"), variable_name = "replicate_id", value_name="tumor_volume_mm3")

df_trx_high_replicates = df_trx_high_replicates[completecases(df_trx_high_replicates), :]

stats_high = combine(groupby(df_trx_high_replicates, :time_d),
                :tumor_volume_mm3 => mean => :vmean,
                :tumor_volume_mm3 => std  => :vstd)

df_trx_high_replicates_comp = leftjoin(df_trx_high_replicates, stats_high, on = :time_d)

CSV.write(path*"tv_trx_high_replicates.csv", df_trx_high_replicates_comp)

## Low-dose group
ids_low = string.(vec(data_trx["A"][12:20, :]))
measurements_low = data_trx["B:DA"][12:20, :]
measurements_low = measurements_low[:, cols_trx]
measurements_low = permutedims(measurements_low)
df_trx_low = DataFrame(measurements_low, :auto)
rename!(df_trx_low, ids_low)
df_trx_low[!, "time_d"] = times_trx
df_trx_low_replicates = stack(df_trx_low, Not("time_d"), variable_name = "replicate_id", 
value_name="tumor_volume_mm3")

df_trx_low_replicates = df_trx_low_replicates[completecases(df_trx_low_replicates), :]

stats_low = combine(groupby(df_trx_low_replicates, :time_d),
                :tumor_volume_mm3 => mean => :vmean,
                :tumor_volume_mm3 => std  => :vstd)

df_trx_low_replicates_comp = leftjoin(df_trx_low_replicates, stats_low, on = :time_d)

CSV.write(path*"tv_trx_low_replicates.csv", df_trx_low_replicates_comp)

## SMOOTH DATA 
path = "/home/lara/Documents/GRK2762/Projects/explorations/Chen_data/"
fname_low = "tv_trx_low_replicates.csv"
fname_high = "tv_trx_high_replicates.csv"

df_trx_low_replicates_comp = CSV.read(path*fname_low, DataFrame)
df_trx_high_replicates_comp = CSV.read(path*fname_high, DataFrame)

## 1) TVs in low-dose group: raw and smoothed data
smoothed_low_dose = []
s = [5000, 7500, 5000, 5000, 5000, 5000, 7500, 7500, 2500]
plt = plot(layout=grid(3,3), size=(900,600), left_margin=10mm, bottom_margin=2mm,
plot_title="Tumor Volumes in Treatment Group (18.5 MBq)", plot_titlefontsize=14)
for idx in eachindex(unique(df_trx_low_replicates_comp.replicate_id))
    df_tmp = df_trx_low_replicates_comp[df_trx_low_replicates_comp.replicate_id .== unique(df_trx_low_replicates_comp.replicate_id)[idx], :]

    spl = Spline1D(df_tmp.time_d, df_tmp.tumor_volume_mm3, s=s[idx])
    xs = df_tmp.time_d
    ys = spl(xs)

    plot!(plt, subplot=idx, xs, ys, label="Smoothed", color=:blue)
    plot!(plt, subplot=idx, df_tmp.time_d, df_tmp.tumor_volume_mm3, markersize=3,
    color=:red, st=:scatter, msc=:white, title="$(string(unique(df_trx_low_replicates_comp.replicate_id)[idx]))",
    label="Data", legend_font_pointsize=8, 
    titlefontsize=12)
    if idx in [7,8,9]
        xlabel!(plt, subplot=idx, "Time [days]", xguidefontsize=11)
    end
    if idx in [1,4,7]
        ylabel!(plt, subplot=idx, L"\mathrm{[mm^3]}", yguidefontrotation=-90, yguidefontsize=11)
    end
    push!(smoothed_low_dose, ys)
end
for i=1:6; plot!(plt[i],xformatter=_->""); end
plt
savefig(plt, path*"TVs_trx_low_replicates_smoothed.svg")

# Add smoothed tumor volume to CSV file
df_trx_low_replicates_comp[!, "tumor_volume_smoothed"] = vcat(smoothed_low_dose...)
CSV.write(path*"tv_trx_low_replicates.csv", df_trx_low_replicates_comp)

## 2) TVs in high-dose group: raw and smoothed data
smoothed_high_dose = []
s = [1500, 2500, 2500, 2500, 4000, 3500, 2500]
plt = plot(layout=grid(2,4), size=(1200,450), left_margin=13mm, bottom_margin=8mm,
plot_title="Tumor Volumes in Treatment Group (29.6 MBq)", plot_titlefontsize=14)
for idx in eachindex(unique(df_trx_high_replicates_comp.replicate_id))
    df_tmp = df_trx_high_replicates_comp[df_trx_high_replicates_comp.replicate_id .== unique(df_trx_high_replicates_comp.replicate_id)[idx], :]

    spl = Spline1D(df_tmp.time_d, df_tmp.tumor_volume_mm3, s=s[idx])
    xs = df_tmp.time_d
    ys = spl(xs)

    plot!(plt, subplot=idx, xs, ys, label="Smoothed", color=:blue)
    plot!(plt, subplot=idx, df_tmp.time_d, df_tmp.tumor_volume_mm3, markersize=3,
    color=:red, st=:scatter, msc=:white, title="$(string(unique(df_trx_high_replicates_comp.replicate_id)[idx]))",
    label="Data", legend_font_pointsize=8, 
    titlefontsize=12)
    # ylabel!(plt, "[ml]", yguidefontrotation=-90, yguidefontsize=11)
    # xlabel!(plt, "Time [days]", xguidefontsize=11)
    if idx in [4,5,6,7]
        xlabel!(plt, subplot=idx, "Time [days]", xguidefontsize=11)
    end
    if idx in [1,5]
        ylabel!(plt, subplot=idx, L"\mathrm{[mm^3]}", yguidefontrotation=-90, yguidefontsize=11)
    end
    push!(smoothed_high_dose, ys)
end
for i=1:4; plot!(plt[i],xformatter=_->""); end
plt
savefig(plt, path*"TVs_trx_high_replicates_smoothed.svg")

# Add smoothed tumor volume to CSV file
df_trx_high_replicates_comp[!, "tumor_volume_smoothed"] = vcat(smoothed_high_dose...)
CSV.write(path*"tv_trx_high_replicates.csv", df_trx_high_replicates_comp)
