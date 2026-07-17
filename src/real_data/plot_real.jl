using Plots
using DataFrames
using CSV
using Dates
using LaTeXStrings
using Statistics

default(
    fontfamily = "Computer Modern",
    titlefontsize = 14,
    guidefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 8,
    framestyle = :box,
    grid = true,
    gridalpha = 0.3,
    margin = 8Plots.mm, 
)


const LW = 2
const ZOOM_START = DateTime(2024, 6, 25)
const ZOOM_END = DateTime(2024, 6, 28)

const COLORS = Dict(
    "SAA" => :royalblue,      
    "RO" => :crimson,         
    "DRO-exact" => :black,           
    "DRO-mccormick" => :darkorange4,    
    "DRO-box" => :darkorange,    
    "naive (n=0)" => :lightgreen,     
    "naive (n=0.5)" => :mediumseagreen, 
    "naive (n=1)" => :forestgreen,    
)

const NAIVE_STRATEGIES = [
    ("naive (n=0)", 0.0, :profit_naive_0),
    ("naive (n=0.5)", 0.5, :profit_naive_05),
    ("naive (n=1)", 1.0, :profit_naive_1),
]

const PROFIT_SERIES = [
    ("naive (n=0)", :profit_naive_0),
    ("naive (n=0.5)", :profit_naive_05),
    ("naive (n=1)", :profit_naive_1),
    ("SAA", :profit_saa),
    ("RO", :profit_ro),
    ("DRO-exact",     :profit_dro_exact),
    ("DRO-mccormick", :profit_dro_mcco),
    ("DRO-box",       :profit_dro_box),
]

const DECISION_SERIES = [
    ("SAA", :n_saa),
    ("RO", :n_ro),
    ("DRO-exact", :n_dro_exact),
    ("DRO-mccormick", :n_dro_mccormick),
    ("DRO-box", :n_dro_box),
]

const METRIC_SERIES = vcat(
    [(label, n, col) for (label, n, col) in NAIVE_STRATEGIES],
    [("SAA", :n_saa, :profit_saa),
     ("RO", :n_ro, :profit_ro),
     ("DRO-exact", :n_dro_exact, :profit_dro_exact),
     ("DRO-mccormick", :n_dro_mccormick, :profit_dro_mcco),
     ("DRO-box", :n_dro_box, :profit_dro_box)],
)

function parse_timestamp_column(col)
    return DateTime.(
        replace.(string.(col), r"\+00:00$" => ""),
        dateformat"yyyy-mm-dd HH:MM:SS",
    )
end

function load_results(results_dir::String)
    results = CSV.read(joinpath(results_dir, "oos_profits.csv"), DataFrame)
    results.timestamp = DateTime.(results.timestamp)
    results.date = Date.(results.timestamp)
    return results
end

function load_real_data(input_file::String)
    data = CSV.read(input_file, DataFrame)
    data.timestamp = parse_timestamp_column(data[!, 1])
    Symbol("r+") in propertynames(data) && rename!(data, Symbol("r+") => :rp)
    Symbol("r-") in propertynames(data) && rename!(data, Symbol("r-") => :rm)
    sort!(data, [:timestamp])
    return data
end

function add_market_columns!(results::DataFrame, real_data::DataFrame)
    real_by_timestamp = Dict(row.timestamp => row for row in eachrow(real_data))
    for col in (:g, :s, :rp, :rm)
        results[!, col] = Vector{Float64}(undef, nrow(results))
    end
    for (i, row) in enumerate(eachrow(results))
        real_row = real_by_timestamp[row.timestamp]
        results[i, :g] = real_row.g
        results[i, :s] = real_row.s
        results[i, :rp] = real_row.rp
        results[i, :rm] = real_row.rm
    end
    return results
end

function add_naive_profits!(results::DataFrame)
    for (_, nomination, col) in NAIVE_STRATEGIES
        results[!, col] = settlement_profit.(nomination, results.g, results.s, results.rp, results.rm)
    end
    return results
end

settlement_profit(n, g, s, rp, rm) = n*s + rm*max(g - n, 0.0) - rp*max(n - g, 0.0)

function print_daily_sharpe_ratios(results::DataFrame, results_dir::String)
    daily = daily_average_profits(results)

    println()
    println("Daily Sharpe ratios: $results_dir ───")
    println("Computed from daily average profits.")

    for (label, column) in PROFIT_SERIES
        values = collect(skipmissing(daily[!, column]))
        values = filter(isfinite, values)

        μ = mean(values)
        σ = std(values)

        sr = σ > 0 ? μ / σ : NaN

        println(rpad(label, 20), "  mean = ", round(μ, digits=4),
                "  std = ", round(σ, digits=4),
                "  SR = ", round(sr, digits=4))
    end

    println()
end

function zoom_window(results::DataFrame)
    return results[
        ZOOM_START .<= results.timestamp .<= ZOOM_END,
        :,
    ]
end

function plot_cumulative_profit(results::DataFrame, plot_dir::String)
    p = plot(
        xlabel = "Date",
        ylabel = "Cumulative profit [EUR/MWh]",
        legend = :topleft,
    )

    for (label, column) in PROFIT_SERIES
        plot!(
            p,
            results.timestamp,
            cumsum(results[!, column]),
            color = COLORS[label],
            lw = LW,
            label = label,
        )
    end

    savefig(p, joinpath(plot_dir, "cumulative_profit.pdf"))
end

function plot_cumulative_profit_zoom(results::DataFrame, plot_dir::String)
    zoomed = zoom_window(results)

    if isempty(zoomed)
        @warn "No observations in zoom window" ZOOM_START ZOOM_END
        zoomed = results
    end

    p = plot(
        xlabel = "Date",
        ylabel = "Cumulative profit [EUR/MWh]",
        legend = :topleft,
    )

    for (label, column) in PROFIT_SERIES
        plot!(
            p,
            zoomed.timestamp,
            cumsum(zoomed[!, column]),
            color = COLORS[label],
            lw = LW,
            label = label,
        )
    end

    savefig(p, joinpath(plot_dir, "cumulative_profit_zoom.pdf"))
end

function plot_decisions(results::DataFrame, plot_dir::String)
    zoomed = zoom_window(results)

    if isempty(zoomed)
        @warn "No observations in decision zoom window" ZOOM_START ZOOM_END
        zoomed = results
    end

    p = plot(
        xlabel = "Date",
        ylabel = "Nomination decision",
        ylims = (0.0, 1.0),
        legend = :topleft,
    )

    plot!(
        p,
        zoomed.timestamp,
        zoomed.g,
        color = :red,
        lw = 3,
        ls = :dash,
        label = "actual generation",
    )

    for (label, column) in DECISION_SERIES
        plot!(
            p,
            zoomed.timestamp,
            zoomed[!, column],
            color = COLORS[label],
            lw = LW,
            label = label,
        )
    end

    savefig(p, joinpath(plot_dir, "decision_zoom.pdf"))
end

function cvar_left_tail(values; alpha=0.05)
    clean_values = collect(skipmissing(values))
    clean_values = filter(isfinite, clean_values)

    if isempty(clean_values)
        return NaN
    end

    sorted_values = sort(clean_values)
    n_tail = max(1, ceil(Int, alpha * length(sorted_values)))

    return mean(sorted_values[1:n_tail])
end

function plot_cvar(results::DataFrame, plot_dir::String; alpha=0.05)
    labels = String[]
    cvars = Float64[]

    for (label, column) in PROFIT_SERIES
        push!(labels, label)
        push!(cvars, cvar_left_tail(results[!, column]; alpha=alpha))
    end

    p = bar(
        labels,
        cvars,
        ylabel = "CVaR $(round(Int, alpha * 100))% profit [EUR/MWh]",
        legend = false,
        color = [COLORS[l] for l in labels],
        xrotation = 35,
    )

    hline!(
        p,
        [0.0],
        color = :gray,
        ls = :dot,
        lw = 1,
        label = false,
    )

    savefig(p, joinpath(plot_dir, "cvar_$(round(Int, alpha * 100))pct_raw_profit.pdf"))
end

function plot_sharpe(results::DataFrame, plot_dir::String)
    daily = daily_average_profits(results)

    labels = String[]
    srs    = Float64[]

    for (label, column) in PROFIT_SERIES
        values = filter(isfinite, collect(skipmissing(daily[!, column])))
        μ = mean(values)
        σ = std(values)
        push!(labels, label)
        push!(srs, σ > 0 ? μ / σ : NaN)
    end

    p = bar(
        labels,
        srs,
        ylabel = "Sharpe ratio",
        legend = false,
        color = [COLORS[l] for l in labels],
        xrotation = 35,
    )

    hline!(p, [0.0], color = :gray, ls = :dot, lw = 1, label = false)

    savefig(p, joinpath(plot_dir, "sharpe_ratio.pdf"))
end

function daily_average_profits(results::DataFrame)
    daily = combine(
        groupby(results, :date),
        [column => mean => column for (_, column) in PROFIT_SERIES]...,
    )

    return daily
end

function generate_real_plots(input_file::String, results_dir::String)
    plot_dir = joinpath(results_dir, "plots")
    mkpath(plot_dir)

    results = load_results(results_dir)
    real_data = load_real_data(input_file)

    add_market_columns!(results, real_data)
    add_naive_profits!(results)
    print_daily_sharpe_ratios(results, results_dir)
    plot_cumulative_profit(results, plot_dir)
    plot_cumulative_profit_zoom(results, plot_dir)
    plot_decisions(results, plot_dir)
    plot_cvar(results, plot_dir)
    plot_sharpe(results, plot_dir)

    println("Saved plots to $plot_dir")
end

generate_real_plots("real_data_hourly.csv", "results/real/hourly")
