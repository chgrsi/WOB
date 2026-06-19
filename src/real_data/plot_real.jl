using Plots
using DataFrames
using CSV
using Dates
using LaTeXStrings
using Statistics

default(
    fontfamily     = "Computer Modern",
    titlefontsize  = 14,
    guidefontsize  = 12,
    tickfontsize   = 10,
    legendfontsize = 9,
    framestyle     = :box,
    grid           = true,
    gridalpha      = 0.3,
    margin         = 5Plots.mm,
)

const K_VAL = 1.2
const TAU_VAL = (K_VAL - 1.0) / (K_VAL + 1.0)

const LW = 2

const ZOOM_START = DateTime(2024, 6, 25)
const ZOOM_END   = DateTime(2024, 6, 28)

const COLORS = Dict(
    "SAA"             => :royalblue,
    "RO"              => :crimson,
    "DRO-exact"       => :black,
    "DRO-mccormick"   => :darkorange4,
    "DRO-box"         => :darkorange,
    "naive (n=0)"     => :lightgreen,
    "naive (n=0.5)"   => :mediumseagreen,
    "naive (n=1)"     => :forestgreen,
)

const NAIVE_STRATEGIES = [
    ("naive (n=0)",   0.0, :profit_naive_0),
    ("naive (n=0.5)", 0.5, :profit_naive_05),
    ("naive (n=1)",   1.0, :profit_naive_1),
]

const PROFIT_SERIES = [
    ("naive (n=0)",     :profit_naive_0),
    ("naive (n=0.5)",   :profit_naive_05),
    ("naive (n=1)",     :profit_naive_1),
    ("SAA",             :profit_saa),
    ("RO",              :profit_ro),
    ("DRO-exact",       :profit_dro_exact),
    ("DRO-mccormick",   :profit_dro_mcco),
    ("DRO-box",         :profit_dro_box),
]

const DECISION_SERIES = [
    ("SAA",             :n_saa),
    ("RO",              :n_ro),
    ("DRO-exact",       :n_dro_exact),
    ("DRO-mccormick",   :n_dro_mccormick),
    ("DRO-box",         :n_dro_box),
]

function parse_timestamp_column(column)
    cleaned = replace.(string.(column), r"\+00:00$" => "")

    return DateTime.(
        cleaned,
        dateformat"yyyy-mm-dd HH:MM:SS",
    )
end

function obs_profit(n::Float64, g::Float64, s::Float64, delta::Float64)
    cp = (s + delta) * TAU_VAL - delta
    cm = (s + delta) * TAU_VAL + delta

    return g * s -
           cp * max(g - n, 0.0) -
           cm * max(n - g, 0.0)
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
    sort!(data, :timestamp)

    return data
end

function add_naive_profits!(results::DataFrame, real_data::DataFrame)
    real_by_timestamp = Dict(row.timestamp => row for row in eachrow(real_data))

    for (_, _, column) in NAIVE_STRATEGIES
        results[!, column] = Vector{Float64}(undef, nrow(results))
    end

    for (i, result_row) in enumerate(eachrow(results))
        real_row = real_by_timestamp[result_row.timestamp]

        for (_, nomination, column) in NAIVE_STRATEGIES
            results[i, column] = obs_profit(
                nomination,
                real_row.g,
                real_row.s,
                real_row.delta,
            )
        end
    end

    return results
end

function add_actual_generation!(results::DataFrame, real_data::DataFrame)
    generation_by_timestamp = Dict(
        row.timestamp => row.g for row in eachrow(real_data)
    )

    results.g = Vector{Float64}(undef, nrow(results))

    for (i, row) in enumerate(eachrow(results))
        results.g[i] = generation_by_timestamp[row.timestamp]
    end

    return results
end

function daily_average_profits(results::DataFrame)
    return combine(
        groupby(results, :date),
        [column => mean => column for (_, column) in PROFIT_SERIES]...,
    )
end

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

        println(
            rpad(label, 20),
            "  mean = ", round(μ, digits = 4),
            "  std = ", round(σ, digits = 4),
            "  SR = ", round(sr, digits = 4),
        )
    end

    println()
end

function zoom_window(results::DataFrame)
    return results[ZOOM_START .<= results.timestamp .<= ZOOM_END, :]
end

function get_zoomed_results(results::DataFrame, warning_message::String)
    zoomed = zoom_window(results)

    if isempty(zoomed)
        @warn warning_message ZOOM_START ZOOM_END
        return results
    end

    return zoomed
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
    zoomed = get_zoomed_results(results, "No observations in zoom window")

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
    zoomed = get_zoomed_results(results, "No observations in decision zoom window")

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

function cvar_left_tail(values; alpha = 0.05)
    clean_values = collect(skipmissing(values))
    clean_values = filter(isfinite, clean_values)

    if isempty(clean_values)
        return NaN
    end

    sorted_values = sort(clean_values)
    n_tail = max(1, ceil(Int, alpha * length(sorted_values)))

    return mean(sorted_values[1:n_tail])
end

function plot_cvar(results::DataFrame, plot_dir::String; alpha = 0.05)
    labels = String[]
    cvars = Float64[]

    for (label, column) in PROFIT_SERIES
        push!(labels, label)
        push!(cvars, cvar_left_tail(results[!, column]; alpha = alpha))
    end

    p = bar(
        labels,
        cvars,
        ylabel = "CVaR $(round(Int, alpha * 100))% profit [EUR/MWh]",
        legend = false,
        color = [COLORS[label] for label in labels],
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

    filename = "cvar_$(round(Int, alpha * 100))pct_raw_profit.pdf"
    savefig(p, joinpath(plot_dir, filename))
end

function plot_sharpe(results::DataFrame, plot_dir::String)
    daily = daily_average_profits(results)

    labels = String[]
    sharpe_ratios = Float64[]

    for (label, column) in PROFIT_SERIES
        values = filter(isfinite, collect(skipmissing(daily[!, column])))

        μ = mean(values)
        σ = std(values)

        push!(labels, label)
        push!(sharpe_ratios, σ > 0 ? μ / σ : NaN)
    end

    p = bar(
        labels,
        sharpe_ratios,
        ylabel = "Sharpe ratio",
        legend = false,
        color = [COLORS[label] for label in labels],
        xrotation = 35,
    )

    hline!(p, [0.0], color = :gray, ls = :dot, lw = 1, label = false)

    savefig(p, joinpath(plot_dir, "sharpe_ratio.pdf"))
end

function generate_real_plots(input_file::String, results_dir::String)
    plot_dir = joinpath(results_dir, "plots")
    mkpath(plot_dir)

    results = load_results(results_dir)
    real_data = load_real_data(input_file)

    add_naive_profits!(results, real_data)
    add_actual_generation!(results, real_data)

    print_daily_sharpe_ratios(results, results_dir)
    plot_cumulative_profit(results, plot_dir)
    plot_cumulative_profit_zoom(results, plot_dir)
    plot_decisions(results, plot_dir)
    plot_cvar(results, plot_dir)
    plot_sharpe(results, plot_dir)

    println("Saved plots to $plot_dir")
end

generate_real_plots("real_data_hourly.csv", "results/real/hourly")
generate_real_plots("real_data_15min.csv", "results/real/15min")
