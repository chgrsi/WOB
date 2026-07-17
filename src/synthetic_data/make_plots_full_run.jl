using Plots
using DataFrames
using CSV
using LaTeXStrings

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

const COLORS = Dict(
    "SAA" => :royalblue,
    "SAA-exact" => :navy,
    "RO-box" => :indianred1,
    "RO-mccormick" => :crimson,
    "DRO-exact" => :black,
    "DRO-approx-mccormick" => :darkorange4,
    "DRO-approx-box" => :darkorange,
)

const LABELS = Dict(
    "SAA" => "SAA",
    "SAA-exact" => "SAA-exact",
    "RO-box" => "RO-box",
    "RO-mccormick" => "RO-mcc",
    "DRO-exact" => "DRO-exact",
    "DRO-approx-mccormick" => "DRO-mcc",
    "DRO-approx-box" => "DRO-box",
)

const DRO_METHODS = ["DRO-exact", "DRO-approx-mccormick", "DRO-approx-box"]
const BENCHMARKS  = ["SAA", "SAA-exact", "RO-mccormick", "RO-box"]

function generate_plots()
    results_dir = "results/full_run"
    plots_dir = "plots/full_run"
    mkpath(plots_dir)
    N_train = 3

    df = CSV.read(joinpath(results_dir, "dro_metrics_N_$(N_train).csv"), DataFrame)

    line_width = 3
    my_xticks = 10.0 .^ (-4:1:4)
    logaxis(ylabel; kw...) = plot(; xscale = :log10, xticks = my_xticks,
                                  xminorgrid = true, minorgridalpha = 0.15,
                                  xlabel = L"Radius $\epsilon$", ylabel = ylabel, kw...)

    for scen in unique(df.Scenario)
        sub = filter(r -> r.Scenario == scen, df)
        bench(m) = only(eachrow(filter(r -> r.Method == m, sub)))

         panels = [(:Profit_Mean, :Profit_Q20, :Profit_Q80, "Mean profit [EUR/MWh]", "profit"),
                  (:N_opt, :N_Q20, :N_Q80, L"Optimal decision $n^{\star}$", "decision"),
                  (:CVaR5, :CVaR5_Q20, :CVaR5_Q80, "CVaR 5% [EUR/MWh]", "cvar"),
                  (:Sharpe, :Sharpe_Q20, :Sharpe_Q80, "Sharpe ratio", "sharpe"),
                  (:Sortino, :Sortino_Q20, :Sortino_Q80, "Sortino ratio", "sortino"),
                  (:NegFreq, :NegFreq_Q20, :NegFreq_Q80, "Negative profit freq", "negfreq"),
                  (:Regret, nothing, nothing, "Ex-post regret [EUR/MWh]", "regret")]

        for (col, lo, hi, ylabel, name) in panels
            p = logaxis(ylabel; legend = :outerright)
            for meth in DRO_METHODS
                # sort by epsilon
                d = sort(filter(r -> r.Method == meth, sub), :Epsilon)
                if lo === nothing
                    plot!(p, d.Epsilon, d[!, col], color = COLORS[meth], lw = line_width, label = LABELS[meth])
                else
                    plot!(p, d.Epsilon, d[!, col],
                          ribbon = (d[!, col] .- d[!, lo], d[!, hi] .- d[!, col]),
                          fillalpha = 0.15, color = COLORS[meth], lw = line_width,
                          label = LABELS[meth])
                end
            end
            for m in BENCHMARKS
                hline!(p, [bench(m)[col]], color = COLORS[m], ls = :dash, lw = 2, label = LABELS[m])
            end
            col == :N_opt && ylims!(p, 0.0, 1.0)
            savefig(p, joinpath(plots_dir, "$(name)_$(scen)_N_$(N_train).pdf"))
        end
    end
end

generate_plots()
