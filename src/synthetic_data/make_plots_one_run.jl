using Plots
using DataFrames
using CSV
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
    margin = 5Plots.mm,
)

function generate_one_run_plots()
    results_dir = "results/one_run"
    plots_dir = "plots/one_run"
    mkpath(plots_dir)

    df = CSV.read(joinpath(results_dir, "dro_metrics.csv"), DataFrame)
    refs = CSV.read(joinpath(results_dir, "references.csv"), DataFrame)

    colors = Dict(
        "DRO-exact" => :black,
        "DRO-approx-mccormick" => :chocolate4,
        "DRO-approx-box" => :chocolate1,
    )

    ref_colors = Dict(
        "SAA" => :dodgerblue,
        "SAA-exact" => :navy,
        "RO-mccormick" => :indianred1,
        "RO-box" => :darkred,
    )

    line_width = 3
    my_xticks = 10.0 .^ (-4:1:4)

    # Decision
    p1 = plot(xscale = :log10, xticks = my_xticks, xminorgrid = true, minorgridalpha = 0.15,
              xlabel = L"Radius $\epsilon$", ylabel = L"Optimal decision $n^{\star}$",
              ylims = (0.0, 1.0), legend = :topleft)
    # Profit
    p2 = plot(xscale = :log10, xticks = my_xticks, xminorgrid = true, minorgridalpha = 0.15,
              xlabel = L"Radius $\epsilon$", ylabel = "Mean profit [EUR/MWh]", legend = :bottomleft)
    # Sharpe
    p3 = plot(xscale = :log10, xticks = my_xticks, xminorgrid = true, minorgridalpha = 0.15,
              xlabel = L"Radius $\epsilon$", ylabel = "Sharpe ratio", legend = :bottomleft)
    # Sortino
    p4 = plot(xscale = :log10, xticks = my_xticks, xminorgrid = true, minorgridalpha = 0.15,
              xlabel = L"Radius $\epsilon$", ylabel = "Sortino ratio", legend = :bottomleft)
    # CVaR
    p5 = plot(xscale = :log10, xticks = my_xticks, xminorgrid = true, minorgridalpha = 0.15,
              xlabel = L"Radius $\epsilon$", ylabel = "CVaR 5% [EUR/MWh]", legend = :bottomleft)

    for meth in ["DRO-exact", "DRO-approx-mccormick", "DRO-approx-box"]
        sub = sort(filter(r -> r.method == meth, df), :epsilon)
        col = colors[meth]
        plot!(p1, sub.epsilon, sub.n, color = col, lw = line_width, label = meth)
        plot!(p2, sub.epsilon, sub.mean, color = col, lw = line_width, label = meth)
        plot!(p3, sub.epsilon, sub.sharpe, color = col, lw = line_width, label = meth)
        plot!(p4, sub.epsilon, sub.sortino, color = col, lw = line_width, label = meth)
        plot!(p5, sub.epsilon, sub.cvar, color = col, lw = line_width, label = meth)
    end

    # benchmark
    for r in eachrow(refs)
        col = ref_colors[r.method]
        hline!(p1, [r.n], color = col, ls = :dash, lw = 2, label = r.method)
        hline!(p2, [r.mean], color = col, ls = :dash, lw = 2, label = r.method)
        hline!(p3, [r.sharpe], color = col, ls = :dash, lw = 2, label = r.method)
        hline!(p4, [r.sortino], color = col, ls = :dash, lw = 2, label = r.method)
        hline!(p5, [r.cvar], color = col, ls = :dash, lw = 2, label = r.method)
    end

    savefig(p1, joinpath(plots_dir, "decision.pdf"))
    savefig(p2, joinpath(plots_dir, "profit.pdf"))
    savefig(p3, joinpath(plots_dir, "sharpe.pdf"))
    savefig(p4, joinpath(plots_dir, "sortino.pdf"))
    savefig(p5, joinpath(plots_dir, "cvar.pdf"))
    println("Saved plots to $plots_dir")

    # table
    println()
    header = rpad("method", 24) * rpad("epsilon", 10) * rpad("n", 8) *
             rpad("p_mean", 10) * rpad("sharpe", 8) * rpad("sortino", 10) * rpad("CVaR", 10) *
             rpad("neg freq", 10) * rpad("p_deficit", 11) * "p_surplus"
    println(header)

    row(name, eps, n, m) = println(
        rpad(name, 24), rpad(eps, 10), 
        rpad(round(n, digits = 3), 8),
        rpad(round(m.mean, digits = 2), 10), 
        rpad(round(m.sharpe, digits = 3), 8),
        rpad(round(m.sortino, digits = 3), 10),
        rpad(round(m.cvar, digits = 2), 10), 
        rpad(round(m.neg_freq, digits = 3), 10),
        rpad(round(m.profit_deficit, digits = 2), 11), 
        round(m.profit_surplus, digits = 2))

    for r in eachrow(refs)
        row(r.method, "-", r.n, r)
    end

    for meth in ["DRO-exact", "DRO-approx-mccormick", "DRO-approx-box"]
        sub = sort(filter(r -> r.method == meth, df), :epsilon)
        j = argmax(sub.sharpe)
        row("$meth", round(sub.epsilon[j], sigdigits = 3), sub.n[j], sub[j, :])
    end
end

generate_one_run_plots()
